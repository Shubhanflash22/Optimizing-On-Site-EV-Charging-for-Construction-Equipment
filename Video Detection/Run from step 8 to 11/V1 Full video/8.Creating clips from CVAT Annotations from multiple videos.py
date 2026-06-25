"""
dataset_builder.py
==================
YOLO + CVAT  ->  ResNet-3D Dataset Builder

Pipeline:
  1. Parse CVAT XML annotations to get per-frame activity labels.
  2. Resample each video from its native FPS to TARGET_FPS using a
     round-nearest map (matches inference-time resampling exactly).
  3. Stream through the video: grab() to skip, read() only on needed frames.
  4. Run YOLO on each needed frame; crop & resize the best detection box.
  5. Slide a 16-frame window over the resampled buffer to emit clips.
  6. Assign contiguous groups of clips (GROUP_SIZE) to train or val
     so that temporal chunks stay together while both splits see every video.
  7. Save clips under  OUTPUT_DIR / {train|val} / {activity} / clip_NNNNN /
     using a single persistent clip counter so numbering never resets across
     videos and files are never overwritten.

Key fixes vs previous version:
  - YOLO model loaded ONCE before the video loop (not once per video).
  - Clip counter passed in and mutated in-place (no per-video reset /
    overwrite on disk).
  - Group-based train/val split is intentional: contiguous temporal chunks
    from every video appear in both splits.
"""

import os
import random
from collections import Counter

import cv2
import numpy as np
from lxml import etree
from tqdm import tqdm
from ultralytics import YOLO

# ============================================================
# Reproducibility
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# Configuration
# ============================================================

# Each tuple is (video_path, cvat_xml_path).
BASE_DIR = r"/data/shubhan_avik_work/Targeted_run"

VIDEO_XML_PAIRS = [
    (f"{BASE_DIR}/Day_2.mp4",   f"{BASE_DIR}/Day_2_annotations.xml"),
    (f"{BASE_DIR}/Day_3.mp4",   f"{BASE_DIR}/Day_3_annotations.xml"),
    (f"{BASE_DIR}/Day_4_1.mp4",   f"{BASE_DIR}/Day_4_1_annotations.xml"),
    (f"{BASE_DIR}/TC_00011.mp4",   f"{BASE_DIR}/TC_00011_annotations.xml"),
    (f"{BASE_DIR}/TC_00012.mp4",   f"{BASE_DIR}/TC_00012_annotations.xml"),
    (f"{BASE_DIR}/TC_00013.mp4",   f"{BASE_DIR}/TC_00013_annotations.xml"),
    (f"{BASE_DIR}/TC_00014.mp4",   f"{BASE_DIR}/TC_00014_annotations.xml"),
    (f"{BASE_DIR}/TC_00015.mp4",   f"{BASE_DIR}/TC_00015_annotations.xml"),
    (f"{BASE_DIR}/TC_00016.mp4",   f"{BASE_DIR}/TC_00016_annotations.xml"),
    (f"{BASE_DIR}/TC_00019.mp4",   f"{BASE_DIR}/TC_00019_annotations.xml"),
    (f"{BASE_DIR}/TC_00021.mp4",   f"{BASE_DIR}/TC_00021_annotations.xml")
]

YOLO_MODEL = f"{BASE_DIR}/best.pt"
OUTPUT_DIR = f"{BASE_DIR}/Dataset_Ten_days"

# --- Paper-aligned clip parameters ---
CLIP_LENGTH = 16      # frames per clip (C3D / ResNet-3D standard)
TARGET_FPS  = 25      # resample target FPS (paper: 25)
CLIP_STRIDE = 3       # resampled-frame stride between clip start points
                      #   stride=3 -> ~81% overlap; reasonable dataset density

# --- Detection ---
MIN_CONFIDENCE = 0.5  # YOLO box confidence threshold
CROP_SIZE      = 112  # spatial crop size fed to ResNet-3D (paper: 112x112)

# --- Train / val split ---
TRAIN_RATIO = 0.85    # probability that a group is assigned to train
GROUP_SIZE  = 20      # number of *clips* per contiguous group before re-rolling
                      #   keeps temporal chunks together within each split

# ============================================================
# CVAT XML parser
# ============================================================

def parse_cvat_xml(xml_path: str) -> dict:
    """
    Parse a CVAT video-annotation XML file.

    Returns
    -------
    dict
        { original_frame_index (int) -> activity_label (str) }
        Only frames that contain at least one annotated bounding box are
        included.  If a frame has multiple boxes the last one wins (CVAT
        typically exports one box per frame for single-object tasks).
    """
    print(f"  Parsing XML: {xml_path}")
    tree = etree.parse(xml_path)
    root = tree.getroot()

    frame_labels = {}
    for image in root.findall(".//image"):
        frame_num = int(image.get("id"))
        for box in image.findall("box"):
            frame_labels[frame_num] = box.get("label")

    label_counts = Counter(frame_labels.values())
    print(f"  Annotated frames : {len(frame_labels)}")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count} frames")

    return frame_labels


# ============================================================
# Resample map builder
# ============================================================

def build_resample_map(original_fps: float,
                       target_fps: float,
                       total_original_frames: int) -> list:
    """
    Build an index map from resampled frame positions to original frame indices.

    Uses round-nearest so that the inference pipeline can apply the identical
    logic and always land on the same frames.

    Parameters
    ----------
    original_fps : float
        Native FPS of the source video.
    target_fps : float
        Desired output FPS (TARGET_FPS, e.g. 25).
    total_original_frames : int
        Total frame count reported by cv2.CAP_PROP_FRAME_COUNT.

    Returns
    -------
    list of int
        resample_map[i] = original frame index for resampled frame i.
        The list is monotonically non-decreasing (may contain duplicates when
        original_fps < target_fps).
    """
    interval = original_fps / target_fps
    resample_map = []
    i = 0
    while True:
        orig_idx = int(round(i * interval))
        if orig_idx >= total_original_frames:
            break
        resample_map.append(orig_idx)
        i += 1
    return resample_map


# ============================================================
# Core extraction function
# ============================================================

def extract_clips_with_yolo(
    video_path:     str,
    model:          YOLO,       # pre-loaded model, shared across all videos
    frame_labels:   dict,
    output_dir:     str,
    clip_counter:   dict,       # {"train": {act: int}, "val": {act: int}}
                                # mutated in-place so numbering is global
    clip_length:    int   = 16,
    clip_stride:    int   = 3,
    min_confidence: float = 0.5,
    crop_size:      int   = 112,
) -> None:
    """
    Single-pass streaming extraction for one video.

    Steps
    -----
    1. Build a resample map for this video's native FPS.
    2. Stream through frames using grab() (no decode) to skip non-target
       frames cheaply; decode only frames referenced by resample_map.
    3. Run YOLO on each decoded frame; keep the highest-confidence box.
    4. Accumulate a frame_buffer of (cropped_img, activity) | None entries.
    5. Slide a clip_length window with clip_stride over the buffer.
    6. Assign each clip to train or val based on GROUP_SIZE block assignment.
    7. Write frames to disk under output_dir/{split}/{activity}/clip_NNNNN/.

    Parameters
    ----------
    video_path : str
        Path to the source MP4.
    model : YOLO
        Pre-loaded Ultralytics YOLO model (loaded once in __main__).
    frame_labels : dict
        Output of parse_cvat_xml — maps original frame index -> label.
    output_dir : str
        Root output directory (OUTPUT_DIR).
    clip_counter : dict
        Persistent counter shared across all videos.
        Structure: {"train": {"digging": 42, ...}, "val": {...}}
        Mutated in-place so clip indices are globally unique.
    clip_length : int
        Number of frames per clip.
    clip_stride : int
        Step size (in resampled frames) between consecutive clip windows.
    min_confidence : float
        Minimum YOLO detection confidence to accept a box.
    crop_size : int
        Output spatial size (height = width) after cropping and resizing.
    """

    # ----------------------------------------------------------
    # Open video
    # ----------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video  : {total_frames} frames @ {original_fps:.2f} FPS")

    # ----------------------------------------------------------
    # Build resample map
    # ----------------------------------------------------------
    resample_map = build_resample_map(original_fps, TARGET_FPS, total_frames)
    print(f"  Resamp : {len(resample_map)} frames @ {TARGET_FPS} FPS "
          f"(interval = {original_fps / TARGET_FPS:.4f})")

    # Pre-create output subdirectories for every known activity
    all_activities = set(frame_labels.values())
    for split in ("train", "val"):
        for act in all_activities:
            os.makedirs(os.path.join(output_dir, split, act.lower()), exist_ok=True)

    # ----------------------------------------------------------
    # Streaming frame buffer fill
    #
    # We use a pointer (map_ptr) into resample_map rather than a set so
    # that duplicate entries (when original_fps < TARGET_FPS) are handled
    # naturally without re-reading the same frame from disk.
    # ----------------------------------------------------------
    frame_buffer          = []   # list of (crop_img, activity_str) | None
    map_ptr               = 0    # next position to fill in resample_map
    current_frame_idx     = 0    # where the video cursor currently sits
    current_frame_decoded = None # cached decode to avoid re-reads for dups

    skipped_no_label     = 0
    skipped_no_detection = 0
    skipped_low_conf     = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("  Scanning frames...")
    pbar = tqdm(total=total_frames, desc="  Frames", leave=False)

    while map_ptr < len(resample_map):
        target_orig_idx = resample_map[map_ptr]

        # --- skip forward using grab() (no decode cost) ---
        while current_frame_idx < target_orig_idx:
            if not cap.grab():
                break
            current_frame_idx += 1
            pbar.update(1)

        if current_frame_idx != target_orig_idx:
            break  # video ended before we reached the target

        # --- decode this frame (once even if resample_map has duplicates) ---
        ret, current_frame_decoded = cap.read()
        if not ret:
            break

        # --- inner loop: handle all resampled entries at this original frame ---
        while map_ptr < len(resample_map) and resample_map[map_ptr] == current_frame_idx:

            activity = frame_labels.get(current_frame_idx)

            if activity is None:
                # Frame was not annotated in CVAT
                skipped_no_label += 1
                frame_buffer.append(None)

            else:
                results = model(current_frame_decoded, imgsz=480, verbose=False)

                if len(results[0].boxes) == 0:
                    # YOLO found no object in this frame
                    skipped_no_detection += 1
                    frame_buffer.append(None)

                else:
                    best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                    conf     = float(best_box.conf[0])

                    if conf < min_confidence:
                        skipped_low_conf += 1
                        frame_buffer.append(None)

                    else:
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                        h, w = current_frame_decoded.shape[:2]

                        # Clamp to frame boundaries
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 > x1 and y2 > y1:
                            crop = cv2.resize(
                                current_frame_decoded[y1:y2, x1:x2],
                                (crop_size, crop_size)
                            )
                            frame_buffer.append((crop, activity))
                        else:
                            # Degenerate box (zero area after clamping)
                            skipped_no_detection += 1
                            frame_buffer.append(None)

            map_ptr += 1

        # cap.read() advanced the cursor by 1
        pbar.update(1)
        current_frame_idx += 1

    # finish progress bar up to total_frames
    remaining = total_frames - current_frame_idx
    if remaining > 0:
        pbar.update(remaining)
    pbar.close()
    cap.release()

    valid = sum(1 for x in frame_buffer if x is not None)
    print(f"  Buffer : {len(frame_buffer)} resampled frames "
          f"({valid} with valid detections)")
    print(f"  Skipped: no_label={skipped_no_label}  "
          f"no_detect={skipped_no_detection}  low_conf={skipped_low_conf}")

    # ----------------------------------------------------------
    # Clip generation with group-based train / val assignment
    #
    # group_id   = clip_index // GROUP_SIZE
    # Each group is randomly assigned to "train" or "val" once, so
    # contiguous temporal chunks stay in the same split.  Because groups are
    # assigned per-clip-index (not per-frame-index), the split is stable
    # regardless of how many clips survive the None / mixed-label filters.
    # ----------------------------------------------------------
    group_split: dict = {}   # group_id -> "train" | "val"
    clip_index  = 0          # counts valid clips emitted *for this video*
                             # (used only for group assignment, not for naming)

    print("  Emitting clips...")
    for i in tqdm(range(0, len(frame_buffer) - clip_length + 1, clip_stride),
                  desc="  Clips", leave=False):

        window = frame_buffer[i : i + clip_length]

        # All 16 slots must contain a valid detection
        if any(x is None for x in window):
            continue

        # All 16 frames must share the same activity label
        acts = {a for _, a in window}
        if len(acts) != 1:
            continue

        activity = acts.pop().lower()

        # --- group assignment ---
        group_id = clip_index // GROUP_SIZE
        if group_id not in group_split:
            group_split[group_id] = "train" if random.random() < TRAIN_RATIO else "val"
        split = group_split[group_id]

        # --- persistent clip counter ensures globally unique folder names ---
        clip_counter[split].setdefault(activity, 0)
        clip_idx  = clip_counter[split][activity]
        clip_dir  = os.path.join(output_dir, split, activity, f"clip_{clip_idx:05d}")
        os.makedirs(clip_dir, exist_ok=True)

        for j, (frame_img, _) in enumerate(window):
            cv2.imwrite(os.path.join(clip_dir, f"frame_{j:03d}.jpg"), frame_img)

        clip_counter[split][activity] += 1
        clip_index += 1

    # Per-video summary
    print(f"  Clips emitted this video: {clip_index}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("YOLO + CVAT  ->  ResNet-3D Dataset Builder")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Videos : {len(VIDEO_XML_PAIRS)}")
    print(f"Clip   : {CLIP_LENGTH} frames @ {TARGET_FPS} FPS  "
          f"(stride={CLIP_STRIDE})")
    print(f"Split  : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)} "
          f"train/val  (group size={GROUP_SIZE} clips)")
    print("=" * 60)

    # ----------------------------------------------------------
    # Load YOLO model once — shared across all 12 videos
    # Loading inside the loop would reload weights 12 times needlessly
    # ----------------------------------------------------------
    print("\nLoading YOLO model (once)...")
    model = YOLO(YOLO_MODEL)
    print("YOLO ready.\n")

    # ----------------------------------------------------------
    # Single persistent clip counter for the entire run
    # Passed by reference into each call so numbering is global
    # and no two clips ever share the same folder path
    # ----------------------------------------------------------
    global_clip_counter = {
        "train": {},   # { activity_name: int }
        "val"  : {},
    }

    # ----------------------------------------------------------
    # Process each video in sequence
    # ----------------------------------------------------------
    for video_idx, (video_path, xml_path) in enumerate(VIDEO_XML_PAIRS):

        print(f"\n{'─' * 60}")
        print(f"Video {video_idx + 1}/{len(VIDEO_XML_PAIRS)} : {video_path}")
        print(f"{'─' * 60}")

        frame_labels = parse_cvat_xml(xml_path)

        if not frame_labels:
            print("  WARNING: no annotations found — skipping.")
            continue

        extract_clips_with_yolo(
            video_path     = video_path,
            model          = model,               # shared, not reloaded
            frame_labels   = frame_labels,
            output_dir     = OUTPUT_DIR,
            clip_counter   = global_clip_counter, # shared, mutated in-place
            clip_length    = CLIP_LENGTH,
            clip_stride    = CLIP_STRIDE,
            min_confidence = MIN_CONFIDENCE,
            crop_size      = CROP_SIZE,
        )

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    total_clips = 0
    for split in ("train", "val"):
        split_total = sum(global_clip_counter[split].values())
        total_clips += split_total
        print(f"\n{split.upper()}  ({split_total} clips):")
        for act, count in sorted(global_clip_counter[split].items()):
            frames = count * CLIP_LENGTH
            print(f"  {act:<20s}: {count:>5d} clips  ({frames:>7d} frames)")

    print(f"\nTotal clips : {total_clips}")
    print(f"Temporal res: {TARGET_FPS} FPS — "
          f"each clip = {CLIP_LENGTH / TARGET_FPS:.2f} s")
    print(f"Dataset root: {OUTPUT_DIR}")
    print("=" * 60)
    print("Done — dataset ready for training.")