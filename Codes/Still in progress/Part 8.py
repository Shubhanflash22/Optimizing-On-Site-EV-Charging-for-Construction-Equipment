import os
import cv2
from pathlib import Path
from lxml import etree
from ultralytics import YOLO
from tqdm import tqdm
import sys

# ============================
# Configuration
# ============================

VIDEO_PATH = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Video construction\Day 2-Oct 21, 2025\Excavator\videos\Day_2.mp4"
CVAT_XML   = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Workflow\8.Creating clips from CVAT Annotations\annotations.xml"
YOLO_MODEL = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Workflow\8.Creating clips from CVAT Annotations\best.pt"
OUTPUT_DIR = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Workflow\8.Creating clips from CVAT Annotations\Dataset_Resnet_3"

# Paper-aligned parameters
CLIP_LENGTH     = 16    # frames per clip  (paper: 16)
TARGET_FPS      = 25    # resample target  (paper: 25)
CLIP_STRIDE     = 3     # how many resampled frames to advance between clips
                        # stride=1 → 94% overlap, ~3x more clips than needed
                        # stride=3 → 81% overlap, reasonable dataset size
MIN_CONFIDENCE  = 0.5
CROP_SIZE       = 112   # paper: 112

# ============================
# CVAT XML parser
# ============================

def parse_cvat_xml(xml_path):
    """Returns dict: original_frame_index -> activity_label."""
    print("Parsing CVAT XML...")
    tree = etree.parse(xml_path)
    root = tree.getroot()

    frame_labels = {}
    for image in root.findall(".//image"):
        frame_num = int(image.get("id"))
        for box in image.findall("box"):
            frame_labels[frame_num] = box.get("label")

    print(f"  Labels for {len(frame_labels)} frames")
    from collections import Counter
    for act, cnt in sorted(Counter(frame_labels.values()).items()):
        print(f"    {act}: {cnt}")
    return frame_labels


# ============================
# Resample map builder
# ============================

def build_resample_map(original_fps, target_fps, total_original_frames):
    """resample_map[i] = original frame index for resampled frame i.
    Uses round-nearest so Part 10 inference can use the identical logic."""
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


# ============================
# Main extraction  —  streaming, memory-efficient
# ============================

def extract_clips_with_yolo(video_path, yolo_model_path, frame_labels, output_dir,
                            clip_length=16, clip_stride=3,
                            min_confidence=0.5, crop_size=112):
    """
    Single sequential pass through the video.  Uses a pointer into the sorted
    resample_map instead of a set:
      - Handles ANY FPS ratio correctly, including cases where round() maps
        two resampled indices to the same original frame (interval < 2).
      - Skips non-target frames with cap.grab() (no decoding cost).
      - Only decodes + runs YOLO on frames we actually need.
    """

    # --- YOLO ---
    print("\nLoading YOLO model...")
    model = YOLO(yolo_model_path)

    # --- video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames at {original_fps:.2f} FPS")

    # --- resample map ---
    resample_map = build_resample_map(original_fps, TARGET_FPS, total_frames)
    print(f"Resampling -> {len(resample_map)} frames at {TARGET_FPS} FPS "
          f"(interval = {original_fps / TARGET_FPS:.4f})")

    # --- output dirs ---
    activities = set(frame_labels.values())
    for act in activities:
        os.makedirs(os.path.join(output_dir, act.lower()), exist_ok=True)
    print(f"Activities: {sorted(activities)}")

    # ---------------------------------------------------------------
    # Pointer-based streaming loop
    #
    # resample_map is sorted (monotonically non-decreasing).
    # map_ptr points to the next resampled frame we need to fill.
    # current_frame_idx tracks where we are in the original video.
    #
    # When current_frame_idx == resample_map[map_ptr]:
    #   - Decode the frame, run YOLO, append crop to buffer.
    #   - Advance map_ptr.  If the NEXT map entry points to the same
    #     original index (duplicate), loop again WITHOUT reading a new frame.
    # When current_frame_idx < resample_map[map_ptr]:
    #   - Use grab() to skip forward cheaply (no decode).
    # ---------------------------------------------------------------

    frame_buffer   = []       # list of (cropped_img, activity) or None
    map_ptr        = 0        # pointer into resample_map
    current_frame_idx = 0
    current_frame_decoded = None  # cache so we don't re-decode for duplicates

    skipped_no_label     = 0
    skipped_no_detection = 0
    skipped_low_conf     = 0

    print("\nProcessing video (streaming)...")
    pbar = tqdm(total=total_frames, desc="Scanning")

    while map_ptr < len(resample_map):
        target_orig_idx = resample_map[map_ptr]

        # --- skip forward to the target frame using grab() (no decode) ---
        while current_frame_idx < target_orig_idx:
            if not cap.grab():
                break                          # video ended early
            current_frame_idx += 1
            pbar.update(1)

        if current_frame_idx != target_orig_idx:
            break                              # couldn't reach target — video too short

        # --- decode this frame (only once, even if map has duplicates here) ---
        if current_frame_decoded is None or current_frame_idx != target_orig_idx:
            ret, current_frame_decoded = cap.read()
            if not ret:
                break
            # cap.read() already advanced the internal position by 1,
            # so we account for that after the inner loop below.

        # --- inner loop: process all resampled entries that map to this original frame ---
        while map_ptr < len(resample_map) and resample_map[map_ptr] == current_frame_idx:
            activity = frame_labels.get(current_frame_idx)

            if activity is None:
                skipped_no_label += 1
                frame_buffer.append(None)
            else:
                results = model(current_frame_decoded, imgsz=480, verbose=False)

                if len(results[0].boxes) == 0:
                    skipped_no_detection += 1
                    frame_buffer.append(None)
                else:
                    best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                    conf = float(best_box.conf[0])

                    if conf < min_confidence:
                        skipped_low_conf += 1
                        frame_buffer.append(None)
                    else:
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                        h, w = current_frame_decoded.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 <= x1 or y2 <= y1:
                            skipped_no_detection += 1
                            frame_buffer.append(None)
                        else:
                            cropped = cv2.resize(
                                current_frame_decoded[y1:y2, x1:x2],
                                (crop_size, crop_size)
                            )
                            frame_buffer.append((cropped, activity))

            map_ptr += 1

        # We consumed this original frame; move position forward
        pbar.update(1)
        current_frame_idx += 1
        current_frame_decoded = None  # invalidate cache

    # finish progress bar
    remaining = total_frames - current_frame_idx
    if remaining > 0:
        pbar.update(remaining)
    pbar.close()
    cap.release()

    valid_count = sum(1 for x in frame_buffer if x is not None)
    print(f"Captured {len(frame_buffer)} resampled frames ({valid_count} with valid detections)")

    # ---------------------------------------------------------------
    # Emit clips  (stride = clip_stride over the resampled buffer)
    # ---------------------------------------------------------------
    print(f"\nEmitting {clip_length}-frame clips (stride={clip_stride}) ...")

    clip_counter = {}

    for i in tqdm(range(0, len(frame_buffer) - clip_length + 1, clip_stride), desc="Saving clips"):
        window = frame_buffer[i: i + clip_length]

        # All 16 entries must be valid detections
        if any(item is None for item in window):
            continue

        # All 16 must share the same activity label
        acts = set(a for _, a in window)
        if len(acts) != 1:
            continue

        activity_name = acts.pop().lower()
        clip_counter.setdefault(activity_name, 0)

        clip_dir = os.path.join(output_dir, activity_name,
                                f"clip_{clip_counter[activity_name]:05d}")
        os.makedirs(clip_dir, exist_ok=True)

        for j, (crop_frame, _) in enumerate(window):
            cv2.imwrite(os.path.join(clip_dir, f"frame_{j:03d}.jpg"), crop_frame)

        clip_counter[activity_name] += 1

    # --- stats ---
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"\n  Resampled frames captured : {len(frame_buffer)}")
    print(f"  Skipped (no label)        : {skipped_no_label}")
    print(f"  Skipped (no detection)    : {skipped_no_detection}")
    print(f"  Skipped (low confidence)  : {skipped_low_conf}")

    total_clips = sum(clip_counter.values())
    print(f"\n  Clips per activity:")
    for act, count in sorted(clip_counter.items()):
        print(f"    {act}: {count} clips  ({count * clip_length} frames)")
    print(f"\n  Total clips  : {total_clips}")
    print(f"  Temporal res : {TARGET_FPS} FPS  (each clip = {clip_length / TARGET_FPS:.2f} s)")
    print(f"  Dataset dir  : {output_dir}")

    return clip_counter


# ============================
# Main
# ============================

if __name__ == "__main__":
    print("=" * 50)
    print("YOLO + CVAT  ->  ResNet Dataset  (streaming, FPS-corrected)")
    print("=" * 50)

    print("\nStep 1: Parsing CVAT annotations...")
    frame_labels = parse_cvat_xml(CVAT_XML)
    if not frame_labels:
        print("ERROR: No labels found!")
        sys.exit(1)

    print("\nStep 2: Extracting clips (resampled to 25 FPS)...")
    clip_stats = extract_clips_with_yolo(
        VIDEO_PATH,
        YOLO_MODEL,
        frame_labels,
        OUTPUT_DIR,
        clip_length    = CLIP_LENGTH,
        clip_stride    = CLIP_STRIDE,
        min_confidence = MIN_CONFIDENCE,
        crop_size      = CROP_SIZE
    )

    if sum(clip_stats.values()) == 0:
        print("\nWARNING: No clips created. Check YOLO detections / CVAT labels / confidence.")
    else:
        print("\n" + "=" * 50)
        print("SUCCESS — dataset ready for Part 9 training")
        print("=" * 50)