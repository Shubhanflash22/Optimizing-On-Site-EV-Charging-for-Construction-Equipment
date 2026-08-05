"""
dataset_builder.py  —  VAL-MANIFEST EDITION
============================================
YOLO + CVAT  ->  ResNet-3D Dataset Builder

Identical to the original Step 8 EXCEPT two JSON files are written
after all clips are emitted:

  OUTPUT_DIR / val_manifest.json
      Full per-clip detail for every val clip:
        - video_path, activity, clip_folder
        - resampled_frame_start / resampled_frame_end
        - original_frame_indices  (the 16 original frame indices)
      Useful for debugging and offline analysis.

  OUTPUT_DIR / val_frame_registry.json
      Compact per-video lookup used by evaluate_val_only_final.py:
        { "Day_2.mp4": [1020, 1021, ...], "TC_00011.mp4": [...], ... }
      Each list is the UNION of all original frame indices that appear
      in any val clip for that video. The evaluate script checks
      `if orig_frame_idx in val_indices` to skip training frames.

No clip-extraction logic is changed.
"""

import os
import json
import random
from collections import Counter, defaultdict

import cv2
import numpy as np
from lxml import etree
from tqdm import tqdm
from ultralytics import YOLO

# ============================================================
# Console summary logger (writes a clean .txt when run finishes)
# ============================================================

SUMMARY_LINES = []

def slog(msg=""):
    """Print AND record a line for the clean summary text file."""
    print(msg)
    SUMMARY_LINES.append(str(msg))

# ============================================================
# Reproducibility
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# Configuration
# ============================================================

BASE_DIR = r"/data/shubhan_avik_work/Targeted_run_3"

VIDEO_XML_PAIRS = [
    (f"{BASE_DIR}/Day_2.mp4",      f"{BASE_DIR}/Day_2_annotations.xml"),
    (f"{BASE_DIR}/Day_3.mp4",      f"{BASE_DIR}/Day_3_annotations.xml"),
    (f"{BASE_DIR}/Day_4_1.mp4",    f"{BASE_DIR}/Day_4_1_annotations.xml"),
    (f"{BASE_DIR}/TC_00011.mp4",   f"{BASE_DIR}/TC_00011_annotations.xml"),
    (f"{BASE_DIR}/TC_00012.mp4",   f"{BASE_DIR}/TC_00012_annotations.xml"),
    (f"{BASE_DIR}/TC_00013.mp4",   f"{BASE_DIR}/TC_00013_annotations.xml"),
    (f"{BASE_DIR}/TC_00014.mp4",   f"{BASE_DIR}/TC_00014_annotations.xml"),
    (f"{BASE_DIR}/TC_00015.mp4",   f"{BASE_DIR}/TC_00015_annotations.xml"),
    (f"{BASE_DIR}/TC_00016.mp4",   f"{BASE_DIR}/TC_00016_annotations.xml"),
    (f"{BASE_DIR}/TC_00019.mp4",   f"{BASE_DIR}/TC_00019_annotations.xml"),
    (f"{BASE_DIR}/TC_00021.mp4",   f"{BASE_DIR}/TC_00021_annotations.xml"),
]

YOLO_MODEL  = f"{BASE_DIR}/best.pt"
OUTPUT_DIR  = f"{BASE_DIR}/Dataset_Ten_days"

CLIP_LENGTH = 16
TARGET_FPS  = 25
CLIP_STRIDE = 3

MIN_CONFIDENCE = 0.5
CROP_SIZE      = 112

TRAIN_RATIO = 0.85
GROUP_SIZE  = 20   # (legacy; no longer used for splitting)

# ============================================================
# Hardcoded continuous 15% TEST windows (seconds), one per video.
# Each window is the single continuous block (= 15% of that video's
# duration) that covers the MOST distinct activities, computed from
# Tasks_Split_by_Video_xlwings.xlsx. Clips inside the window become the
# 'val' (test) set; clips fully outside become 'train'; clips straddling
# the boundary are dropped to prevent train/test frame leakage.
# ============================================================

TEST_WINDOWS_SEC = {
    "Day_2.mp4":    (0,    498),
    "Day_3.mp4":    (0,    1485),
    "Day_4_1.mp4":  (38,   143),
    "TC_00011.mp4": (56,   98),
    "TC_00012.mp4": (0,    968),
    "TC_00013.mp4": (0,    9),
    "TC_00014.mp4": (812,  1391),
    "TC_00015.mp4": (135,  164),
    "TC_00016.mp4": (0,    1126),
    "TC_00019.mp4": (398,  1475),
    "TC_00021.mp4": (0,    882),
}

# ============================================================
# CVAT XML parser  (unchanged)
# ============================================================

def parse_cvat_xml(xml_path: str) -> dict:
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
# Resample map builder  (unchanged)
# ============================================================

def build_resample_map(original_fps: float,
                       target_fps: float,
                       total_original_frames: int) -> list:
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
# Core extraction function  —  returns val_entries for manifest
# ============================================================

def extract_clips_with_yolo(
    video_path:     str,
    model:          YOLO,
    frame_labels:   dict,
    output_dir:     str,
    clip_counter:   dict,
    clip_length:    int   = 16,
    clip_stride:    int   = 3,
    min_confidence: float = 0.5,
    crop_size:      int   = 112,
    test_window:    tuple = (0.0, 0.0),
) -> list:
    """
    Same as original, but RETURNS a list of manifest entries for val clips.

    Each entry is a dict:
        {
          "video_path"            : str,          # absolute path to source video
          "activity"              : str,
          "split"                 : "val",
          "clip_folder"           : str,          # relative to output_dir
          "resampled_frame_start" : int,          # index into resampled timeline
          "resampled_frame_end"   : int,          # inclusive
          "original_frame_indices": [int, ...]    # 16 original frame indices
        }

    Train clips are NOT recorded (we only need val for evaluation filtering).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps  = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video  : {total_frames} frames @ {original_fps:.2f} FPS")

    resample_map = build_resample_map(original_fps, TARGET_FPS, total_frames)
    print(f"  Resamp : {len(resample_map)} frames @ {TARGET_FPS} FPS "
          f"(interval = {original_fps / TARGET_FPS:.4f})")

    all_activities = set(frame_labels.values())
    for split in ("train", "val"):
        for act in all_activities:
            os.makedirs(os.path.join(output_dir, split, act.lower()), exist_ok=True)

    # ----------------------------------------------------------
    # Streaming frame buffer fill  (identical to original)
    # We also record the ORIGINAL frame index for every resampled slot
    # so we can log it in the manifest later.
    # ----------------------------------------------------------
    frame_buffer      = []   # (crop_img, activity_str) | None
    orig_idx_buffer   = []   # original frame index for each resampled slot
    map_ptr               = 0
    current_frame_idx     = 0
    current_frame_decoded = None

    skipped_no_label     = 0
    skipped_no_detection = 0
    skipped_low_conf     = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("  Scanning frames...")
    pbar = tqdm(total=total_frames, desc="  Frames", leave=False)

    while map_ptr < len(resample_map):
        target_orig_idx = resample_map[map_ptr]

        while current_frame_idx < target_orig_idx:
            if not cap.grab():
                break
            current_frame_idx += 1
            pbar.update(1)

        if current_frame_idx != target_orig_idx:
            break

        ret, current_frame_decoded = cap.read()
        if not ret:
            break

        while map_ptr < len(resample_map) and resample_map[map_ptr] == current_frame_idx:
            activity = frame_labels.get(current_frame_idx)

            if activity is None:
                skipped_no_label += 1
                frame_buffer.append(None)
                orig_idx_buffer.append(current_frame_idx)
            else:
                results = model(current_frame_decoded, imgsz=480, verbose=False)

                if len(results[0].boxes) == 0:
                    skipped_no_detection += 1
                    frame_buffer.append(None)
                    orig_idx_buffer.append(current_frame_idx)
                else:
                    best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                    conf     = float(best_box.conf[0])

                    if conf < min_confidence:
                        skipped_low_conf += 1
                        frame_buffer.append(None)
                        orig_idx_buffer.append(current_frame_idx)
                    else:
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                        h, w = current_frame_decoded.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 > x1 and y2 > y1:
                            crop = cv2.resize(
                                current_frame_decoded[y1:y2, x1:x2],
                                (crop_size, crop_size)
                            )
                            frame_buffer.append((crop, activity))
                            orig_idx_buffer.append(current_frame_idx)
                        else:
                            skipped_no_detection += 1
                            frame_buffer.append(None)
                            orig_idx_buffer.append(current_frame_idx)

            map_ptr += 1

        pbar.update(1)
        current_frame_idx += 1

    remaining = total_frames - current_frame_idx
    if remaining > 0:
        pbar.update(remaining)
    pbar.close()
    cap.release()

    valid = sum(1 for x in frame_buffer if x is not None)
    print(f"  Buffer : {len(frame_buffer)} resampled frames ({valid} with valid detections)")
    print(f"  Skipped: no_label={skipped_no_label}  "
          f"no_detect={skipped_no_detection}  low_conf={skipped_low_conf}")

    # ----------------------------------------------------------
    # Clip generation  (same group logic as original)
    # NEW: collect manifest entries for val clips
    # ----------------------------------------------------------
    clip_index        = 0
    val_entries       = []   # ← NEW: returned to caller
    win_start, win_end = test_window
    n_train = n_val = n_straddle = 0

    print(f"  Test window (15% continuous): {win_start}s – {win_end}s")
    print("  Emitting clips...")
    for i in tqdm(range(0, len(frame_buffer) - clip_length + 1, clip_stride),
                  desc="  Clips", leave=False):

        window    = frame_buffer[i : i + clip_length]
        orig_idxs = orig_idx_buffer[i : i + clip_length]

        if any(x is None for x in window):
            continue

        acts = {a for _, a in window}
        if len(acts) != 1:
            continue

        activity = acts.pop().lower()

        # ── Deterministic split by hardcoded continuous test window ──
        clip_start_sec = orig_idxs[0]  / original_fps
        clip_end_sec   = orig_idxs[-1] / original_fps

        if clip_start_sec >= win_start and clip_end_sec <= win_end:
            split = "val"          # fully inside the 15% continuous test block
            n_val += 1
        elif clip_end_sec <= win_start or clip_start_sec >= win_end:
            split = "train"        # fully outside the test block
            n_train += 1
        else:
            n_straddle += 1
            continue               # straddles boundary → drop to avoid leakage

        clip_counter[split].setdefault(activity, 0)
        clip_idx  = clip_counter[split][activity]
        clip_dir  = os.path.join(output_dir, split, activity, f"clip_{clip_idx:05d}")
        os.makedirs(clip_dir, exist_ok=True)

        for j, (frame_img, _) in enumerate(window):
            cv2.imwrite(os.path.join(clip_dir, f"frame_{j:03d}.jpg"), frame_img)

        # ── NEW: record val clips in the manifest ────────────────
        if split == "val":
            rel_clip_folder = os.path.join("val", activity, f"clip_{clip_idx:05d}")
            val_entries.append({
                "video_path"             : video_path,
                "activity"               : activity,
                "split"                  : "val",
                "clip_folder"            : rel_clip_folder,
                # resampled timeline position (index into resample_map)
                "resampled_frame_start"  : i,
                "resampled_frame_end"    : i + clip_length - 1,
                # the 16 ORIGINAL frame indices in the source video
                "original_frame_indices" : orig_idxs,
            })
        # ─────────────────────────────────────────────────────────

        clip_counter[split][activity] += 1
        clip_index += 1

    print(f"  Clips emitted this video: {clip_index}  "
          f"(val entries recorded: {len(val_entries)})")
    slog(f"  {os.path.basename(video_path):<14s} window={win_start}-{win_end}s  "
         f"train={n_train}  val={n_val}  dropped(straddle)={n_straddle}")
    return val_entries


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("YOLO + CVAT  ->  ResNet-3D Dataset Builder  [val-manifest edition]")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Videos : {len(VIDEO_XML_PAIRS)}")
    print(f"Clip   : {CLIP_LENGTH} frames @ {TARGET_FPS} FPS  "
          f"(stride={CLIP_STRIDE})")
    print("Split  : continuous 15% test window per video (hardcoded)")
    print("=" * 60)
    slog("STEP 8 — Dataset Builder (continuous 15% test split)")
    slog(f"Output : {OUTPUT_DIR}")
    slog(f"Videos : {len(VIDEO_XML_PAIRS)}")
    slog("Per-video split (continuous 15% test window):")

    print("\nLoading YOLO model (once)...")
    model = YOLO(YOLO_MODEL)
    print("YOLO ready.\n")

    global_clip_counter = {"train": {}, "val": {}}

    # ── Collect val manifest entries across ALL videos ───────────
    all_val_entries = []

    for video_idx, (video_path, xml_path) in enumerate(VIDEO_XML_PAIRS):

        print(f"\n{'─' * 60}")
        print(f"Video {video_idx + 1}/{len(VIDEO_XML_PAIRS)} : {video_path}")
        print(f"{'─' * 60}")

        frame_labels = parse_cvat_xml(xml_path)

        if not frame_labels:
            print("  WARNING: no annotations found — skipping.")
            continue

        val_entries = extract_clips_with_yolo(
            video_path     = video_path,
            model          = model,
            frame_labels   = frame_labels,
            output_dir     = OUTPUT_DIR,
            clip_counter   = global_clip_counter,
            clip_length    = CLIP_LENGTH,
            clip_stride    = CLIP_STRIDE,
            min_confidence = MIN_CONFIDENCE,
            crop_size      = CROP_SIZE,
            test_window    = TEST_WINDOWS_SEC.get(os.path.basename(video_path), (0.0, 0.0)),
        )
        all_val_entries.extend(val_entries)

    # ── Save val_manifest.json  (full per-clip detail) ───────────
    manifest_path = os.path.join(OUTPUT_DIR, "val_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_val_entries, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"VAL MANIFEST saved → {manifest_path}")
    print(f"  Total val clips recorded : {len(all_val_entries)}")

    # ── Derive val_frame_registry.json from the manifest ─────────
    # This is the compact lookup used by evaluate_val_only_final.py:
    #   { "Day_2.mp4": [orig_frame_idx, ...], ... }
    # Each list is the UNION of all original frame indices that appear
    # in any val clip for that video (duplicates removed, order preserved).
    registry: dict = defaultdict(set)
    by_video    = defaultdict(int)
    by_activity = defaultdict(int)

    for e in all_val_entries:
        v_name = os.path.basename(e["video_path"])
        registry[v_name].update(e["original_frame_indices"])
        by_video[v_name] += 1
        by_activity[e["activity"]] += 1

    # Convert sets → sorted lists for JSON serialisation
    registry_serialisable = {k: sorted(v) for k, v in registry.items()}

    registry_path = os.path.join(OUTPUT_DIR, "val_frame_registry.json")
    with open(registry_path, "w") as f:
        json.dump(registry_serialisable, f, indent=2)

    print(f"\nVAL FRAME REGISTRY saved → {registry_path}")
    total_val_frames = sum(len(v) for v in registry_serialisable.values())
    print(f"  Videos in registry       : {len(registry_serialisable)}")
    print(f"  Total unique val frames  : {total_val_frames}")

    print("\n  Val clips per video:")
    for vname, cnt in sorted(by_video.items()):
        print(f"    {vname}: {cnt} clips  ({len(registry_serialisable.get(vname, []))} unique frames)")
    print("\n  Val clips per activity:")
    for act, cnt in sorted(by_activity.items()):
        print(f"    {act}: {cnt}")

    # ── Final dataset summary (unchanged) ─────────────────────────
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
    print("Done — dataset ready for training. Manifest ready for evaluation.")

    # ── Record final summary lines ───────────────────────────────
    slog("")
    slog("FINAL DATASET SUMMARY")
    for split in ("train", "val"):
        split_total = sum(global_clip_counter[split].values())
        slog(f"{split.upper()}  ({split_total} clips):")
        for act, count in sorted(global_clip_counter[split].items()):
            slog(f"  {act:<20s}: {count:>5d} clips")
    slog(f"Total clips : {total_clips}")
    slog(f"Total val clips recorded : {len(all_val_entries)}")
    slog(f"Total unique val frames  : {total_val_frames}")
    slog(f"Dataset root: {OUTPUT_DIR}")

    # ── Write clean console summary to a notepad .txt ────────────
    summary_path = os.path.join(BASE_DIR, "step8_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(SUMMARY_LINES) + "\n")
    print(f"\nClean summary written → {summary_path}")
