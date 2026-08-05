"""
8b_add_clip_manifest.py  -  Video-tagged dataset builder (LOVO enabler)
=======================================================================
This is Step 8 with ONE addition: it writes

    DATASET_DIR / all_clips_manifest.json

recording EVERY emitted clip (train AND val) with its source video, so
trial_5.py can do leave-one-video-out (LOVO) cross-validation.

Each manifest entry:
    {
      "clip_folder"           : "train/digging/clip_00012",   # relative to DATASET_DIR
      "video"                 : "TC_00011.mp4",               # source video basename
      "activity"              : "digging",
      "split"                 : "train" | "val",
      "original_frame_indices": [ ... 16 original frame indices ... ]
    }

Everything else is byte-for-byte the original Step 8 logic (same YOLO crop,
same continuous-15% split, same clip folder naming), so it is fully
backward compatible with Steps 9-11 - it only ADDS a file, renames nothing.

Run this ONCE on the server (it re-extracts clips, ~hours). After it
finishes, trial_5.py auto-detects the manifest and switches to LOVO CV.
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---- Configuration (identical to your Step 8) ----------------------------
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


def parse_cvat_xml(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    frame_labels = {}
    for image in root.findall(".//image"):
        frame_num = int(image.get("id"))
        for box in image.findall("box"):
            frame_labels[frame_num] = box.get("label")
    return frame_labels


def build_resample_map(original_fps, target_fps, total_original_frames):
    interval = original_fps / target_fps
    resample_map, i = [], 0
    while True:
        orig_idx = int(round(i * interval))
        if orig_idx >= total_original_frames:
            break
        resample_map.append(orig_idx)
        i += 1
    return resample_map


def extract_clips(video_path, model, frame_labels, output_dir, clip_counter, test_window):
    """Same extraction as Step 8. RETURNS a list of manifest entries for
    EVERY clip emitted from this video (train + val)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resample_map = build_resample_map(original_fps, TARGET_FPS, total_frames)

    all_activities = set(frame_labels.values())
    for split in ("train", "val"):
        for act in all_activities:
            os.makedirs(os.path.join(output_dir, split, act.lower()), exist_ok=True)

    frame_buffer, orig_idx_buffer = [], []
    map_ptr = current_frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        ret, frame = cap.read()
        if not ret:
            break
        while map_ptr < len(resample_map) and resample_map[map_ptr] == current_frame_idx:
            activity = frame_labels.get(current_frame_idx)
            if activity is None:
                frame_buffer.append(None); orig_idx_buffer.append(current_frame_idx)
            else:
                results = model(frame, imgsz=480, verbose=False)
                if len(results[0].boxes) == 0:
                    frame_buffer.append(None); orig_idx_buffer.append(current_frame_idx)
                else:
                    best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                    conf = float(best_box.conf[0])
                    if conf < MIN_CONFIDENCE:
                        frame_buffer.append(None); orig_idx_buffer.append(current_frame_idx)
                    else:
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            crop = cv2.resize(frame[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))
                            frame_buffer.append((crop, activity)); orig_idx_buffer.append(current_frame_idx)
                        else:
                            frame_buffer.append(None); orig_idx_buffer.append(current_frame_idx)
            map_ptr += 1
        pbar.update(1)
        current_frame_idx += 1
    pbar.close()
    cap.release()

    entries = []
    win_start, win_end = test_window
    v_name = os.path.basename(video_path)

    for i in tqdm(range(0, len(frame_buffer) - CLIP_LENGTH + 1, CLIP_STRIDE),
                  desc="  Clips", leave=False):
        window = frame_buffer[i:i + CLIP_LENGTH]
        orig_idxs = orig_idx_buffer[i:i + CLIP_LENGTH]
        if any(x is None for x in window):
            continue
        acts = {a for _, a in window}
        if len(acts) != 1:
            continue
        activity = acts.pop().lower()

        clip_start_sec = orig_idxs[0] / original_fps
        clip_end_sec = orig_idxs[-1] / original_fps
        if clip_start_sec >= win_start and clip_end_sec <= win_end:
            split = "val"
        elif clip_end_sec <= win_start or clip_start_sec >= win_end:
            split = "train"
        else:
            continue  # straddle -> drop (prevents leakage)

        clip_counter[split].setdefault(activity, 0)
        clip_idx = clip_counter[split][activity]
        clip_dir = os.path.join(output_dir, split, activity, f"clip_{clip_idx:05d}")
        os.makedirs(clip_dir, exist_ok=True)
        for j, (frame_img, _) in enumerate(window):
            cv2.imwrite(os.path.join(clip_dir, f"frame_{j:03d}.jpg"), frame_img)

        entries.append({
            "clip_folder": os.path.join(split, activity, f"clip_{clip_idx:05d}").replace("\\", "/"),
            "video": v_name,
            "activity": activity,
            "split": split,
            "original_frame_indices": [int(x) for x in orig_idxs],
        })
        clip_counter[split][activity] += 1

    print(f"  {v_name}: {len(entries)} clips recorded")
    return entries


if __name__ == "__main__":
    print("=" * 60)
    print("8b - Video-tagged dataset builder (writes all_clips_manifest.json)")
    print("=" * 60)
    model = YOLO(YOLO_MODEL)

    clip_counter = {"train": {}, "val": {}}
    all_entries = []
    for vi, (video_path, xml_path) in enumerate(VIDEO_XML_PAIRS):
        print(f"\nVideo {vi+1}/{len(VIDEO_XML_PAIRS)}: {os.path.basename(video_path)}")
        frame_labels = parse_cvat_xml(xml_path)
        if not frame_labels:
            print("  WARNING: no annotations - skipping.")
            continue
        entries = extract_clips(
            video_path, model, frame_labels, OUTPUT_DIR, clip_counter,
            TEST_WINDOWS_SEC.get(os.path.basename(video_path), (0.0, 0.0)))
        all_entries.extend(entries)

    manifest_path = os.path.join(OUTPUT_DIR, "all_clips_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    by_video = Counter(e["video"] for e in all_entries)
    by_split = Counter(e["split"] for e in all_entries)
    print(f"\nWrote {len(all_entries)} clip entries -> {manifest_path}")
    print(f"  Splits : {dict(by_split)}")
    print("  Per-video clip counts:")
    for v, c in sorted(by_video.items()):
        print(f"    {v}: {c}")
    print("\nDone. trial_5.py will now auto-detect the manifest and use LOVO CV.")
