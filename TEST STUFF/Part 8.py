import os
import cv2
import csv
from pathlib import Path
from lxml import etree
from ultralytics import YOLO
from tqdm import tqdm
import sys

# ============================
# Configuration
# ============================

# Paths
VIDEO_PATH = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Video construction\Day 2-Oct 21, 2025\Excavator\videos\Day_2.mp4"
CVAT_XML = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Workflow\8.Creating clips from CVAT Annotations\annotations.xml"
YOLO_MODEL = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Workflow\8.Creating clips from CVAT Annotations\best.pt"
OUTPUT_DIR = r"C:\Users\shubh\Desktop\New folder\Dataset_Resnet_2"

# ResNet parameters
CLIP_LENGTH = 16  # number of frames per clip
MIN_CONFIDENCE = 0.5  # minimum YOLO confidence to accept detection
CROP_SIZE = 112  # resize cropped images to 112x112 (matches ResNet input)

# ============================
# Parse CVAT XML to get frame->activity mapping
# ============================

def parse_cvat_xml(xml_path):
    """
    Returns: dict mapping frame_number -> activity_label
    """
    print("Parsing CVAT XML...")
    tree = etree.parse(xml_path)
    root = tree.getroot()

    frame_labels = {}

    for image in root.findall(".//image"):
        frame_num = int(image.get("id"))
        for box in image.findall("box"):
            label = box.get("label")
            frame_labels[frame_num] = label  # overwrite if multiple boxes

    print(f"  Found labels for {len(frame_labels)} frames")

    from collections import Counter
    activity_counts = Counter(frame_labels.values())
    print("  Activity distribution:")
    for activity, count in sorted(activity_counts.items()):
        print(f"    {activity}: {count} frames")

    return frame_labels

# ============================
# Run YOLO and extract crops
# ============================

# ============================
# Extract clips with YOLO
# ============================

def extract_clips_with_yolo(video_path, yolo_model_path, frame_labels, output_dir, 
                            clip_length=16, stride=None, min_confidence=0.5, crop_size=112):
    """
    Processes video, crops excavator, organizes clips by activity.
    """

    print("\nLoading YOLO model...")
    model = YOLO(yolo_model_path)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if stride is None:
        # stride = int(fps/2)
        stride = 10

    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, stride={stride}")

    # Create activity directories
    activities = set(frame_labels.values())
    for activity in activities:
        os.makedirs(os.path.join(output_dir, activity.lower()), exist_ok=True)

    frame_buffer = []   # (cropped_frame, activity, frame_num)
    clip_counter = {}

    frame_num = 0
    skipped_no_label = 0
    skipped_no_detection = 0
    skipped_low_conf = 0

    print("\nProcessing video...")
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while frame_num < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break

            activity = frame_labels.get(frame_num)
            if activity is None:
                skipped_no_label += 1
                frame_num += stride
                pbar.update(stride)
                continue

            results = model(frame, imgsz=480, verbose=False)
            if len(results[0].boxes) == 0:
                skipped_no_detection += 1
                frame_num += stride
                pbar.update(stride)
                continue

            best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            conf = float(best_box.conf[0])
            if conf < min_confidence:
                skipped_low_conf += 1
                frame_num += stride
                pbar.update(stride)
                continue

            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                skipped_no_detection += 1
                frame_num += stride
                pbar.update(stride)
                continue

            cropped = cv2.resize(frame[y1:y2, x1:x2], (crop_size, crop_size))
            frame_buffer.append((cropped, activity, frame_num))

            # Create clip if enough frames
            if len(frame_buffer) >= clip_length:
                for i in range(len(frame_buffer) - clip_length + 1):
                    clip_frames = frame_buffer[i:i+clip_length]
                    clip_activities = [a for _, a, _ in clip_frames]
                    if len(set(clip_activities)) == 1:
                        act_name = clip_activities[0].lower()
                        clip_counter.setdefault(act_name, 0)
                        clip_dir = os.path.join(output_dir, act_name, f"clip_{clip_counter[act_name]:05d}")
                        os.makedirs(clip_dir, exist_ok=True)

                        for j, (cf, _, _) in enumerate(clip_frames):
                            cv2.imwrite(os.path.join(clip_dir, f"frame_{j:03d}.jpg"), cf)

                        clip_counter[act_name] += 1
                        break

                if len(frame_buffer) >= clip_length + stride:
                    frame_buffer = frame_buffer[stride:]

            frame_num += stride
            pbar.update(stride)

    cap.release()
    
    # Print statistics
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)

    print("\n📊 Frame Statistics:")
    print(f"  Total frames processed: {frame_num}")
    print(f"  Skipped (no label): {skipped_no_label}")
    print(f"  Skipped (no detection): {skipped_no_detection}")
    print(f"  Skipped (low confidence): {skipped_low_conf}")
    print(f"  Successfully processed: {frame_num - skipped_no_label - skipped_no_detection - skipped_low_conf}")

    print("\n📁 Dataset Statistics:")
    total_clips = sum(clip_counter.values())
    for activity, count in sorted(clip_counter.items()):
        print(f"  {activity}: {count} clips ({count * clip_length} frames)")

    print(f"\n✅ Total clips created: {total_clips}")
    print(f"✅ Dataset saved to: {output_dir}")

    return clip_counter

# ============================
# Main execution
# ============================

if __name__ == "__main__":
    print("="*50)
    print("YOLO + CVAT → ResNet Dataset Generator")
    print("="*50)

    print("\nStep 1: Parsing CVAT annotations...")
    frame_labels = parse_cvat_xml(CVAT_XML)
    if not frame_labels:
        print("❌ ERROR: No labels found in CVAT XML!")
        sys.exit(1)

    print("\nStep 2: Extracting clips with YOLO...")
    clip_stats = extract_clips_with_yolo(
        VIDEO_PATH,
        YOLO_MODEL,
        frame_labels,
        OUTPUT_DIR,
        clip_length=CLIP_LENGTH,
        stride=None,  # defaults to 1 second (fps)
        min_confidence=MIN_CONFIDENCE,
        crop_size=CROP_SIZE
    )

    if sum(clip_stats.values()) == 0:
        print("\n⚠️ WARNING: No clips were created!")
        print("Check if YOLO detects excavators, CVAT labels match video, or confidence is too high")
    else:
        print("\n✅ SUCCESS! Dataset ready for ResNet training")
