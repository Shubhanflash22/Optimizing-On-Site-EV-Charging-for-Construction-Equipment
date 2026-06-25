import cv2
import csv
import os
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = r"C:\Users\shubh\Desktop\Research work with AVIK\Test\Day_2 - Trim.mp4"
CSV_PATH = r"C:\Users\shubh\Desktop\Research work with AVIK\Test\detections.csv"
OUTPUT_PATH = r"C:\Users\shubh\Desktop\Research work with AVIK\Test\Day_2 - DEEPSORT.mp4" 
TRACK_OUTPUT_CSV = r"C:\Users\shubh\Desktop\Test\Track_Output.csv"
SAVE_DIR = r"C:\Users\shubh\Desktop\Test\Tubes"
 
tracker = DeepSort(max_age=35, n_init=4)

detections_per_frame = {}
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame = int(row["frame"])
        det = [
            [int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])],
            float(row["confidence"]),
            row["class"]
        ]
        detections_per_frame.setdefault(frame, []).append(det)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 1
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

track_csv = open(TRACK_OUTPUT_CSV, 'w', newline='')
csv_writer = csv.writer(track_csv)
csv_writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2"])

track_frame_buffers = {}
CLIP_LENGTH = 16

while True:
    ret, frame = cap.read()
    if not ret:
        break
    dets = detections_per_frame.get(frame_idx, [])
    tracks = tracker.update_tracks(dets, frame=frame)
    for t in tracks:
        if not t.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        roi = frame[y1:y2, x1:x2]
        if roi.size != 0:
            resized_roi = cv2.resize(roi, (112, 112))
            track_folder = os.path.join(SAVE_DIR, f"track_{t.track_id}")
            os.makedirs(track_folder, exist_ok=True)
            cv2.imwrite(os.path.join(track_folder, f"{frame_idx:06d}.jpg"), resized_roi)
            if t.track_id not in track_frame_buffers:
                track_frame_buffers[t.track_id] = []
            track_frame_buffers[t.track_id].append({
                'frame_idx': frame_idx,
                'crop': resized_roi.copy()
            })
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{t.track_id}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        csv_writer.writerow([frame_idx, t.track_id, x1, y1, x2, y2])
    out.write(frame)
    if frame_idx % 10 == 0:
        print(f"{frame_idx} frames processed...")
    frame_idx += 1
cap.release()
out.release()
track_csv.close()
print("\nSaving frame sequence metadata for activity recognition...")
metadata_path = os.path.join(SAVE_DIR, "track_metadata.csv")
with open(metadata_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["track_id", "total_frames", "frame_folder"])
    for track_id, frames in track_frame_buffers.items():
        track_folder = os.path.join(SAVE_DIR, f"track_{track_id}")
        writer.writerow([track_id, len(frames), track_folder])
print(f"Tracking completed. Video saved at: {OUTPUT_PATH}")
print(f"Tracking data saved at: {TRACK_OUTPUT_CSV}")
print(f"Frame crops saved in: {SAVE_DIR}")
print(f"Metadata saved at: {metadata_path}")
print(f"\nTotal tracks: {len(track_frame_buffers)}")
for track_id, frames in track_frame_buffers.items():
    print(f"  Track {track_id}: {len(frames)} frames")