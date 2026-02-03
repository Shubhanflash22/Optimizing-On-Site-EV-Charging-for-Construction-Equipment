import os
import cv2
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from torchvision.models.video import r3d_18 # Correct import

# ============================
# Configuration (edit paths)
# ============================

SAVE_DIR = r"C:\Users\shubh\Desktop\DELETE AFTER USE\Tubes_10min"
METADATA_PATH = os.path.join(SAVE_DIR, "track_metadata.csv")
ACTIVITY_OUTPUT_CSV = r"C:\Users\shubh\Desktop\DELETE AFTER USE\Activity_Output_1_50_Day3.csv"
ACTIVITY_VISUAL_CSV = r"C:\Users\shubh\Desktop\DELETE AFTER USE\Activity_Visual_1_50_Day3.csv"
MODEL_PATH = r"C:\Users\shubh\Desktop\DELETE AFTER USE\resnet3d_best_kinetics.pth"

FPS = 59.94           # frames per second of video
CLIP_LENGTH = 16   # frames per clip for the 3D model
CROP_SIZE = 112    # model input spatial size (H=W=112)
STRIDE = 3         # Downsampling stride (Matches Training!)
VOTING_SECONDS = 1.0
VOTING_WINDOW = int(VOTING_SECONDS * FPS)

ACTIVITY_NAMES = {
    0: 'digging',
    1: 'idling',
    2: 'loading',
    3: 'swinging',
    4: 'travelling'
}
NUM_CLASSES = 5

# ============================
# Start of pipeline
# ============================

print("="*70)
print("Starting flattened Activity Recognition pipeline")
print("="*70)
start_time_total = time.time()

# 1) Device selection
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"[INFO] Torch device set to: {device} (cuda available: {use_cuda})")

# 2) Model creation and loading
print("[INFO] Instantiating model (ResNet3D-18) ...")
# Use standard r3d_18 structure
model = r3d_18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[INFO] Model weights loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        print("[WARNING] Using randomly initialized model (Predictions will be garbage).")
else:
    print(f"[WARNING] Model file not found at {MODEL_PATH}. Using randomly initialized model.")

model.eval()
print("[INFO] Model set to eval() mode.\n")

# 3) Check metadata file
if not os.path.exists(METADATA_PATH):
    print(f"[ERROR] Metadata file not found at: {METADATA_PATH}")
    raise SystemExit(1)

print(f"[INFO] Reading metadata CSV: {METADATA_PATH}")
with open(METADATA_PATH, 'r', newline='') as mf:
    reader = csv.DictReader(mf)
    tracks = list(reader)

print(f"[INFO] Found {len(tracks)} tracks in metadata.")
print()

all_results = []   # will collect dicts for CSV output
visual_timeline_rows = []

# 4) Iterate over tracks (linear)
track_counter = 0
for track_info in tracks:
    track_counter += 1
    try:
        track_id = int(track_info.get('track_id', track_counter))
    except:
        track_id = track_counter
    track_folder = track_info.get('frame_folder', None)
    
    print("-"*60)
    print(f"[TRACK {track_counter}] track_id={track_id}")

    if not track_folder or not os.path.exists(track_folder):
        print(f"[TRACK {track_counter} WARNING] track folder missing: {track_folder}. Skipping.")
        continue

    # 4a) Load frames from folder
    t0 = time.time()
    # Sort by number in filename to ensure correct order
    import re
    frame_files = sorted(
        [f for f in os.listdir(track_folder) if f.lower().endswith('.jpg') or f.lower().endswith('.png')],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )
    
    frames = []
    frame_indices = []

    for idx, fname in enumerate(frame_files):
        full_path = os.path.join(track_folder, fname)
        img = cv2.imread(full_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
        except:
            continue
        frames.append(img)
        
        # Parse frame number
        try:
            name_no_ext = os.path.splitext(fname)[0]
            # Assumes format like "frame_00105.jpg" or "105.jpg"
            frame_idx = int(re.search(r'\d+', name_no_ext).group())
        except:
            frame_idx = idx
        frame_indices.append(frame_idx)

    t1 = time.time()
    if len(frames) == 0:
        continue

    # 4b) Prepare sliding window clip indices (CORRECTED TEMPORAL SAMPLING)
    num_frames = len(frames)
    
    # We need to cover a span of (16-1)*4 + 1 = 61 frames to get 16 frames at stride 4
    window_span = (CLIP_LENGTH - 1) * STRIDE + 1
    step_size = STRIDE # We slide the window by 4 frames at a time
    
    # Calculate how many clips we can fit
    if num_frames < window_span:
        num_clips = 1 # We will pad it
    else:
        num_clips = (num_frames - window_span) // step_size + 1

    predictions_by_frame_idx = {}
    total_inference_time = 0.0

    # 4c) Iterate clips
    for i in range(num_clips):
        # Calculate the start index for this window
        start_idx = i * step_size
        
        # FIX: Select 16 frames spaced by STRIDE (e.g., 0, 4, 8... 60)
        clip_indices_local = [start_idx + k*STRIDE for k in range(CLIP_LENGTH)]
        
        # Gather frames, handling padding if near the end or if video is short
        clip_frames = []
        valid_indices_for_mapping = [] # To map result back to specific frames
        
        for local_idx in clip_indices_local:
            if local_idx < num_frames:
                clip_frames.append(frames[local_idx])
                valid_indices_for_mapping.append(frame_indices[local_idx])
            else:
                # Padding: repeat the last valid frame
                clip_frames.append(frames[-1])
                # Do not append to valid_indices_for_mapping (don't label non-existent frames)

        # FIX: Add Normalization (Matches Training)
        clip_np = np.array(clip_frames).astype(np.float32) / 255.0
        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        clip_np = (clip_np - mean) / std

        # Convert to Tensor (B, C, T, H, W)
        clip_tensor = torch.from_numpy(clip_np).float().permute(3, 0, 1, 2).unsqueeze(0)
        clip_tensor = clip_tensor.to(device)

        # Inference
        t_inf0 = time.time()
        try:
            with torch.no_grad():
                output = model(clip_tensor)
                _, predicted = torch.max(output, dim=1)
                pred_label = int(predicted.item())
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            pred_label = -1
        
        total_inference_time += (time.time() - t_inf0)

        # Assign prediction to the frames we actually looked at
        for fi in valid_indices_for_mapping:
            predictions_by_frame_idx[fi] = pred_label

    print(f"[TRACK {track_counter}] Processed {num_clips} clips. Time: {total_inference_time:.2f}s")

    # 4e) Sort and smooth
    sorted_frame_items = sorted(predictions_by_frame_idx.items(), key=lambda x: x[0])
    if not sorted_frame_items: continue

    sorted_frame_idxs = [fi for fi, _ in sorted_frame_items]
    raw_predictions = [p for _, p in sorted_frame_items]

    # Majority Voting
    smoothed_predictions = []
    n = len(raw_predictions)
    # Calculate window size in terms of list indices (not frames)
    # Since we sampled every 4th frame, the list is already downsampled.
    # 1 second = 60 frames. Downsampled list = 15 items.
    list_window = int(VOTING_SECONDS * (FPS / STRIDE)) 
    
    for i in range(n):
        start = max(0, i - list_window // 2)
        end = min(n, i + list_window // 2)
        window = raw_predictions[start:end]
        if not window:
            smoothed_predictions.append(raw_predictions[i])
            continue
        most_common, _ = Counter(window).most_common(1)[0]
        smoothed_predictions.append(most_common)

    # 4g) Save per-frame results
    for i, frame_idx in enumerate(sorted_frame_idxs):
        all_results.append({
            'track_id': track_id,
            'frame': int(frame_idx),
            'activity_label': int(smoothed_predictions[i]),
            'activity_name': ACTIVITY_NAMES.get(int(smoothed_predictions[i]), "unknown"),
            'raw_prediction': int(raw_predictions[i])
        })

    # 4h) Create Timeline Segments
    current_activity = None
    start_frame = None
    prev_frame = None

    for frame_idx, activity in zip(sorted_frame_idxs, smoothed_predictions):
        if activity != current_activity:
            if current_activity is not None:
                duration = (prev_frame - start_frame) / FPS
                visual_timeline_rows.append({
                    'track_id': track_id,
                    'activity': ACTIVITY_NAMES.get(int(current_activity), "unknown"),
                    'start_frame': start_frame,
                    'end_frame': prev_frame,
                    'duration_sec': round(duration, 2),
                    'start_time_sec': round(start_frame / FPS, 2),
                    'end_time_sec': round(prev_frame / FPS, 2)
                })
            current_activity = activity
            start_frame = frame_idx
        prev_frame = frame_idx

    # Close last segment
    if current_activity is not None and start_frame is not None:
        duration = (prev_frame - start_frame) / FPS
        visual_timeline_rows.append({
            'track_id': track_id,
            'activity': ACTIVITY_NAMES.get(int(current_activity), "unknown"),
            'start_frame': start_frame,
            'end_frame': prev_frame,
            'duration_sec': round(duration, 2),
            'start_time_sec': round(start_frame / FPS, 2),
            'end_time_sec': round(prev_frame / FPS, 2)
        })

# ============================
# Save CSV outputs
# ============================
if all_results:
    print("[OUTPUT] Saving per-frame activity results to CSV...")
    with open(ACTIVITY_OUTPUT_CSV, 'w', newline='') as outf:
        fieldnames = ['track_id', 'frame', 'activity_label', 'activity_name', 'raw_prediction']
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved: {ACTIVITY_OUTPUT_CSV}")

if visual_timeline_rows:
    print("[OUTPUT] Saving visual timeline CSV...")
    with open(ACTIVITY_VISUAL_CSV, 'w', newline='') as outf:
        fieldnames = ['track_id', 'activity', 'start_frame', 'end_frame', 'duration_sec', 'start_time_sec', 'end_time_sec']
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(visual_timeline_rows)
    print(f"Saved: {ACTIVITY_VISUAL_CSV}")

print("="*70)
print(f"Pipeline finished in {time.time() - start_time_total:.2f}s")