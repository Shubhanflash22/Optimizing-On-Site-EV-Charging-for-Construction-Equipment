"""
3D ResNet Inference for Excavator Activity Recognition  —  FPS-corrected
=========================================================================

FIXES applied vs the original Part_10_v1:
  1. Frame extraction now uses the SAME round-nearest resampling as Part 8,
     so the model sees exactly the same temporal grid it was trained on.
  2. Majority-voting window is computed from TARGET_FPS (not hard-coded to 50).
     Paper: "each activity lasts at least 2 s"  →  window = 2 * TARGET_FPS.
  3. Cycle-time and productivity calculations use TARGET_FPS consistently.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision.models.video import r3d_18
import json
from scipy import stats
from scipy.signal import savgol_filter
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd

# ============================
# Configuration
# ============================
# --- IDLING PHYSICS CONFIG (From Part 6) ---
IDLE_WINDOW = 40
DIST_THRESHOLD = 0.2
AREA_THRESHOLD_PERCENT = 0.5
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 2
ROLL_MEDIAN_WINDOW = 5

BASE_DIR   = Path("/data/shubhan_avik_work/Targeted_run")

MODEL_PATH  = BASE_DIR / "resnet3d_best_ten_days.pth"

# All videos to process — add/remove as needed
VIDEO_FILES = sorted(BASE_DIR.glob("*.mp4"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Load Model + Config
# ============================
print("=" * 60)
print("Loading Model")
print("=" * 60)

checkpoint     = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
config         = checkpoint['config']

CLIP_LENGTH    = config['clip_length']      # 16
CROP_SIZE      = config['crop_size']        # 112
TARGET_FPS     = config['target_fps']       # 25   ← single source of truth
NUM_CLASSES    = config['num_classes']
ACTIVITY_NAMES = checkpoint['activity_names']

print(f"  Clip Length : {CLIP_LENGTH}")
print(f"  Crop Size   : {CROP_SIZE}x{CROP_SIZE}")
print(f"  Target FPS  : {TARGET_FPS}")
print(f"  Classes     : {NUM_CLASSES}  →  {ACTIVITY_NAMES}")

model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print(f"  Model loaded  (epoch {checkpoint['epoch']}, val-acc {checkpoint['val_acc']:.2f}%)")

YOLO_PATH = BASE_DIR / "best.pt"
if not YOLO_PATH.exists():
    print(f"ERROR: YOLO model not found at {YOLO_PATH}")
    exit()
print("Loading YOLO model...")
yolo_model = YOLO(str(YOLO_PATH))
# ============================
# Normalization stats  (must match training)
# ============================
MEAN = np.load(str(BASE_DIR / "dataset_mean.npy"))
STD  = np.load(str(BASE_DIR / "dataset_std.npy"))

# ============================
# 3. IDLING LOGIC (Copied from Part 6)
# ============================

def smooth_signal_med(signal):
    """Your exact smoothing function from Part 6"""
    n = len(signal)
    if n < 3: return signal.copy()
    w = SAVGOL_WINDOW
    if w >= n: w = n - 1 if (n - 1) % 2 == 1 else n - 2
    if w < 3: w = 3
    
    sg = savgol_filter(signal, w, SAVGOL_POLYORDER, mode="interp")
    s = pd.Series(sg)
    s = s.rolling(window=min(ROLL_MEDIAN_WINDOW, n), center=True, min_periods=1).median()
    return s.values

def compute_idling_mask(bbox_list):
    """
    Takes list of (x1, y1, x2, y2).
    Returns boolean array: True = Physically Idling.
    """
    if not bbox_list: return []
    
    # Convert to Dataframe/Arrays
    data = np.array(bbox_list) # shape (N, 4)
    x1, y1 = data[:, 0], data[:, 1]
    x2, y2 = data[:, 2], data[:, 3]
    
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = (x2 - x1) * (y2 - y1)
    
    # Smooth
    cx_s = smooth_signal_med(cx)
    cy_s = smooth_signal_med(cy)
    area_s = smooth_signal_med(area)
    
    area_mean = np.mean(area_s)
    area_std_thresh = (AREA_THRESHOLD_PERCENT / 100.0) * area_mean
    
    n = len(data)
    idle_mask = np.zeros(n, dtype=bool)
    
    # Sliding window statistics
    if n >= IDLE_WINDOW:
        for i in range(0, n - IDLE_WINDOW + 1):
            cxw = cx_s[i : i+IDLE_WINDOW]
            cyw = cy_s[i : i+IDLE_WINDOW]
            aw  = area_s[i : i+IDLE_WINDOW]
            
            dist = np.sqrt(np.diff(cxw)**2 + np.diff(cyw)**2)
            dA = np.abs(np.diff(aw))
            
            if np.std(dist) < DIST_THRESHOLD and np.std(dA) < area_std_thresh:
                idle_mask[i : i+IDLE_WINDOW] = True
    else:
        # Fallback for very short videos
        dist = np.sqrt(np.diff(cx_s)**2 + np.diff(cy_s)**2)
        dA = np.abs(np.diff(area_s))
        if np.std(dist) < DIST_THRESHOLD and np.std(dA) < area_std_thresh:
            idle_mask[:] = True
            
    return idle_mask

def extract_frames_generator(video_path, target_fps=25):
    """
    Yields frames resampled to target_fps using 'Round-Nearest' logic.
    Uses cap.grab() to skip frames efficiently without decoding them.
    Saves RAM by not loading the whole video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval       = original_fps / target_fps

    print(f"\nVideo: {total_frames} frames at {original_fps:.2f} FPS")
    print(f"Resampling to {target_fps} FPS (Streaming Mode)")

    # 1. Pre-calculate the exact frame indices we need
    target_indices = []
    i = 0
    while True:
        idx = int(round(i * interval))
        if idx >= total_frames:
            break
        target_indices.append(idx)
        i += 1
    
    print(f"  → Will process {len(target_indices)} resampled frames")

    # 2. Stream through video
    current_frame_idx = 0
    map_ptr = 0
    cached_frame = None
    
    # Progress bar
    pbar = tqdm(total=len(target_indices), desc="Inference Streaming")
    
    while map_ptr < len(target_indices):
        target_orig_idx = target_indices[map_ptr]
        
        # Skip frames efficiently until we reach the target
        while current_frame_idx < target_orig_idx:
            if not cap.grab(): # Fast skip
                break 
            current_frame_idx += 1
            cached_frame = None # Invalidate cache if we moved
            
        if current_frame_idx != target_orig_idx:
            break # End of video
            
        # Decode the frame if we haven't already
        if cached_frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            
            # === ADD THESE LINES (YOLO DETECTION & CROP) ===
            # Run YOLO on the raw frame
            results = yolo_model(frame, imgsz=480, verbose=False)
            
            # Default fallback: If no excavator found, resize the whole frame
            # (matches training crop size of 112)
            processed_frame = cv2.resize(frame, (112, 112))
            h, w = frame.shape[:2]
            best_bbox = [0, 0, w, h]

            if len(results[0].boxes) > 0:
                # Get the highest confidence detection
                best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                
                # Safety checks: Ensure coordinates are within image bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # If the box is valid, crop and resize
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    processed_frame = cv2.resize(crop, (112, 112))
                    best_bbox = [x1, y1, x2, y2]
            
            cached_frame = processed_frame
            cached_bbox = best_bbox
            # ===============================================

            # cap.read() advances index by 1 automatically
            current_frame_idx += 1
        
        # Yield the frame (RGB)
        # Note: If multiple resampled frames map to the same original frame,
        # we yield the same cached_frame multiple times without re-reading.
        # yield cv2.cvtColor(cached_frame, cv2.COLOR_BGR2RGB)
        yield cv2.cvtColor(cached_frame, cv2.COLOR_BGR2RGB), cached_bbox
        pbar.update(1)
        
        map_ptr += 1
        
        # If the next target is DIFFERENT, we clear the cache
        if map_ptr < len(target_indices) and target_indices[map_ptr] != target_orig_idx:
            cached_frame = None
            
    pbar.close()
    cap.release()


# ============================
# Preprocessing  (identical to validation path in Part 9)
# ============================

def preprocess_clip(frames, crop_size=112):
    """Resize→center-crop→normalize a list of 16 RGB frames into a model tensor."""
    processed = []
    for frame in frames:
        img = cv2.resize(frame, (128, 128))
        img = img[8:120, 8:120]                          # center crop to 112
        img = img.astype(np.float32) / 255.0
        processed.append(img)

    frames_np = np.array(processed)                      # (L, H, W, C)
    frames_np = (frames_np - MEAN) / STD                 # same stats as training

    clip_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()  # (C,L,H,W)
    return clip_tensor.unsqueeze(0)                      # (1,C,L,H,W)


# ============================
# Prediction  (sliding-window, stride=1 over resampled frames)
# ============================

def predict_frame_activities(frame_generator, model, clip_length=16, crop_size=112, device='cuda'):
    """Hybrid Prediction: AI + Physics Override"""
    clip_buffer = deque(maxlen=clip_length)
    
    # Store data for post-processing
    raw_predictions = []
    confidences = []
    bbox_history = [] 
    
    print("\nStep 1: AI Inference + Tracking...")
    
    for frame, bbox in frame_generator:
        clip_buffer.append(frame)
        bbox_history.append(bbox)
        
        if len(clip_buffer) < clip_length:
            if not raw_predictions: continue
        
        # AI Prediction
        frames_np = np.array(list(clip_buffer)).astype(np.float32) / 255.0
        frames_np = (frames_np - MEAN) / STD 
        clip_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(clip_tensor)
            probs   = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
            
        raw_predictions.append(pred.item())
        confidences.append(conf.item())

    # Pad start to match lengths
    while len(raw_predictions) < len(bbox_history):
        raw_predictions.insert(0, raw_predictions[0] if raw_predictions else 0)
        confidences.insert(0, confidences[0] if confidences else 0.0)

    print("\nStep 2: Applying Physics-Based Correction (Part 6 Logic)...")
    
    idle_mask = compute_idling_mask(bbox_history)
    
    final_predictions = []
    corrected_count = 0
    
    if 'idling' in ACTIVITY_NAMES:
        idling_idx = ACTIVITY_NAMES.index('idling')
    else:
        print("WARNING: 'idling' class not found. Skipping correction.")
        return raw_predictions, confidences

    for i, pred_idx in enumerate(raw_predictions):
        if idle_mask[i]:
            pred_label = ACTIVITY_NAMES[pred_idx]
            # if pred_label in ['travelling', 'swinging', 'digging']:
            if pred_label in ['travelling']:
                final_predictions.append(idling_idx)
                if pred_idx != idling_idx: corrected_count += 1
            else:
                final_predictions.append(pred_idx)
        else:
            final_predictions.append(pred_idx)
            
    print(f"  → Physics Engine corrected {corrected_count} frames to 'Idling'")

    return final_predictions, confidences


# ============================
# Majority voting  —  window derived from TARGET_FPS, not hard-coded
# ============================

def apply_majority_voting(predictions, target_fps=25, min_activity_duration_s=2.0):
    """
    Paper: "each activity lasts at least 2 s during the operation"
    Window = min_activity_duration_s × target_fps   →  50 frames at 25 FPS.

    The ORIGINAL code hard-coded window_size=50 but the actual FPS during
    inference was ~30, making the real window only 1.67 s.  Now both the
    window calculation and the FPS are consistent at TARGET_FPS.
    """
    window_size = int(min_activity_duration_s * target_fps)   # 2 * 25 = 50
    print(f"\nMajority voting  (window = {window_size} frames = {min_activity_duration_s} s at {target_fps} FPS)")

    smoothed    = []
    half_window = window_size // 2

    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end   = min(len(predictions), i + half_window + 1)
        window = predictions[start:end]
        most_common = stats.mode(window, keepdims=True)[0][0]
        smoothed.append(most_common)

    changes = sum(1 for a, b in zip(predictions, smoothed) if a != b)
    print(f"  Corrected {changes} frames ({100*changes/max(len(predictions),1):.1f}%)")
    return smoothed


# ============================
# Cycle-time & productivity  (paper Section 4.5, Eq. 3)
# ============================

def calculate_cycle_times(predictions, activity_names, fps):
    """Cycle = digging_start[i]  →  digging_start[i+1].  fps = TARGET_FPS."""
    print(f"\nCalculating work cycles (fps={fps})...")
    if 'digging' not in activity_names:
        print("  WARNING: 'digging' not found")
        return []

    digging_idx   = activity_names.index('digging')
    digging_starts = [i for i in range(len(predictions))
                      if predictions[i] == digging_idx and
                         (i == 0 or predictions[i-1] != digging_idx)]

    print(f"  Found {len(digging_starts)} digging events")
    cycles = []

    for idx in range(len(digging_starts) - 1):
        start_frame = digging_starts[idx]
        end_frame   = digging_starts[idx + 1]
        duration_s  = (end_frame - start_frame) / fps          # correct fps here

        cycle_preds = predictions[start_frame:end_frame]
        activity_counts = defaultdict(int)
        for p in cycle_preds:
            activity_counts[activity_names[p]] += 1

        cycles.append({
            'cycle_number'        : idx + 1,
            'start_frame'         : start_frame,
            'end_frame'           : end_frame,
            'duration_frames'     : end_frame - start_frame,
            'duration_seconds'    : duration_s,
            'activity_counts'     : dict(activity_counts),
            'activity_percentages': {a: c/len(cycle_preds)*100 for a, c in activity_counts.items()}
        })

    if cycles:
        avg = np.mean([c['duration_seconds'] for c in cycles])
        print(f"  Avg cycle time : {avg:.2f} s   →  {3600/avg:.1f} cycles/hr")
    return cycles


def calculate_productivity(cycles, bucket_payload_lcy=1.5):
    """Paper Eq. 3:  Productivity = (cycles / hr) × bucket_payload."""
    if not cycles:
        print("\nNo complete cycles  →  productivity = 0")
        return 0.0

    total_s         = sum(c['duration_seconds'] for c in cycles)
    cycles_per_hour = len(cycles) / (total_s / 3600.0)
    productivity    = cycles_per_hour * bucket_payload_lcy

    print(f"\nProductivity:")
    print(f"  Cycles          : {len(cycles)}")
    print(f"  Total time      : {total_s/60:.1f} min")
    print(f"  Cycles / hr     : {cycles_per_hour:.2f}")
    print(f"  Bucket payload  : {bucket_payload_lcy} LCY")
    print(f"  Productivity    : {productivity:.2f} LCY/hr")
    return productivity


# ============================
# Visualisation & saving  (unchanged logic, just uses TARGET_FPS)
# ============================

def visualize_predictions(predictions, activity_names, fps, save_path=None):
    time_s = np.arange(len(predictions)) / fps
    plt.figure(figsize=(15, 6))
    plt.plot(time_s, predictions, linewidth=0.5, alpha=0.7)
    colors = plt.cm.Set3(np.linspace(0, 1, len(activity_names)))
    for i, act in enumerate(activity_names):
        plt.axhline(y=i, color=colors[i], linestyle='--', alpha=0.3, linewidth=0.5)
        plt.text(0, i, f' {act}', va='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.5))
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')
    plt.title('Excavator Activity Recognition')
    plt.yticks(range(len(activity_names)), activity_names)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")


def save_results(predictions, confidences, cycles, productivity, activity_names, fps, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # frame CSV
    with open(save_dir / "frame_predictions_idling_included.csv", 'w') as f:
        f.write("Frame,Time_s,Activity,Confidence\n")
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            f.write(f"{i},{i/fps:.3f},{activity_names[pred]},{conf:.4f}\n")

    # cycles JSON
    with open(save_dir / "cycles_idling_included.json", 'w') as f:
        json.dump(cycles, f, indent=2)

    # summary JSON
    activity_counts = defaultdict(int)
    for p in predictions:
        activity_counts[activity_names[p]] += 1

    summary = {
        'total_frames'          : len(predictions),
        'duration_seconds'      : len(predictions) / fps,
        'fps'                   : fps,
        'activity_distribution' : {a: c/len(predictions)*100 for a, c in activity_counts.items()},
        'num_cycles'            : len(cycles),
        'avg_cycle_time_seconds': np.mean([c['duration_seconds'] for c in cycles]) if cycles else 0,
        'productivity_lcy_per_hour': productivity,
        'avg_confidence'        : float(np.mean(confidences))
    }
    with open(save_dir / "summary_idling_included.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {save_dir}")
    print(f"  Duration : {summary['duration_seconds']/60:.2f} min")
    for a, pct in sorted(summary['activity_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {a:12s}: {pct:6.2f}%")
    print(f"  Cycles   : {len(cycles)}   Avg confidence: {summary['avg_confidence']*100:.1f}%")


# ============================
# Main inference pipeline
# ============================

def main(video_path, model_path, output_dir, bucket_payload=1.5):
    print("=" * 60)
    print("EXCAVATOR ACTIVITY RECOGNITION  —  INFERENCE  (FPS-fixed)")
    print("=" * 60)

    # 1. Extract frames at TARGET_FPS (same grid as training)
    frame_gen = extract_frames_generator(video_path, TARGET_FPS)

    # 2. Dense prediction
    raw_preds, confidences = predict_frame_activities(
        frame_gen, model, CLIP_LENGTH, CROP_SIZE, DEVICE)

    # 3. Majority voting  (window = 2 s × TARGET_FPS = 50 frames)
    smoothed = apply_majority_voting(raw_preds, target_fps=TARGET_FPS, min_activity_duration_s=2.0)

    # 4. Cycles
    cycles = calculate_cycle_times(smoothed, ACTIVITY_NAMES, fps=TARGET_FPS)

    # 5. Productivity
    productivity = calculate_productivity(cycles, bucket_payload)

    # 6. Save & visualise
    output_dir = Path(output_dir)
    visualize_predictions(smoothed, ACTIVITY_NAMES, TARGET_FPS, output_dir / "activity_timeline_idling_included.png")
    save_results(smoothed, confidences, cycles, productivity, ACTIVITY_NAMES, TARGET_FPS, output_dir)

    print(f"\nDone.  Results in {output_dir}")
    return smoothed, cycles, productivity


# ============================
# Entry point
# ============================
if __name__ == "__main__":
    if not VIDEO_FILES:
        print(f"ERROR: No .mp4 files found in {BASE_DIR}")
    else:
        print(f"Found {len(VIDEO_FILES)} video(s) to process:")
        for v in VIDEO_FILES:
            print(f"  {v.name}")

        for video_path in VIDEO_FILES:
            video_stem = video_path.stem          # e.g. "Day_3"
            output_dir = BASE_DIR / f"{video_stem}_pred"
            output_dir.mkdir(exist_ok=True)

            print(f"{chr(61)*60}")
            print(f"Processing: {video_path.name}")
            print(f"Output dir: {output_dir}")
            print(f"{chr(61)*60}")

            predictions, cycles, productivity = main(
                video_path     = video_path,
                model_path     = MODEL_PATH,
                output_dir     = output_dir,
                bucket_payload = 1.5
            )