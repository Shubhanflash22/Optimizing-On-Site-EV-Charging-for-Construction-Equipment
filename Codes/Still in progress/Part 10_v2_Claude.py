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
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# ============================
# Configuration
# ============================
MODEL_PATH  = Path(r"/mnt/nvme_data/Avik_Shubhan_codes_data/resnet3d_best_kinetics_2.pth")
VIDEO_PATH  = Path(r"/mnt/nvme_data/Avik_Shubhan_codes_data/Day_2.mp4")
OUTPUT_DIR  = Path(r"/mnt/nvme_data/Avik_Shubhan_codes_data/")
OUTPUT_DIR.mkdir(exist_ok=True)

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

# ============================
# Normalization stats  (must match training)
# ============================
MEAN = np.load("/mnt/nvme_data/Avik_Shubhan_codes_data/dataset_mean.npy")
STD  = np.load("/mnt/nvme_data/Avik_Shubhan_codes_data/dataset_std.npy")

# ============================
# Frame extraction  —  matches Part 8 resampling exactly
# ============================

def extract_frames_generator(video_path, target_fps=25):
    """
    Yield frames resampled to target_fps using the SAME round-nearest logic
    as Part 8's build_resample_map.  This guarantees the model sees the same
    temporal grid during inference that it saw during training.

    Original Part 10 used:  frame_skip = int(original_fps / target_fps)  →  2
        → effective FPS = 59.94 / 2 = 29.97   ← WRONG (training was at 25)

    Fixed version uses:     orig_idx = round(i * original_fps / target_fps)
        → effective FPS = 25.00 exactly        ← CORRECT
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval       = original_fps / target_fps   # e.g. 2.3976

    print(f"\nVideo: {total_frames} frames at {original_fps:.2f} FPS")
    print(f"Resampling to {target_fps} FPS  (interval = {interval:.4f})")

    # --- read all frames (avoids cv2 seek drift) ---
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    # --- yield in resampled order ---
    i = 0
    while True:
        orig_idx = int(round(i * interval))
        if orig_idx >= len(all_frames):
            break
        yield cv2.cvtColor(all_frames[orig_idx], cv2.COLOR_BGR2RGB)
        i += 1


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
    """Dense prediction: one label per resampled frame."""
    clip_buffer = deque(maxlen=clip_length)
    predictions = []
    confidences = []

    for frame in frame_generator:
        clip_buffer.append(frame)
        if len(clip_buffer) < clip_length:
            continue

        clip_tensor = preprocess_clip(list(clip_buffer), crop_size).to(device)

        with torch.no_grad():
            outputs = model(clip_tensor)
            probs   = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)

        predictions.append(pred.item())
        confidences.append(conf.item())

        if len(predictions) % 100 == 0:
            print(f"  Predicted {len(predictions)} frames", end='\r')

    print()
    return predictions, confidences


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
    with open(save_dir / "frame_predictions.csv", 'w') as f:
        f.write("Frame,Time_s,Activity,Confidence\n")
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            f.write(f"{i},{i/fps:.3f},{activity_names[pred]},{conf:.4f}\n")

    # cycles JSON
    with open(save_dir / "cycles.json", 'w') as f:
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
    with open(save_dir / "summary.json", 'w') as f:
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
    visualize_predictions(smoothed, ACTIVITY_NAMES, TARGET_FPS, output_dir / "activity_timeline.png")
    save_results(smoothed, confidences, cycles, productivity, ACTIVITY_NAMES, TARGET_FPS, output_dir)

    print(f"\nDone.  Results in {output_dir}")
    return smoothed, cycles, productivity


# ============================
# Entry point
# ============================
if __name__ == "__main__":
    if not VIDEO_PATH.exists():
        print(f"ERROR: Video not found at {VIDEO_PATH}")
    else:
        predictions, cycles, productivity = main(
            video_path     = VIDEO_PATH,
            model_path     = MODEL_PATH,
            output_dir     = OUTPUT_DIR,
            bucket_payload = 1.5          # LCY  — adjust for your excavator
        )
