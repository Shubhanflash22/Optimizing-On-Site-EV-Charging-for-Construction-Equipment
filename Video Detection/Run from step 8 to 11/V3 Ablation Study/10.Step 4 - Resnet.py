"""
evaluate_val_only_final.py
==========================
Runs full continuous-stream inference (identical to Step 11's pipeline),
then filters to val-only frames via the Step-8 manifest for clean metrics.

Post-processing is IDENTICAL to Step 11's apply_post_processing(),
locked to the best hyperparameters from the grid search:

    min_activity_duration_s : 2.0
    dist_threshold          : 0.05   ← was 0.2 in Gemini/my version (WRONG)
    idle_window             : 36     ← was 40 in Gemini/my version (WRONG)
    fsm_min_dwell_seconds   : 1.0
    enable_fsm              : False  ← FSM off (best params say false)
    override_travelling_only: True

Differences vs Gemini's evaluate_val_only.py:
  1. IDLE_WINDOW fixed 40 → 36
  2. DIST_THRESHOLD fixed 0.2 → 0.05
  3. smooth_signal_med now takes window_len param (matches Step 11)
  4. compute_idling_mask now takes idle_window, dist_threshold params (matches Step 11)
  5. Full apply_post_processing() function lifted verbatim from Step 11
  6. ActivityFSM class lifted verbatim from Step 11 (enable_fsm=False so won't fire,
     but must be present so the code path is identical)
  7. override_travelling_only logic uses the full if/else from Step 11
  8. min_activity_duration_s explicit (2.0) instead of hardcoded magic number
  9. Metrics: sklearn classification_report + confusion matrix (kept from Gemini,
     it's better than Step 11's manual table for a val-only eval)
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
from collections import deque, Counter
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
from lxml import etree
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================
# Console summary logger (writes a clean .txt when eval finishes)
# ============================
SUMMARY_LINES = []

def slog(msg=""):
    """Print AND record a line for the clean summary text file."""
    print(msg)
    SUMMARY_LINES.append(str(msg))

# ============================
# Best Hyperparameters (locked)
# ============================
BEST_PARAMS = {
    "min_activity_duration_s" : 2.0,
    "dist_threshold"          : 0.05,   # Step 11 best — NOT 0.2
    "idle_window"             : 36,     # Step 11 best — NOT 40
    "fsm_min_dwell_seconds"   : 1.0,
    "enable_fsm"              : False,
    "override_travelling_only": True,
}

# ============================
# Paths
# ============================
BASE_DIR      = Path("/data/shubhan_avik_work/Targeted_run_3")
MODEL_PATH    = BASE_DIR / "resnet3d_best_ten_days.pth"
YOLO_PATH     = BASE_DIR / "best.pt"
REGISTRY_PATH = BASE_DIR / "Dataset_Ten_days" / "val_frame_registry.json"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_FILES   = sorted(BASE_DIR.glob("*.mp4"))

# Static signal-processing constants (match Step 11)
SAVGOL_POLYORDER       = 2
ROLL_MEDIAN_WINDOW     = 5
AREA_THRESHOLD_PERCENT = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# ActivityFSM  (verbatim from Step 11)
# enable_fsm=False so it won't fire, but the class must exist
# so the pipeline is truly identical to Step 11
# ============================
class ActivityFSM:
    """Finite State Machine for activity sequence validation."""

    TRANSITIONS = {
        'travelling': {'idling', 'swinging'},
        'idling':     {'travelling', 'digging', 'swinging'},
        'digging':    {'loading', 'swinging'},
        'loading':    {'swinging'},
        'swinging':   {'digging', 'loading', 'idling', 'travelling'}
    }

    OVERRIDE_CONF_THRESHOLD = 0.95

    def __init__(self, target_fps=25, min_dwell=2.0):
        self.target_fps        = target_fps
        self.min_dwell_seconds = min_dwell
        self.min_dwell_frames  = int(self.min_dwell_seconds * target_fps)

    def clean_sequence(self, predictions, confidences, activity_names):
        pred_names = [activity_names[p] for p in predictions]
        stage1     = self._enforce_dwell_and_transitions(pred_names, confidences)
        stage2     = self._repair_impossible_sequences(stage1)
        cleaned    = self._final_smoothing(stage2)
        return [activity_names.index(name) for name in cleaned]

    def _enforce_dwell_and_transitions(self, pred_names, confidences):
        cleaned     = []
        current     = pred_names[0]
        state_start = 0
        for i, (pred, conf) in enumerate(zip(pred_names, confidences)):
            dwell = i - state_start
            if dwell < self.min_dwell_frames:
                cleaned.append(current)
                continue
            if pred != current:
                allowed = self.TRANSITIONS.get(current, set())
                if pred in allowed or conf > self.OVERRIDE_CONF_THRESHOLD:
                    current     = pred
                    state_start = i
            cleaned.append(current)
        return cleaned

    def _repair_impossible_sequences(self, predictions):
        repaired = predictions.copy()
        for i in range(1, len(repaired) - 1):
            prev, curr, nxt = repaired[i-1], repaired[i], repaired[i+1]
            if prev == 'digging'  and curr == 'travelling' and nxt == 'loading':
                repaired[i] = 'swinging'
            elif prev == 'loading' and curr == 'travelling':
                repaired[i] = 'swinging'
            elif prev == 'loading' and curr == 'digging':
                repaired[i] = 'swinging'
        return repaired

    def _final_smoothing(self, predictions, window=5):
        smoothed = []
        half = window // 2
        for i in range(len(predictions)):
            start = max(0, i - half)
            end   = min(len(predictions), i + half + 1)
            mode  = Counter(predictions[start:end]).most_common(1)[0][0]
            smoothed.append(mode)
        return smoothed

# ============================
# Signal Processing  (verbatim from Step 11)
# smooth_signal_med takes window_len — NOT a module-level constant
# ============================
def smooth_signal_med(signal, window_len):
    n = len(signal)
    if n < 3:
        return signal.copy()
    w = window_len
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    if w < 3:
        w = 3
    sg = savgol_filter(signal, w, SAVGOL_POLYORDER, mode="interp")
    s  = pd.Series(sg)
    s  = s.rolling(window=min(ROLL_MEDIAN_WINDOW, n), center=True, min_periods=1).median()
    return s.values

def compute_idling_mask(bbox_list, idle_window, dist_threshold):
    """Verbatim from Step 11 — parameterised idle_window & dist_threshold."""
    if not bbox_list:
        return []

    data   = np.array(bbox_list)
    cx     = (data[:, 0] + data[:, 2]) / 2.0
    cy     = (data[:, 1] + data[:, 3]) / 2.0
    area   = (data[:, 2] - data[:, 0]) * (data[:, 3] - data[:, 1])

    cx_s   = smooth_signal_med(cx,   11)
    cy_s   = smooth_signal_med(cy,   11)
    area_s = smooth_signal_med(area, 11)

    area_mean       = np.mean(area_s)
    area_std_thresh = (AREA_THRESHOLD_PERCENT / 100.0) * area_mean

    n         = len(data)
    idle_mask = np.zeros(n, dtype=bool)
    w_size    = int(idle_window)

    if n >= w_size:
        for i in range(0, n - w_size + 1):
            dist = np.sqrt(np.diff(cx_s[i:i+w_size])**2 + np.diff(cy_s[i:i+w_size])**2)
            dA   = np.abs(np.diff(area_s[i:i+w_size]))
            if np.std(dist) < dist_threshold and np.std(dA) < area_std_thresh:
                idle_mask[i:i+w_size] = True
    else:
        dist = np.sqrt(np.diff(cx_s)**2 + np.diff(cy_s)**2)
        dA   = np.abs(np.diff(area_s))
        if np.std(dist) < dist_threshold and np.std(dA) < area_std_thresh:
            idle_mask[:] = True

    return idle_mask

# ============================
# Post-processing  (verbatim from Step 11's apply_post_processing)
# ============================
def apply_post_processing(raw_preds, confs, bboxes, activity_names, params, fps):
    """Exact replica of Step 11's apply_post_processing()."""

    # Step 1 — Physics-based idling correction
    idle_mask = compute_idling_mask(bboxes, params['idle_window'], params['dist_threshold'])

    if 'idling' in activity_names:
        idling_idx      = activity_names.index('idling')
        corrected_preds = []
        for i, pred_idx in enumerate(raw_preds):
            if idle_mask[i]:
                pred_label = activity_names[pred_idx]
                if params['override_travelling_only']:
                    # Only flip travelling → idling (best params: True)
                    corrected_preds.append(idling_idx if pred_label == 'travelling' else pred_idx)
                else:
                    # Flip everything except idling itself
                    corrected_preds.append(pred_idx if pred_label == 'idling' else idling_idx)
            else:
                corrected_preds.append(pred_idx)
    else:
        corrected_preds = list(raw_preds)

    # Step 2 — Majority-voting temporal smoothing
    window_size = int(params['min_activity_duration_s'] * fps)
    half_window = window_size // 2
    smoothed    = []
    for i in range(len(corrected_preds)):
        start = max(0, i - half_window)
        end   = min(len(corrected_preds), i + half_window + 1)
        most_common = stats.mode(corrected_preds[start:end], keepdims=True)[0][0]
        smoothed.append(most_common)

    # Step 3 — FSM (best params: enable_fsm=False, so this block won't fire)
    if params['enable_fsm']:
        fsm      = ActivityFSM(target_fps=fps, min_dwell=params['fsm_min_dwell_seconds'])
        smoothed = fsm.clean_sequence(smoothed, confs, activity_names)

    return smoothed

# ============================
# Ground-truth loader (CVAT XML)
# ============================
def load_ground_truth_labels(video_name):
    xml_path = BASE_DIR / f"{video_name.replace('.mp4', '_annotations.xml')}"
    if not xml_path.exists():
        return {}
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    frame_labels = {}
    for image in root.findall(".//image"):
        f_num = int(image.get("id"))
        for box in image.findall("box"):
            frame_labels[f_num] = box.get("label").lower()
    return frame_labels

# ============================
# Plotting Helpers
# ============================
COLOR_MAP = {
    'digging': '#d62728', 'idling': '#7f7f7f', 
    'loading': '#1f77b4', 'swinging': '#ff7f0e', 'travelling': '#2ca02c'
}

def plot_heatmap(y_true, y_pred, title, filename):
    if not y_true: return
    # Forces 5x5 grid even if a class is missing in a specific video
    cm = confusion_matrix(y_true, y_pred, labels=range(len(ACTIVITY_NAMES)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIVITY_NAMES, yticklabels=ACTIVITY_NAMES)
    plt.ylabel('Ground Truth')
    plt.xlabel('Pipeline Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_timeline(t_indices, s_preds, gt_map, title, filename):
    if not t_indices: return
    plot_frames = min(len(t_indices), len(s_preds))
    time_axes = np.arange(plot_frames) / TARGET_FPS 
    
    gt_numerical, pred_numerical = [], []
    for idx, target_orig_idx in enumerate(t_indices[:plot_frames]):
        pred_name = ACTIVITY_NAMES[s_preds[idx]]
        gt_name = gt_map.get(target_orig_idx, 'idling').lower()
        pred_numerical.append(ACTIVITY_NAMES.index(pred_name))
        gt_numerical.append(ACTIVITY_NAMES.index(gt_name) if gt_name in ACTIVITY_NAMES else 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    
    for ax, num_list, ylabel in [(ax1, gt_numerical, "Ground Truth"), (ax2, pred_numerical, "Pipeline Pred")]:
        for act in ACTIVITY_NAMES:
            mask = np.array([ACTIVITY_NAMES[p] == act for p in num_list])
            if np.any(mask):
                ax.scatter(time_axes[mask], [1]*np.sum(mask), color=COLOR_MAP.get(act, '#000000'), 
                           label=act if ax == ax1 else "", marker='|', s=500, linewidths=3)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([])
        if ax == ax1: ax.set_title(title)
        if ax == ax2: ax.set_xlabel("Time (seconds)")
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, 1.3), ncol=5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ============================
# Model & asset initialisation
# ============================
print("=" * 60)
print("LOADING MODELS & ASSETS")
print("=" * 60)

checkpoint     = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
config         = checkpoint['config']
CLIP_LENGTH    = config['clip_length']
CROP_SIZE      = config['crop_size']
TARGET_FPS     = config['target_fps']
NUM_CLASSES    = config['num_classes']
ACTIVITY_NAMES = checkpoint['activity_names']

print(f"  Clip length : {CLIP_LENGTH}")
print(f"  Target FPS  : {TARGET_FPS}")
print(f"  Classes     : {ACTIVITY_NAMES}")

model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE).eval()

yolo_model = YOLO(str(YOLO_PATH))

MEAN = np.load(str(BASE_DIR / "dataset_mean.npy"))
STD  = np.load(str(BASE_DIR / "dataset_std.npy"))

if not REGISTRY_PATH.exists():
    raise FileNotFoundError(
        f"val_frame_registry.json not found at {REGISTRY_PATH}\n"
        "Run Step 8 (8_Creating_clips_with_val_manifest.py) first."
    )
with open(REGISTRY_PATH) as f:
    val_frame_registry = json.load(f)

print(f"\nVal manifest loaded — {len(val_frame_registry)} videos registered.")
print(f"\nBest params locked in:")
for k, v in BEST_PARAMS.items():
    print(f"  {k}: {v}")

# ============================
# Main inference + evaluation loop
# ============================
global_y_true = []
global_y_pred = []
overall_timeline_target_indices = []
overall_timeline_smoothed_preds = []
overall_timeline_gt_labels = {}
timeline_offset = 0

for video_path in VIDEO_FILES:
    v_name = video_path.name

    if v_name not in val_frame_registry:
        print(f"\n[SKIP] {v_name} — not in val manifest (no val clips from this video)")
        continue

    val_indices = set(val_frame_registry[v_name])
    gt_labels   = load_ground_truth_labels(v_name)

    print(f"\n{'='*60}")
    print(f"Processing: {v_name}")
    print(f"  Val frames to evaluate : {len(val_indices)}")
    print(f"{'='*60}")

    # ── Build resampled-frame index list (same formula as Step 8 & Step 10) ──
    cap          = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval     = original_fps / TARGET_FPS

    target_indices = []
    i = 0
    while True:
        idx = int(round(i * interval))
        if idx >= total_frames:
            break
        target_indices.append(idx)
        i += 1

    # ── Full continuous-stream inference (same as Step 11 / Gemini) ──────────
    raw_predictions = []
    confidences     = []
    bbox_history    = []
    clip_buffer     = deque(maxlen=CLIP_LENGTH)

    current_frame_idx = 0

    print("  Running continuous AI inference...")
    for target_orig_idx in tqdm(target_indices, desc="  Frames"):
        while current_frame_idx < target_orig_idx:
            cap.grab()
            current_frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            break
        current_frame_idx += 1

        # YOLO crop (identical to Step 10 / Step 11)
        results         = yolo_model(frame, imgsz=480, verbose=False)
        processed_frame = cv2.resize(frame, (CROP_SIZE, CROP_SIZE))
        h, w            = frame.shape[:2]
        best_bbox       = [0, 0, w, h]

        if len(results[0].boxes) > 0:
            best_box        = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            x1, y1          = max(0, x1), max(0, y1)
            x2, y2          = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                processed_frame = cv2.resize(frame[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))
                best_bbox       = [x1, y1, x2, y2]

        clip_buffer.append(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        bbox_history.append(best_bbox)

        # Only predict once the buffer is full; pad start like Step 11
        if len(clip_buffer) < CLIP_LENGTH:
            if not raw_predictions:
                continue

        frames_np  = np.array(list(clip_buffer)).astype(np.float32) / 255.0
        frames_np  = (frames_np - MEAN) / STD
        clip_tensor = (torch.from_numpy(frames_np)
                       .permute(3, 0, 1, 2).float().unsqueeze(0).to(DEVICE))

        with torch.no_grad():
            outputs      = model(clip_tensor)
            probs        = torch.softmax(outputs, dim=1)
            conf, pred   = probs.max(1)

        raw_predictions.append(pred.item())
        confidences.append(conf.item())

    cap.release()

    # Pad start to align lengths (identical to Step 11)
    while len(raw_predictions) < len(bbox_history):
        raw_predictions.insert(0, raw_predictions[0] if raw_predictions else 0)
        confidences.insert(0, confidences[0] if confidences else 0.0)

    # ── Apply full Step 11 post-processing with best params ──────────────────
    print("  Applying post-processing (idling physics + majority voting + FSM)...")
    smoothed_preds = apply_post_processing(
        raw_predictions, confidences, bbox_history,
        ACTIVITY_NAMES, BEST_PARAMS, TARGET_FPS
    )

    # ── Filter to val frames only via manifest ───────────────────────────────
    print("  Filtering to val frames only...")
    video_y_true = []
    video_y_pred = []

    for map_idx, orig_frame_idx in enumerate(target_indices):
        if map_idx >= len(smoothed_preds):
            break
        if orig_frame_idx not in val_indices:
            continue                                   # skip training frames
        gt_name = gt_labels.get(orig_frame_idx)
        if gt_name not in ACTIVITY_NAMES:
            continue                                   # unannotated frame
        video_y_true.append(int(ACTIVITY_NAMES.index(gt_name)))
        video_y_pred.append(int(smoothed_preds[map_idx]))

    correct = sum(p == t for p, t in zip(video_y_pred, video_y_true))
    total   = len(video_y_true)
    print(f"  Val frames evaluated : {total}  |  Correct : {correct}  "
          f"|  Acc : {100*correct/total:.2f}%" if total > 0 else "  No val frames found")
    slog(f"{v_name:<14s} val_frames={total:>6d}  acc="
         + (f"{100*correct/total:.2f}%" if total > 0 else "N/A"))

    global_y_true.extend(video_y_true)
    global_y_pred.extend(video_y_pred)

    # 1. Extract pure video-level metrics for the per-video heatmap
    video_y_true, video_y_pred = [], []
    for map_idx, target_orig_idx in enumerate(target_indices):
        if map_idx >= len(smoothed_preds): break
        if target_orig_idx in val_indices:
            gt_name = gt_labels.get(target_orig_idx)
            if gt_name in ACTIVITY_NAMES:
                video_y_true.append(ACTIVITY_NAMES.index(gt_name))
                video_y_pred.append(smoothed_preds[map_idx])

    # 2. Generate & Save Per-Video Heatmap & Timeline
    plot_heatmap(video_y_true, video_y_pred, f"Validation Matrix: {v_name}", 
                 PLOTS_DIR / f"heatmap_{Path(v_name).stem}.png")
    plot_timeline(target_indices, smoothed_preds, gt_labels, f"Timeline: {v_name}", 
                  PLOTS_DIR / f"timeline_{Path(v_name).stem}.png")

    # 3. Accumulate data for the single giant overall timeline
    if target_indices:
        for orig_idx, label in gt_labels.items():
            overall_timeline_gt_labels[orig_idx + timeline_offset] = label
        for idx in target_indices:
            overall_timeline_target_indices.append(idx + timeline_offset)
        overall_timeline_smoothed_preds.extend(smoothed_preds)
        
        # Add 5 seconds of blank space between videos on the overall timeline
        timeline_offset += (target_indices[-1] + int(original_fps * 5))

# ============================
# Final Results
# ============================
print("\n" + "=" * 60)
print("FINAL RESULTS — VAL SET ONLY (training frames excluded)")
print("=" * 60)
print(f"Best params used:")
for k, v in BEST_PARAMS.items():
    print(f"  {k}: {v}")
print()

if len(global_y_true) == 0:
    print("ERROR: No val frames were evaluated. Check manifest and paths.")
else:
    overall_correct = int(sum(p == t for p, t in zip(global_y_pred, global_y_true)))
    overall_total   = len(global_y_true)
    print(f"Overall Accuracy : {100 * overall_correct / overall_total:.2f}%  "
          f"({overall_correct}/{overall_total} frames)\n")

    report_str = classification_report(
        global_y_true, global_y_pred,
        target_names=ACTIVITY_NAMES,
        digits=4
    )
    print(report_str)

    slog("")
    slog("STEP 10 — FINAL VAL-ONLY EVALUATION (15% continuous test block)")
    slog(f"Overall Accuracy : {100 * overall_correct / overall_total:.2f}%  "
         f"({overall_correct}/{overall_total} frames)")
    slog("")
    slog("Classification report:")
    slog(report_str)

    print("Confusion Matrix (rows=GT, cols=Pred):")
    cm = confusion_matrix(global_y_true, global_y_pred,
                          labels=list(range(len(ACTIVITY_NAMES))))
    header = "         " + "  ".join(f"{a[:6]:>6}" for a in ACTIVITY_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {ACTIVITY_NAMES[i]:<10}" + "  ".join(f"{v:>6}" for v in row))

    # Save results JSON
    out = {
        "best_params"     : BEST_PARAMS,
        "overall_accuracy": round(100 * overall_correct / overall_total, 4),
        "total_val_frames": int(overall_total),
        "per_class"       : {}
    }
    for idx, act in enumerate(ACTIVITY_NAMES):
        act_true  = [t for t in global_y_true if t == idx]
        act_pred  = [p for t, p in zip(global_y_true, global_y_pred) if t == idx]
        act_corr  = int(sum(p == idx for p in act_pred))
        out["per_class"][act] = {
            "total"   : int(len(act_true)),
            "correct" : act_corr,
            "accuracy": round(100 * act_corr / len(act_true), 4) if act_true else None
        }
    print("\nGenerating Overall Macro Visualizations...")
    
    # Generate & Save Overall Heatmap
    plot_heatmap(global_y_true, global_y_pred, "Overall Validation Confusion Matrix", 
                 PLOTS_DIR / "heatmap_OVERALL.png")
    
    # Generate & Save Overall Timeline (All videos stitched together)
    plot_timeline(overall_timeline_target_indices, overall_timeline_smoothed_preds, 
                  overall_timeline_gt_labels, "Overall Aggregate Timeline", 
                  PLOTS_DIR / "timeline_OVERALL.png")
        
    print(f"\n[Success] All 24 images plotted and saved in: {PLOTS_DIR}")
    slog(f"Plots saved in: {PLOTS_DIR}")

    # ── Write clean console summary to a notepad .txt ────────────
    summary_path = BASE_DIR / "step10_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(SUMMARY_LINES) + "\n")
    print(f"Clean summary written → {summary_path}")

#    out_path = BASE_DIR / "val_only_results_final.json"
#    with open(out_path, "w") as f:
#        json.dump(out, f, indent=2)
#    print(f"\nResults saved → {out_path}")

print("=" * 60)
print("hi")