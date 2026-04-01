"""
Combined 3D ResNet Inference & Evaluation Pipeline (Grid Search)
================================================================
Combines:
  1. Inference (AI + Physics + Post-processing)
  2. Ground Truth Evaluation (vs Tasks.xlsx)

It runs the heavy AI model ONCE, then iterates through the parameter grid
to test different post-processing settings.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import json
import csv
import itertools
import re
from pathlib import Path
from scipy import stats
from scipy.signal import savgol_filter
from collections import deque, defaultdict
from tqdm import tqdm
from ultralytics import YOLO

# ============================
# Activity FSM Class
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
        self.target_fps = target_fps
        self.min_dwell_seconds = min_dwell  # Set from argument
        self.min_dwell_frames = int(self.min_dwell_seconds * target_fps)
    
    def clean_sequence(self, predictions, confidences, activity_names):
        """Main cleaning pipeline."""
        # Convert indices to names for processing
        pred_names = [activity_names[p] for p in predictions]
        
        # Step 1: Enforce dwell + transitions
        stage1 = self._enforce_dwell_and_transitions(pred_names, confidences)
        
        # Step 2: Repair impossible patterns
        stage2 = self._repair_impossible_sequences(stage1)
        
        # Step 3: Final smoothing
        cleaned_names = self._final_smoothing(stage2)
        
        # Convert back to indices
        return [activity_names.index(name) for name in cleaned_names]
    
    def _enforce_dwell_and_transitions(self, pred_names, confidences):
        cleaned = []
        current_state = pred_names[0]
        state_start = 0
        
        for i, (pred, conf) in enumerate(zip(pred_names, confidences)):
            dwell = i - state_start
            
            # Rule 1: Minimum dwell
            if dwell < self.min_dwell_frames:
                cleaned.append(current_state)
                continue
            
            # Rule 2: Valid transitions
            if pred != current_state:
                allowed = self.TRANSITIONS.get(current_state, set())
                if pred in allowed or conf > self.OVERRIDE_CONF_THRESHOLD:
                    current_state = pred
                    state_start = i
            
            cleaned.append(current_state)
        
        return cleaned
    
    def _repair_impossible_sequences(self, predictions):
        repaired = predictions.copy()
        for i in range(1, len(repaired) - 1):
            prev, curr, next_ = repaired[i-1], repaired[i], repaired[i+1]
            
            # Pattern: Digging -> Travelling -> Loading (Impossible)
            if prev == 'digging' and curr == 'travelling' and next_ == 'loading':
                repaired[i] = 'swinging'
            # Pattern: Loading -> Travelling (Need to retract/swing first)
            elif prev == 'loading' and curr == 'travelling':
                repaired[i] = 'swinging'
            # Pattern: Loading -> Digging (Impossible without dumping)
            elif prev == 'loading' and curr == 'digging':
                repaired[i] = 'swinging'
                
        return repaired
        
    def _final_smoothing(self, predictions, window=5):
        from collections import Counter
        smoothed = []
        half = window // 2
        for i in range(len(predictions)):
            start = max(0, i - half)
            end = min(len(predictions), i + half + 1)
            # Find most common in window
            mode = Counter(predictions[start:end]).most_common(1)[0][0]
            smoothed.append(mode)
        return smoothed

# ============================
# 1. CONFIGURATION & GRID
# ============================

HYPERPARAM_GRID = {
     'min_activity_duration_s': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
     'dist_threshold': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6],
     'idle_window': [20, 24, 30, 36, 40, 48, 56, 64, 80],
     'fsm_min_dwell_seconds': [1.0, 1.5, 2.0, 2.5, 3.0],
     'enable_fsm': [True, False],
     'override_travelling_only': [True, False]
}

# --- PATHS ---
MODEL_PATH      = Path(r"/mnt/nvme1/avik_shubhan/resnet3d/resnet3d_best_kinetics_2.pth")
VIDEO_PATH      = Path(r"/mnt/nvme1/avik_shubhan/resnet3d/Day_3.mp4")
YOLO_PATH       = Path(r"/mnt/nvme1/avik_shubhan/resnet3d/best.pt")
TASKS_XLSX      = Path(r"/mnt/nvme1/avik_shubhan/resnet3d/Tasks.xlsx")
OUTPUT_BASE_DIR = Path(r"/mnt/nvme1/avik_shubhan/resnet3d_1/optimization_runs")

# --- STATIC CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVGOL_POLYORDER = 2
ROLL_MEDIAN_WINDOW = 5
AREA_THRESHOLD_PERCENT = 0.5
BUCKET_PAYLOAD = 1.5  # LCY

# Normalization stats (must match training)
MEAN = np.load("/mnt/nvme1/avik_shubhan/resnet3d/dataset_mean.npy")
STD  = np.load("/mnt/nvme1/avik_shubhan/resnet3d/dataset_std.npy")

ALL_CLASSES = ["digging", "idling", "loading", "swinging", "travelling"]
ACTIVITY_COLORS = {
    "digging"   : "#E74C3C",
    "idling"    : "#95A5A6",
    "loading"   : "#2ECC71",
    "swinging"  : "#3498DB",
    "travelling": "#F39C12",
}

# ============================
# 2. UTILITY FUNCTIONS (Physics & Signal)
# ============================

def smooth_signal_med(signal, window_len):
    """Smooths signal for idling detection."""
    n = len(signal)
    if n < 3: return signal.copy()
    w = window_len
    if w >= n: w = n - 1 if (n - 1) % 2 == 1 else n - 2
    if w < 3: w = 3
    
    sg = savgol_filter(signal, w, SAVGOL_POLYORDER, mode="interp")
    s = pd.Series(sg)
    s = s.rolling(window=min(ROLL_MEDIAN_WINDOW, n), center=True, min_periods=1).median()
    return s.values

def compute_idling_mask(bbox_list, idle_window, dist_threshold):
    """
    Returns boolean array: True = Physically Idling.
    Dynamically uses idle_window and dist_threshold from grid.
    """
    if not bbox_list: return []
    
    # Convert to Dataframe/Arrays
    data = np.array(bbox_list) # shape (N, 4)
    x1, y1 = data[:, 0], data[:, 1]
    x2, y2 = data[:, 2], data[:, 3]
    
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = (x2 - x1) * (y2 - y1)
    
    # Smooth (using a fixed small window for the signal itself, logic from Part 6)
    # Note: SAVGOL_WINDOW was 11 in original, we keep it logic-internal or pass it.
    # We will use a default 11 for the SavGol part as it's signal noise reduction.
    cx_s = smooth_signal_med(cx, 11)
    cy_s = smooth_signal_med(cy, 11)
    area_s = smooth_signal_med(area, 11)
    
    area_mean = np.mean(area_s)
    area_std_thresh = (AREA_THRESHOLD_PERCENT / 100.0) * area_mean
    
    n = len(data)
    idle_mask = np.zeros(n, dtype=bool)
    
    # Sliding window statistics using the Grid Parameter 'idle_window'
    w_size = int(idle_window)
    
    if n >= w_size:
        for i in range(0, n - w_size + 1):
            cxw = cx_s[i : i+w_size]
            cyw = cy_s[i : i+w_size]
            aw  = area_s[i : i+w_size]
            
            dist = np.sqrt(np.diff(cxw)**2 + np.diff(cyw)**2)
            dA = np.abs(np.diff(aw))
            
            if np.std(dist) < dist_threshold and np.std(dA) < area_std_thresh:
                idle_mask[i : i+w_size] = True
    else:
        # Fallback for very short videos
        dist = np.sqrt(np.diff(cx_s)**2 + np.diff(cy_s)**2)
        dA = np.abs(np.diff(area_s))
        if np.std(dist) < dist_threshold and np.std(dA) < area_std_thresh:
            idle_mask[:] = True
            
    return idle_mask

# ============================
# 3. INFERENCE CORE (Generator & Model)
# ============================

def extract_frames_generator(video_path, yolo_model, target_fps=25):
    """Yields (frame, bbox) tuples at target_fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval     = original_fps / target_fps

    print(f"  Source: {total_frames} frames @ {original_fps:.2f} FPS")
    print(f"  Target: Resampling to {target_fps} FPS")

    target_indices = []
    i = 0
    while True:
        idx = int(round(i * interval))
        if idx >= total_frames: break
        target_indices.append(idx)
        i += 1
    
    current_frame_idx = 0
    map_ptr = 0
    cached_frame = None
    cached_bbox = [0, 0, 0, 0] # format x1,y1,x2,y2
    
    pbar = tqdm(total=len(target_indices), desc="  Extracting & Detecting")
    
    while map_ptr < len(target_indices):
        target_orig_idx = target_indices[map_ptr]
        
        while current_frame_idx < target_orig_idx:
            if not cap.grab(): break
            current_frame_idx += 1
            cached_frame = None 
            
        if current_frame_idx != target_orig_idx: break 
            
        if cached_frame is None:
            ret, frame = cap.read()
            if not ret: break
            
            # YOLO
            results = yolo_model(frame, imgsz=480, verbose=False)
            processed_frame = cv2.resize(frame, (112, 112))
            h, w = frame.shape[:2]
            best_bbox = [0, 0, w, h]

            if len(results[0].boxes) > 0:
                best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    processed_frame = cv2.resize(crop, (112, 112))
                    best_bbox = [x1, y1, x2, y2]
            
            cached_frame = processed_frame
            cached_bbox = best_bbox
            current_frame_idx += 1
        
        yield cv2.cvtColor(cached_frame, cv2.COLOR_BGR2RGB), cached_bbox
        pbar.update(1)
        map_ptr += 1
        
        if map_ptr < len(target_indices) and target_indices[map_ptr] != target_orig_idx:
            cached_frame = None
            
    pbar.close()
    cap.release()

def run_heavy_inference(model, yolo_model, video_path, config):
    """
    Runs the ResNet model ONCE. Returns raw logits/indices and bbox history.
    This allows us to re-run post-processing logic without reloading video.
    """
    clip_length = config['clip_length']
    clip_buffer = deque(maxlen=clip_length)
    
    raw_predictions = []
    confidences = []
    bbox_history = []
    
    frame_gen = extract_frames_generator(video_path, yolo_model, config['target_fps'])
    
    print("\n[Phase 1] Running Heavy AI Inference...")
    
    for frame, bbox in frame_gen:
        clip_buffer.append(frame)
        bbox_history.append(bbox)
        
        # Need full buffer to predict
        if len(clip_buffer) < clip_length:
            # Pad beginning with first prediction once available, or 0
            if not raw_predictions: continue
        
        frames_np = np.array(list(clip_buffer)).astype(np.float32) / 255.0
        frames_np = (frames_np - MEAN) / STD 
        clip_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(clip_tensor)
            probs   = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
            
        raw_predictions.append(pred.item())
        confidences.append(conf.item())

    # Pad start to match lengths (simple padding)
    while len(raw_predictions) < len(bbox_history):
        raw_predictions.insert(0, raw_predictions[0] if raw_predictions else 0)
        confidences.insert(0, confidences[0] if confidences else 0.0)
        
    return raw_predictions, confidences, bbox_history

# ============================
# 4. POST-PROCESSING (Grid Applied Here)
# ============================

def apply_post_processing(raw_preds, confs, bboxes, activity_names, params, fps):
    """
    Applies Idling logic, Majority Voting (Smoothing), and optional FSM.
    """
    # 1. Physics / Idling Correction
    idle_mask = compute_idling_mask(bboxes, params['idle_window'], params['dist_threshold'])
    
    if 'idling' in activity_names:
        idling_idx = activity_names.index('idling')
        corrected_preds = []
        
        for i, pred_idx in enumerate(raw_preds):
            if idle_mask[i]:
                pred_label = activity_names[pred_idx]
                # Apply override logic
                if params['override_travelling_only']:
                    if pred_label == 'travelling':
                        corrected_preds.append(idling_idx)
                    else:
                        corrected_preds.append(pred_idx)
                else:
                    # If not overriding only travelling, we might override everything (if logic dictates)
                    # But per original Part 10, it was often just travelling/swinging/digging.
                    # We stick to the prompt's param `override_travelling_only` as the switch.
                    # If False, we assume standard behavior (override all motion classes? or none?)
                    # For safety, let's assume False means "Apply to all non-idling classes"
                    if pred_label != 'idling':
                         corrected_preds.append(idling_idx)
                    else:
                         corrected_preds.append(pred_idx)
            else:
                corrected_preds.append(pred_idx)
    else:
        corrected_preds = list(raw_preds)

    # 2. Majority Voting
    window_sec = params['min_activity_duration_s']
    window_size = int(window_sec * fps)
    
    smoothed = []
    half_window = window_size // 2
    for i in range(len(corrected_preds)):
        start = max(0, i - half_window)
        end   = min(len(corrected_preds), i + half_window + 1)
        window = corrected_preds[start:end]
        most_common = stats.mode(window, keepdims=True)[0][0]
        smoothed.append(most_common)

    # --- 3. FSM Cleaning ---
    if params['enable_fsm']:
        # Pass the grid parameter directly during creation
        fsm = ActivityFSM(target_fps=fps, min_dwell=params['fsm_min_dwell_seconds'])
        
        # Run FSM
        smoothed = fsm.clean_sequence(smoothed, confs, activity_names)

    return smoothed

def calculate_cycles_and_prod(predictions, activity_names, fps, bucket_payload):
    if 'digging' not in activity_names: return [], 0.0

    digging_idx = activity_names.index('digging')
    digging_starts = [i for i in range(len(predictions))
                      if predictions[i] == digging_idx and
                         (i == 0 or predictions[i-1] != digging_idx)]

    cycles = []
    for idx in range(len(digging_starts) - 1):
        start = digging_starts[idx]
        end   = digging_starts[idx + 1]
        duration = (end - start) / fps
        cycles.append({'duration_seconds': duration})

    if not cycles: return [], 0.0

    total_s = sum(c['duration_seconds'] for c in cycles)
    cycles_per_hr = len(cycles) / (total_s / 3600.0)
    prod = cycles_per_hr * bucket_payload
    return cycles, prod

# ============================
# 5. EVALUATION LOGIC
# ============================

def parse_time_range(raw: str):
    parts = re.split(r'\s*-\s*', raw.strip(), maxsplit=1)
    times = []
    for p in parts:
        p = p.strip().replace(" ", "")
        m, s = p.split(":")
        times.append(int(m) * 60 + int(s))
    return float(times[0]), float(times[1])

def load_gt_from_xlsx(xlsx_path):
    xls = pd.ExcelFile(xlsx_path)
    result = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
        segments = []
        for idx in range(1, len(df)):
            time_raw  = str(df.iloc[idx, 0]).strip()
            label_raw = str(df.iloc[idx, 1]).strip().lower()
            if time_raw in ("nan", "") or label_raw in ("nan", ""): continue
            try:
                s, e = parse_time_range(time_raw)
                if label_raw in ALL_CLASSES:
                    segments.append((s, e, label_raw))
            except: continue
        result[sheet] = segments
    return result

def evaluate_run(predictions_idx, confidences, activity_names, fps, gt_data, output_dir, run_name):
    """
    Compares the current prediction list against ground truth.
    Generates CSVs and plots in output_dir.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert list of indices to list of dicts for evaluation func
    pred_dicts = {}
    for i, (p_idx, conf) in enumerate(zip(predictions_idx, confidences)):
        pred_dicts[i] = {
            "time_s": i / fps,
            "activity": activity_names[p_idx],
            "confidence": conf
        }
        
    max_pred_time = len(predictions_idx) / fps
    all_comparison = []

    # Iterate over sheets (Day 2, Day 3, etc.)
    for sheet_name, raw_segs in gt_data.items():
        # Check overlap
        if not raw_segs: continue
        sheet_start = raw_segs[0][0]
        sheet_end = raw_segs[-1][1]
        
        # If no overlap with video time, skip (likely different day)
        if sheet_start > max_pred_time: continue
        
        # Build Map
        gt_map = {}
        for s, e, lab in raw_segs:
            # Clip to video duration
            e = min(e, max_pred_time)
            if s >= e: continue
            
            sf = int(round(s * fps))
            ef = int(round(e * fps))
            for f in range(sf, ef):
                gt_map[f] = lab
                
        # Compare
        sheet_comparison = []
        for f in sorted(gt_map.keys()):
            if f in pred_dicts:
                gt_lab = gt_map[f]
                pred_lab = pred_dicts[f]["activity"]
                match = (gt_lab == pred_lab)
                sheet_comparison.append((f, gt_lab, pred_lab, match))
                
        if not sheet_comparison: continue
        all_comparison.extend(sheet_comparison)

        # Plot Timeline for this sheet
        plot_timeline(raw_segs, pred_dicts, fps, output_dir / f"{run_name}_{sheet_name}_timeline.png", f"{run_name} - {sheet_name}")

    # Calculate Global Metrics
    if not all_comparison:
        print(f"    Warning: No overlapping ground truth found for evaluation.")
        return 0.0

    correct = sum(1 for *_, m in all_comparison if m)
    total = len(all_comparison)
    accuracy = correct / total if total > 0 else 0
    
    # Confusion Matrix
    cm = np.zeros((len(ALL_CLASSES), len(ALL_CLASSES)), dtype=int)
    idx_map = {c: i for i, c in enumerate(ALL_CLASSES)}
    for _, gt, pred, _ in all_comparison:
        if gt in idx_map and pred in idx_map:
            cm[idx_map[gt]][idx_map[pred]] += 1
            
    plot_confusion(cm, ALL_CLASSES, output_dir / f"{run_name}_confusion.png")
    
    return accuracy

def plot_confusion(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums
    ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=8, color=color)
            
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_title("Confusion Matrix")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_timeline(segments, predictions, fps, save_path, title):
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # GT Bars
    for s, e, lab in segments:
        ax.barh(1.4, e-s, left=s, height=0.6, color=ACTIVITY_COLORS.get(lab, '#ccc'))
        
    # Pred Bars (RLE)
    pred_list = sorted(predictions.items())
    if not pred_list: return
    
    curr_lab = pred_list[0][1]['activity']
    start_f = pred_list[0][0]
    
    for f, d in pred_list:
        lab = d['activity']
        if lab != curr_lab:
            # draw prev
            end_s = f / fps
            start_s = start_f / fps
            ax.barh(0.4, end_s-start_s, left=start_s, height=0.6, color=ACTIVITY_COLORS.get(curr_lab, '#ccc'))
            curr_lab = lab
            start_f = f
    # last
    ax.barh(0.4, (pred_list[-1][0] - start_f)/fps, left=start_f/fps, height=0.6, color=ACTIVITY_COLORS.get(curr_lab, '#ccc'))
    
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(["Pred", "GT"])
    ax.set_title(title)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def get_inference_data(model, yolo, video_path, config, cache_dir):
    """
    Checks for 'raw_inference_cache.xlsx'. 
    If found: Loads prediction data from Excel.
    If not: Runs AI inference, saves to Excel, then returns data.
    """
    cache_path = cache_dir / "raw_inference_cache.xlsx"
    
    if cache_path.exists():
        print(f"\n[Cache] Found existing data at {cache_path}")
        print("       Loading raw predictions from Excel (Skipping AI)...")
        
        df = pd.read_excel(cache_path)
        
        # Convert columns back to lists
        raw_preds = df['Raw_Pred_Idx'].tolist()
        confs     = df['Confidence'].tolist()
        
        # Reconstruct bbox list from columns [x1, y1, x2, y2]
        bboxes = df[['x1', 'y1', 'x2', 'y2']].values.tolist()
        
        print(f"       Loaded {len(raw_preds)} frames.")
        return raw_preds, confs, bboxes
    
    else:
        print(f"\n[Cache] No cache found. Running Heavy Inference...")
        
        # Run the heavy AI model
        raw_preds, confs, bboxes = run_heavy_inference(model, yolo, video_path, config)       
        print(f"[Cache] Saving results to {cache_path}...")
        
        # Prepare DataFrame
        # bboxes is a list of [x1, y1, x2, y2], so we split it into columns
        bbox_np = np.array(bboxes)
        
        df = pd.DataFrame({
            'Frame': range(len(raw_preds)),
            'Raw_Pred_Idx': raw_preds,
            'Confidence': confs,
            'x1': bbox_np[:, 0],
            'y1': bbox_np[:, 1],
            'x2': bbox_np[:, 2],
            'y2': bbox_np[:, 3]
        })
        
        df.to_excel(cache_path, index=False)
        print("       Save complete.")
        
        return raw_preds, confs, bboxes

# ============================
# 6. MAIN EXECUTION
# ============================

def main():
    print("="*60)
    print("RESNET 3D OPTIMIZATION PIPELINE")
    print("="*60)
    
    OUTPUT_BASE_DIR.mkdir(exist_ok=True, parents=True)

    # 1. Load Models
    print("\n[Step 1] Loading Models...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']
    target_fps = config['target_fps']
    activity_names = checkpoint['activity_names']
    
    # --- FIX: Use torchvision directly, no torch.hub/pytorchvideo needed ---
    from torchvision.models.video import r3d_18
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE).eval()
    
    yolo = YOLO(str(YOLO_PATH))

    # 2. Run Heavy Inference Once
    print(f"\n[Step 2] Processing Video: {VIDEO_PATH.name}")
    raw_preds, confs, bboxes = get_inference_data(model, yolo, VIDEO_PATH, config, OUTPUT_BASE_DIR)
    
    # 3. Load Ground Truth
    print(f"\n[Step 3] Loading Ground Truth from {TASKS_XLSX.name}")
    gt_data = load_gt_from_xlsx(TASKS_XLSX)

    # 4. Grid Search Loop
    print(f"\n[Step 4] Starting Grid Search over {len(list(itertools.product(*HYPERPARAM_GRID.values())))} combinations...")
    
    keys = list(HYPERPARAM_GRID.keys())
    master_report_path = OUTPUT_BASE_DIR / "master_optimization_report.csv"
    
    # Initialize master report file
    with open(master_report_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys + ['accuracy_percent', 'cycles_per_hr', 'productivity_lcy_hr', 'run_folder'])

    param_combinations = list(itertools.product(*HYPERPARAM_GRID.values()))
    
    for i, values in enumerate(param_combinations):
        params = dict(zip(keys, values))
        run_id = f"Run_{i+1:03d}"
        print(f"\n--- {run_id} Parameters ---")
        print(json.dumps(params, indent=2))
        
        # A. Post-Process
        final_preds = apply_post_processing(raw_preds, confs, bboxes, activity_names, params, target_fps)
        
        # B. Metrics
        cycles, prod = calculate_cycles_and_prod(final_preds, activity_names, target_fps, BUCKET_PAYLOAD)
        
        # C. Evaluate
        run_dir = OUTPUT_BASE_DIR / run_id
        acc = evaluate_run(final_preds, confs, activity_names, target_fps, gt_data, run_dir, run_id)
        
        # D. Save Prediction CSV for this run
        with open(run_dir / "predictions.csv", 'w') as f:
            f.write("Frame,Activity\n")
            for idx, p in enumerate(final_preds):
                f.write(f"{idx},{activity_names[p]}\n")
        
        # E. Log to Master Report
        with open(master_report_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Safe conversion of values
            acc_str = f"{acc*100:.2f}"
            cycles_str = f"{len(cycles)/(len(final_preds)/target_fps/3600):.2f}" if final_preds else "0"
            prod_str = f"{prod:.2f}"
            
            writer.writerow(list(values) + [acc_str, cycles_str, prod_str, run_id])
            
        print(f"  -> Accuracy: {acc*100:.1f}% | Prod: {prod:.1f} LCY/hr")

    print(f"\nDONE. All results saved to {OUTPUT_BASE_DIR}")
if __name__ == "__main__":
    if not VIDEO_PATH.exists():
        print(f"ERROR: Video {VIDEO_PATH} does not exist.")
    elif not TASKS_XLSX.exists():
        print(f"ERROR: GT File {TASKS_XLSX} does not exist.")
    else:
        main()