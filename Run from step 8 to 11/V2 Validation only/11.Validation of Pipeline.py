"""
Combined 3D ResNet Inference & Evaluation Pipeline (Global Grid Search)
=======================================================================
Combines:
  1. Inference (AI + Physics + Post-processing)
  2. Ground Truth Evaluation (vs Tasks.xlsx)

Upgrades:
 - Dataset-level Hyperparameter Tuning (Evaluates across all videos)
 - RAM Caching (Avoids reading disk repeatedly during the grid search)
 - Dynamic Early Stopping (Stops poor runs if they fall below the Top 15 baseline)
 - Selective Plotting (Only generates heavy visualizations for the Top 5 results)
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
        self.min_dwell_seconds = min_dwell  
        self.min_dwell_frames = int(self.min_dwell_seconds * target_fps)
    
    def clean_sequence(self, predictions, confidences, activity_names):
        """Main cleaning pipeline."""
        pred_names = [activity_names[p] for p in predictions]
        
        # Step 1: Enforce dwell + transitions
        stage1 = self._enforce_dwell_and_transitions(pred_names, confidences)
        
        # Step 2: Repair impossible patterns
        stage2 = self._repair_impossible_sequences(stage1)
        
        # Step 3: Final smoothing
        cleaned_names = self._final_smoothing(stage2)
        
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
            
            if prev == 'digging' and curr == 'travelling' and next_ == 'loading':
                repaired[i] = 'swinging'
            elif prev == 'loading' and curr == 'travelling':
                repaired[i] = 'swinging'
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
BASE_DIR        = Path("/data/shubhan_avik_work/Targeted_run_2")
MODEL_PATH      = BASE_DIR / "resnet3d_best_ten_days.pth"
YOLO_PATH       = BASE_DIR / "best.pt"
TASKS_XLSX      = BASE_DIR / "Tasks_Split_by_Video_xlwings.xlsx"
OUTPUT_BASE_DIR = BASE_DIR / "optimization_runs"
VIDEO_FILES     = sorted(BASE_DIR.glob("*.mp4"))

# --- STATIC CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVGOL_POLYORDER = 2
ROLL_MEDIAN_WINDOW = 5
AREA_THRESHOLD_PERCENT = 0.5
BUCKET_PAYLOAD = 1.5  # LCY

MEAN = np.load(str(BASE_DIR / "dataset_mean.npy"))
STD  = np.load(str(BASE_DIR / "dataset_std.npy"))

ALL_CLASSES = ["digging", "idling", "loading", "swinging", "travelling"]
ACTIVITY_COLORS = {
    "digging"   : "#E74C3C",
    "idling"    : "#95A5A6",
    "loading"   : "#2ECC71",
    "swinging"  : "#3498DB",
    "travelling": "#F39C12",
}

# ============================
# 2. UTILITY FUNCTIONS
# ============================

def smooth_signal_med(signal, window_len):
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
    if not bbox_list: return []
    
    data = np.array(bbox_list)
    x1, y1 = data[:, 0], data[:, 1]
    x2, y2 = data[:, 2], data[:, 3]
    
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = (x2 - x1) * (y2 - y1)
    
    cx_s = smooth_signal_med(cx, 11)
    cy_s = smooth_signal_med(cy, 11)
    area_s = smooth_signal_med(area, 11)
    
    area_mean = np.mean(area_s)
    area_std_thresh = (AREA_THRESHOLD_PERCENT / 100.0) * area_mean
    
    n = len(data)
    idle_mask = np.zeros(n, dtype=bool)
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
        dist = np.sqrt(np.diff(cx_s)**2 + np.diff(cy_s)**2)
        dA = np.abs(np.diff(area_s))
        if np.std(dist) < dist_threshold and np.std(dA) < area_std_thresh:
            idle_mask[:] = True
            
    return idle_mask

# ============================
# 3. INFERENCE CORE
# ============================

def extract_frames_generator(video_path, yolo_model, target_fps=25):
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
    cached_bbox = [0, 0, 0, 0]
    
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
    clip_length = config['clip_length']
    clip_buffer = deque(maxlen=clip_length)
    
    raw_predictions = []
    confidences = []
    bbox_history = []
    
    frame_gen = extract_frames_generator(video_path, yolo_model, config['target_fps'])
    
    print("  [Running AI] Processing frames through 3D ResNet...")
    
    for frame, bbox in frame_gen:
        clip_buffer.append(frame)
        bbox_history.append(bbox)
        
        if len(clip_buffer) < clip_length:
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

    while len(raw_predictions) < len(bbox_history):
        raw_predictions.insert(0, raw_predictions[0] if raw_predictions else 0)
        confidences.insert(0, confidences[0] if confidences else 0.0)
        
    return raw_predictions, confidences, bbox_history

def get_inference_data(model, yolo, video_path, config, cache_dir):
    cache_path = cache_dir / f"{video_path.stem}_raw_inference_cache.xlsx"
    
    if cache_path.exists():
        print(f"  [Cache Hit] Loading raw predictions from {cache_path.name}")
        df = pd.read_excel(cache_path)
        raw_preds = df['Raw_Pred_Idx'].tolist()
        confs     = df['Confidence'].tolist()
        bboxes = df[['x1', 'y1', 'x2', 'y2']].values.tolist()
        return raw_preds, confs, bboxes
    else:
        print(f"  [Cache Miss] Running Heavy Inference on {video_path.name}")
        raw_preds, confs, bboxes = run_heavy_inference(model, yolo, video_path, config)       
        
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
        return raw_preds, confs, bboxes

# ============================
# 4. POST-PROCESSING
# ============================

def apply_post_processing(raw_preds, confs, bboxes, activity_names, params, fps):
    idle_mask = compute_idling_mask(bboxes, params['idle_window'], params['dist_threshold'])
    
    if 'idling' in activity_names:
        idling_idx = activity_names.index('idling')
        corrected_preds = []
        
        for i, pred_idx in enumerate(raw_preds):
            if idle_mask[i]:
                pred_label = activity_names[pred_idx]
                if params['override_travelling_only']:
                    if pred_label == 'travelling':
                        corrected_preds.append(idling_idx)
                    else:
                        corrected_preds.append(pred_idx)
                else:
                    if pred_label != 'idling':
                         corrected_preds.append(idling_idx)
                    else:
                         corrected_preds.append(pred_idx)
            else:
                corrected_preds.append(pred_idx)
    else:
        corrected_preds = list(raw_preds)

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

    if params['enable_fsm']:
        fsm = ActivityFSM(target_fps=fps, min_dwell=params['fsm_min_dwell_seconds'])
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

def evaluate_run(predictions_idx, confidences, activity_names, fps, gt_data, output_dir, run_name, save_plots=True):
    """
    Compares the current prediction list against ground truth.
    Optional plot saving to save I/O time during large sweeps.
    """
    pred_dicts = {}
    for i, (p_idx, conf) in enumerate(zip(predictions_idx, confidences)):
        pred_dicts[i] = {
            "time_s": i / fps,
            "activity": activity_names[p_idx],
            "confidence": conf
        }
        
    max_pred_time = len(predictions_idx) / fps
    all_comparison = []

    for sheet_name, raw_segs in gt_data.items():
        if not raw_segs: continue
        sheet_start = raw_segs[0][0]
        
        if sheet_start > max_pred_time: continue
        
        gt_map = {}
        for s, e, lab in raw_segs:
            e = min(e, max_pred_time)
            if s >= e: continue
            
            sf = int(round(s * fps))
            ef = int(round(e * fps))
            for f in range(sf, ef):
                gt_map[f] = lab
                
        sheet_comparison = []
        for f in sorted(gt_map.keys()):
            if f in pred_dicts:
                gt_lab = gt_map[f]
                pred_lab = pred_dicts[f]["activity"]
                match = (gt_lab == pred_lab)
                sheet_comparison.append((f, gt_lab, pred_lab, match))
                
        if not sheet_comparison: continue
        all_comparison.extend(sheet_comparison)

        if save_plots:
            output_dir.mkdir(exist_ok=True, parents=True)
            plot_timeline(raw_segs, pred_dicts, fps, output_dir / f"{run_name}_{sheet_name}_timeline.png", f"{run_name} - {sheet_name}")

    if not all_comparison:
        return 0.0

    correct = sum(1 for *_, m in all_comparison if m)
    total = len(all_comparison)
    accuracy = correct / total if total > 0 else 0
    
    if save_plots:
        output_dir.mkdir(exist_ok=True, parents=True)
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
    
    for s, e, lab in segments:
        ax.barh(1.4, e-s, left=s, height=0.6, color=ACTIVITY_COLORS.get(lab, '#ccc'))
        
    pred_list = sorted(predictions.items())
    if not pred_list: return
    
    curr_lab = pred_list[0][1]['activity']
    start_f = pred_list[0][0]
    
    for f, d in pred_list:
        lab = d['activity']
        if lab != curr_lab:
            end_s = f / fps
            start_s = start_f / fps
            ax.barh(0.4, end_s-start_s, left=start_s, height=0.6, color=ACTIVITY_COLORS.get(curr_lab, '#ccc'))
            curr_lab = lab
            start_f = f
            
    ax.barh(0.4, (pred_list[-1][0] - start_f)/fps, left=start_f/fps, height=0.6, color=ACTIVITY_COLORS.get(curr_lab, '#ccc'))
    
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(["Pred", "GT"])
    ax.set_title(title)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# ============================
# 6. MAIN EXECUTION
# ============================

def main():
    print("="*60)
    print("RESNET 3D OPTIMIZATION PIPELINE — GLOBAL TUNING + DYNAMIC STOP")
    print("="*60)

    if not VIDEO_FILES:
        print("ERROR: No .mp4 files found")
        return

    OUTPUT_BASE_DIR.mkdir(exist_ok=True, parents=True)

    # =========================
    # Load Models
    # =========================
    print("\n[Step 1] Loading Models...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']
    target_fps = config['target_fps']
    activity_names = checkpoint['activity_names']

    from torchvision.models.video import r3d_18
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE).eval()

    yolo = YOLO(str(YOLO_PATH))

    # =========================
    # Ground Truth & RAM Cache
    # =========================
    print("\n[Step 2] Loading Ground Truth & Caching AI Predictions...")
    gt_data = load_gt_from_xlsx(TASKS_XLSX)
    
    memory_cache = {}
    valid_videos = []

    for video_path in VIDEO_FILES:
        video_stem = video_path.stem
        
        sheet_name = next((s for s in gt_data.keys() if s.lower() == video_stem.lower()), None)
        if not sheet_name:
            print(f"  [SKIPPING] No Ground Truth sheet found for {video_stem}")
            continue
            
        valid_videos.append((video_path, video_stem, sheet_name))
        
        video_output_dir = OUTPUT_BASE_DIR / video_stem
        video_output_dir.mkdir(exist_ok=True, parents=True)
        memory_cache[video_stem] = get_inference_data(
            model, yolo, video_path, config, video_output_dir
        )

    # =========================
    # Grid Setup
    # =========================
    keys = list(HYPERPARAM_GRID.keys())
    param_combinations = list(itertools.product(*HYPERPARAM_GRID.values()))
    print(f"\n[Step 3] Grid size: {len(param_combinations)} combinations on {len(valid_videos)} videos.")

    global_results = []
    top_15_scores = [] 
    
    # Give the run time to stabilize before early stopping
    MIN_VIDEOS_BEFORE_STOP = max(3, len(valid_videos) // 3)

    # =========================================================
    # PARAMETER LOOP
    # =========================================================
    for i, values in enumerate(param_combinations):
        params = dict(zip(keys, values))
        run_id = f"PARAM_SET_{i+1:03d}"

        print("\n" + "="*70)
        print(f"RUNNING: {run_id}")
        
        # Determine Dynamic Threshold (Minimum score in the Top 15 list)
        dynamic_threshold = 0.0
        if len(top_15_scores) >= 15:
            dynamic_threshold = min(top_15_scores)
            print(f"Dynamic Threshold Active: Must maintain > {dynamic_threshold*100:.2f}% net weighted accuracy")
        else:
            print(f"Dynamic Threshold Inactive: Building baseline ({len(top_15_scores)}/15 completed)")
        print("="*70)

        per_video_acc = []
        per_video_prod = []
        per_video_len = []
        video_details = []

        stopped_early = False
        stop_reason = None
        
        running_correct_frames = 0
        running_total_frames = 0
        videos_done = 0

        # =========================================
        # VIDEO LOOP (Using RAM Cache)
        # =========================================
        for video_path, video_stem, sheet_name in valid_videos:
            
            raw_preds, confs, bboxes = memory_cache[video_stem]

            final_preds = apply_post_processing(
                raw_preds, confs, bboxes,
                activity_names, params, target_fps
            )

            cycles, prod = calculate_cycles_and_prod(
                final_preds, activity_names, target_fps, BUCKET_PAYLOAD
            )

            gt_for_video = {sheet_name: gt_data[sheet_name]}
            
            acc = evaluate_run(
                final_preds, confs, activity_names, target_fps,
                gt_for_video, OUTPUT_BASE_DIR, run_id, 
                save_plots=False 
            )

            video_len = len(final_preds)

            per_video_acc.append(acc)
            per_video_prod.append(prod)
            per_video_len.append(video_len)

            video_details.append({
                "video": video_stem,
                "acc": acc,
                "prod": prod,
                "len": video_len
            })

            # Apples-to-Apples Weighted Accuracy calculation
            running_correct_frames += (acc * video_len)
            running_total_frames += video_len
            running_net_weighted_acc = running_correct_frames / running_total_frames
            
            videos_done += 1

            print(f"  {video_stem}: acc={acc*100:.2f}% | net_running={running_net_weighted_acc*100:.2f}%")

            # =========================
            # DYNAMIC EARLY STOPPING
            # =========================
            if len(top_15_scores) >= 15 and videos_done >= MIN_VIDEOS_BEFORE_STOP:
                if running_net_weighted_acc < dynamic_threshold:
                    stopped_early = True
                    stop_reason = f"Net running avg ({running_net_weighted_acc*100:.1f}%) fell below Top-15 threshold ({dynamic_threshold*100:.1f}%)"
                    print(f"  [EARLY STOP] {stop_reason}")
                    break

        # =========================================
        # AGGREGATION
        # =========================================
        if len(per_video_acc) == 0:
            continue

        macro_acc = float(np.mean(per_video_acc))
        net_weighted_accuracy = float(running_correct_frames / running_total_frames)
        avg_prod = float(np.mean(per_video_prod))

        # Update the Top 15 rolling baseline
        if not stopped_early:
            top_15_scores.append(net_weighted_accuracy)
            top_15_scores = sorted(top_15_scores, reverse=True)[:15]

        global_results.append({
            "params": params,
            "run_id": run_id,
            "macro_acc": macro_acc,
            "net_weighted_accuracy": net_weighted_accuracy,
            "avg_prod": avg_prod,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "videos_evaluated": videos_done,
            "total_videos": len(valid_videos),
            "video_details": video_details
        })

    # =========================
    # SORT & VISUALIZE TOP 5
    # =========================
    global_results.sort(key=lambda x: x["net_weighted_accuracy"], reverse=True)
    top5 = global_results[:5]

    print("\n" + "="*60)
    print("TOP 5 PARAMETER SETS (Generating Output Plots...)")
    print("="*60)

    top5_dir = OUTPUT_BASE_DIR / "TOP_5_BEST_RUNS"
    top5_dir.mkdir(exist_ok=True)

    for rank, r in enumerate(top5):
        print(f"\n[{rank+1}] RUN: {r['run_id']} | Net Weighted Acc: {round(r['net_weighted_accuracy']*100, 2)}%")
        print("Params:", r["params"])
        
        run_viz_dir = top5_dir / f"Rank_{rank+1}_{r['run_id']}"
        run_viz_dir.mkdir(exist_ok=True)
        
        for video_path, video_stem, sheet_name in valid_videos:
            raw_preds, confs, bboxes = memory_cache[video_stem]
            final_preds = apply_post_processing(
                raw_preds, confs, bboxes, activity_names, r["params"], target_fps
            )
            
            evaluate_run(
                final_preds, confs, activity_names, target_fps,
                {sheet_name: gt_data[sheet_name]}, run_viz_dir, f"{r['run_id']}_{video_stem}",
                save_plots=True
            )

    # =========================
    # SAVE OUTPUTS
    # =========================
    with open(OUTPUT_BASE_DIR / "top5_params.json", "w") as f:
        json.dump(top5, f, indent=2)

    pd.DataFrame(global_results).to_csv(OUTPUT_BASE_DIR / "all_param_results.csv", index=False)

    print("\nDONE.")
    print("Saved JSON/CSV reports and Top 5 visualizations in:", OUTPUT_BASE_DIR)

if __name__ == "__main__":
    if not TASKS_XLSX.exists():
        print(f"ERROR: GT file not found: {TASKS_XLSX}")
    else:
        main()
