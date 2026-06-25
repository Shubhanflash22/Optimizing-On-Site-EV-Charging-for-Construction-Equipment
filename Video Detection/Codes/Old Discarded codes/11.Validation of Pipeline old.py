"""
Part 11 - Ground Truth Validation (CSV Based)
=============================================
Compares model predictions against the Ground Truth CSV file.

1. Loads 'frame_predictions.csv' (Model Output)
2. Loads 'Tasks.xlsx - Day2.csv' (Ground Truth)
3. Aligns them frame-by-frame
4. Generates Accuracy, Confusion Matrix, and Timeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ============================
# 1. CONFIGURATION
# ============================

# Path to the Prediction CSV (Output from Part 10)
PRED_CSV_PATH = Path(r"C:\Users\shubh\Desktop\DELETE AFTER USE\FINAL Codes and results from Remote\frame_predictions.csv")

# Path to the Ground Truth CSV (The file you just uploaded)
GT_CSV_PATH   = Path(r"C:\Users\shubh\Desktop\DELETE AFTER USE\Tasks.xlsx") 

# FPS used in Part 10 (Critical for time-to-frame conversion)
FPS = 25 

# ============================
# 2. PARSING FUNCTIONS
# ============================

def parse_time_str(time_str):
    """
    Parses 'MM:SS' or 'HH:MM:SS' into total seconds.
    Handles typos like spaces or extra chars.
    """
    # Remove non-numeric/colon chars (e.g. spaces)
    clean_str = re.sub(r'[^\d:]', '', str(time_str))
    parts = list(map(int, clean_str.split(':')))
    
    if len(parts) == 2:   # MM:SS
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3: # HH:MM:SS
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def load_ground_truth_from_csv(file_path, total_frames, fps):
    """
    Reads the Task file (CSV or Excel) and converts it to a frame-by-frame label array.
    """
    file_path = Path(file_path)
    print(f"Loading Ground Truth from: {file_path.name}...")
    
    # 1. Load Data based on file extension
    if file_path.suffix.lower() == '.xlsx':
        # Requires: pip install openpyxl
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        # Fallback to CSV with robust encoding (handles Windows Excel CSVs)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Initialize Ground Truth array
    gt_labels = ["unknown"] * total_frames
    valid_mask = [False] * total_frames
    
    mapped_count = 0
    
    # Iterate through rows
    # We use iloc to access columns by position (0=Time, 1=Activity) 
    # to avoid issues if column names change.
    for index, row in df.iterrows():
        # 1. Parse Time Column (Column 0)
        time_range = str(row.iloc[0]) 
        if '-' not in time_range: 
            continue # Skip invalid rows
            
        try:
            start_str, end_str = time_range.split('-')
            start_sec = parse_time_str(start_str)
            end_sec = parse_time_str(end_str)
            
            # 2. Parse Activity Column (Column 1)
            activity = str(row.iloc[1]).strip().lower()
            
            # 3. Convert to Frames
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            
            # Clip to video bounds
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            
            # 4. Fill Array
            if end_frame > start_frame:
                for f in range(start_frame, end_frame):
                    gt_labels[f] = activity
                    valid_mask[f] = True
                mapped_count += 1
        except Exception as e:
            print(f"Skipping row {index}: {e}")
            continue
            
    print(f"  → Parsed {mapped_count} activity segments")
    return gt_labels, valid_mask

# ============================
# 3. MAIN PIPELINE
# ============================

def main():
    print("="*60)
    print("AUTOMATED VALIDATION REPORT")
    print("="*60)
    
    # --- Step 1: Load Predictions ---
    if not PRED_CSV_PATH.exists():
        print(f"❌ Error: Prediction file not found at {PRED_CSV_PATH}")
        return
        
    print("Loading Model Predictions...")
    df_pred = pd.read_csv(PRED_CSV_PATH)
    pred_labels = df_pred['Activity'].str.lower().tolist()
    total_frames = len(pred_labels)
    print(f"  → Loaded {total_frames} frames from model output")
    
    # --- Step 2: Load Ground Truth ---
    if not GT_CSV_PATH.exists():
        print(f"❌ Error: Ground Truth file not found at {GT_CSV_PATH}")
        return
        
    gt_labels, valid_mask = load_ground_truth_from_csv(GT_CSV_PATH, total_frames, FPS)
    
    # --- Step 3: Align Data ---
    y_true = []
    y_pred = []
    
    # Only compare frames where we have a Ground Truth label
    # (Ignores gaps or unlabelled sections in the Excel file)
    for i in range(total_frames):
        if valid_mask[i]:
            y_true.append(gt_labels[i])
            y_pred.append(pred_labels[i])
            
    if len(y_true) == 0:
        print("❌ Error: No overlapping valid frames found! Check FPS or CSV format.")
        return

    print(f"\nValidating on {len(y_true)} labeled frames ({len(y_true)/25/60:.1f} minutes)...")

    # --- Step 4: Calculate Metrics ---
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🏆 OVERALL ACCURACY: {acc*100:.2f}%")
    
    # Detailed Report
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    classes = sorted(list(set(y_true + y_pred)))
    print(classification_report(y_true, y_pred, labels=classes, digits=4))
    
    # --- Step 5: Visualization ---
    
    # A. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted (Model)')
    plt.ylabel('Actual (Ground Truth)')
    plt.title(f'Confusion Matrix (Acc: {acc*100:.1f}%)')
    
    matrix_path = PRED_CSV_PATH.parent / "validation_matrix.png"
    plt.savefig(matrix_path)
    print(f"Saved Confusion Matrix: {matrix_path}")
    
    # B. Timeline Compare (First 5 mins)
    plt.figure(figsize=(15, 5))
    limit = (len(y_true))
    
    # Map labels to integers
    label_map = {name: i for i, name in enumerate(classes)}
    y_t_num = [label_map[l] for l in y_true[:limit]]
    y_p_num = [label_map[l] for l in y_pred[:limit]]
    time_axis = np.arange(limit) / FPS
    
    plt.step(time_axis, y_t_num, where='post', label='Ground Truth', lw=2)
    plt.step(time_axis, y_p_num, where='post', label='Prediction', lw=2, alpha=0.7, linestyle='--')
    
    plt.yticks(list(label_map.values()), list(label_map.keys()))
    plt.xlabel("Time (seconds)")
    plt.title("Timeline Comparison (First 5 Minutes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    timeline_path = PRED_CSV_PATH.parent / "validation_timeline.png"
    plt.savefig(timeline_path)
    print(f"Saved Timeline Plot: {timeline_path}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()