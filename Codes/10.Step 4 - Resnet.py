import os
import cv2
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from itertools import groupby

# ============================
# Configuration (edit paths)
# ============================
SAVE_DIR = r"C:\Users\shubh\Desktop\Test\Tubes"
METADATA_PATH = os.path.join(SAVE_DIR, "track_metadata.csv")
ACTIVITY_OUTPUT_CSV = r"C:\Users\shubh\Desktop\Test\Activity_Output.csv"
ACTIVITY_VISUAL_CSV = r"C:\Users\shubh\Desktop\Test\Activity_Visual.csv"
MODEL_PATH = r"C:\Users\shubh\Desktop\Test\best_activity_model.pth"

FPS = 59           # frames per second of video
CLIP_LENGTH = 16   # frames per clip for the 3D model
CROP_SIZE = 112    # model input spatial size (H=W=112)
STRIDE = 16        # clip stride (frames). Consider lowering to 4 for overlap.
VOTING_SECONDS = 2.0
VOTING_WINDOW = int(VOTING_SECONDS * FPS)

ACTIVITY_NAMES = {
    0: 'digging',
    1: 'loading',
    2: 'swinging'
}

# ============================
# Model classes (unchanged)
# ============================

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual; out = self.relu(out)
        return out

class Bottleneck3D(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual; out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super(ResNet3D, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 2, 2))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

def resnet3d_50(num_classes=3):
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], num_classes=num_classes)

# ============================
# Start of linear procedural workflow
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
print("[INFO] Instantiating model (ResNet3D-50) ...")
model = resnet3d_50(num_classes=len(ACTIVITY_NAMES))
model = model.to(device)
print("[INFO] Model instantiated.")

if os.path.exists(MODEL_PATH):
    try:
        print(f"[INFO] Loading model weights from: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[INFO] Model weights loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        print("[WARNING] Continuing with randomly initialized model.")
else:
    print(f"[WARNING] Model file not found at {MODEL_PATH}. Using random initialization.")

model.eval()
print("[INFO] Model set to eval() mode.\n")

# 3) Check metadata file
if not os.path.exists(METADATA_PATH):
    print(f"[ERROR] Metadata file not found at: {METADATA_PATH}")
    print("Please generate tracking metadata (step 2) and ensure METADATA_PATH is correct.")
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
    print(f"[TRACK {track_counter}] frame_folder={track_folder}")

    if not track_folder or not os.path.exists(track_folder):
        print(f"[TRACK {track_counter} WARNING] track folder missing: {track_folder}. Skipping.")
        continue

    # 4a) Load frames from folder (sequential, verbose)
    t0 = time.time()
    print(f"[TRACK {track_counter}] Loading frame files from folder...")
    frame_files = sorted([f for f in os.listdir(track_folder) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
    print(f"[TRACK {track_counter}] Found {len(frame_files)} image files.")
    frames = []
    frame_indices = []

    for idx, fname in enumerate(frame_files):
        if (idx + 1) % 100 == 0:
            print(f"[TRACK {track_counter}]   Loaded {idx+1}/{len(frame_files)} frames so far...")
        full_path = os.path.join(track_folder, fname)
        img = cv2.imread(full_path)
        if img is None:
            print(f"[TRACK {track_counter} WARNING] Failed to read image: {full_path} (skipping)")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to model input size to avoid huge tensors
        try:
            img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
        except Exception as e:
            print(f"[TRACK {track_counter} ERROR] Resize failed for {full_path}: {e}")
            # fallback: skip this frame
            continue
        frames.append(img)
        # try to parse frame index from filename, fallback to idx
        name_no_ext = os.path.splitext(fname)[0]
        try:
            frame_idx = int(name_no_ext)
        except:
            frame_idx = idx
        frame_indices.append(frame_idx)

    t1 = time.time()
    print(f"[TRACK {track_counter}] Finished loading frames in {t1 - t0:.2f}s. Total valid frames: {len(frames)}")
    if len(frames) == 0:
        print(f"[TRACK {track_counter}] No valid frames found, skipping track.")
        continue

    # 4b) Prepare sliding window clip indices
    num_frames = len(frames)
    print(f"[TRACK {track_counter}] num_frames = {num_frames}")
    stride = STRIDE
    print(f"[TRACK {track_counter}] Using CLIP_LENGTH={CLIP_LENGTH}, STRIDE={stride}")
    num_clips = max(1, (num_frames - CLIP_LENGTH) // stride + 1) if num_frames >= CLIP_LENGTH else 1
    print(f"[TRACK {track_counter}] Calculated num_clips = {num_clips}")

    # We'll produce a raw prediction list aligned with sorted frame_indices
    # For simplicity, predictions_by_frame_idx will be dict: frame_idx -> predicted_label
    predictions_by_frame_idx = {}

    # 4c) Iterate clips (non-overlapping or overlapping depending on stride)
    clip_counter = 0
    total_inference_time = 0.0
    for i in range(num_clips):
        clip_counter += 1
        start_idx = i * stride
        end_idx = min(start_idx + CLIP_LENGTH, num_frames)
        clip_frame_indices = frame_indices[start_idx:end_idx]
        clip_frames = frames[start_idx:end_idx]

        print(f"[TRACK {track_counter}] Processing clip {clip_counter}/{num_clips}: frames {start_idx} to {end_idx-1} (len={len(clip_frames)})")

        # pad if clip is shorter than CLIP_LENGTH
        if len(clip_frames) < CLIP_LENGTH:
            pad_count = CLIP_LENGTH - len(clip_frames)
            print(f"[TRACK {track_counter}]   Padding clip with {pad_count} frames (repeat last frame)")
            last_frame = clip_frames[-1]
            for _ in range(pad_count):
                clip_frames.append(last_frame)

        # convert to numpy array and normalize
        clip_np = np.array(clip_frames).astype(np.float32) / 255.0  # shape: (L, H, W, C)
        # convert to tensor shape (B, C, L, H, W)
        clip_tensor = torch.from_numpy(clip_np).float().permute(3, 0, 1, 2).unsqueeze(0)
        print(f"[TRACK {track_counter}]   Clip tensor shape before to(device): {clip_tensor.shape}")

        clip_tensor = clip_tensor.to(device)

        # inference
        t_inf0 = time.time()
        try:
            with torch.no_grad():
                output = model(clip_tensor)
                _, predicted = torch.max(output, dim=1)
                pred_label = int(predicted.item())
        except Exception as e:
            print(f"[TRACK {track_counter} ERROR] Inference failed on clip {clip_counter}: {e}")
            pred_label = -1  # invalid
        t_inf1 = time.time()
        inf_time = t_inf1 - t_inf0
        total_inference_time += inf_time
        print(f"[TRACK {track_counter}]   Inference time for clip {clip_counter}: {inf_time:.3f}s  -> predicted: {pred_label}")

        # assign prediction to the frames that belong to this clip (use clip_frame_indices)
        for fi in clip_frame_indices:
            predictions_by_frame_idx[fi] = pred_label

    # 4d) Handle remaining frames if any (when num_frames < CLIP_LENGTH or final remainder)
    if num_frames < CLIP_LENGTH:
        print(f"[TRACK {track_counter}] num_frames < CLIP_LENGTH, doing single padded inference across all frames...")
        clip_frames = frames[:]  # all frames
        if len(clip_frames) < CLIP_LENGTH:
            pad_count = CLIP_LENGTH - len(clip_frames)
            last_frame = clip_frames[-1]
            for _ in range(pad_count):
                clip_frames.append(last_frame)
        clip_np = np.array(clip_frames).astype(np.float32) / 255.0
        clip_tensor = torch.from_numpy(clip_np).float().permute(3,0,1,2).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                output = model(clip_tensor)
                _, predicted = torch.max(output, dim=1)
                pred_label = int(predicted.item())
        except Exception as e:
            print(f"[TRACK {track_counter} ERROR] Inference failed on padded short clip: {e}")
            pred_label = -1
        print(f"[TRACK {track_counter}]   Padded inference predicted: {pred_label}")
        for fi in frame_indices:
            predictions_by_frame_idx[fi] = pred_label

    print(f"[TRACK {track_counter}] Total inference time for track (sum of clips): {total_inference_time:.3f}s")
    print(f"[TRACK {track_counter}] Predictions assigned to {len(predictions_by_frame_idx)} frames (may be less than total frames if some clips errored).")

    # 4e) Sort frame indices and build raw prediction list in time order
    sorted_frame_items = sorted(predictions_by_frame_idx.items(), key=lambda x: x[0])
    if not sorted_frame_items:
        print(f"[TRACK {track_counter}] No predictions available after inference. Skipping saving for this track.")
        continue

    sorted_frame_idxs = [fi for fi, _ in sorted_frame_items]
    raw_predictions = [p for _, p in sorted_frame_items]
    print(f"[TRACK {track_counter}] Built raw_predictions array of length {len(raw_predictions)}")

    # 4f) Apply majority voting smoothing (inline implementation)
    print(f"[TRACK {track_counter}] Applying majority voting with window size = {VOTING_WINDOW} frames (~{VOTING_SECONDS:.1f}s)")
    smoothed_predictions = []
    n = len(raw_predictions)
    for i in range(n):
        start = max(0, i - VOTING_WINDOW // 2)
        end = min(n, i + VOTING_WINDOW // 2)
        window = raw_predictions[start:end]
        # If window empty (shouldn't happen), fallback to raw_predictions[i]
        if not window:
            smoothed_predictions.append(raw_predictions[i])
            continue
        counter = Counter(window)
        most_common, _ = counter.most_common(1)[0]
        smoothed_predictions.append(most_common)
    print(f"[TRACK {track_counter}] Completed majority voting.")

    # 4g) Save per-frame results to all_results list
    for i, frame_idx in enumerate(sorted_frame_idxs):
        raw_pred = raw_predictions[i]
        smoothed_pred = smoothed_predictions[i]
        all_results.append({
            'track_id': track_id,
            'frame': int(frame_idx),
            'activity_label': int(smoothed_pred),
            'activity_name': ACTIVITY_NAMES.get(int(smoothed_pred), "unknown"),
            'raw_prediction': int(raw_pred)
        })

    # 4h) Create visual timeline segments for this track (continuous same-activity runs)
    segments = []
    prev_frame_idx = None
    current_activity = None
    start_frame = None
    # iterate through (frame_idx, smoothed_pred) in order
    for frame_idx, activity in zip(sorted_frame_idxs, smoothed_predictions):
        if activity != current_activity:
            if current_activity is not None:
                # close previous segment: end at prev_frame
                prev_frame = prev_frame_idx
                duration_sec = (prev_frame - start_frame + 1) / float(FPS)
                segments.append({
                    'track_id': track_id,
                    'activity': ACTIVITY_NAMES.get(int(current_activity), "unknown"),
                    'start_frame': int(start_frame),
                    'end_frame': int(prev_frame),
                    'duration_sec': round(duration_sec, 2),
                    'start_time_sec': round(start_frame / float(FPS), 2),
                    'end_time_sec': round(prev_frame / float(FPS), 2)
                })
            # start new segment
            current_activity = activity
            start_frame = int(frame_idx)
        prev_frame_idx = int(frame_idx)

    # close last segment if exists
    if current_activity is not None:
        last_frame = prev_frame_idx
        duration_sec = (last_frame - start_frame + 1) / float(FPS)
        segments.append({
            'track_id': track_id,
            'activity': ACTIVITY_NAMES.get(int(current_activity), "unknown"),
            'start_frame': int(start_frame),
            'end_frame': int(last_frame),
            'duration_sec': round(duration_sec, 2),
            'start_time_sec': round(start_frame / float(FPS), 2),
            'end_time_sec': round(last_frame / float(FPS), 2)
        })

    print(f"[TRACK {track_counter}] Built {len(segments)} timeline segments for visualization.")
    visual_timeline_rows.extend(segments)

    # 4i) Print summary counts for this track
    track_activity_counts = Counter(smoothed_predictions)
    print(f"[TRACK {track_counter}] Activity summary (smoothed):")
    for activity_id, cnt in sorted(track_activity_counts.items()):
        dur = cnt / float(FPS)
        pct = (cnt / float(len(smoothed_predictions))) * 100.0
        print(f"   {ACTIVITY_NAMES.get(int(activity_id),'unknown')}: {cnt} frames ({dur:.2f}s, {pct:.1f}%)")

    # end of per-track loop
    print(f"[TRACK {track_counter}] Completed processing.\n")

# ============================
# After all tracks: Save CSV outputs (inline)
# ============================
if all_results:
    print("[OUTPUT] Saving per-frame activity results to CSV...")
    try:
        with open(ACTIVITY_OUTPUT_CSV, 'w', newline='') as outf:
            fieldnames = ['track_id', 'frame', 'activity_label', 'activity_name', 'raw_prediction']
            writer = csv.DictWriter(outf, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        print(f"[OUTPUT] Saved activity CSV to: {ACTIVITY_OUTPUT_CSV}")
    except Exception as e:
        print(f"[ERROR] Failed to save activity CSV: {e}")
else:
    print("[OUTPUT] No per-frame results to save.")

if visual_timeline_rows:
    print("[OUTPUT] Saving visual timeline CSV...")
    try:
        with open(ACTIVITY_VISUAL_CSV, 'w', newline='') as outf:
            fieldnames = ['track_id', 'activity', 'start_frame', 'end_frame', 'duration_sec', 'start_time_sec', 'end_time_sec']
            writer = csv.DictWriter(outf, fieldnames=fieldnames)
            writer.writeheader()
            for row in visual_timeline_rows:
                writer.writerow(row)
        print(f"[OUTPUT] Saved visual timeline CSV to: {ACTIVITY_VISUAL_CSV}")
    except Exception as e:
        print(f"[ERROR] Failed to save visual timeline CSV: {e}")
else:
    print("[OUTPUT] No visual timeline rows to save.")

end_time_total = time.time()
print("="*70)
print(f"Flattened pipeline finished in {end_time_total - start_time_total:.2f}s (wall time).")
print("="*70)
