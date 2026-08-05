"""
3D Action Recognition - TRIAL 5: Incremental Accuracy-Improvement STUDY
=======================================================================
Goal: push val accuracy past the ~67% plateau from Trial 4, WITHOUT leaving
the ResNet video-model family (r3d_18 / r2plus1d_18) or your Step 8->11
workflow.

This is NOT a blind grid search. It is a one-factor-at-a-time (OFAT)
ablation: we start from a LOCKED baseline recipe (the best-performing
settings proven in Trial 4) and change EXACTLY ONE thing per experiment,
so every accuracy delta is attributable to that single change. A running
comparison table records what helped, what hurt, and by how much.

Locked baseline (from Trial 4 verified main-effects):
  backbone=r2plus1d_18, stride=2, unfreeze=full, lr_mode=flat,
  label_smoothing=0.15, ssl=off, dropout=0.5, batch=16, cosine LR.

Experiments are grouped into the four phases we discussed:
  PHASE 0  Honest metric        - leave-one-video-out (LOVO) CV if a
                                  video-tagged manifest exists, else the
                                  fixed continuous-15% split (auto-fallback).
  PHASE 1  Kill overfitting     - clip de-duplication, stronger augmentation
                                  (mixup / cutmix / random-erase / temporal
                                  jitter), focal loss for rare classes.
  PHASE 2  Squeeze the family   - r2+1d vs r3d, 32-frame clips, test-time
                                  augmentation, seed ensembling.
  PHASE 3  Motion (two-stream)  - a second ResNet on OpenCV optical flow,
                                  fused with the RGB stream.

Everything stays on torchvision video ResNets + OpenCV you already use.
AMP stays OFF (FP32), matching the rest of your pipeline.

Data / video identity for LOVO:
  If DATASET_DIR/all_clips_manifest.json exists (written by the Step-8
  add-on), every clip is tagged with its source video and LOVO CV turns on
  automatically. Otherwise the study runs on the existing train/ + val/
  fixed split with a clear warning.

Outputs -> /data/shubhan_avik_work/Trial5/
  per-experiment checkpoints + histories, trial5_comparison.csv
  (the incremental "what worked" table), and trial5_study_summary.txt.
Resume-safe: completed experiments are skipped if their row already exists
in trial5_comparison.csv.
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
from torchvision.models.video import (
    r3d_18, R3D_18_Weights,
    r2plus1d_18, R2Plus1D_18_Weights,
    mc3_18, MC3_18_Weights,
)
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
import copy
import gc
import os
import sys
import time
from dataclasses import dataclass, field, asdict, replace as dc_replace
from collections import defaultdict, Counter

# ============================
# Console summary logger
# ============================
SUMMARY_LINES = []

def slog(msg=""):
    print(msg)
    SUMMARY_LINES.append(str(msg))


def compute_mean_std(dataset, max_samples=None):
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    count = 0
    indices = range(len(dataset))
    if max_samples is not None:
        indices = random.sample(list(indices), min(max_samples, len(dataset)))
    for idx in tqdm(indices, desc="Computing dataset mean/std"):
        frames, _ = dataset[idx]
        frames = frames.numpy().reshape(3, -1)
        channel_sum += frames.mean(axis=1)
        channel_sq_sum += (frames ** 2).mean(axis=1)
        count += 1
    mean = channel_sum / count
    std = np.sqrt(channel_sq_sum / count - mean ** 2)
    return mean.astype(np.float32), std.astype(np.float32)


# ============================
# 1. Configuration
# ============================
SOURCE_DIR  = Path("/data/shubhan_avik_work/Targeted_run_3").resolve()
DATASET_DIR = SOURCE_DIR / "Dataset_Ten_days"
TRAIN_DIR   = DATASET_DIR / "train"
VAL_DIR     = DATASET_DIR / "val"

# Video-tagged manifest (written by the Step-8 add-on 8b_add_clip_manifest.py).
# If present -> LOVO CV is available. If absent -> fixed split (auto-fallback).
MANIFEST_PATH = DATASET_DIR / "all_clips_manifest.json"

TRIAL_DIR      = Path("/data/shubhan_avik_work/Trial5").resolve()
MODEL_SAVE_DIR = TRIAL_DIR
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_CSV = MODEL_SAVE_DIR / "trial5_comparison.csv"

NUM_CLASSES = 5
CROP_SIZE   = 112
TARGET_FPS  = 25
BATCH_SIZE  = 16
NUM_WORKERS = 4

ACTIVITY_NAMES  = ['digging', 'idling', 'loading', 'swinging', 'travelling']
ACTIVITY_TO_IDX = {name: idx for idx, name in enumerate(ACTIVITY_NAMES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Study-level controls -------------------------------------------------
# EVAL_PROTOCOL: "auto" -> LOVO if manifest exists else fixed
#                "lovo" -> force leave-one-video-out (errors if no manifest)
#                "fixed"-> force existing train/ + val/ split
EVAL_PROTOCOL = "auto"
# For speed during development set QUICK_MODE=True to cap clips/epochs.
QUICK_MODE          = False
QUICK_MAX_PER_CLASS = 400     # clips/class cap when QUICK_MODE
QUICK_EPOCHS        = 3
# LOVO is expensive (K folds). Optionally restrict to a subset of held-out
# videos to keep wall-clock sane; empty list = use all videos as folds.
LOVO_FOLD_VIDEOS = []          # e.g. ["Day_3.mp4", "TC_00016.mp4"]

# ---- Progressive-unfreeze schedule / LRs (kept from Trial 4) --------------
FULL_UNFREEZE_STAGE = {"layer1", "layer2", "layer3", "layer4", "fc"}
LAYER_LR = {"layer1": 5e-6, "layer2": 8e-6, "layer3": 1.5e-5, "layer4": 3e-5, "fc": 1e-3}
FLAT_BASE_LR = 1e-5
FLAT_FC_LR   = 1e-3
WEIGHT_DECAY = 1e-4

# ---- Mutable globals the reused data/model code reads. The study runner
#      overwrites these from the active RecipeConfig before each fold. ------
CLIP_LENGTH = 16
DROPOUT_P   = 0.5
NUM_EPOCHS  = 24
EARLY_STOP_PATIENCE = 3
# augmentation switches (augment_frame_stack + dataset read these)
AUG_FLIP             = True
AUG_COLORSHIFT       = True
AUG_SHEAR            = True
AUG_RANDOM_ERASE     = False
AUG_TEMPORAL_JITTER  = False


def progressive_stages(num_epochs):
    """Staged unfreeze schedule scaled to num_epochs (kept from Trial 4)."""
    stages = [
        (0, 3,  {"fc"}),
        (3, 7,  {"layer4", "fc"}),
        (7, 12, {"layer3", "layer4", "fc"}),
        (12, 18, {"layer2", "layer3", "layer4", "fc"}),
        (18, num_epochs, {"layer1", "layer2", "layer3", "layer4", "fc"}),
    ]
    return stages


# ============================================================
# RecipeConfig - one experiment = baseline + exactly one change
# ============================================================
@dataclass
class RecipeConfig:
    name: str = "baseline"
    phase: str = "P0"
    desc: str = ""
    # --- locked baseline values (best from Trial 4) ---
    backbone: str = "r2plus1d_18"        # r3d_18 | r2plus1d_18 | mc3_18
    stride: int = 1                      # stored clips are 16 frames => stride 1 uses them all
    clip_length: int = 16                # fixed by Step 8 extraction; >16 needs re-extraction
    dropout: float = 0.5
    unfreeze_mode: str = "full"          # full | progressive
    lr_mode: str = "flat"                # flat | layerwise
    label_smoothing: float = 0.15
    loss: str = "ce"                     # ce | focal
    focal_gamma: float = 2.0
    sampler: str = "weighted"            # weighted | none
    # --- Phase-1 overfitting knobs ---
    dedup_factor: int = 1                # keep every Nth clip per class (1=all)
    aug_flip: bool = True
    aug_colorshift: bool = True
    aug_shear: bool = True
    aug_random_erase: bool = False
    aug_temporal_jitter: bool = False
    mixup_alpha: float = 0.0             # 0 disables
    cutmix_alpha: float = 0.0            # 0 disables
    # --- Phase-2/3 knobs ---
    stream: str = "rgb"                  # rgb | flow | two
    tta: bool = False                    # test-time augmentation at eval
    # --- training budget ---
    epochs: int = 24
    patience: int = 3
    seeds: tuple = (42,)                 # >1 seed => seed-ensemble experiment


# The locked baseline the whole study is measured against.
BASELINE = RecipeConfig(
    name="baseline", phase="P0",
    desc="Trial-4 best recipe (r2+1d, full, flat, ls0.15); stride=1 since stored clips are 16 frames",
)


def make_experiments():
    """Ordered OFAT experiments. Each flips exactly ONE knob vs BASELINE
    (except the explicitly-combined final runs), so deltas are attributable."""
    B = BASELINE
    exps = [B]

    # ---- PHASE 1: kill overfitting -------------------------------------
    exps += [
        dc_replace(B, name="p1_dedup2", phase="P1", dedup_factor=2,
                   desc="De-dup clips 2x (halve ~81%-overlap redundancy)"),
        dc_replace(B, name="p1_dedup3", phase="P1", dedup_factor=3,
                   desc="De-dup clips 3x"),
        dc_replace(B, name="p1_random_erase", phase="P1", aug_random_erase=True,
                   desc="Add random erasing augmentation"),
        # NOTE: temporal-stride jitter needs clips longer than 16 frames
        # (stored clips are exactly 16), so it is omitted until re-extraction.
        dc_replace(B, name="p1_mixup", phase="P1", mixup_alpha=0.2,
                   desc="Add MixUp (alpha=0.2)"),
        dc_replace(B, name="p1_cutmix", phase="P1", cutmix_alpha=1.0,
                   desc="Add CutMix (alpha=1.0)"),
        dc_replace(B, name="p1_focal", phase="P1", loss="focal",
                   desc="Focal loss (gamma=2) for rare-class recall"),
    ]

    # ---- PHASE 2: squeeze the ResNet family ----------------------------
    exps += [
        dc_replace(B, name="p2_r3d18", phase="P2", backbone="r3d_18",
                   desc="Backbone r3d_18 (sanity vs r2+1d)"),
        dc_replace(B, name="p2_mc3_18", phase="P2", backbone="mc3_18",
                   desc="Backbone mc3_18 (mixed-conv variant)"),
        dc_replace(B, name="p2_tta", phase="P2", tta=True,
                   desc="Test-time augmentation (hflip) at eval"),
        dc_replace(B, name="p2_ensemble3", phase="P2", seeds=(42, 43, 44),
                   desc="3-seed logit ensemble"),
    ]

    # ---- PHASE 3: optical-flow two-stream ------------------------------
    exps += [
        dc_replace(B, name="p3_flow_only", phase="P3", stream="flow",
                   desc="Single ResNet on optical flow only"),
        dc_replace(B, name="p3_two_stream", phase="P3", stream="two",
                   desc="RGB + optical-flow two-stream fusion"),
    ]

    # ---- FINAL: stack everything that helped (edit after seeing results)
    exps += [
        dc_replace(B, name="final_combined", phase="FINAL",
                   dedup_factor=2, aug_random_erase=True, mixup_alpha=0.2,
                   loss="focal", stream="two", tta=True, seeds=(42, 43, 44),
                   desc="Combined best-guess recipe (tune from the deltas above)"),
    ]
    return exps


EXPERIMENTS = make_experiments()

print("=" * 70)
print("TRIAL 5 - INCREMENTAL ACCURACY STUDY (ResNet family, OFAT ablation)")
print("=" * 70)
print(f"  Dataset       : {DATASET_DIR}")
print(f"  Manifest      : {MANIFEST_PATH} (exists={MANIFEST_PATH.exists()})")
print(f"  Eval protocol : {EVAL_PROTOCOL}")
print(f"  Experiments   : {len(EXPERIMENTS)}")
print(f"  Output dir    : {MODEL_SAVE_DIR}")
print(f"  Device        : {DEVICE}")
print(f"  QUICK_MODE    : {QUICK_MODE}")


# ============================
# 2. Reproducibility
# ============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================
# 3. Shared frame augmentation (operates on an already-read frame stack)
# ============================
def augment_frame_stack(frames_np, crop_size, train):
    """frames_np: (clip_length, H, W, 3) uint8 RGB, already center/random
    cropped to crop_size x crop_size. Applies the same flip/color-shift/
    shear augmentation used in the original pipeline. Returns float32 in [0,1]."""
    frames_np = frames_np.astype(np.float32) / 255.0
    if train:
        if AUG_FLIP and random.random() > 0.5:
            frames_np = np.flip(frames_np, axis=2).copy()
        if AUG_COLORSHIFT and random.random() > 0.5:
            shift = np.random.uniform(-0.1, 0.1, (1, 1, 1, 3))
            frames_np = np.clip(frames_np + shift, 0, 1)
        if AUG_SHEAR and random.random() > 0.3:
            shear_factor = random.uniform(-0.15, 0.15)
            M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
            for i in range(len(frames_np)):
                frames_np[i] = cv2.warpAffine(
                    frames_np[i], M, (crop_size, crop_size), borderMode=cv2.BORDER_REFLECT
                )
        if AUG_RANDOM_ERASE and random.random() > 0.5:
            # Erase one rectangle, consistent across the whole clip (temporal coherence)
            eh = random.randint(crop_size // 8, crop_size // 3)
            ew = random.randint(crop_size // 8, crop_size // 3)
            ty = random.randint(0, crop_size - eh)
            tx = random.randint(0, crop_size - ew)
            frames_np[:, ty:ty + eh, tx:tx + ew, :] = random.random()
    return frames_np


def crop_frame(img, crop_size, train):
    img = cv2.resize(img, (128, 128))
    if train:
        top = random.randint(0, 16)
        left = random.randint(0, 16)
    else:
        top, left = 8, 8
    return img[top:top + crop_size, left:left + crop_size]


# ============================
# 4. Labeled dataset (supervised train/val) -- reads pre-extracted jpg frames
# ============================
class VideoClipDataset(Dataset):
    def __init__(self, root_dir, activity_names, video_list,
                 clip_length=16, crop_size=112, stride=1, train=True,
                 normalize=True, mean=None, std=None):
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.stride = stride
        self.train = train
        self.samples = []
        self.normalize = normalize
        self.mean = mean
        self.std = std

        for video_path in video_list:
            video_path = Path(video_path)
            label_name = video_path.parent.name
            if label_name not in ACTIVITY_TO_IDX:
                continue
            frame_files = sorted([str(f) for f in video_path.glob('*.jpg')])
            if len(frame_files) > 0:
                self.samples.append((frame_files, ACTIVITY_TO_IDX[label_name]))

        print(f"  -> Loaded {len(self.samples)} samples "
              f"({'train' if train else 'val'}, stride={stride})")
        labels = [label for _, label in self.samples]
        for idx, activity in enumerate(ACTIVITY_NAMES):
            print(f"     {activity:12s}: {labels.count(idx):4d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_files, label = self.samples[idx]
        total_frames = len(frame_files)
        # Effective stride: optionally jittered for temporal augmentation.
        eff_stride = self.stride
        if AUG_TEMPORAL_JITTER and self.train and self.stride > 1:
            eff_stride = random.randint(1, self.stride)
        required_span = (self.clip_length - 1) * eff_stride + 1
        if total_frames >= required_span:
            if self.train:
                start_idx = random.randint(0, total_frames - required_span)
            else:
                start_idx = (total_frames - required_span) // 2
        else:
            start_idx = 0

        frames = []
        for i in range(self.clip_length):
            current_idx = min(start_idx + i * eff_stride, total_frames - 1)
            img = cv2.imread(frame_files[current_idx])
            if img is None:
                img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = crop_frame(img, self.crop_size, self.train)
            frames.append(img)

        frames_np = augment_frame_stack(np.array(frames), self.crop_size, self.train)
        if self.normalize:
            frames_np = (frames_np - self.mean) / self.std
        clip_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()
        return clip_tensor, label


def collect_clips(split_dir):
    clips = []
    for label in ACTIVITY_NAMES:
        label_dir = split_dir / label
        if label_dir.exists():
            clips.extend([v for v in label_dir.iterdir() if v.is_dir()])
        else:
            print(f"WARNING: {label_dir} not found")
    return clips


# ============================
# 5. Unlabeled RAW VIDEO dataset for SSL (#6) -- reads directly from .mp4
# ============================
class UnlabeledRawVideoClipDataset(Dataset):
    """Scans SSL_VIDEO_DIR for .mp4 files, excludes any whose stem is in
    SSL_EXCLUDE_VIDEO_STEMS, and builds a virtual index of
    (video_path, frame_count) so __getitem__ can pick a random start
    frame per call. Two independently-augmented views are returned per
    clip for SimCLR-style contrastive pretraining. No pre-extracted
    frames needed -- reads directly via cv2.VideoCapture."""

    def __init__(self, video_dir, clip_length, crop_size, stride,
                 mean, std, clips_per_video=150, exclude_stems=None):
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.stride = stride
        self.mean = mean
        self.std = std
        exclude_stems = set(exclude_stems or [])

        video_dir = Path(video_dir)
        all_videos = sorted(video_dir.glob("*.mp4"))
        self.video_frame_counts = {}
        required_span = (clip_length - 1) * stride + 1

        for vp in all_videos:
            if vp.stem in exclude_stems:
                print(f"  -> Excluding {vp.name} from SSL pretraining (in SSL_EXCLUDE_VIDEO_STEMS)")
                continue
            cap = cv2.VideoCapture(str(vp))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n_frames >= required_span:
                self.video_frame_counts[str(vp)] = n_frames

        # Virtual index: each usable video contributes `clips_per_video`
        # entries (a random start frame is drawn fresh in __getitem__,
        # so this just controls how many samples-per-epoch each video gets).
        self.index = []
        for vp, n_frames in self.video_frame_counts.items():
            self.index.extend([vp] * clips_per_video)

        print(f"  -> {len(self.video_frame_counts)} usable raw videos, "
              f"{len(self.index)} virtual clips/epoch for SSL")

    def __len__(self):
        return len(self.index)

    def _read_clip(self, video_path, start_frame, train):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        last_good = None
        for i in range(self.clip_length):
            target = start_frame + i * self.stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = last_good if last_good is not None else \
                    np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = crop_frame(frame, self.crop_size, train)
                last_good = frame
            frames.append(frame)
        cap.release()
        return augment_frame_stack(np.array(frames), self.crop_size, train)

    def __getitem__(self, idx):
        video_path = self.index[idx]
        n_frames = self.video_frame_counts[video_path]
        required_span = (self.clip_length - 1) * self.stride + 1
        start_frame = random.randint(0, max(n_frames - required_span, 0))

        view1 = self._read_clip(video_path, start_frame, train=True)
        view2 = self._read_clip(video_path, start_frame, train=True)
        view1 = (view1 - self.mean) / self.std
        view2 = (view2 - self.mean) / self.std
        t1 = torch.from_numpy(view1).permute(3, 0, 1, 2).float()
        t2 = torch.from_numpy(view2).permute(3, 0, 1, 2).float()
        return t1, t2


# ============================
# 6. Backbone construction
# ============================
def make_backbone(backbone_name, pretrained=True):
    if backbone_name == "r3d_18":
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        return r3d_18(weights=weights)
    elif backbone_name == "r2plus1d_18":
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        return r2plus1d_18(weights=weights)
    elif backbone_name == "mc3_18":
        weights = MC3_18_Weights.KINETICS400_V1 if pretrained else None
        return mc3_18(weights=weights)
    raise ValueError(f"Unknown backbone: {backbone_name}")


def build_model(backbone_name, ssl_state_dict=None):
    model = make_backbone(backbone_name, pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(in_features, NUM_CLASSES))

    if ssl_state_dict is not None:
        missing, unexpected = model.load_state_dict(ssl_state_dict, strict=False)
        slog(f"    Loaded SSL-pretrained backbone (fc mismatches expected: "
             f"missing={len(missing)}, unexpected={len(unexpected)})")

    for param in model.parameters():
        param.requires_grad = False
    return model


def apply_unfreeze(model, unfrozen_substrings):
    for name, param in model.named_parameters():
        param.requires_grad = any(s in name for s in unfrozen_substrings)


def build_param_groups(model, lr_mode):
    """lr_mode == 'layerwise' (#4): per-layer LR from LAYER_LR.
    lr_mode == 'flat': single base_params group (FLAT_BASE_LR) + fc
    group (FLAT_FC_LR), matching the original trial.py scheme."""
    if lr_mode == "layerwise":
        buckets = {k: [] for k in LAYER_LR}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            for group in ["layer4", "layer3", "layer2", "layer1", "fc"]:
                if group in name:
                    buckets[group].append(param)
                    break
        return [{"params": p, "lr": LAYER_LR[g], "weight_decay": WEIGHT_DECAY}
                for g, p in buckets.items() if len(p) > 0]

    elif lr_mode == "flat":
        fc_params, base_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (fc_params if "fc" in name else base_params).append(param)
        groups = []
        if base_params:
            groups.append({"params": base_params, "lr": FLAT_BASE_LR, "weight_decay": WEIGHT_DECAY})
        if fc_params:
            groups.append({"params": fc_params, "lr": FLAT_FC_LR, "weight_decay": WEIGHT_DECAY})
        return groups

    raise ValueError(f"Unknown lr_mode: {lr_mode}")


# ==========================================================================
# TRIAL-5 STUDY MACHINERY  (losses, mix, models, flow, folds, trainer)
# ==========================================================================

# ---- Loss functions -------------------------------------------------------
class FocalLoss(nn.Module):
    """Multi-class focal loss. Down-weights easy examples to help rare classes
    (e.g. 'loading')."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def build_criterion(recipe, class_weights=None):
    w = None
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    if recipe.loss == "focal":
        return FocalLoss(gamma=recipe.focal_gamma, weight=w)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=recipe.label_smoothing)


# ---- MixUp / CutMix (single-stream only) ----------------------------------
def apply_mix(clips, labels, recipe):
    """Returns (clips, target_a, target_b, lam). No-op returns lam=1."""
    if recipe.mixup_alpha <= 0 and recipe.cutmix_alpha <= 0:
        return clips, labels, labels, 1.0
    B = clips.size(0)
    perm = torch.randperm(B, device=clips.device)
    do_cut = recipe.cutmix_alpha > 0 and (recipe.mixup_alpha <= 0 or random.random() < 0.5)
    if do_cut:
        lam = float(np.random.beta(recipe.cutmix_alpha, recipe.cutmix_alpha))
        _, _, _, H, W = clips.shape
        r = np.sqrt(1.0 - lam)
        cw, ch = int(W * r), int(H * r)
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        clips[:, :, :, y1:y2, x1:x2] = clips[perm][:, :, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    else:
        lam = float(np.random.beta(recipe.mixup_alpha, recipe.mixup_alpha))
        clips = lam * clips + (1.0 - lam) * clips[perm]
    return clips, labels, labels[perm], lam


# ---- Models: single-stream + optical-flow two-stream ----------------------
class TwoStreamModel(nn.Module):
    """RGB ResNet + Flow ResNet, late feature-concat fusion. Stays in the
    ResNet family (two torchvision video backbones)."""
    def __init__(self, backbone_name, dropout=0.5):
        super().__init__()
        rgb = make_backbone(backbone_name, pretrained=True)
        flow = make_backbone(backbone_name, pretrained=True)
        self.feat_rgb = rgb.fc.in_features
        self.feat_flow = flow.fc.in_features
        rgb.fc = nn.Identity()
        flow.fc = nn.Identity()
        self.rgb = rgb
        self.flow = flow
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feat_rgb + self.feat_flow, NUM_CLASSES),
        )

    def forward(self, rgb_x, flow_x):
        f1 = self.rgb(rgb_x)
        f2 = self.flow(flow_x)
        return self.classifier(torch.cat([f1, f2], dim=1))


def build_recipe_model(recipe):
    if recipe.stream == "two":
        return TwoStreamModel(recipe.backbone, dropout=recipe.dropout)
    model = make_backbone(recipe.backbone, pretrained=True)  # rgb or flow (both 3ch)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=recipe.dropout),
                             nn.Linear(in_features, NUM_CLASSES))
    return model


def setup_trainable(model, recipe):
    """Freeze all, unfreeze layer1-4 + head (matches Trial-4 'full' unfreeze).
    Head is 'classifier' for two-stream, else 'fc'. Returns optimizer groups."""
    for p in model.parameters():
        p.requires_grad = False
    head = "classifier" if isinstance(model, TwoStreamModel) else "fc"
    unfreeze = {"layer1", "layer2", "layer3", "layer4", head}
    for name, p in model.named_parameters():
        if any(s in name for s in unfreeze):
            p.requires_grad = True

    if recipe.lr_mode == "layerwise":
        order = ["layer4", "layer3", "layer2", "layer1", head]
        lrmap = dict(LAYER_LR); lrmap[head] = LAYER_LR["fc"]
        buckets = {g: [] for g in order}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            for g in order:
                if g in name:
                    buckets[g].append(p); break
        return [{"params": ps, "lr": lrmap[g], "weight_decay": WEIGHT_DECAY}
                for g, ps in buckets.items() if ps]

    head_params, base_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if head in name else base_params).append(p)
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": FLAT_BASE_LR, "weight_decay": WEIGHT_DECAY})
    if head_params:
        groups.append({"params": head_params, "lr": FLAT_FC_LR, "weight_decay": WEIGHT_DECAY})
    return groups


# ---- Optical flow + RGB/flow dataset (consistent per-clip crop) ------------
def compute_flow_clip(gray_frames):
    """gray_frames: list of (H,W) uint8. Returns (T,H,W,3) float32 in [-1,1]
    where channels are (flow_u, flow_v, magnitude)."""
    flows = []
    prev = gray_frames[0]
    for t in range(1, len(gray_frames)):
        f = cv2.calcOpticalFlowFarneback(prev, gray_frames[t], None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(f)
        prev = gray_frames[t]
    if not flows:
        h, w = gray_frames[0].shape
        flows = [np.zeros((h, w, 2), np.float32)]
    flows = [flows[0]] + flows            # pad first to length T
    flow = np.stack(flows, axis=0)        # (T,H,W,2)
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    stacked = np.stack([u, v, mag], axis=-1)
    return np.clip(stacked / 20.0, -1.0, 1.0).astype(np.float32)


class RGBFlowClipDataset(Dataset):
    """Reads a clip's 16 jpgs ONCE with a single consistent crop/flip for the
    whole clip, then emits RGB and/or optical-flow tensors. Used for
    stream='flow' and stream='two'."""
    RESIZE = 128

    def __init__(self, clip_dirs, recipe, train, mean, std, want_rgb, want_flow):
        self.clip_length = recipe.clip_length
        self.crop_size = CROP_SIZE
        self.stride = recipe.stride
        self.train = train
        self.mean, self.std = mean, std
        self.want_rgb, self.want_flow = want_rgb, want_flow
        self.samples = []
        for d in clip_dirs:
            d = Path(d)
            label_name = d.parent.name
            if label_name not in ACTIVITY_TO_IDX:
                continue
            frames = sorted(str(f) for f in d.glob("*.jpg"))
            if frames:
                self.samples.append((frames, ACTIVITY_TO_IDX[label_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_files, label = self.samples[idx]
        total = len(frame_files)
        span = (self.clip_length - 1) * self.stride + 1
        if total >= span:
            start = random.randint(0, total - span) if self.train else (total - span) // 2
        else:
            start = 0
        rs = self.RESIZE
        if self.train:
            top = random.randint(0, rs - self.crop_size)
            left = random.randint(0, rs - self.crop_size)
        else:
            top = left = (rs - self.crop_size) // 2
        do_flip = self.train and AUG_FLIP and random.random() > 0.5

        rgb_list, gray_list = [], []
        for i in range(self.clip_length):
            ci = min(start + i * self.stride, total - 1)
            img = cv2.imread(frame_files[ci])
            if img is None:
                img = np.zeros((self.crop_size, self.crop_size, 3), np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (rs, rs))[top:top + self.crop_size, left:left + self.crop_size]
                if do_flip:
                    img = np.flip(img, axis=1).copy()
            rgb_list.append(img)
            if self.want_flow:
                gray_list.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

        out_rgb = out_flow = None
        if self.want_rgb:
            rgb = np.array(rgb_list).astype(np.float32) / 255.0
            rgb = (rgb - self.mean) / self.std
            out_rgb = torch.from_numpy(rgb).permute(3, 0, 1, 2).float()
        if self.want_flow:
            flow = compute_flow_clip(gray_list)
            out_flow = torch.from_numpy(flow).permute(3, 0, 1, 2).float()

        if self.want_rgb and self.want_flow:
            return (out_rgb, out_flow), label
        return (out_flow if self.want_flow else out_rgb), label


def make_dataset(clip_dirs, recipe, train, mean, std):
    if recipe.stream == "rgb":
        return VideoClipDataset(None, ACTIVITY_NAMES, clip_dirs, recipe.clip_length,
                                CROP_SIZE, recipe.stride, train=train,
                                normalize=True, mean=mean, std=std)
    want_rgb = recipe.stream == "two"
    return RGBFlowClipDataset(clip_dirs, recipe, train, mean, std,
                              want_rgb=want_rgb, want_flow=True)


# ---- Clip-dir collection, de-dup, LOVO folds ------------------------------
def collect_clip_dirs_from_split(split_dir):
    dirs = []
    for label in ACTIVITY_NAMES:
        d = split_dir / label
        if d.exists():
            dirs += sorted(p for p in d.iterdir() if p.is_dir())
    return dirs


def dedup_dirs(dirs, factor):
    """Keep every factor-th clip within each activity (sorted) to cut the
    ~81%-overlap redundancy from CLIP_STRIDE=3 extraction."""
    if factor <= 1:
        return list(dirs)
    by_act = defaultdict(list)
    for p in dirs:
        by_act[Path(p).parent.name].append(p)
    kept = []
    for act, ps in by_act.items():
        kept += sorted(ps)[::factor]
    return kept


def cap_per_class(dirs, n):
    by_act = defaultdict(list)
    for p in dirs:
        by_act[Path(p).parent.name].append(p)
    kept = []
    for act, ps in by_act.items():
        kept += sorted(ps)[:n]
    return kept


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def lovo_folds(manifest):
    videos = sorted({e["video"] for e in manifest})
    if LOVO_FOLD_VIDEOS:
        videos = [v for v in videos if v in LOVO_FOLD_VIDEOS]
    folds = []
    for held in videos:
        train_dirs = [DATASET_DIR / e["clip_folder"] for e in manifest if e["video"] != held]
        val_dirs = [DATASET_DIR / e["clip_folder"] for e in manifest if e["video"] == held]
        folds.append((held, train_dirs, val_dirs))
    return folds


def resolve_folds(recipe):
    proto = EVAL_PROTOCOL
    have = MANIFEST_PATH.exists()
    if proto == "auto":
        proto = "lovo" if have else "fixed"
    if proto == "lovo":
        if not have:
            raise FileNotFoundError(
                f"EVAL_PROTOCOL=lovo but {MANIFEST_PATH} missing. Run the Step-8 "
                f"add-on (8b_add_clip_manifest.py) to enable LOVO CV.")
        return "lovo", lovo_folds(load_manifest())
    td = collect_clip_dirs_from_split(TRAIN_DIR)
    vd = collect_clip_dirs_from_split(VAL_DIR)
    return "fixed", [("fixed", td, vd)]


# ---- Evaluation (optional TTA) + per-fold training ------------------------
def evaluate_recipe(model, loader, recipe, criterion):
    model.eval()
    all_probs, all_labels, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            if recipe.stream == "two":
                rgb, flow = inputs
                rgb, flow = rgb.to(DEVICE), flow.to(DEVICE)
                logits = model(rgb, flow)
                if recipe.tta:
                    logits = (logits + model(torch.flip(rgb, dims=[4]),
                                             torch.flip(flow, dims=[4]))) / 2
            else:
                clips = inputs.to(DEVICE)
                logits = model(clips)
                if recipe.tta:
                    logits = (logits + model(torch.flip(clips, dims=[4]))) / 2
            loss = criterion(logits, labels.to(DEVICE))
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return (total_loss / max(n, 1),
            np.concatenate(all_probs), np.concatenate(all_labels))


def run_recipe_fold(recipe, seed, train_loader, val_loader, loss_class_weights, tag):
    """Train one model (one seed, one fold). Returns (best_acc, best_probs,
    val_labels) where best_probs are val softmax at the best epoch."""
    gc.collect(); torch.cuda.empty_cache()
    set_seed(seed)
    model = build_recipe_model(recipe).to(DEVICE)
    groups = setup_trainable(model, recipe)
    optimizer = optim.Adam(groups)
    epochs = QUICK_EPOCHS if QUICK_MODE else recipe.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = build_criterion(recipe, loss_class_weights)

    best_acc, best_probs, best_labels, no_improve = 0.0, None, None, 0
    for epoch in range(epochs):
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"[{tag}|s{seed}] ep{epoch+1}/{epochs}")
        for inputs, labels in loop:
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            if recipe.stream == "two":
                rgb, flow = inputs
                outputs = model(rgb.to(DEVICE), flow.to(DEVICE))
                loss = criterion(outputs, labels)
            else:
                clips = inputs.to(DEVICE)
                clips, ya, yb, lam = apply_mix(clips, labels, recipe)
                outputs = model(clips)
                loss = lam * criterion(outputs, ya) + (1 - lam) * criterion(outputs, yb)
            loss.backward(); optimizer.step()
            run_loss += loss.item() * labels.size(0)
            _, pred = outputs.max(1)
            seen += labels.size(0); correct += (pred == labels).sum().item()
            loop.set_postfix(loss=loss.item(), acc=100 * correct / max(seen, 1))
        scheduler.step()

        _, probs, vlabels = evaluate_recipe(model, val_loader, recipe, criterion)
        acc = float((probs.argmax(1) == vlabels).mean() * 100)
        slog(f"    [{tag}|s{seed}] epoch {epoch+1}: train_acc={100*correct/max(seen,1):.2f}% "
             f"val_acc={acc:.2f}%")
        if acc > best_acc:
            best_acc, best_probs, best_labels, no_improve = acc, probs, vlabels, 0
        else:
            no_improve += 1
            if no_improve >= recipe.patience:
                slog(f"    [{tag}|s{seed}] early stop at epoch {epoch+1}")
                break

    del model; gc.collect(); torch.cuda.empty_cache()
    return best_acc, best_probs, best_labels


# ============================
# 7. SimCLR-style SSL pretraining (#6)
# ============================
class SSLModel(nn.Module):
    def __init__(self, backbone_name, proj_dim=128):
        super().__init__()
        backbone = make_backbone(backbone_name, pretrained=True)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features), nn.ReLU(inplace=True),
            nn.Linear(in_features, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.projector(self.backbone(x)), dim=1)


def nt_xent_loss(z1, z2, temperature):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    targets = torch.cat([torch.arange(batch_size, 2 * batch_size),
                          torch.arange(0, batch_size)]).to(z.device)
    return F.cross_entropy(sim, targets)


def run_ssl_pretrain(backbone_name, stride, mean, std, tag):
    ckpt_path = MODEL_SAVE_DIR / f"ssl_backbone_{tag}.pth"
    if ckpt_path.exists():
        slog(f"  [SSL] Reusing existing checkpoint for {tag} -> {ckpt_path}")
        return ckpt_path

    if not SSL_VIDEO_DIR.exists():
        slog(f"  [SSL] SSL_VIDEO_DIR not found -- skipping SSL for {tag}.")
        return None

    ssl_dataset = UnlabeledRawVideoClipDataset(
        SSL_VIDEO_DIR, CLIP_LENGTH, CROP_SIZE, stride, mean, std,
        clips_per_video=SSL_CLIPS_PER_VIDEO, exclude_stems=SSL_EXCLUDE_VIDEO_STEMS
    )
    if len(ssl_dataset) < SSL_MIN_CLIPS:
        slog(f"  [SSL] Only {len(ssl_dataset)} usable clips for {tag} "
             f"(< {SSL_MIN_CLIPS}) -- skipping.")
        return None

    slog(f"  [SSL] Pretraining {tag} on {len(ssl_dataset)} virtual clips/epoch "
         f"for {SSL_EPOCHS} epochs...")
    ssl_loader = DataLoader(ssl_dataset, batch_size=SSL_BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    model = SSLModel(backbone_name, proj_dim=SSL_PROJ_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=SSL_LR, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SSL_EPOCHS)

    for epoch in range(SSL_EPOCHS):
        model.train()
        running_loss, n_batches = 0.0, 0
        loop = tqdm(ssl_loader, desc=f"[SSL {tag}] Epoch {epoch+1}/{SSL_EPOCHS}")
        for view1, view2 in loop:
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
            optimizer.zero_grad()
            loss = nt_xent_loss(model(view1), model(view2), SSL_TEMPERATURE)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            loop.set_postfix(loss=loss.item())
        scheduler.step()
        slog(f"  [SSL {tag}] Epoch {epoch+1}/{SSL_EPOCHS}: avg_loss={running_loss/max(n_batches,1):.4f}")

    torch.save(model.backbone.state_dict(), ckpt_path)
    slog(f"  [SSL] Saved -> {ckpt_path}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ckpt_path


# ============================
# 8. Evaluation
# ============================
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for clips, labels in dataloader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * clips.size(0)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(NUM_CLASSES), zero_division=0
    )
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(dataloader.dataset), accuracy, precision, recall


# ============================
# 9. Single supervised run
# ============================
def run_supervised_trial(backbone_name, stride, ssl_on, unfreeze_mode, lr_mode, label_smoothing, seed,
                          ssl_state_dict, train_loader, val_loader, tag):
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(seed)

    slog("")
    slog("=" * 70)
    slog(f"RUN: {tag}")
    slog("=" * 70)

    model = build_model(backbone_name, ssl_state_dict=ssl_state_dict).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_loss, best_val_acc, epochs_no_improve, best_state = float('inf'), 0.0, 0, None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'val_precision': [], 'val_recall': [], 'stage': []}

    current_stage_substrings, optimizer, scheduler = None, None, None

    for epoch in range(NUM_EPOCHS):
        if unfreeze_mode == "full":
            stage_substrings = FULL_UNFREEZE_STAGE
            stage_end = NUM_EPOCHS
            is_final_stage = True
        else:  # progressive
            stage_substrings, stage_end = None, NUM_EPOCHS
            for start, end, substrings in PROGRESSIVE_STAGES:
                if start <= epoch < end:
                    stage_substrings, stage_end = substrings, end
                    break
            is_final_stage = (stage_substrings == PROGRESSIVE_STAGES[-1][2])

        if stage_substrings != current_stage_substrings:
            current_stage_substrings = stage_substrings
            apply_unfreeze(model, current_stage_substrings)
            param_groups = build_param_groups(model, lr_mode)
            trainable = sum(p.numel() for g in param_groups for p in g["params"])
            total = sum(p.numel() for p in model.parameters())
            slog(f"  --> Stage change at epoch {epoch+1}: unfreezing "
                 f"{sorted(current_stage_substrings)} ({trainable:,}/{total:,}, "
                 f"{100*trainable/total:.1f}%) [lr_mode={lr_mode}]")
            optimizer = optim.Adam(param_groups)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(stage_end - epoch, 1)
            )

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch+1}/{NUM_EPOCHS}")
        for clips, labels in loop:
            clips, labels = clips.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * clips.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total
        val_loss, val_acc, val_precision, val_recall = evaluate_model(model, val_loader, criterion, DEVICE)
        scheduler.step()

        slog(f"  Epoch {epoch+1}/{NUM_EPOCHS} [stage={sorted(current_stage_substrings)}]: "
             f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
             f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc * 100)
        history['val_precision'].append(val_precision.tolist())
        history['val_recall'].append(val_recall.tolist())
        history['stage'].append(sorted(current_stage_substrings))

        current_val_acc = val_acc * 100
        if current_val_acc > best_val_acc or (current_val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc, best_val_loss, epochs_no_improve = current_val_acc, val_loss, 0
            best_state = {'epoch': epoch + 1, 'model_state_dict': copy.deepcopy(model.state_dict()),
                          'val_loss': val_loss, 'val_acc': current_val_acc,
                          'val_precision': val_precision.tolist(), 'val_recall': val_recall.tolist()}
            slog(f"  New best: val_acc={current_val_acc:.2f}% val_loss={val_loss:.4f}")
        else:
            epochs_no_improve += 1
            slog(f"  No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
            if is_final_stage and epochs_no_improve >= EARLY_STOP_PATIENCE:
                slog(f"  Early stopping at epoch {epoch+1}")
                break

    checkpoint = {
        'epoch': best_state['epoch'], 'model_state_dict': best_state['model_state_dict'],
        'val_loss': best_state['val_loss'], 'val_acc': best_state['val_acc'],
        'val_precision': best_state['val_precision'], 'val_recall': best_state['val_recall'],
        'activity_names': ACTIVITY_NAMES, 'backbone': backbone_name, 'stride': stride,
        'ssl_on': ssl_on, 'unfreeze_mode': unfreeze_mode, 'lr_mode': lr_mode,
        'label_smoothing': label_smoothing, 'seed': seed,
        'config': {'clip_length': CLIP_LENGTH, 'crop_size': CROP_SIZE, 'num_classes': NUM_CLASSES,
                   'batch_size': BATCH_SIZE, 'dropout_p': DROPOUT_P, 'label_smoothing': label_smoothing},
    }
    save_path = MODEL_SAVE_DIR / f"resnet3d_{tag}.pth"
    torch.save(checkpoint, save_path)
    with open(MODEL_SAVE_DIR / f"history_{tag}.json", 'w') as f:
        json.dump(history, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'tag': tag, 'backbone': backbone_name, 'stride': stride, 'ssl_on': ssl_on,
        'unfreeze_mode': unfreeze_mode, 'lr_mode': lr_mode, 'label_smoothing': label_smoothing,
        'seed': seed, 'best_val_loss': best_val_loss, 'best_val_acc': best_val_acc,
        'epochs_run': len(history['train_loss']), 'checkpoint_path': str(save_path),
    }


# ==========================================================================
# STUDY RUNNER  (OFAT ablation with incremental comparison table)
# ==========================================================================
def get_mean_std():
    """RGB dataset mean/std, computed once from the fixed train split and
    cached. Shared across folds/experiments for comparability."""
    mp = MODEL_SAVE_DIR / "dataset_mean.npy"
    sp = MODEL_SAVE_DIR / "dataset_std.npy"
    if mp.exists() and sp.exists():
        return np.load(str(mp)), np.load(str(sp))
    dirs = collect_clip_dirs_from_split(TRAIN_DIR)
    ds = VideoClipDataset(None, ACTIVITY_NAMES, dirs, 16, CROP_SIZE, 1,
                          train=True, normalize=False)
    mean, std = compute_mean_std(ds, max_samples=200)
    np.save(str(mp), mean)
    np.save(str(sp), std)
    return mean, std


def per_class_recall(labels, preds):
    rec = {}
    for i, act in enumerate(ACTIVITY_NAMES):
        m = labels == i
        rec[act] = float((preds[m] == i).mean() * 100) if m.sum() > 0 else float("nan")
    return rec


def run_experiment(recipe, mean, std):
    """Run one recipe across all folds (fixed=1, LOVO=K) and seeds; return a
    result row. Seeds>1 are logit-ensembled per fold."""
    # Push recipe knobs into the module globals the data/aug code reads.
    globals().update(
        CLIP_LENGTH=recipe.clip_length,
        DROPOUT_P=recipe.dropout,
        NUM_EPOCHS=recipe.epochs,
        EARLY_STOP_PATIENCE=recipe.patience,
        AUG_FLIP=recipe.aug_flip,
        AUG_COLORSHIFT=recipe.aug_colorshift,
        AUG_SHEAR=recipe.aug_shear,
        AUG_RANDOM_ERASE=recipe.aug_random_erase,
        AUG_TEMPORAL_JITTER=recipe.aug_temporal_jitter,
    )
    proto, folds = resolve_folds(recipe)
    fold_accs, pooled_labels, pooled_preds = [], [], []
    t0 = time.time()

    for fi, (fold_name, train_dirs, val_dirs) in enumerate(folds):
        train_dirs = dedup_dirs(train_dirs, recipe.dedup_factor)
        if QUICK_MODE:
            train_dirs = cap_per_class(train_dirs, QUICK_MAX_PER_CLASS)
            val_dirs = cap_per_class(val_dirs, QUICK_MAX_PER_CLASS)

        train_ds = make_dataset(train_dirs, recipe, True, mean, std)
        val_ds = make_dataset(val_dirs, recipe, False, mean, std)
        if len(train_ds) == 0 or len(val_ds) == 0:
            slog(f"  Fold [{fold_name}] empty (train={len(train_ds)} val={len(val_ds)}) - skip")
            continue

        targets = [lbl for _, lbl in train_ds.samples]
        counts = np.bincount(targets, minlength=NUM_CLASSES)
        inv = 1.0 / np.maximum(counts, 1)

        loss_w = None
        if recipe.sampler == "weighted":
            sw = [inv[t] for t in targets]
            sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                      num_workers=NUM_WORKERS, pin_memory=True)
        else:
            loss_w = inv * NUM_CLASSES / inv.sum()   # feed imbalance to the loss instead
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        slog(f"  Fold {fi+1}/{len(folds)} [{fold_name}]: train={len(train_ds)} val={len(val_ds)}")

        sum_probs, ref_labels = None, None
        for seed in recipe.seeds:
            _, probs, vlabels = run_recipe_fold(
                recipe, seed, train_loader, val_loader, loss_w,
                tag=f"{recipe.name}_{fold_name}")
            if probs is None:
                continue
            sum_probs = probs if sum_probs is None else sum_probs + probs
            ref_labels = vlabels
        if sum_probs is None:
            continue

        fold_preds = sum_probs.argmax(1)
        fold_acc = float((fold_preds == ref_labels).mean() * 100)
        fold_accs.append(fold_acc)
        pooled_labels.append(ref_labels)
        pooled_preds.append(fold_preds)
        slog(f"  Fold {fi+1} [{fold_name}] acc={fold_acc:.2f}%")

    pooled_labels = np.concatenate(pooled_labels) if pooled_labels else np.array([])
    pooled_preds = np.concatenate(pooled_preds) if pooled_preds else np.array([])
    recalls = per_class_recall(pooled_labels, pooled_preds) if len(pooled_labels) else {}

    result = {
        "name": recipe.name, "phase": recipe.phase, "desc": recipe.desc,
        "protocol": proto, "n_folds": len(fold_accs),
        "val_acc_mean": float(np.mean(fold_accs)) if fold_accs else float("nan"),
        "val_acc_std": float(np.std(fold_accs)) if fold_accs else float("nan"),
        "n_seeds": len(recipe.seeds),
        "minutes": round((time.time() - t0) / 60.0, 1),
    }
    for act in ACTIVITY_NAMES:
        result[f"recall_{act}"] = round(recalls.get(act, float("nan")), 2)
    result["params"] = json.dumps({k: v for k, v in asdict(recipe).items()
                                    if k not in ("name", "phase", "desc")})
    return result


def load_done_names():
    if COMPARISON_CSV.exists():
        try:
            return set(pd.read_csv(COMPARISON_CSV)["name"].tolist())
        except Exception:
            return set()
    return set()


def append_result_row(result):
    row = pd.DataFrame([result])
    if COMPARISON_CSV.exists():
        row.to_csv(COMPARISON_CSV, mode="a", header=False, index=False)
    else:
        row.to_csv(COMPARISON_CSV, index=False)


def select_experiments():
    """Choose which experiments to run this invocation:
      --light         : only single-stream, 1-seed experiments (GPU-friendly;
                        good to run now while the GPU is shared).
      --only a,b,c    : only the named experiments.
      (no flag)       : all experiments.
    All modes are resume-safe (already-completed rows are skipped)."""
    if "--light" in sys.argv:
        sel = [r for r in EXPERIMENTS if r.stream == "rgb" and len(r.seeds) == 1]
        slog(f"MODE --light: {len(sel)} single-stream/1-seed experiments")
        return sel
    for a in sys.argv:
        if a.startswith("--only"):
            names = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
            wanted = {n.strip() for n in names.split(",")}
            sel = [r for r in EXPERIMENTS if r.name in wanted]
            slog(f"MODE --only: {[r.name for r in sel]}")
            return sel
    return EXPERIMENTS


def main():
    mean, std = get_mean_std()
    slog(f"Dataset mean={mean}  std={std}")

    exps = select_experiments()
    done = load_done_names()
    if done:
        slog(f"Resume: {len(done)} experiment(s) already in "
             f"{COMPARISON_CSV.name} -> will skip them.")

    for i, recipe in enumerate(exps):
        if recipe.name in done:
            slog(f"[{i+1}/{len(exps)}] SKIP {recipe.name} (already done)")
            continue
        slog("")
        slog("=" * 70)
        slog(f"[{i+1}/{len(exps)}] EXPERIMENT: {recipe.name}  ({recipe.phase})")
        slog(f"  {recipe.desc}")
        slog("=" * 70)
        try:
            result = run_experiment(recipe, mean, std)
        except Exception as e:
            slog(f"  ERROR in {recipe.name}: {e}")
            continue
        append_result_row(result)                 # save incrementally (resume-safe)
        slog(f"  => {recipe.name}: val_acc={result['val_acc_mean']:.2f} "
             f"+/- {result['val_acc_std']:.2f}%  ({result['minutes']} min)")

    # ---- Final ranked comparison with deltas vs baseline ----
    if not COMPARISON_CSV.exists():
        slog("No results produced.")
        return
    df = pd.read_csv(COMPARISON_CSV)
    base = df[df["name"] == "baseline"]
    base_acc = float(base["val_acc_mean"].iloc[0]) if len(base) else float("nan")
    df["delta_vs_baseline"] = df["val_acc_mean"] - base_acc
    df["helped"] = df["delta_vs_baseline"].apply(
        lambda d: "" if pd.isna(d) else ("YES" if d > 0.3 else ("no" if d < -0.3 else "~")))
    df = df.sort_values("val_acc_mean", ascending=False)
    df.to_csv(MODEL_SAVE_DIR / "trial5_comparison_ranked.csv", index=False)

    show = ["name", "phase", "val_acc_mean", "val_acc_std", "delta_vs_baseline",
            "helped", "recall_loading", "minutes"]
    slog("")
    slog("=" * 100)
    slog(f"TRIAL 5 COMPARISON  (baseline = {base_acc:.2f}%)  sorted best-first")
    slog("=" * 100)
    slog(df[show].to_string(index=False))

    summary_path = MODEL_SAVE_DIR / "trial5_study_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(SUMMARY_LINES) + "\n")
    print(f"\nComparison CSV -> {COMPARISON_CSV}")
    print(f"Ranked CSV     -> {MODEL_SAVE_DIR / 'trial5_comparison_ranked.csv'}")
    print(f"Text summary   -> {summary_path}")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()