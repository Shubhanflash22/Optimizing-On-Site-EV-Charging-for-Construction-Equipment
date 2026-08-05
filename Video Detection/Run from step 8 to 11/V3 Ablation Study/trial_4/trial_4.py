"""
3D Action Recognition — TRIAL 4: Full Factorial Ablation
===========================================================

Every design choice from the last two sweeps, tested as an independent
on/off axis instead of bundled together, so the final table tells you
exactly which factor moves accuracy and which don't:

  Axis 1 - backbone      : r3d_18  vs  r2plus1d_18                (2)
  Axis 2 - stride        : 1       vs  2                          (2)
  Axis 3 - SSL pretrain  : off     vs  on  (domain-adapted, #6)   (2)
  Axis 4 - unfreeze mode : "full" (all layers unfrozen epoch 0,
                            i.e. your original Trial 5 style)
                            vs "progressive" (staged opening, #7)  (2)
  Axis 5 - LR mode       : "flat" (single base_params LR + fc LR,
                            i.e. your original script's scheme)
                            vs "layerwise" (per-layer LR, #4)      (2)
  Axis 6 - label smooth  : 0.05 / 0.1 / 0.15 / 0.2                (4)
  Axis 7 - seed          : 42, 43, 44                             (3)

  Total runs = 2 x 2 x 2 x 2 x 2 x 4 x 3 = 384

AMP stays OFF throughout (FP32) -- unchanged, per earlier instruction.

Folder layout:
  Labeled data  : /data/shubhan_avik_work/Targeted_run_3/Dataset_Ten_days
                  (train/ and val/ from Step 8 -- read-only)
  Raw videos for SSL pretraining (#6, OPTIONAL):
                  SSL_VIDEO_DIR below -- point this at a folder of
                  raw .mp4 files (e.g. your Targeted_run_2 folder).
                  Read directly via cv2.VideoCapture -- no pre-
                  extracted frames needed, no new library needed.
  Outputs       : /data/shubhan_avik_work/Trial4/
                  (SSL checkpoints, per-run checkpoints + histories,
                   full_results.csv, aggregated tables, summary txt)

This is a LOT of runs (96 supervised + up to 4 SSL pretrains) --
expect multi-day wall clock on one GPU, as intended.
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
)
from sklearn.metrics import precision_recall_fscore_support
import json
import copy
import gc

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
SOURCE_DIR = Path("/data/shubhan_avik_work/Targeted_run_3").resolve()
DATASET_DIR = SOURCE_DIR / "Dataset_Ten_days"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

# Raw .mp4 folder for SSL pretraining (#6). Point at e.g. Targeted_run_2.
SSL_VIDEO_DIR = Path("/data/shubhan_avik_work/Targeted_run_2").resolve()
# Name (stem, no extension) any videos you KNOW feed the val split so
# SSL pretraining doesn't quietly train on val footage. Empty by
# default -- fill in from your Step 8 train/val video mapping if known.
SSL_EXCLUDE_VIDEO_STEMS = []   # e.g. ["Day_4_1", "TC_00019"]

TRIAL_DIR = Path("/data/shubhan_avik_work/Trial4").resolve()
MODEL_SAVE_DIR = TRIAL_DIR
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 5
CLIP_LENGTH = 16
CROP_SIZE = 112
TARGET_FPS = 25

BATCH_SIZE = 16
NUM_WORKERS = 4

ACTIVITY_NAMES = ['digging', 'idling', 'loading', 'swinging', 'travelling']
ACTIVITY_TO_IDX = {name: idx for idx, name in enumerate(ACTIVITY_NAMES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- The six axes ---
BACKBONES = ["r3d_18", "r2plus1d_18"]
STRIDES = [1, 2]
SSL_OPTIONS = [True, False]
UNFREEZE_MODE_OPTIONS = ["full", "progressive"]
LR_MODE_OPTIONS = ["flat", "layerwise"]
# LABEL_SMOOTHING_OPTIONS = [0.05, 0.1, 0.15, 0.2]
LABEL_SMOOTHING_OPTIONS = [0.05, 0.15]  # reduced for speed
SEEDS = [42] # reduced for speed
# SEEDS = [42, 43, 44]

# --- Fixed knobs (not axes) ---
DROPOUT_P = 0.5
NUM_EPOCHS = 24
EARLY_STOP_PATIENCE = 3

# --- unfreeze_mode == "progressive" schedule (epoch ranges, [start,end)) ---
PROGRESSIVE_STAGES = [
    (0, 3,  {"fc"}),
    (3, 7,  {"layer4", "fc"}),
    (7, 12, {"layer3", "layer4", "fc"}),
    (12, 18, {"layer2", "layer3", "layer4", "fc"}),
    (18, None, {"layer1", "layer2", "layer3", "layer4", "fc"}),
]
PROGRESSIVE_STAGES[-1] = (PROGRESSIVE_STAGES[-1][0], NUM_EPOCHS, PROGRESSIVE_STAGES[-1][2])
FULL_UNFREEZE_STAGE = {"layer1", "layer2", "layer3", "layer4", "fc"}

# --- lr_mode == "layerwise" ---
LAYER_LR = {
    "layer1": 5e-6, "layer2": 8e-6, "layer3": 1.5e-5, "layer4": 3e-5, "fc": 1e-3,
}
# --- lr_mode == "flat" (matches your original trial.py / trial_2.py scheme) ---
FLAT_BASE_LR = 1e-5
FLAT_FC_LR = 1e-3
WEIGHT_DECAY = 1e-4

# --- SSL pretraining (#6) ---
SSL_EPOCHS = 15
SSL_BATCH_SIZE = 16
SSL_LR = 1e-4
SSL_TEMPERATURE = 0.2
SSL_PROJ_DIM = 128
SSL_MIN_CLIPS = 50
SSL_CLIPS_PER_VIDEO = 150   # virtual samples drawn per source video per epoch

print("Configuration:")
print(f"  Backbones: {BACKBONES}")
print(f"  Strides: {STRIDES}")
print(f"  SSL options: {SSL_OPTIONS}")
print(f"  Unfreeze modes: {UNFREEZE_MODE_OPTIONS}")
print(f"  LR modes: {LR_MODE_OPTIONS}")
print(f"  Label smoothing options: {LABEL_SMOOTHING_OPTIONS}")
print(f"  Seeds: {SEEDS}")
total_runs = (len(BACKBONES) * len(STRIDES) * len(SSL_OPTIONS)
              * len(UNFREEZE_MODE_OPTIONS) * len(LR_MODE_OPTIONS)
              * len(LABEL_SMOOTHING_OPTIONS) * len(SEEDS))
print(f"  TOTAL SUPERVISED RUNS: {total_runs}")
print(f"  SSL video dir: {SSL_VIDEO_DIR} (exists={SSL_VIDEO_DIR.exists()})")
print(f"  Device: {DEVICE}")


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
        if random.random() > 0.5:
            frames_np = np.flip(frames_np, axis=2).copy()
        if random.random() > 0.5:
            shift = np.random.uniform(-0.1, 0.1, (1, 1, 1, 3))
            frames_np = np.clip(frames_np + shift, 0, 1)
        if random.random() > 0.3:
            shear_factor = random.uniform(-0.15, 0.15)
            M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
            for i in range(len(frames_np)):
                frames_np[i] = cv2.warpAffine(
                    frames_np[i], M, (crop_size, crop_size), borderMode=cv2.BORDER_REFLECT
                )
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
        required_span = (self.clip_length - 1) * self.stride + 1
        if total_frames >= required_span:
            if self.train:
                start_idx = random.randint(0, total_frames - required_span)
            else:
                start_idx = (total_frames - required_span) // 2
        else:
            start_idx = 0

        frames = []
        for i in range(self.clip_length):
            current_idx = min(start_idx + i * self.stride, total_frames - 1)
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


# ============================
# 10. Main factorial sweep
# ============================
print("\n" + "=" * 70)
print(f"STARTING FULL FACTORIAL SWEEP -- {total_runs} supervised runs")
print("=" * 70)

all_results = []

for stride in STRIDES:
    slog(f"\n### Building labeled data loaders for stride={stride} ###")
    train_videos, val_videos = collect_clips(TRAIN_DIR), collect_clips(VAL_DIR)

    mean_path = MODEL_SAVE_DIR / f"dataset_mean_stride{stride}.npy"
    std_path = MODEL_SAVE_DIR / f"dataset_std_stride{stride}.npy"
    if mean_path.exists() and std_path.exists():
        mean, std = np.load(str(mean_path)), np.load(str(std_path))
    else:
        stats_dataset = VideoClipDataset(TRAIN_DIR, ACTIVITY_NAMES, train_videos,
                                          CLIP_LENGTH, CROP_SIZE, stride, train=True, normalize=False)
        mean, std = compute_mean_std(stats_dataset, max_samples=200)
        np.save(str(mean_path), mean)
        np.save(str(std_path), std)
    slog(f"  Mean: {mean}  Std: {std}")

    train_dataset = VideoClipDataset(TRAIN_DIR, ACTIVITY_NAMES, train_videos, CLIP_LENGTH,
                                      CROP_SIZE, stride, train=True, normalize=True, mean=mean, std=std)
    val_dataset = VideoClipDataset(VAL_DIR, ACTIVITY_NAMES, val_videos, CLIP_LENGTH,
                                    CROP_SIZE, stride, train=False, normalize=True, mean=mean, std=std)

    targets = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    for backbone_name in BACKBONES:
        # SSL checkpoint computed once per (backbone, stride), reused across
        # unfreeze_mode/lr_mode/seed for ssl_on=True runs.
        ssl_ckpt_path = run_ssl_pretrain(backbone_name, stride, mean, std,
                                          tag=f"{backbone_name}_stride{stride}")

        for ssl_on in SSL_OPTIONS:
            ssl_state_dict = None
            if ssl_on:
                if ssl_ckpt_path is None:
                    slog(f"  [SKIP] ssl_on=True requested for {backbone_name}/stride{stride} "
                         f"but no SSL checkpoint available -- skipping this branch.")
                    continue
                ssl_state_dict = torch.load(ssl_ckpt_path, map_location=DEVICE)

            for unfreeze_mode in UNFREEZE_MODE_OPTIONS:
                for lr_mode in LR_MODE_OPTIONS:
                    for label_smoothing in LABEL_SMOOTHING_OPTIONS:
                        for seed in SEEDS:
                            tag = (f"{backbone_name}_stride{stride}_ssl{ssl_on}_"
                                   f"{unfreeze_mode}_{lr_mode}_ls{label_smoothing}_seed{seed}")
                            result = run_supervised_trial(
                                backbone_name, stride, ssl_on, unfreeze_mode, lr_mode,
                                label_smoothing, seed,
                                ssl_state_dict, train_loader, val_loader, tag
                            )
                            all_results.append(result)

# ============================
# 11. Comprehensive comparison tables
# ============================
df = pd.DataFrame(all_results)
csv_path = MODEL_SAVE_DIR / "full_results.csv"
df.to_csv(csv_path, index=False)

slog("")
slog("=" * 100)
slog("FULL PER-RUN TABLE (all 96 runs), sorted by best_val_acc")
slog("=" * 100)
slog(df.sort_values("best_val_acc", ascending=False).to_string(index=False))

# --- Aggregated over seeds, per full config combo ---
agg_cols = ["backbone", "stride", "ssl_on", "unfreeze_mode", "lr_mode", "label_smoothing"]
agg = df.groupby(agg_cols)["best_val_acc"].agg(["mean", "std", "count"]).reset_index()
agg = agg.rename(columns={"mean": "mean_val_acc", "std": "std_val_acc", "count": "n_seeds"})
agg = agg.sort_values("mean_val_acc", ascending=False)
agg_path = MODEL_SAVE_DIR / "aggregated_by_config.csv"
agg.to_csv(agg_path, index=False)

slog("")
slog("=" * 100)
slog("AGGREGATED BY FULL CONFIG (mean +/- std over seeds), sorted best-first")
slog("=" * 100)
slog(agg.to_string(index=False))

# --- Main-effect marginal tables: isolate each axis's average impact ---
slog("")
slog("=" * 100)
slog("MAIN EFFECTS -- average best_val_acc marginalized over every other axis")
slog("(this is the 'what actually works' summary)")
slog("=" * 100)
for factor in ["backbone", "stride", "ssl_on", "unfreeze_mode", "lr_mode", "label_smoothing"]:
    marginal = df.groupby(factor)["best_val_acc"].agg(["mean", "std", "count"]).reset_index()
    marginal = marginal.sort_values("mean", ascending=False)
    slog(f"\n-- Main effect: {factor} --")
    slog(marginal.to_string(index=False))

# --- Two-way interaction: unfreeze_mode x lr_mode, and ssl_on x backbone ---
slog("\n" + "=" * 100)
slog("SELECTED INTERACTIONS")
slog("=" * 100)
for pair in [("unfreeze_mode", "lr_mode"), ("ssl_on", "backbone"), ("ssl_on", "unfreeze_mode"),
             ("label_smoothing", "unfreeze_mode"), ("label_smoothing", "ssl_on")]:
    pivot = df.pivot_table(values="best_val_acc", index=pair[0], columns=pair[1], aggfunc="mean")
    slog(f"\n-- {pair[0]} x {pair[1]} (mean best_val_acc) --")
    slog(pivot.to_string())

best_overall = df.loc[df["best_val_acc"].idxmax()]
slog("")
slog(f"BEST SINGLE RUN: {best_overall['tag']} "
     f"(val_acc={best_overall['best_val_acc']:.2f}%, val_loss={best_overall['best_val_loss']:.4f})")
slog(f"Checkpoint: {best_overall['checkpoint_path']}")

summary_path = MODEL_SAVE_DIR / "trial4_sweep_summary.txt"
with open(summary_path, 'w') as f:
    f.write("\n".join(SUMMARY_LINES) + "\n")

print(f"\nFull results CSV      -> {csv_path}")
print(f"Aggregated-by-config  -> {agg_path}")
print(f"Text summary          -> {summary_path}")
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)