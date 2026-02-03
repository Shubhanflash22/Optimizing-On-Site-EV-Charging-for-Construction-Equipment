import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from torchvision.models.video import r3d_18, R3D_18_Weights
from torch.cuda.amp import autocast, GradScaler  # For Mixed Precision

# ============================
# 1. Configuration
# ============================
BASE_DIR = Path("/mnt/nvme_data/Avik_Shubhan_codes_data").resolve()
DATASET_DIR = BASE_DIR / "Dataset_Resnet_2"
MODEL_SAVE_DIR = BASE_DIR

NUM_CLASSES = 5
CLIP_LENGTH = 16        
CROP_SIZE = 112         
STRIDE = 2              # Effective FPS: ~30 (if input is 60fps)

BATCH_SIZE = 16 
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4    
NUM_WORKERS = 4         # Increase for faster data loading
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIVITY_NAMES = ['digging', 'idling', 'loading', 'swinging', 'travelling']
ACTIVITY_TO_IDX = {name: idx for idx, name in enumerate(ACTIVITY_NAMES)}

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

set_seed(SEED)

# ============================
# 3. Robust Dataset Class
# ============================
class VideoClipDataset(Dataset):
    def __init__(self, root_dir, activity_names, video_list, clip_length=16, crop_size=112, stride=2, train=True):
        self.root_dir = Path(root_dir)
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.stride = stride
        self.train = train
        self.samples = []

        print(f"Processing {len(video_list)} videos for {'Train' if train else 'Val'}...")

        for video_path in video_list:
            video_path = Path(video_path)
            label_name = video_path.parent.name 
            
            if label_name not in ACTIVITY_TO_IDX:
                continue

            frame_files = sorted([str(f) for f in video_path.glob('*.jpg')])
            
            # [IMPROVEMENT] Accept ALL videos, even short ones (we will loop them)
            if len(frame_files) > 0:
                self.samples.append((frame_files, ACTIVITY_TO_IDX[label_name]))

        print(f"  -> Generated {len(self.samples)} clips.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_files, label = self.samples[idx]
        total_frames = len(frame_files)
        
        # Calculate span needed for one clip
        required_span = (self.clip_length - 1) * self.stride + 1

        # --- Temporal Sampling ---
        if total_frames >= required_span:
            if self.train:
                max_start = total_frames - required_span
                start_idx = random.randint(0, max_start)
            else:
                start_idx = (total_frames - required_span) // 2
        else:
            # Video is too short; start at 0 and loop later
            start_idx = 0

        # --- Frame Selection with Looping ---
        selected_frames = []
        for i in range(self.clip_length):
            current_idx = start_idx + (i * self.stride)
            
            # [IMPROVEMENT] Loop video if we go out of bounds
            if current_idx >= total_frames:
                current_idx = current_idx % total_frames
                
            selected_frames.append(frame_files[current_idx])

        # --- Load & Resize ---
        frames = []
        for f in selected_frames:
            img = cv2.imread(f)
            if img is None:
                img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.crop_size, self.crop_size))
            frames.append(img)

        frames_np = np.array(frames) # (L, H, W, C)

        # --- Augmentation (Train Only) ---
        if self.train:
            # Horizontal Flip
            if random.random() > 0.5:
                frames_np = np.flip(frames_np, axis=2).copy()
            # Color Noise (Prevent overfitting to exact pixel values)
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.02, frames_np.shape)
                frames_np = frames_np + noise
                frames_np = np.clip(frames_np, 0, 1) if frames_np.max() <= 1.0 else frames_np

        # --- Normalization ---
        # Ensure float32 0-1 range first
        if frames_np.max() > 1.0:
             frames_np = frames_np.astype(np.float32) / 255.0

        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        frames_np = (frames_np - mean) / std

        # Permute to Torch format (C, L, H, W)
        clip_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()
        
        return clip_tensor, label

# ============================
# 4. Data Preparation (No Leakage)
# ============================
print("\n" + "="*60)
print("Grouping Videos for Split...")
print("="*60)

all_videos = []
for label in ACTIVITY_NAMES:
    label_dir = DATASET_DIR / label
    if label_dir.exists():
        videos = [v for v in label_dir.iterdir() if v.is_dir()]
        all_videos.extend(videos)

# Shuffle and Split by VIDEO ID
random.shuffle(all_videos)
split_idx = int(0.8 * len(all_videos))
train_videos = all_videos[:split_idx]
val_videos = all_videos[split_idx:]

print(f"Total Videos: {len(all_videos)}")
print(f"Training Videos: {len(train_videos)}")
print(f"Validation Videos: {len(val_videos)}")

train_dataset = VideoClipDataset(DATASET_DIR, ACTIVITY_NAMES, train_videos, 
                               CLIP_LENGTH, CROP_SIZE, STRIDE, train=True)

val_dataset = VideoClipDataset(DATASET_DIR, ACTIVITY_NAMES, val_videos, 
                             CLIP_LENGTH, CROP_SIZE, STRIDE, train=False)

# Weighted Sampler
targets = [label for _, label in train_dataset.samples]
class_counts = np.bincount(targets)
class_weights = 1. / np.maximum(class_counts, 1)
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ============================
# 5. Model & Training Setup
# ============================
print("\nInitializing Model...")
weights = R3D_18_Weights.KINETICS400_V1
model = r3d_18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Freezing Strategy
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

model = model.to(DEVICE)

# Optimizer & Scheduler
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler() # Mixed Precision

# ============================
# 6. Training Loop
# ============================
print("\nStarting Training...")
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # --- Train ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    
    for clips, labels in loop:
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed Precision Forward
        with autocast():
            outputs = model(clips)
            loss = criterion(outputs, labels)
        
        # Mixed Precision Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * clips.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * correct / total
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(DEVICE), labels.to(DEVICE)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * clips.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_dataset)
    
    # Step Scheduler
    scheduler.step()
    
    print(f"Results Epoch {epoch+1}:")
    print(f"  Train Acc: {epoch_acc:.2f}% | Loss: {epoch_loss:.4f}")
    print(f"  Val Acc:   {val_acc:.2f}% | Loss: {val_loss:.4f}")
    
    # --- Checkpointing (Best Only) ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = Path(MODEL_SAVE_DIR) / "resnet3d_best_kinetics_1.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            "accuracy": val_acc,
            "activity_names": ACTIVITY_NAMES,
            "stride": STRIDE
        }, save_path)
        print(f"✅ New Best Model Saved (Loss: {val_loss:.4f})")

print("\nTraining Complete.")