import cv2

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

from torch.utils.data.sampler import WeightedRandomSampler

import numpy as np

from pathlib import Path

from tqdm import tqdm

import random



# Import official 3D ResNet and Weights

from torchvision.models.video import r3d_18, R3D_18_Weights



# ============================

# Configuration

# ============================



#DATASET_DIR = r"C:\Users\shubh\Desktop\New folder\Dataset_Resnet_2"

#MODEL_SAVE_DIR = r"C:\Users\shubh\Desktop\New folder"

BASE_DIR = Path("/mnt/nvme_data/Avik_Shubhan_codes_data").resolve()

DATASET_DIR = BASE_DIR / "Dataset_Resnet_2"

MODEL_SAVE_DIR = BASE_DIR



# Architecture Constants (Matches Chen et al. Paper)

NUM_CLASSES = 5

CLIP_LENGTH = 16        # Paper specifies 16 frames

CROP_SIZE = 112         # Paper specifies 112x112 input

STRIDE = 4              # 60 FPS Video / 4 = 15 FPS (Matches Kinetics Weights)



# Training Hyperparameters

BATCH_SIZE = 16

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4    # Low LR for fine-tuning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Activity Classes

ACTIVITY_NAMES = ['digging', 'idling', 'loading', 'swinging', 'travelling']

ACTIVITY_TO_IDX = {name: idx for idx, name in enumerate(ACTIVITY_NAMES)}



print("="*60)

print("3D ResNet Activity Recognition Training (Kinetics-400 Fine-tuning)")

print("="*60)

print(f"  Dataset: {DATASET_DIR}")

print(f"  Device: {DEVICE}")

print(f"  Clip Length: {CLIP_LENGTH} (Effective FPS: ~15)")

print(f"  Crop Size: {CROP_SIZE}")

print(f"  Stride: {STRIDE}")

print("="*60)



# ============================

# Dataset class (Fixed for Temporal Sampling)

# ============================



class VideoClipDataset(Dataset):

    def __init__(self, root_dir, clip_length=16, crop_size=112, stride=4, train=True):

        self.root_dir = Path(root_dir)

        self.clip_length = clip_length

        self.crop_size = crop_size

        self.stride = stride

        self.train = train  # Flag to enable/disable augmentation

        self.samples = []



        print(f"\nLoading dataset from: {root_dir}")

        for label_name in ACTIVITY_NAMES:

            label_dir = self.root_dir / label_name

            if not label_dir.exists():

                print(f"  ⚠️  Warning: {label_name} folder not found, skipping...")

                continue



            clip_count = 0

            for video_folder in label_dir.iterdir():

                if video_folder.is_dir():

                    # Get all frames and sort them numerically

                    frame_files = sorted([str(f) for f in video_folder.glob('*.jpg')])

                    if len(frame_files) > 0:

                        self.samples.append((frame_files, ACTIVITY_TO_IDX[label_name]))

                        clip_count += 1

            print(f"✓ {label_name}: {clip_count} clips")



        if len(self.samples) == 0:

            raise ValueError("No samples found! Check dataset directory.")

        print(f"\nTotal samples loaded: {len(self.samples)}")



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx):

        frame_files, label = self.samples[idx]

        total_frames = len(frame_files)



        # --- 1. RANDOM TEMPORAL SAMPLING ---

        # Instead of always taking the first frames, pick a random start.

        # This ensures the model sees the "middle" of the action.

        # required_span = self.clip_length * self.stride
        required_span = (self.clip_length - 1) * self.stride + 1



        if self.train and total_frames > required_span:

            # Random start for training

            max_start = total_frames - required_span

            start_idx = random.randint(0, max_start)

        else:

            # Center crop or start for validation/short videos

            if total_frames > required_span:

                start_idx = (total_frames - required_span) // 2

            else:

                start_idx = 0



        # --- 2. FRAME SELECTION WITH LOOPING ---

        selected_frames = []

        for i in range(self.clip_length):

            # Calculate index with stride

            current_idx = start_idx + (i * self.stride)



            # Handle looping for short videos

            if current_idx >= total_frames:

                current_idx = current_idx % total_frames



            selected_frames.append(frame_files[current_idx])



        # --- 3. LOAD & RESIZE ---

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



        # --- 4. AUGMENTATION (Random Horizontal Flip) ---

        if self.train and random.random() > 0.5:

            frames_np = np.flip(frames_np, axis=2).copy()



        # --- 5. NORMALIZATION (Kinetics-400 Stats) ---

        frames_np = frames_np.astype(np.float32) / 255.0

        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)

        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

        frames_np = (frames_np - mean) / std



        # --- 6. PERMUTE TO TORCH FORMAT (C, L, H, W) ---

        clip_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()



        return clip_tensor, label



# ============================

# Prepare Datasets

# ============================



print("\n" + "="*60)

print("Preparing Data Loaders...")

print("="*60)



# Full dataset

full_dataset = VideoClipDataset(DATASET_DIR, CLIP_LENGTH, CROP_SIZE, STRIDE, train=True)



# Split indices

train_size = int(0.8 * len(full_dataset))

val_size = len(full_dataset) - train_size

train_subset, val_subset = random_split(

    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)

)



# Create distinct datasets for Train (with aug) and Val (no aug)

# We share the underlying samples list to save memory but control flags

# train_dataset = VideoClipDataset(DATASET_DIR, CLIP_LENGTH, CROP_SIZE, STRIDE, train=True)

# train_dataset.samples = [full_dataset.samples[i] for i in train_subset.indices]



# val_dataset = VideoClipDataset(DATASET_DIR, CLIP_LENGTH, CROP_SIZE, STRIDE, train=False)

# val_dataset.samples = [full_dataset.samples[i] for i in val_subset.indices]

from torch.utils.data import Subset

# Keep the full dataset
full_dataset = VideoClipDataset(DATASET_DIR, CLIP_LENGTH, CROP_SIZE, STRIDE, train=True)

# Split indices
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Train subset (with augmentation)
train_dataset = Subset(full_dataset, train_subset.indices)

# Validation subset (no augmentation)
full_dataset.train = False  # Turn off augmentation
val_dataset = Subset(full_dataset, val_subset.indices)


# Weighted Sampler to handle class imbalance

# targets = [label for _, label in train_dataset.samples]
targets = [full_dataset.samples[i][1] for i in train_subset.indices] 

class_counts = np.bincount(targets)

class_weights = 1. / np.maximum(class_counts, 1) # avoid divide by zero

sample_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)



# DataLoaders

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)



# Loss Function Weights

# weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# criterion = nn.CrossEntropyLoss(weight=weights_tensor)

criterion = nn.CrossEntropyLoss()



print(f"✓ Train batches: {len(train_loader)}")

print(f"✓ Val batches:   {len(val_loader)}")



# ============================

# Model Initialization (Kinetics-400)

# ============================



print("\n" + "="*60)

print("Initializing Model...")

print("="*60)



# 1. Load Weights

weights = R3D_18_Weights.KINETICS400_V1

model = r3d_18(weights=weights)



# 2. Modify Head for 5 Classes

in_features = model.fc.in_features

model.fc = nn.Linear(in_features, NUM_CLASSES)



# 3. Freezing Strategy

# Freeze everything first

for param in model.parameters():

    param.requires_grad = False



# Unfreeze the last residual block (layer4) and the classification head (fc)

# This allows the model to adapt high-level features without breaking low-level motion features.

print("🔓 Unfreezing 'layer4' and 'fc' for fine-tuning...")

for name, param in model.named_parameters():

    if "layer4" in name or "fc" in name:

        param.requires_grad = True



model = model.to(DEVICE)



# Optimizer (Only optimize unfrozen params)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)



# ============================

# Training Loop

# ============================



print("\n" + "="*60)

print("Starting Training...")

print("="*60)



best_val_loss = float('inf')



for epoch in range(NUM_EPOCHS):

    # --- TRAIN ---

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0



    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

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



        loop.set_postfix(loss=loss.item())



    epoch_loss = running_loss / len(train_dataset)

    epoch_acc = 100 * correct / total



    # --- VALIDATION ---

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0



    with torch.no_grad():

        for clips, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):

            clips, labels = clips.to(DEVICE), labels.to(DEVICE)

            outputs = model(clips)

            loss = criterion(outputs, labels)



            val_loss += loss.item() * clips.size(0)

            _, predicted = outputs.max(1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()



    val_loss /= len(val_dataset)

    val_acc = 100 * val_correct / val_total



    print(f"\nResults Epoch {epoch+1}:")

    print(f"  Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}%")



    # --- SAVE BEST MODEL ---

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        save_path = Path(MODEL_SAVE_DIR) / "resnet3d_best_kinetics.pth"

        torch.save({

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "loss": val_loss,

            "accuracy": val_acc,

            "activity_names": ACTIVITY_NAMES,

            "stride": STRIDE, # Save stride so inference knows!

            "clip_length": CLIP_LENGTH

        }, save_path)

        print(f"✅ Saved Best Model to {save_path}")



print("\nTraining Complete.")
