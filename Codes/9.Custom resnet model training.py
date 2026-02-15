"""
3D ResNet Training for Excavator Activity Recognition
======================================================

This implementation follows the methodology described in the paper's Section 4.4 and 5.1.

Paper Reference Points:
- Model: 3D ResNet (ResNet architecture with 3D convolutions)
- Input: 16 × 112 × 112 video frames
- FPS: 25 FPS (all video clips fixed at this rate)
- Batch size: 16
- Learning rate: 0.001
- Activities: digging, loading, swinging (paper has 3, this code has 5)
- Augmentation: flipping, channel shifting, frame shearing
- Pre-training: Fine-tuned from Kinetics-400 weights
- Post-processing: Majority voting for temporal smoothing

DEVIATIONS & JUSTIFICATIONS:
1. Using 5 classes instead of 3 - Extended task scope (valid assumption)
2. PyTorch R3D-18 instead of custom ResNet - Equivalent architecture (valid)
3. Adam optimizer - Not specified in paper, but standard choice (valid)
4. Mixed precision training - Not in paper, but improves efficiency (valid)
"""

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
from torch.cuda.amp import autocast, GradScaler  # NOT in paper, but improves training speed
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix  # Paper uses precision/recall
import json

def compute_mean_std(dataset, max_samples=None):
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    count = 0

    indices = range(len(dataset))
    if max_samples is not None:
        indices = random.sample(list(indices), min(max_samples, len(dataset)))

    for idx in tqdm(indices, desc="Computing dataset mean/std"):
        frames, _ = dataset[idx]   # (C, T, H, W)
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

# File paths - Adjust to your system
BASE_DIR = Path("/mnt/nvme1/avik_shubhan/resnet3d/").resolve()
DATASET_DIR = BASE_DIR / "Dataset_Resnet_3"
MODEL_SAVE_DIR = BASE_DIR

# Model Architecture Parameters
# PAPER: "network takes 16 × 112 × 112 video frames as input"
NUM_CLASSES = 5          # DEVIATION: Paper has 3 classes (digging, loading, swinging)
                         # JUSTIFICATION: Extended to 5 activities for more comprehensive monitoring
CLIP_LENGTH = 16         # MATCHES PAPER: 16 frames per clip
CROP_SIZE = 112          # MATCHES PAPER: 112×112 spatial resolution

# Temporal Sampling Parameters
# PAPER: "all video clips are fixed at 25 FPS for training"
TARGET_FPS = 25          # MATCHES PAPER: 25 FPS target
STRIDE = 1               # CORRECTED: With 25 FPS video, stride=1 gives proper temporal sampling
                         # PAPER SAYS: "temporal strides are 1 for first conv, 2 for others"
                         # NOTE: That refers to MODEL architecture, not data loading
                         # The R3D-18 model already implements this internally

# Training Hyperparameters
# PAPER: "batch size of the model was set to 16"
BATCH_SIZE = 16          # MATCHES PAPER: Batch size of 16

# PAPER: "learning rate was set to 0.001"
LEARNING_RATE = 1e-3     # MATCHES PAPER: 0.001 learning rate
                         # CORRECTED from original 1e-4

NUM_EPOCHS = 20          # NOT specified in paper, but reasonable for fine-tuning
NUM_WORKERS = 4          # NOT in paper - system optimization
SEED = 42                # NOT in paper - for reproducibility

# Hardware configuration
# PAPER: "NVIDIA GeForce GTX 1080GPU and 32 gigabytes" for training
# ASSUMPTION: Using available GPU is valid
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Activity Labels
# PAPER has: ['digging', 'loading', 'swinging']
# EXTENDED to include 'idling' and 'travelling' for comprehensive activity recognition
ACTIVITY_NAMES = ['digging', 'idling', 'loading', 'swinging', 'travelling']
ACTIVITY_TO_IDX = {name: idx for idx, name in enumerate(ACTIVITY_NAMES)}

print(f"Configuration:")
print(f"  Target FPS: {TARGET_FPS} (matches paper)")
print(f"  Clip Length: {CLIP_LENGTH} frames (matches paper)")
print(f"  Spatial Size: {CROP_SIZE}×{CROP_SIZE} (matches paper)")
print(f"  Batch Size: {BATCH_SIZE} (matches paper)")
print(f"  Learning Rate: {LEARNING_RATE} (matches paper)")
print(f"  Device: {DEVICE}")

# ============================
# 2. Reproducibility Setup
# ============================
# NOT explicitly in paper, but essential for reproducible research
def set_seed(seed):
    """
    Set random seeds for reproducibility across runs.
    JUSTIFICATION: Scientific best practice, not mentioned in paper but critical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Sacrifices speed for reproducibility

set_seed(SEED)

# ============================
# 3. Dataset Class - Aligned with Paper
# ============================
class VideoClipDataset(Dataset):
    """
    Dataset for loading video clips for 3D ResNet training.
    
    PAPER ALIGNMENT:
    - Input: 16×112×112 frames (matches paper Section 4.4)
    - Augmentation: Flip, channel shift, shear (matches paper Section 4.5)
    - Temporal sampling: Uses stride to achieve target FPS
    
    DEVIATIONS:
    - Handles variable-length videos by padding (paper doesn't specify)
    - Random temporal crops during training (standard practice)
    """
    
    def __init__(self, root_dir, activity_names, video_list, 
                 clip_length=16, crop_size=112, stride=1, train=True,
                normalize=True, mean=None, std=None):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing activity folders
            activity_names: List of activity class names
            video_list: List of video paths to include
            clip_length: Number of frames per clip (PAPER: 16)
            crop_size: Spatial dimension after cropping (PAPER: 112)
            stride: Frame sampling stride (CORRECTED: 1 for 25fps)
            train: Whether this is training set (affects augmentation)
        """
        self.root_dir = Path(root_dir)
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.stride = stride
        self.train = train
        self.samples = []
        self.normalize = normalize
        self.mean = mean
        self.std = std

        print(f"\n{'='*60}")
        print(f"Building Dataset ({'Train' if train else 'Validation'})")
        print(f"{'='*60}")
        print(f"Processing {len(video_list)} videos...")

        # Build sample list from video directories
        for video_path in video_list:
            video_path = Path(video_path)
            label_name = video_path.parent.name  # Folder name = activity label
            
            # Skip if not a valid activity
            if label_name not in ACTIVITY_TO_IDX:
                continue

            # Get all frame files in this video directory
            frame_files = sorted([str(f) for f in video_path.glob('*.jpg')])
            
            # PAPER doesn't specify handling of short videos
            # ASSUMPTION: Include all videos, handle short ones via padding
            if len(frame_files) > 0:
                self.samples.append((frame_files, ACTIVITY_TO_IDX[label_name]))

        print(f"  → Loaded {len(self.samples)} video samples")
        
        # Print class distribution
        labels = [label for _, label in self.samples]
        for idx, activity in enumerate(ACTIVITY_NAMES):
            count = labels.count(idx)
            print(f"  → {activity:12s}: {count:4d} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and process a single video clip.
        
        PAPER ALIGNMENT:
        - Returns 16 frames of size 112×112
        - Applies augmentation (flip, channel shift, shear)
        - Normalizes using mean/std
        
        Returns:
            clip_tensor: (C, L, H, W) = (3, 16, 112, 112)
            label: Integer class label
        """
        frame_files, label = self.samples[idx]
        total_frames = len(frame_files)
        
        # Calculate how many frames we need to sample
        # With stride=1 and clip_length=16, we need 16 consecutive frames
        required_span = (self.clip_length - 1) * self.stride + 1

        # ===== TEMPORAL SAMPLING =====
        # PAPER: Not explicitly specified, but standard practice
        # ASSUMPTION: Random crops during training, center crop during validation
        if total_frames >= required_span:
            if self.train:
                # Random temporal crop - helps model generalize to different phases
                max_start = total_frames - required_span
                start_idx = random.randint(0, max_start)
            else:
                # Center crop for validation - consistent evaluation
                start_idx = (total_frames - required_span) // 2
        else:
            # Video too short - start at beginning and pad
            # PAPER: Doesn't mention this case
            # ASSUMPTION: Start at 0 and handle padding below
            start_idx = 0

        # ===== FRAME SELECTION =====
        # Select frames according to stride
        selected_frames = []
        for i in range(self.clip_length):
            current_idx = start_idx + (i * self.stride)
            
            # Handle videos shorter than required span
            # ASSUMPTION: Repeat last frame rather than looping
            # JUSTIFICATION: More natural than jumping back to start
            if current_idx >= total_frames:
                current_idx = total_frames - 1
                
            selected_frames.append(frame_files[current_idx])

        # ===== LOAD AND RESIZE FRAMES =====
        frames = []
        for f in selected_frames:
            # Read frame
            img = cv2.imread(f)
            
            # Handle corrupted/missing frames
            # PAPER: Doesn't specify error handling
            # ASSUMPTION: Use black frame if load fails
            if img is None:
                img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB (OpenCV loads as BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # SPATIAL PROCESSING
                # PAPER: "network takes 16 × 112 × 112 video frames as input"
                # STANDARD PRACTICE: Resize larger, then crop for augmentation
                if self.train:
                    # Training: Resize to 128, then random crop to 112
                    # JUSTIFICATION: Provides scale augmentation
                    img = cv2.resize(img, (128, 128))
                    
                    # Random spatial crop
                    top = random.randint(0, 16)   # 128 - 112 = 16
                    left = random.randint(0, 16)
                    img = img[top:top+self.crop_size, left:left+self.crop_size]
                else:
                    # Validation: Resize to 128, then center crop to 112
                    # Ensures consistent evaluation
                    img = cv2.resize(img, (128, 128))
                    img = img[8:120, 8:120]  # Center crop
                    
            frames.append(img)

        # Convert to numpy array: (L, H, W, C)
        frames_np = np.array(frames).astype(np.float32) / 255.0  # Normalize to [0, 1]

        # ===== DATA AUGMENTATION =====
        # PAPER Section 4.5: "data augmentation technique is used to extend 
        #                     the size of the dataset by flipping the video frames, 
        #                     shifting the image channels, and shearing the frame size"
        
        if self.train:
            # 1. HORIZONTAL FLIP
            # PAPER: "flipping the video frames"
            # MATCHES PAPER
            if random.random() > 0.5:
                frames_np = np.flip(frames_np, axis=2).copy()  # Flip along width
            
            # 2. CHANNEL SHIFTING
            # PAPER: "shifting the image channels"
            # INTERPRETATION: Brightness/contrast variation via channel-wise shifts
            # MATCHES PAPER
            if random.random() > 0.5:
                # Random shift for each channel
                shift = np.random.uniform(-0.1, 0.1, (1, 1, 1, 3))
                frames_np = np.clip(frames_np + shift, 0, 1)
            
            # 3. SHEARING
            # PAPER: "shearing the frame size"
            # INTERPRETATION: Affine shear transformation
            # MATCHES PAPER
            if random.random() > 0.3:  # 70% probability
                shear_factor = random.uniform(-0.15, 0.15)
                
                # Create shear transformation matrix
                M = np.array([[1, shear_factor, 0], 
                              [0, 1, 0]], dtype=np.float32)
                
                # Apply same shear to all frames in the clip
                # JUSTIFICATION: Maintains temporal consistency
                for i in range(len(frames_np)):
                    frames_np[i] = cv2.warpAffine(
                        frames_np[i], M, 
                        (self.crop_size, self.crop_size),
                        borderMode=cv2.BORDER_REFLECT  # Handle borders gracefully
                    )

        # ===== NORMALIZATION =====
        # PAPER: Doesn't specify normalization values
        # ASSUMPTION: Using Kinetics-400 dataset statistics (standard for pre-trained models)
        # JUSTIFICATION: Model was pre-trained on Kinetics, so same normalization expected
        #mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        #std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        if self.normalize:
            frames_np = (frames_np - self.mean) / self.std

        #frames_np = (frames_np - mean) / std

        # ===== FORMAT FOR PYTORCH =====
        # Convert from (L, H, W, C) to (C, L, H, W)
        # PyTorch video models expect (Channel, Time, Height, Width)
        clip_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()
        
        return clip_tensor, label

# ============================
# 4. Data Preparation - No Data Leakage
# ============================
print("\n" + "="*60)
print("Preparing Dataset Split")
print("="*60)

# Collect all video directories
all_videos = []
for label in ACTIVITY_NAMES:
    label_dir = DATASET_DIR / label
    if label_dir.exists():
        # Each subdirectory is one video (contains frame images)
        videos = [v for v in label_dir.iterdir() if v.is_dir()]
        all_videos.extend(videos)
    else:
        print(f"WARNING: Directory not found for activity '{label}'")

print(f"Total videos found: {len(all_videos)}")

# CRITICAL: Split by VIDEO, not by frames
# PAPER: Doesn't specify split strategy
# ASSUMPTION: 80/20 train/val split at video level
# JUSTIFICATION: Prevents data leakage (frames from same video in both sets)
random.shuffle(all_videos)
split_idx = int(0.8 * len(all_videos))
train_videos = all_videos[:split_idx]
val_videos = all_videos[split_idx:]

print(f"Training videos: {len(train_videos)} (80%)")
print(f"Validation videos: {len(val_videos)} (20%)")

# Create datasets
# CORRECTED: stride=1 to work with 25 FPS video
#train_dataset = VideoClipDataset(
#    DATASET_DIR, ACTIVITY_NAMES, train_videos, 
#    CLIP_LENGTH, CROP_SIZE, STRIDE, train=True
#)

#val_dataset = VideoClipDataset(
#    DATASET_DIR, ACTIVITY_NAMES, val_videos, 
#    CLIP_LENGTH, CROP_SIZE, STRIDE, train=False
#)

# Temporary dataset WITHOUT normalization
stats_dataset = VideoClipDataset(
    DATASET_DIR, ACTIVITY_NAMES, train_videos,
    CLIP_LENGTH, CROP_SIZE, STRIDE,
    train=True,
    normalize=False
)

mean, std = compute_mean_std(stats_dataset, max_samples=200)

print("Computed mean:", mean)
print("Computed std :", std)

np.save("dataset_mean.npy", mean)
np.save("dataset_std.npy", std)

mean = np.load("dataset_mean.npy")
std = np.load("dataset_std.npy")

train_dataset = VideoClipDataset(
    DATASET_DIR, ACTIVITY_NAMES, train_videos,
    CLIP_LENGTH, CROP_SIZE, STRIDE,
    train=True,
    normalize=True,
    mean=mean,
    std=std
)

val_dataset = VideoClipDataset(
    DATASET_DIR, ACTIVITY_NAMES, val_videos,
    CLIP_LENGTH, CROP_SIZE, STRIDE,
    train=False,
    normalize=True,
    mean=mean,
    std=std
)

# ===== CLASS BALANCING =====
# PAPER: Doesn't mention class balancing
# ASSUMPTION: Use weighted sampling to handle class imbalance
# JUSTIFICATION: Common practice for imbalanced datasets, improves learning

print("\nSetting up weighted sampler for class balance...")
targets = [label for _, label in train_dataset.samples]
class_counts = np.bincount(targets, minlength=NUM_CLASSES)
class_weights = 1.0 / np.maximum(class_counts, 1)  # Avoid division by zero

print("Class weights:")
for idx, activity in enumerate(ACTIVITY_NAMES):
    print(f"  {activity:12s}: {class_weights[idx]:.4f}")

sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(
    sample_weights, 
    num_samples=len(sample_weights), 
    replacement=True
)

# Create data loaders
# PAPER: batch_size=16
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,  # Use weighted sampler instead of shuffle
    num_workers=NUM_WORKERS, 
    pin_memory=True  # Faster GPU transfer
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,  # No shuffling for validation
    num_workers=NUM_WORKERS, 
    pin_memory=True
)

# ============================
# 5. Model Setup
# ============================
print("\n" + "="*60)
print("Initializing Model")
print("="*60)

# PAPER: "3D ResNet" fine-tuned from Kinetics-400
# USING: PyTorch's r3d_18 with Kinetics-400 pre-trained weights
# JUSTIFICATION: 
#   - R3D-18 implements 3D ResNet architecture
#   - Same pre-training dataset as mentioned in paper
#   - Equivalent to paper's approach

print("Loading R3D-18 with Kinetics-400 pre-trained weights...")
weights = R3D_18_Weights.KINETICS400_V1  # MATCHES PAPER: Pre-trained on Kinetics-400
model = r3d_18(weights=weights)

# Replace final layer for our number of classes
# PAPER has 3 classes, we have 5
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
print(f"  → Modified final layer: {in_features} → {NUM_CLASSES} classes")

# ===== FINE-TUNING STRATEGY =====
# PAPER: "The training of the activity recognition model started from 
#         finetuning the 3D ResNet work developed by Kay et al."
# PAPER: "The finetuning is to efficiently initialize the parameters 
#         of the model and avoid overfitting"

# STRATEGY: Freeze early layers, train deeper layers + classifier
# JUSTIFICATION: 
#   - Preserves learned low-level features
#   - Prevents overfitting on small dataset
#   - Standard fine-tuning practice

print("\nApplying layer freezing strategy:")
print("  → Freezing all layers initially...")
for param in model.parameters():
    param.requires_grad = False

print("  → Unfreezing layer4 (deeper features) and fc (classifier)...")
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
        print(f"     ✓ {name}")

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
      f"({100*trainable_params/total_params:.1f}%)")

# Move model to GPU
model = model.to(DEVICE)

# ============================
# 6. Training Setup
# ============================
print("\n" + "="*60)
print("Configuring Training")
print("="*60)

# OPTIMIZER
# PAPER: Doesn't specify optimizer
# USING: Adam optimizer
# JUSTIFICATION: 
#   - Industry standard for deep learning
#   - Works well with learning rate 0.001
#   - Better than SGD for fine-tuning in most cases

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LEARNING_RATE  # MATCHES PAPER: 0.001
)
print(f"Optimizer: Adam with LR={LEARNING_RATE} (paper specifies 0.001)")

# LEARNING RATE SCHEDULER
# PAPER: Doesn't specify LR schedule
# USING: ReduceLROnPlateau
# JUSTIFICATION:
#   - Reduces LR when validation loss plateaus
#   - Helps model converge to better minimum
#   - Standard practice for fine-tuning

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Minimize validation loss
    factor=0.5,      # Reduce LR by half
    patience=3,      # Wait 3 epochs before reducing
    verbose=True
)
print("LR Scheduler: ReduceLROnPlateau (not in paper, but improves convergence)")

# LOSS FUNCTION
# PAPER: Doesn't specify loss function
# USING: Cross-Entropy Loss
# JUSTIFICATION: Standard for multi-class classification
criterion = nn.CrossEntropyLoss()
print("Loss: CrossEntropyLoss (standard for classification)")

# MIXED PRECISION TRAINING
# PAPER: Not mentioned
# USING: Automatic Mixed Precision (AMP)
# JUSTIFICATION:
#   - Faster training without accuracy loss
#   - Reduces memory usage
#   - Modern best practice
scaler = GradScaler()
print("Using Mixed Precision Training (speeds up training)")

# ============================
# 7. Evaluation Function
# ============================
def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on validation set.
    
    PAPER Section 5.1: Uses precision and recall metrics
    MATCHES PAPER: Computes TP, FP, FN for precision/recall calculation
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Overall accuracy
        precision: Per-class precision
        recall: Per-class recall
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for clips, labels in dataloader:
            clips = clips.to(device)
            labels = labels.to(device)
            
            outputs = model(clips)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * clips.size(0)
            
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Calculate metrics
    # PAPER Equations (4) and (5):
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, 
        average=None,  # Per-class metrics
        labels=range(NUM_CLASSES),
        zero_division=0
    )
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy, precision, recall

# ============================
# 8. Training Loop
# ============================
print("\n" + "="*60)
print("Starting Training")
print("="*60)
print(f"Total epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE} (matches paper)")
print(f"Batches per epoch: {len(train_loader)}")
print("="*60 + "\n")

best_val_loss = float('inf')
best_val_acc = 0.0
training_history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'val_precision': [], 'val_recall': []
}

for epoch in range(NUM_EPOCHS):
    # ===== TRAINING PHASE =====
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    
    for clips, labels in loop:
        # Move data to GPU
        clips = clips.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        # NOT IN PAPER: Using AMP for efficiency
        with autocast():
            outputs = model(clips)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * clips.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item(), acc=100*correct/total)
    
    # Calculate epoch metrics
    train_loss = running_loss / len(train_dataset)
    train_acc = 100.0 * correct / total
    
    # ===== VALIDATION PHASE =====
    val_loss, val_acc, val_precision, val_recall = evaluate_model(
        model, val_loader, criterion, DEVICE
    )
    
    # Step scheduler based on validation loss
    scheduler.step(val_loss)
    
    # ===== LOGGING =====
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary")
    print(f"{'='*60}")
    print(f"Training   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
    print(f"Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc*100:.2f}%")
    print(f"\nPer-Class Metrics (Validation):")
    print(f"{'Activity':<12} {'Precision':<12} {'Recall':<12}")
    print(f"{'-'*36}")
    
    # PAPER Table 3 format: Reports precision and recall per activity
    # MATCHES PAPER: Same metrics as reported in Table 3
    for i, activity in enumerate(ACTIVITY_NAMES):
        print(f"{activity:<12} {val_precision[i]*100:>10.1f}%  {val_recall[i]*100:>10.1f}%")
    
    avg_precision = np.mean(val_precision)
    avg_recall = np.mean(val_recall)
    print(f"{'-'*36}")
    print(f"{'Average':<12} {avg_precision*100:>10.1f}%  {avg_recall*100:>10.1f}%")
    print(f"{'='*60}\n")
    
    # Save training history
    training_history['train_loss'].append(train_loss)
    training_history['train_acc'].append(train_acc)
    training_history['val_loss'].append(val_loss)
    training_history['val_acc'].append(val_acc*100)
    training_history['val_precision'].append(val_precision.tolist())
    training_history['val_recall'].append(val_recall.tolist())
    
    # ===== MODEL CHECKPOINTING =====
    # PAPER: Doesn't specify checkpointing strategy
    # STRATEGY: Save best model based on validation loss
    # JUSTIFICATION: Prevents overfitting, standard practice
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc * 100
        
        # Save complete checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc * 100,
            'val_precision': val_precision.tolist(),
            'val_recall': val_recall.tolist(),
            'activity_names': ACTIVITY_NAMES,
            'config': {
                'clip_length': CLIP_LENGTH,
                'crop_size': CROP_SIZE,
                'stride': STRIDE,
                'target_fps': TARGET_FPS,
                'num_classes': NUM_CLASSES,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE
            }
        }
        
        save_path = MODEL_SAVE_DIR / "resnet3d_best_kinetics_2.pth"
        torch.save(checkpoint, save_path)
        print(f"✅ New Best Model Saved!")
        print(f"   → Path: {save_path}")
        print(f"   → Val Loss: {val_loss:.4f}")
        print(f"   → Val Acc: {val_acc*100:.2f}%\n")

# ============================
# 9. Training Complete - Save Results
# ============================
print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# PAPER Section 5.1 Table 3: Reports final precision/recall
# Compare with paper's results:
#   - Digging: 95% precision, 86% recall
#   - Swinging: 86% precision, 93% recall  
#   - Loading: 84% precision, 80% recall
#   - Average: 87.6% accuracy

print("\nPAPER COMPARISON:")
print("Paper achieved 87.6% average accuracy on 3 classes")
print(f"This implementation: {best_val_acc:.2f}% on {NUM_CLASSES} classes")

# Save training history
history_path = MODEL_SAVE_DIR / "training_history.json"
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"\nTraining history saved to: {history_path}")

print("\n" + "="*60)
print("ALIGNMENT WITH PAPER - SUMMARY")
print("="*60)
print("✅ MATCHES PAPER:")
print("   - Model: 3D ResNet (R3D-18)")
print("   - Pre-training: Kinetics-400")
print("   - Input: 16 × 112 × 112 frames")
print("   - FPS: 25 (corrected)")
print("   - Batch size: 16")
print("   - Learning rate: 0.001 (corrected)")
print("   - Augmentation: Flip, channel shift, shear")
print("   - Metrics: Precision, Recall, Accuracy")
print("\n📝 VALID ASSUMPTIONS:")
print("   - Extended to 5 activity classes")
print("   - Adam optimizer (not specified in paper)")
print("   - ReduceLROnPlateau scheduler")
print("   - Mixed precision training")
print("   - Weighted sampling for class balance")
print("   - 80/20 train/val split")
print("="*60)
