import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.models.video import r3d_18, mc3_18  

# ============================
# Configuration
# ============================

DATASET_DIR = r"C:\Users\shubh\Desktop\New folder\Dataset_Resnet_2"
MODEL_SAVE_PATH = r"C:\Users\shubh\Desktop\New folder\resnet3d_best_mc3_18.pth"
NUM_CLASSES = 5
CLIP_LENGTH = 32
CROP_SIZE = 160
STRIDE = 4
BATCH_SIZE = 4
NUM_EPOCHS = 20
EPOCHS_PER_STAGE = 5
LEARNING_RATE = 1e-4
MIN_CONFIDENCE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All activities from CVAT (lowercase to match folder names)
ACTIVITY_NAMES = ['digging', 'idling', 'loading', 'swinging', 'travelling']
ACTIVITY_TO_IDX = {name: idx for idx, name in enumerate(ACTIVITY_NAMES)}

print("="*60)
print("3D ResNet Activity Recognition Training")
print("="*60)
print("\nConfiguration:")
print(f"  Dataset: {DATASET_DIR}")
print(f"  Device: {DEVICE}")
print(f"  Activities: {ACTIVITY_NAMES}")
print(f"  Num Classes: {NUM_CLASSES}")
print(f"  Clip Length: {CLIP_LENGTH} frames")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"EPOCHS per stage: {EPOCHS_PER_STAGE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print("="*60)

# ============================
# Dataset class
# ============================

class VideoClipDataset(Dataset):
    def __init__(self, root_dir, clip_length=16, crop_size=112):
        self.root_dir = Path(root_dir)
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.samples = []  # list of (list_of_frame_paths, label)
        
        print(f"\nLoading dataset from: {root_dir}")
        for label_name in ACTIVITY_NAMES:
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                print(f"  ⚠️  Warning: {label_name} folder not found, skipping...")
                continue
            clip_count = 0
            for video_folder in label_dir.iterdir():
                if video_folder.is_dir():
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
        frames = []

        # Sample frames with stride
        for f in frame_files[::STRIDE][:CLIP_LENGTH]:
#        for f in frame_files[:CLIP_LENGTH]:
            img = cv2.imread(f)
            if img is None:
                img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.crop_size, self.crop_size))
            frames.append(img)
        # Pad if less than CLIP_LENGTH
        while len(frames) < CLIP_LENGTH:
            frames.append(frames[-1])

        frames_np = np.array(frames).astype(np.float32) / 255.0

        # Pretrained 3D ResNet normalization (optional)
        # mean = np.array([0.43216, 0.394666, 0.37645])
        # std = np.array([0.22803, 0.22145, 0.216989])
        # frames_np = (frames_np - mean) / std
        # FIX → ensure float32 always
        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        frames_np = ((frames_np - mean) / std).astype(np.float32)
        clip_tensor = torch.from_numpy(frames_np).permute(3,0,1,2)  # (C,L,H,W)
        return clip_tensor, label

# ============================
# Model classes
# ============================

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=4, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=6):
        super(ResNet3D, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2,2,2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2,2,2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2,2,2))
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
    
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet3d_18(num_classes=NUM_CLASSES):
    return ResNet3D(BasicBlock3D, [2,2,2,2], num_classes=num_classes)

# ============================
# DataLoader
# ============================

print("\n" + "="*60)
print("Loading Dataset...")
print("="*60)

dataset = VideoClipDataset(DATASET_DIR, CLIP_LENGTH, CROP_SIZE)
# ============================
# Train / Validation Split
# ============================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # reproducibility
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


if len(dataset) == 0:
    print("\n❌ ERROR: Dataset is empty!")
    print("Make sure you ran yolo_cvat_to_resnet.py first")
    exit(1)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


print(f"\n✓ Train batches: {len(train_loader)}")
print(f"✓ Validation batches: {len(val_loader)}")


print(f"\n✓ DataLoader created with {len(train_loader)} batches")

# ============================
# Model, Loss, Optimizer
# ============================

print("\n" + "="*60)
print("Initializing Model...")
print("="*60)

# Use pretrained 3D ResNet
#model = r3d_18(pretrained=True) 
model = mc3_18(pretrained=True)

# Replace final fc for our 5 classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(DEVICE)

# Freeze all layers first
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze layer3, layer4, and fc
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True

# Create a new optimizer for the trainable params
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)


# # Optional: freeze layers for faster fine-tuning
# for name, param in model.named_parameters():
#     if "fc" not in name:  # only train final layer
#         param.requires_grad = False

# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
best_loss = float('inf')
start_epoch = 0
if Path(MODEL_SAVE_PATH).exists():
    print(f"\n🔁 Loading existing model: {MODEL_SAVE_PATH}")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print("⚠️ Optimizer mismatch — using fresh optimizer")

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss = checkpoint.get('loss', float('inf'))
    print(f"➡️ Resuming from epoch {start_epoch}")
else:
    print("\n🆕 No checkpoint found — starting fresh")
    
if start_epoch > 5:
    print("⚠️ Checkpoint is past warmup — enabling full fine-tuning")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer.param_groups[0]['params'] = [p for p in model.parameters() if p.requires_grad]

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel: Pretrained 3D ResNet-18 (mc3_18)")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# ============================
# Training loop
# ============================

print("\n" + "="*60)
print("Starting Training...")
print("="*60)
criterion = nn.CrossEntropyLoss()

for epoch in range(start_epoch, max(NUM_EPOCHS, start_epoch + 1)):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*60}")
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        clips = clips.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * clips.size(0)
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * correct / total
    
    print(f"\nEpoch {epoch+1} Results:")
    print(f"  Loss: {epoch_loss:.4f}")
    print(f"  Accuracy: {epoch_acc:.2f}%")
    if epoch == 10:
        print("\n🔓 Unfreezing full model for fine-tuning...\n")
        for param in model.parameters():
            param.requires_grad = True
        # NEW optimizer with smaller LR
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # ============================
    # Validation
    # ============================
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for clips, labels in val_loader:
            clips= clips.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * clips.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_loss /= len(val_dataset)
    val_acc = 100 * val_correct / val_total
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.2f}%")
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            'loss':val_loss,
            'accuracy': epoch_acc,
            'activity_names': ACTIVITY_NAMES,
        }, MODEL_SAVE_PATH)
        print(f"  ✓ Saved new best model (loss: {val_loss:.4f})")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\n✅ Best model saved at: {MODEL_SAVE_PATH}")
print(f"   Best loss: {best_loss:.4f}")
print(f"\nActivity classes: {ACTIVITY_NAMES}")