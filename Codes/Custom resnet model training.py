import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# ============================
# Configuration
# ============================

DATASET_DIR = r"C:\Users\shubh\Desktop\Research work with AVIK\Test\dataset"  # path to your labeled dataset
MODEL_SAVE_PATH = r"C:\Users\shubh\Desktop\Research work with AVIK\Test\best_activity_model.pth"

NUM_CLASSES = 3
CLIP_LENGTH = 16
CROP_SIZE = 112
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIVITY_NAMES = ['digging', 'loading', 'swinging']
ACTIVITY_TO_IDX = {name: idx for idx, name in enumerate(ACTIVITY_NAMES)}

# ============================
# Dataset class
# ============================

class VideoClipDataset(Dataset):
    def __init__(self, root_dir, clip_length=16, crop_size=112):
        self.root_dir = Path(root_dir)
        self.clip_length = clip_length
        self.crop_size = crop_size

        self.samples = []  # list of (list_of_frame_paths, label)
        for label_name in ACTIVITY_NAMES:
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                continue
            for video_folder in label_dir.iterdir():
                if video_folder.is_dir():
                    frame_files = sorted([str(f) for f in video_folder.glob('*.jpg')])
                    if len(frame_files) > 0:
                        self.samples.append((frame_files, ACTIVITY_TO_IDX[label_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_files, label = self.samples[idx]
        frames = []

        for f in frame_files[:self.clip_length]:  # take first CLIP_LENGTH frames
            img = cv2.imread(f)
            if img is None:
                img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.crop_size, self.crop_size))
            frames.append(img)

        # pad if less than CLIP_LENGTH
        while len(frames) < self.clip_length:
            frames.append(frames[-1])

        frames_np = np.array(frames).astype(np.float32) / 255.0  # (L,H,W,C)
        clip_tensor = torch.from_numpy(frames_np).permute(3,0,1,2)  # (C,L,H,W)
        return clip_tensor, label

# ============================
# Model classes (same as before)
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
        if stride !=1 or self.in_planes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x,1); x = self.fc(x)
        return x

def resnet3d_50(num_classes=3):
    return ResNet3D(Bottleneck3D, [3,4,6,3], num_classes=num_classes)

# ============================
# DataLoader
# ============================

dataset = VideoClipDataset(DATASET_DIR, CLIP_LENGTH, CROP_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ============================
# Model, Loss, Optimizer
# ============================

model = resnet3d_50(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================
# Training loop
# ============================

best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
    model.train()
    running_loss = 0.0
    for clips, labels in tqdm(dataloader, desc="Training"):
        clips = clips.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * clips.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"[INFO] Saved new best model to {MODEL_SAVE_PATH}")

print("\nTraining complete.")
print(f"Best model saved at: {MODEL_SAVE_PATH}")
