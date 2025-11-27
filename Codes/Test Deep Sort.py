import os
import cv2
import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

# ============================================================================
# Configuration
# ============================================================================
SAVE_DIR = r"C:\Users\shubh\Desktop\Test\Tubes"
METADATA_PATH = os.path.join(SAVE_DIR, "track_metadata.csv")
ACTIVITY_OUTPUT_CSV = r"C:\Users\shubh\Desktop\Test\Activity_Output.csv"
ACTIVITY_VISUAL_CSV = r"C:\Users\shubh\Desktop\Test\Activity_Visual.csv"
MODEL_PATH = r"C:\Users\shubh\Desktop\Test\best_activity_model.pth"  # Your trained model

FPS = 59  # Match your video FPS
CLIP_LENGTH = 16
CROP_SIZE = 112
CLIP_STRIDE = 1  # Dense sampling as per paper (apply kernels on EVERY 16 frames)

# Activity labels
ACTIVITY_NAMES = {
    0: 'digging',
    1: 'loading', 
    2: 'swinging'
}

# ============================================================================
# 3D ResNet-18 Architecture (Kay et al.)
# Paper uses R3D-18, NOT ResNet-50
# ============================================================================

try:
    from torchvision.models.video import r3d_18
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("WARNING: torchvision not available, using custom R3D-18 implementation")


class BasicBlock3D(nn.Module):
    """Basic 3D Residual Block for R3D-18"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
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


class R3D_18(nn.Module):
    """
    3D ResNet-18 (Kay et al.)
    Used in the paper for activity recognition
    """
    def __init__(self, num_classes=3):
        super(R3D_18, self).__init__()
        self.in_planes = 64
        
        # Stem: temporal stride = 1 (as per paper)
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), 
                                    stride=(1, 2, 2), padding=(1, 1, 1))
        
        # Residual layers: temporal stride = 2 for layers 2-4
        self.layer1 = self._make_layer(BasicBlock3D, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock3D, 128, 2, stride=(2, 2, 2))
        self.layer3 = self._make_layer(BasicBlock3D, 256, 2, stride=(2, 2, 2))
        self.layer4 = self._make_layer(BasicBlock3D, 512, 2, stride=(2, 2, 2))
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
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


def get_r3d18_model(num_classes=3, pretrained=True):
    """
    Get R3D-18 model (Kay et al.) as used in the paper
    
    Args:
        num_classes: Number of activity classes (default: 3)
        pretrained: Load pretrained Kinetics weights
    
    Returns:
        model: R3D-18 model
    """
    if TORCHVISION_AVAILABLE:
        # Use torchvision's official R3D-18 (recommended)
        model = r3d_18(pretrained=pretrained)
        # Replace final layer for 3 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        # Use custom implementation
        print("Using custom R3D-18 implementation")
        return R3D_18(num_classes=num_classes)


# ============================================================================
# Majority Voting for Error Correction
# ============================================================================

def majority_voting(predictions, window_size=50):
    """
    Apply majority voting to smooth predictions.
    Each activity should last at least 2 seconds (based on paper).
    
    Args:
        predictions: List of frame-level predictions
        window_size: Window size in frames (e.g., 50 frames for 2s at 25fps)
    
    Returns:
        Smoothed predictions
    """
    if len(predictions) == 0:
        return []
    
    corrected = []
    for i in range(len(predictions)):
        start = max(0, i - window_size // 2)
        end = min(len(predictions), i + window_size // 2 + 1)
        window = predictions[start:end]
        
        # Get most common prediction in window
        counter = Counter(window)
        most_common = counter.most_common(1)[0][0]
        corrected.append(most_common)
    
    return corrected


# ============================================================================
# Activity Recognition Pipeline
# ============================================================================

class ActivityRecognitionPipeline:
    def __init__(self, model_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load R3D-18 model (Kay et al.) as per paper
        print("Initializing R3D-18 model...")
        self.model = get_r3d18_model(num_classes=3, pretrained=False)
        
        if model_path and os.path.exists(model_path):
            print(f"Loading trained model from {model_path}")
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("✓ Model loaded successfully!")
            except Exception as e:
                print(f"WARNING: Could not load model weights: {e}")
                print("Using random initialization. Please train the model first.")
        else:
            print("WARNING: No trained model found. Using random initialization.")
            print("Please train the model first or provide a valid model path.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.clip_length = CLIP_LENGTH
        self.crop_size = CROP_SIZE
        self.stride = CLIP_STRIDE  # Dense sampling: stride=1
    
    def load_frames_from_folder(self, track_folder):
        """Load all frames for a track from folder"""
        frame_files = sorted([f for f in os.listdir(track_folder) if f.endswith('.jpg')])
        
        frames = []
        frame_indices = []
        for frame_file in frame_files:
            frame_path = os.path.join(track_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                # Extract frame index from filename
                frame_idx = int(frame_file.replace('.jpg', ''))
                frame_indices.append(frame_idx)
        
        return np.array(frames), frame_indices
    
    def prepare_clip(self, frames):
        """Prepare 16-frame clip for model input"""
        # Ensure we have exactly 16 frames
        if len(frames) < self.clip_length:
            # Pad with last frame
            pad_frames = [frames[-1]] * (self.clip_length - len(frames))
            frames = np.concatenate([frames, pad_frames])
        
        frames = frames[:self.clip_length]
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor: (C, L, H, W)
        frames_tensor = torch.from_numpy(frames).float()
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (L, H, W, C) -> (C, L, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
        
        return frames_tensor
    
    def predict_track(self, track_folder):
        """
        Predict activities for an entire track using DENSE sliding window
        Paper: "kernels are applied on every 16 frames"
        This means stride should be small (1-8), not 16!
        
        Returns:
            frame_predictions: Dict mapping frame_idx to list of predictions (for voting)
        """
        print(f"\nProcessing track: {track_folder}")
        
        # Load all frames for this track
        frames, frame_indices = self.load_frames_from_folder(track_folder)
        
        if len(frames) == 0:
            print(f"  No frames found in {track_folder}")
            return {}
        
        print(f"  Total frames: {len(frames)}")
        print(f"  Using stride: {self.stride} (dense sampling)")
        
        # Store multiple predictions per frame (for majority voting)
        frame_predictions_multi = {idx: [] for idx in frame_indices}
        
        # Dense sliding window with stride=1 or stride=8
        num_clips = max(1, (len(frames) - self.clip_length) // self.stride + 1)
        
        print(f"  Processing {num_clips} clips...")
        
        for i in range(num_clips):
            start_idx = i * self.stride
            end_idx = min(start_idx + self.clip_length, len(frames))
            
            # Handle last clip
            if end_idx - start_idx < self.clip_length:
                start_idx = max(0, len(frames) - self.clip_length)
                end_idx = len(frames)
            
            clip_frames = frames[start_idx:end_idx]
            clip_frame_indices = frame_indices[start_idx:end_idx]
            
            # Prepare clip
            clip_tensor = self.prepare_clip(clip_frames)
            clip_tensor = clip_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(clip_tensor)
                _, predicted = output.max(1)
                activity = predicted.item()
            
            # Assign prediction to all frames in this clip
            for frame_idx in clip_frame_indices:
                frame_predictions_multi[frame_idx].append(activity)
        
        # Aggregate predictions: take most common prediction per frame
        frame_predictions = {}
        for frame_idx, preds in frame_predictions_multi.items():
            if len(preds) > 0:
                # Majority vote within this frame's predictions
                counter = Counter(preds)
                frame_predictions[frame_idx] = counter.most_common(1)[0][0]
        
        print(f"  ✓ Predictions made for {len(frame_predictions)} frames")
        return frame_predictions
    
    def process_all_tracks(self):
        """Process all tracks and save results"""
        if not os.path.exists(METADATA_PATH):
            print(f"ERROR: Metadata file not found at {METADATA_PATH}")
            print("Please run Step 2 first to generate tracking data.")
            return
        
        print("=" * 70)
        print("Starting Activity Recognition (Step 4)")
        print("Model: R3D-18 (Kay et al.) - as per paper Section 4.4")
        print(f"Clip length: {CLIP_LENGTH} frames")
        print(f"Stride: {CLIP_STRIDE} (dense sampling)")
        print("=" * 70)
        
        # Read metadata
        with open(METADATA_PATH, 'r') as f:
            reader = csv.DictReader(f)
            tracks = list(reader)
        
        print(f"\nFound {len(tracks)} tracks to process")
        
        all_results = []
        
        for track_info in tracks:
            track_id = int(track_info['track_id'])
            track_folder = track_info['frame_folder']
            
            if not os.path.exists(track_folder):
                print(f"WARNING: Track folder not found: {track_folder}")
                continue
            
            # Get predictions with dense sampling
            frame_predictions = self.predict_track(track_folder)
            
            if not frame_predictions:
                continue
            
            # Sort by frame index
            sorted_frames = sorted(frame_predictions.items())
            frame_nums = [f for f, _ in sorted_frames]
            raw_predictions = [p for _, p in sorted_frames]
            
            # Apply majority voting (2 seconds minimum activity duration)
            # Paper: "each activity lasts at least 2 seconds during operation"
            voting_window = int(2.0 * FPS)
            smoothed_predictions = majority_voting(raw_predictions, window_size=voting_window)
            
            # Save results
            for i, (frame_idx, activity) in enumerate(zip(frame_nums, smoothed_predictions)):
                all_results.append({
                    'track_id': track_id,
                    'frame': frame_idx,
                    'activity_label': activity,
                    'activity_name': ACTIVITY_NAMES[activity],
                    'raw_prediction': raw_predictions[i]
                })
            
            # Print summary
            activity_counts = Counter(smoothed_predictions)
            print(f"\n  Track {track_id} Activity Summary (after majority voting):")
            for activity_id, count in sorted(activity_counts.items()):
                duration_sec = count / FPS
                percentage = (count / len(smoothed_predictions)) * 100
                print(f"    {ACTIVITY_NAMES[activity_id]}: {count} frames "
                      f"({duration_sec:.2f}s, {percentage:.1f}%)")
        
        # Save to CSV
        if all_results:
            print(f"\n{'=' * 70}")
            print(f"Saving results to {ACTIVITY_OUTPUT_CSV}")
            
            with open(ACTIVITY_OUTPUT_CSV, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'track_id', 'frame', 'activity_label', 'activity_name', 'raw_prediction'
                ])
                writer.writeheader()
                writer.writerows(all_results)
            
            # Create visual timeline CSV
            self.create_visual_timeline(all_results)
            
            print(f"✓ Activity recognition completed!")
            print(f"✓ Results saved to: {ACTIVITY_OUTPUT_CSV}")
            print(f"✓ Visual timeline saved to: {ACTIVITY_VISUAL_CSV}")
        else:
            print("\nNo results to save.")
    
    def create_visual_timeline(self, results):
        """Create a timeline of activity segments for visualization"""
        from itertools import groupby
        
        timeline_rows = []
        
        for track_id, group in groupby(results, key=lambda x: x['track_id']):
            group_list = list(group)
            
            # Find continuous segments of same activity
            segments = []
            current_activity = None
            start_frame = None
            
            for item in group_list:
                if item['activity_label'] != current_activity:
                    if current_activity is not None:
                        # Save previous segment
                        duration_sec = (item['frame'] - start_frame) / FPS
                        segments.append({
                            'track_id': track_id,
                            'activity': ACTIVITY_NAMES[current_activity],
                            'start_frame': start_frame,
                            'end_frame': item['frame'] - 1,
                            'duration_sec': round(duration_sec, 2),
                            'start_time_sec': round(start_frame / FPS, 2),
                            'end_time_sec': round((item['frame'] - 1) / FPS, 2)
                        })
                    
                    current_activity = item['activity_label']
                    start_frame = item['frame']
            
            # Save last segment
            if current_activity is not None and group_list:
                last_frame = group_list[-1]['frame']
                duration_sec = (last_frame - start_frame + 1) / FPS
                segments.append({
                    'track_id': track_id,
                    'activity': ACTIVITY_NAMES[current_activity],
                    'start_frame': start_frame,
                    'end_frame': last_frame,
                    'duration_sec': round(duration_sec, 2),
                    'start_time_sec': round(start_frame / FPS, 2),
                    'end_time_sec': round(last_frame / FPS, 2)
                })
            
            timeline_rows.extend(segments)
        
        # Save timeline
        with open(ACTIVITY_VISUAL_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'track_id', 'activity', 'start_frame', 'end_frame', 
                'duration_sec', 'start_time_sec', 'end_time_sec'
            ])
            writer.writeheader()
            writer.writerows(timeline_rows)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline with R3D-18
    pipeline = ActivityRecognitionPipeline(
        model_path=MODEL_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process all tracks
    pipeline.process_all_tracks()
    
    print("\n" + "=" * 70)
    print("Activity Recognition Pipeline Complete!")
    print("Architecture: R3D-18 (Kay et al.)")
    print(f"Sampling: Dense (stride={CLIP_STRIDE})")
    print("Post-processing: Majority voting (2s minimum duration)")
    print("=" * 70)