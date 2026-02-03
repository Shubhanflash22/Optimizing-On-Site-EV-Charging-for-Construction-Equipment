"""
3D ResNet Inference for Excavator Activity Recognition
=======================================================

This implements the inference and post-processing methodology from the paper.

PAPER REFERENCE - Section 4.4 and 4.5:
"At last, each frame is labeled to indicate the excavator activity after 
correcting the recognition errors with majority voting, as shown in Fig. 5. 
This is because excavators usually work continuously and cannot change their 
activity states in a short period of time. It was observed that each activity 
lasts at least 2 s during the operation."

PAPER REFERENCE - Section 4.5:
"The time for each cycle is measured following the workflow in Fig. 7. 
After the activity recognition, each video frame is labeled to indicate 
the activity of the excavator in the frame."

KEY FEATURES:
1. Frame-by-frame prediction on video
2. Majority voting for temporal smoothing
3. Cycle time calculation (digging → swinging → loading)
4. Productivity calculation
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision.models.video import r3d_18
import json
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt

# ============================
# Configuration
# ============================
# Load model checkpoint to get configuration
MODEL_PATH = Path(r"C:\Users\shubh\Desktop\DELETE AFTER USE\resnet3d_best_kinetics_2.pth")
VIDEO_PATH = Path(r"C:\Users\shubh\Desktop\DELETE AFTER USE\delete.mp4")  # UPDATE THIS
OUTPUT_DIR = Path(r"C:\Users\shubh\Desktop\DELETE AFTER USE")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Load Model and Configuration
# ============================
print("="*60)
print("Loading Model")
print("="*60)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
config = checkpoint['config']

# Extract configuration
# PAPER: 16 × 112 × 112 input frames
CLIP_LENGTH = config['clip_length']      # 16
CROP_SIZE = config['crop_size']          # 112
STRIDE = config['stride']                # 1
TARGET_FPS = config['target_fps']        # 25
NUM_CLASSES = config['num_classes']      # 5
ACTIVITY_NAMES = checkpoint['activity_names']

print(f"Model Configuration:")
print(f"  Clip Length: {CLIP_LENGTH} frames")
print(f"  Crop Size: {CROP_SIZE}×{CROP_SIZE}")
print(f"  Target FPS: {TARGET_FPS}")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Activities: {ACTIVITY_NAMES}")

# Initialize model
model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

print(f"✅ Model loaded from: {MODEL_PATH}")
print(f"   Trained for {checkpoint['epoch']} epochs")
print(f"   Best validation accuracy: {checkpoint['val_acc']:.2f}%")

# ============================
# Helper Functions
# ============================

# def extract_frames_from_video(video_path, target_fps=25):
#     """
#     Extract frames from video at target FPS.
    
#     PAPER: "all video clips are fixed at 25 FPS for training"
#     MATCHES PAPER: Resamples video to 25 FPS
    
#     Args:
#         video_path: Path to input video
#         target_fps: Target frame rate (PAPER: 25)
    
#     Returns:
#         frames: List of RGB frames as numpy arrays
#         original_fps: Original video FPS
#     """
#     cap = cv2.VideoCapture(str(video_path))
    
#     if not cap.isOpened():
#         raise ValueError(f"Cannot open video: {video_path}")
    
#     # Get video properties
#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     print(f"\nVideo Properties:")
#     print(f"  Original FPS: {original_fps:.2f}")
#     print(f"  Total frames: {total_frames}")
#     print(f"  Duration: {total_frames/original_fps:.2f}s")
    
#     # Calculate frame sampling rate
#     # PAPER: Fixed at 25 FPS
#     frame_skip = max(1, int(original_fps / target_fps))
    
#     print(f"  Resampling to {target_fps} FPS (skip every {frame_skip} frames)")
    
#     frames = []
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Sample frames to match target FPS
#         if frame_count % frame_skip == 0:
#             # Convert BGR to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
        
#         frame_count += 1
    
#     cap.release()
    
#     print(f"  Extracted {len(frames)} frames at ~{target_fps} FPS")
    
#     return frames, original_fps

def extract_frames_generator(video_path, target_fps=25):
    """
    Generator that yields frames from video at target FPS.
    Memory-efficient: does NOT store all frames in RAM.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(original_fps / target_fps))
    frame_count = 0

    print(f"\nVideo Properties:")
    print(f"  Original FPS: {original_fps:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            # Yield frame as RGB
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1

    cap.release()


def preprocess_clip(frames, crop_size=112):
    """
    Preprocess a clip of frames for model input.
    
    PAPER: 16 × 112 × 112 frames, normalized
    MATCHES PAPER: Same preprocessing as training
    
    Args:
        frames: List of numpy arrays (H, W, 3) in RGB
        crop_size: Target spatial size (PAPER: 112)
    
    Returns:
        clip_tensor: (1, C, L, H, W) tensor ready for model
    """
    processed_frames = []
    
    for frame in frames:
        # Resize with center crop (same as validation)
        # MATCHES TRAINING: Resize to 128, center crop to 112
        frame_resized = cv2.resize(frame, (128, 128))
        frame_cropped = frame_resized[8:120, 8:120]  # Center crop
        
        # Normalize to [0, 1]
        frame_norm = frame_cropped.astype(np.float32) / 255.0
        
        processed_frames.append(frame_norm)
    
    # Stack into array (L, H, W, C)
    frames_np = np.array(processed_frames)
    
    # Normalize using same stats as training
    # PAPER: Doesn't specify, using Kinetics-400 stats (same as training)
    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
    frames_np = (frames_np - mean) / std
    
    # Convert to tensor (C, L, H, W)
    clip_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()
    
    # Add batch dimension (1, C, L, H, W)
    clip_tensor = clip_tensor.unsqueeze(0)
    
    return clip_tensor


# def predict_frame_activities(frames, model, clip_length=16, crop_size=112, device='cuda'):
#     """
#     Predict activity for each frame in the video.
    
#     PAPER: "convolutional kernels are applied on every 16 frames to extract 
#             the spatiotemporal information of the activity"
    
#     STRATEGY: Sliding window with stride=1 over all frames
#     JUSTIFICATION: Dense predictions for every frame
    
#     Args:
#         frames: List of video frames (RGB numpy arrays)
#         model: Trained 3D ResNet model
#         clip_length: Number of frames per clip (PAPER: 16)
#         crop_size: Spatial size (PAPER: 112)
#         device: Computation device
    
#     Returns:
#         predictions: List of predicted class indices (one per frame)
#         confidences: List of confidence scores (one per frame)
#     """
#     num_frames = len(frames)
#     predictions = []
#     confidences = []
    
#     print(f"\nPredicting activities for {num_frames} frames...")
    
#     # We need to predict for each frame
#     # STRATEGY: For each frame, create a clip centered on that frame
#     for i in range(num_frames):
#         # Get clip centered on frame i
#         # PAPER: Uses 16 frames per clip
        
#         # Calculate start and end indices for clip
#         half_clip = clip_length // 2
#         start_idx = max(0, i - half_clip)
#         end_idx = start_idx + clip_length
        
#         # Handle edge cases
#         if end_idx > num_frames:
#             end_idx = num_frames
#             start_idx = max(0, end_idx - clip_length)
        
#         # Extract clip
#         clip_frames = frames[start_idx:end_idx]
        
#         # Pad if necessary (for frames near start/end)
#         while len(clip_frames) < clip_length:
#             if start_idx == 0:
#                 # Pad at beginning with first frame
#                 clip_frames.insert(0, frames[0])
#             else:
#                 # Pad at end with last frame
#                 clip_frames.append(frames[-1])
        
#         # Preprocess clip
#         clip_tensor = preprocess_clip(clip_frames, crop_size)
#         clip_tensor = clip_tensor.to(device)
        
#         # Predict
#         with torch.no_grad():
#             outputs = model(clip_tensor)
#             probabilities = torch.softmax(outputs, dim=1)
#             confidence, predicted = probabilities.max(1)
        
#         predictions.append(predicted.item())
#         confidences.append(confidence.item())
        
#         # Progress indicator
#         if (i + 1) % 100 == 0 or (i + 1) == num_frames:
#             print(f"  Processed {i+1}/{num_frames} frames", end='\r')
    
#     print()  # New line after progress
    
#     return predictions, confidences
def predict_frame_activities(frame_generator, model, clip_length=16, crop_size=112, device='cuda'):
    """
    Predict activity for each frame on-the-fly (memory-efficient)
    """
    from collections import deque
    clip_buffer = deque(maxlen=clip_length)
    predictions = []
    confidences = []

    # Pre-fill buffer with first frame repeatedly if needed
    for frame in frame_generator:
        clip_buffer.append(frame)
        if len(clip_buffer) < clip_length:
            continue  # wait until buffer is full

        # Preprocess clip
        clip_tensor = preprocess_clip(list(clip_buffer), crop_size)
        clip_tensor = clip_tensor.to(device)

        # Predict
        with torch.no_grad():
            outputs = model(clip_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)

        predictions.append(pred.item())
        confidences.append(conf.item())

        # Progress print
        if len(predictions) % 100 == 0:
            print(f"Processed {len(predictions)} frames", end='\r')

    print()
    return predictions, confidences


def apply_majority_voting(predictions, window_size=25):
    """
    Apply majority voting to smooth predictions.
    
    PAPER Section 4.4: "each frame is labeled to indicate the excavator 
    activity after correcting the recognition errors with majority voting"
    
    PAPER: "It was observed that each activity lasts at least 2 s during 
    the operation"
    
    CALCULATION: At 25 FPS, 2 seconds = 50 frames
    ASSUMPTION: Use window_size=25 frames (1 second) as reasonable default
    JUSTIFICATION: Balances smoothing with responsiveness
    
    Args:
        predictions: List of frame predictions (before smoothing)
        window_size: Size of voting window (default: 25 frames = 1s at 25fps)
    
    Returns:
        smoothed: List of smoothed predictions
    """
    print(f"\nApplying majority voting (window size: {window_size} frames)...")
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(predictions)):
        # Define window around current frame
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        
        # Get predictions in window
        window = predictions[start:end]
        
        # Find most common prediction (mode)
        # PAPER: "correcting the recognition errors with majority voting"
        # MATCHES PAPER: Takes majority vote within temporal window
        most_common = stats.mode(window, keepdims=True)[0][0]
        smoothed.append(most_common)
    
    # Calculate how many predictions changed
    changes = sum(1 for i in range(len(predictions)) if predictions[i] != smoothed[i])
    print(f"  Corrected {changes} frames ({100*changes/len(predictions):.1f}%)")
    
    return smoothed


def calculate_cycle_times(predictions, activity_names, fps=25):
    """
    Calculate excavator work cycle times.
    
    PAPER Section 4.5: "one excavator working cycle is broken down into 
    digging, swinging, and loading"
    
    PAPER: "The total time of one cycle is the difference between the 
    start times of two adjacent digging activities."
    
    MATCHES PAPER: Identifies cycles based on digging activity
    
    Args:
        predictions: List of frame predictions
        activity_names: List of activity class names
        fps: Frame rate (PAPER: 25)
    
    Returns:
        cycles: List of cycle information dictionaries
    """
    print(f"\nCalculating work cycles...")
    
    # Find 'digging' class index
    if 'digging' not in activity_names:
        print("  WARNING: 'digging' activity not found, cannot calculate cycles")
        return []
    
    digging_idx = activity_names.index('digging')
    
    # Find start of each digging activity
    # PAPER: "The total time of one cycle is the difference between the 
    #         start times of two adjacent digging activities"
    digging_starts = []
    
    for i in range(len(predictions)):
        # Detect transition into digging
        if predictions[i] == digging_idx:
            # Check if this is start of digging (previous frame was different)
            if i == 0 or predictions[i-1] != digging_idx:
                digging_starts.append(i)
    
    print(f"  Found {len(digging_starts)} digging events")
    
    # Calculate cycles
    cycles = []
    
    for idx in range(len(digging_starts) - 1):
        start_frame = digging_starts[idx]
        end_frame = digging_starts[idx + 1]
        
        # Duration in frames
        duration_frames = end_frame - start_frame
        
        # Duration in seconds
        # PAPER: Calculates time as frames / FPS
        duration_seconds = duration_frames / fps
        
        # Count activities in this cycle
        cycle_predictions = predictions[start_frame:end_frame]
        activity_counts = defaultdict(int)
        
        for pred in cycle_predictions:
            activity_counts[activity_names[pred]] += 1
        
        # Calculate percentage time in each activity
        activity_percentages = {
            activity: (count / len(cycle_predictions)) * 100
            for activity, count in activity_counts.items()
        }
        
        cycles.append({
            'cycle_number': idx + 1,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'duration_frames': duration_frames,
            'duration_seconds': duration_seconds,
            'activity_counts': dict(activity_counts),
            'activity_percentages': activity_percentages
        })
    
    # PAPER: Reports cycle times
    if cycles:
        avg_cycle_time = np.mean([c['duration_seconds'] for c in cycles])
        print(f"  Average cycle time: {avg_cycle_time:.2f} seconds")
        print(f"  Cycles per hour: {3600 / avg_cycle_time:.1f}")
    
    return cycles


def calculate_productivity(cycles, bucket_payload_lcy=1.5):
    """
    Calculate excavator productivity.
    
    PAPER Equation (3):
    Productivity (LCY/hr) = Cycles/hr × Average bucket payload (LCY/Cycle)
    
    MATCHES PAPER: Uses cycle count and bucket payload
    
    Args:
        cycles: List of cycle information from calculate_cycle_times()
        bucket_payload_lcy: Bucket capacity in loose cubic yards (default: 1.5)
    
    Returns:
        productivity: Productivity in LCY/hr
    """
    if not cycles:
        print("\nCannot calculate productivity: No complete cycles detected")
        return 0.0
    
    # Total time in hours
    total_duration_seconds = sum(c['duration_seconds'] for c in cycles)
    total_hours = total_duration_seconds / 3600.0
    
    # Cycles per hour
    cycles_per_hour = len(cycles) / total_hours
    
    # PAPER Equation (3): Productivity = Cycles/hr × Bucket payload
    productivity = cycles_per_hour * bucket_payload_lcy
    
    print(f"\nProductivity Calculation:")
    print(f"  Total cycles: {len(cycles)}")
    print(f"  Total time: {total_duration_seconds/60:.1f} minutes")
    print(f"  Cycles per hour: {cycles_per_hour:.2f}")
    print(f"  Bucket payload: {bucket_payload_lcy:.2f} LCY")
    print(f"  Productivity: {productivity:.2f} LCY/hr")
    
    return productivity


def visualize_predictions(predictions, activity_names, fps=25, save_path=None):
    """
    Visualize activity predictions over time.
    
    Creates a timeline plot showing predicted activities.
    
    Args:
        predictions: List of frame predictions
        activity_names: List of activity class names
        fps: Frame rate
        save_path: Path to save visualization (optional)
    """
    # Create time axis in seconds
    time_seconds = np.arange(len(predictions)) / fps
    
    # Create plot
    plt.figure(figsize=(15, 6))
    
    # Plot predictions
    plt.plot(time_seconds, predictions, linewidth=0.5, alpha=0.7)
    
    # Add color bands for each activity
    colors = plt.cm.Set3(np.linspace(0, 1, len(activity_names)))
    
    for i, activity in enumerate(activity_names):
        plt.axhline(y=i, color=colors[i], linestyle='--', alpha=0.3, linewidth=0.5)
        plt.text(0, i, f' {activity}', verticalalignment='center', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.5))
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Activity', fontsize=12)
    plt.title('Excavator Activity Recognition Over Time', fontsize=14, fontweight='bold')
    plt.yticks(range(len(activity_names)), activity_names)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved to: {save_path}")
    
    return plt


def save_results(predictions, confidences, cycles, productivity, 
                activity_names, fps, save_dir):
    """
    Save all results to files.
    
    Saves:
    1. Frame-by-frame predictions (CSV)
    2. Cycle information (JSON)
    3. Summary statistics (JSON)
    
    Args:
        predictions: List of frame predictions
        confidences: List of prediction confidences
        cycles: List of cycle information
        productivity: Calculated productivity
        activity_names: List of activity names
        fps: Frame rate
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 1. Frame-by-frame predictions
    predictions_csv = save_dir / "frame_predictions.csv"
    with open(predictions_csv, 'w') as f:
        f.write("Frame,Time_s,Activity,Confidence\n")
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            time_s = i / fps
            activity = activity_names[pred]
            f.write(f"{i},{time_s:.3f},{activity},{conf:.4f}\n")
    
    print(f"\n✅ Frame predictions saved to: {predictions_csv}")
    
    # 2. Cycle information
    cycles_json = save_dir / "cycles.json"
    with open(cycles_json, 'w') as f:
        json.dump(cycles, f, indent=2)
    
    print(f"✅ Cycle information saved to: {cycles_json}")
    
    # 3. Summary statistics
    # Calculate activity distribution
    activity_counts = defaultdict(int)
    for pred in predictions:
        activity_counts[activity_names[pred]] += 1
    
    activity_percentages = {
        activity: (count / len(predictions)) * 100
        for activity, count in activity_counts.items()
    }
    
    summary = {
        'total_frames': len(predictions),
        'duration_seconds': len(predictions) / fps,
        'fps': fps,
        'activity_distribution': activity_percentages,
        'num_cycles': len(cycles),
        'avg_cycle_time_seconds': np.mean([c['duration_seconds'] for c in cycles]) if cycles else 0,
        'productivity_lcy_per_hour': productivity,
        'avg_confidence': np.mean(confidences)
    }
    
    summary_json = save_dir / "summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved to: {summary_json}")
    
    # Print summary to console
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Duration: {summary['duration_seconds']/60:.2f} minutes")
    print(f"\nActivity Distribution:")
    for activity, percentage in sorted(activity_percentages.items(), 
                                      key=lambda x: x[1], reverse=True):
        print(f"  {activity:12s}: {percentage:6.2f}%")
    print(f"\nWork Cycles: {len(cycles)}")
    if cycles:
        print(f"Average Cycle Time: {summary['avg_cycle_time_seconds']:.2f}s")
        print(f"Productivity: {productivity:.2f} LCY/hr")
    print(f"\nAverage Confidence: {summary['avg_confidence']*100:.2f}%")
    print(f"{'='*60}")


# ============================
# Main Inference Pipeline
# ============================
def main(video_path, model_path, output_dir, 
         majority_voting_window=25, bucket_payload=1.5):
    """
    Complete inference pipeline.
    
    PAPER WORKFLOW (Section 4.4, 4.5, Figure 7):
    1. Extract frames from video
    2. Predict activity for each frame
    3. Apply majority voting for smoothing
    4. Calculate cycle times
    5. Calculate productivity
    
    Args:
        video_path: Path to input video
        model_path: Path to trained model
        output_dir: Directory for saving results
        majority_voting_window: Window size for majority voting (frames)
        bucket_payload: Excavator bucket capacity (LCY)
    """
    print("="*60)
    print("EXCAVATOR ACTIVITY RECOGNITION - INFERENCE")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # STEP 1: Extract frames
    # PAPER: "all video clips are fixed at 25 FPS"
    # frames, original_fps = extract_frames_from_video(video_path, TARGET_FPS)
    # frame_generator = extract_frames_generator(video_path, TARGET_FPS)
    frame_generator = extract_frames_generator(video_path, TARGET_FPS)
    
    # STEP 2: Predict activities
    # PAPER: "convolutional kernels are applied on every 16 frames"
    # raw_predictions, confidences = predict_frame_activities(
    #     frames, model, CLIP_LENGTH, CROP_SIZE, DEVICE
    # )
    raw_predictions, confidences = predict_frame_activities(
    frame_generator, model, CLIP_LENGTH, CROP_SIZE, DEVICE
    )
    # STEP 3: Apply majority voting
    # PAPER: "correcting the recognition errors with majority voting"
    smoothed_predictions = apply_majority_voting(
        raw_predictions, majority_voting_window
    )
    
    # STEP 4: Calculate cycles
    # PAPER Figure 7: "The total time of one cycle is the difference 
    #                  between the start times of two adjacent digging activities"
    cycles = calculate_cycle_times(smoothed_predictions, ACTIVITY_NAMES, TARGET_FPS)
    
    # STEP 5: Calculate productivity
    # PAPER Equation (3): Productivity = Cycles/hr × Bucket payload
    productivity = calculate_productivity(cycles, bucket_payload)
    
    # STEP 6: Visualize and save results
    viz_path = output_dir / "activity_timeline.png"
    visualize_predictions(smoothed_predictions, ACTIVITY_NAMES, TARGET_FPS, viz_path)
    
    save_results(smoothed_predictions, confidences, cycles, productivity,
                ACTIVITY_NAMES, TARGET_FPS, output_dir)
    
    print(f"\n{'='*60}")
    print("✅ INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    
    return smoothed_predictions, cycles, productivity


# ============================
# Run Inference
# ============================
if __name__ == "__main__":
    # UPDATE THESE PATHS
    VIDEO_PATH = Path(r"C:\Users\shubh\Desktop\DELETE AFTER USE\delete.mp4")
    
    if not VIDEO_PATH.exists():
        print(f"\n❌ ERROR: Video not found at {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the script")
    else:
        # Run inference
        predictions, cycles, productivity = main(
            video_path=VIDEO_PATH,
            model_path=MODEL_PATH,
            output_dir=OUTPUT_DIR,
            majority_voting_window=25,  # 1 second at 25 FPS
            bucket_payload=1.5          # LCY - adjust for your excavator
        )
        
        # PAPER Section 5.1: "it is worth noting that the model was also 
        #                     applied on a 60.2 min video to recognize 
        #                     excavator's activities in the implementation 
        #                     stage and achieved the accuracy of 92.5%"
        
        print("\nPAPER COMPARISON:")
        print("Paper tested on 60.2 min video and achieved 92.5% accuracy")
        print("Your results are saved above for comparison")
