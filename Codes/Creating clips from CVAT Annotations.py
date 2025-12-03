import os
import cv2
import csv
from pathlib import Path
from lxml import etree
from ultralytics import YOLO
from tqdm import tqdm

# ============================
# Configuration
# ============================

# Paths
VIDEO_PATH = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Day_2.mp4"
CVAT_XML = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\output_cvat.xml"
YOLO_MODEL = r"C:\Users\shubh\Desktop\Research work with AVIK\Test\yolo_excavator_custom\weights\best.pt"
OUTPUT_DIR = r"C:\Users\shubh\Desktop\Research work with AVIK\Test\dataset"

# ResNet parameters
CLIP_LENGTH = 16  # number of frames per clip
STRIDE = 8  # how many frames to skip between clips (8 = 50% overlap)
MIN_CONFIDENCE = 0.5  # minimum YOLO confidence to accept detection
CROP_SIZE = 112  # resize cropped images to 112x112 (matches ResNet input)

# ============================
# Parse CVAT XML to get frame->activity mapping
# ============================

def parse_cvat_xml(xml_path):
    """
    Returns: dict mapping frame_number -> activity_label
    """
    print("Parsing CVAT XML...")
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    frame_labels = {}
    
    # Parse tracks
    for track in root.findall(".//track"):
        label = track.get("label")
        
        # Get all boxes in this track
        boxes = track.findall("box")
        if len(boxes) < 2:
            continue
            
        # Find start and end frames
        start_frame = None
        end_frame = None
        
        for box in boxes:
            frame_num = int(box.get("frame"))
            outside = int(box.get("outside", "0"))
            
            if outside == 0:  # track is active
                if start_frame is None:
                    start_frame = frame_num
            else:  # track ends
                end_frame = frame_num
                break
        
        # If no explicit end, use last box
        if end_frame is None and len(boxes) > 0:
            end_frame = int(boxes[-1].get("frame"))
        
        # Fill in all frames in this range
        if start_frame is not None and end_frame is not None:
            for frame in range(start_frame, end_frame):
                frame_labels[frame] = label
    
    print(f"  Found labels for {len(frame_labels)} frames")
    
    # Print activity distribution
    from collections import Counter
    activity_counts = Counter(frame_labels.values())
    print("  Activity distribution:")
    for activity, count in sorted(activity_counts.items()):
        print(f"    {activity}: {count} frames")
    
    return frame_labels

# ============================
# Run YOLO and extract crops
# ============================

def extract_clips_with_yolo(video_path, yolo_model_path, frame_labels, output_dir, 
                            clip_length=16, stride=8, min_conf=0.5, crop_size=112):
    """
    Process video with YOLO, crop excavator, organize by activity into clips
    """
    
    # Load YOLO model
    print("\nLoading YOLO model...")
    model = YOLO(yolo_model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    
    # Create output directories for each activity
    activities = set(frame_labels.values())
    for activity in activities:
        os.makedirs(os.path.join(output_dir, activity.lower()), exist_ok=True)
    
    print(f"Found activities: {sorted(activities)}")
    
    # Store frames temporarily for creating clips
    frame_buffer = []  # stores (cropped_frame, activity, frame_num)
    clip_counter = {}  # counter per activity
    
    frame_num = 0
    skipped_no_label = 0
    skipped_no_detection = 0
    skipped_low_conf = 0
    
    print("\nProcessing video...")
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get activity label for this frame
            activity = frame_labels.get(frame_num)
            
            if activity is None:
                skipped_no_label += 1
                frame_num += 1
                pbar.update(1)
                continue
            
            # Run YOLO detection
            results = model(frame, imgsz=480, verbose=False)
            
            if len(results[0].boxes) == 0:
                skipped_no_detection += 1
                frame_num += 1
                pbar.update(1)
                continue
            
            # Get best detection
            best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            conf = float(best_box.conf[0])
            
            if conf < min_conf:
                skipped_low_conf += 1
                frame_num += 1
                pbar.update(1)
                continue
            
            # Crop excavator region
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            
            # Ensure valid crop coordinates
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                skipped_no_detection += 1
                frame_num += 1
                pbar.update(1)
                continue
            
            cropped = frame[y1:y2, x1:x2]
            
            # Resize to standard size
            if cropped.size > 0:
                cropped = cv2.resize(cropped, (crop_size, crop_size))
                
                # Add to buffer
                frame_buffer.append((cropped, activity, frame_num))
                
                # Try to create clips from buffer
                if len(frame_buffer) >= clip_length:
                    # Check if we have enough consecutive frames with same activity
                    for i in range(len(frame_buffer) - clip_length + 1):
                        clip_frames = frame_buffer[i:i+clip_length]
                        clip_activities = [a for _, a, _ in clip_frames]
                        
                        # All frames must have same activity
                        if len(set(clip_activities)) == 1:
                            activity_name = clip_activities[0].lower()
                            
                            if activity_name not in clip_counter:
                                clip_counter[activity_name] = 0
                            
                            # Create clip directory
                            clip_dir = os.path.join(output_dir, activity_name, 
                                                   f"clip_{clip_counter[activity_name]:05d}")
                            os.makedirs(clip_dir, exist_ok=True)
                            
                            # Save frames
                            for j, (crop_frame, _, _) in enumerate(clip_frames):
                                frame_path = os.path.join(clip_dir, f"frame_{j:03d}.jpg")
                                cv2.imwrite(frame_path, crop_frame)
                            
                            clip_counter[activity_name] += 1
                            
                            # Remove processed frames (move by stride)
                            break
                    
                    # Slide window by stride
                    if len(frame_buffer) >= clip_length + stride:
                        frame_buffer = frame_buffer[stride:]
            
            frame_num += 1
            pbar.update(1)
    
    cap.release()
    
    # Print statistics
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    
    print("\n📊 Frame Statistics:")
    print(f"  Total frames processed: {frame_num}")
    print(f"  Skipped (no label): {skipped_no_label}")
    print(f"  Skipped (no detection): {skipped_no_detection}")
    print(f"  Skipped (low confidence): {skipped_low_conf}")
    print(f"  Successfully processed: {frame_num - skipped_no_label - skipped_no_detection - skipped_low_conf}")
    
    print("\n📁 Dataset Statistics:")
    total_clips = 0
    for activity, count in sorted(clip_counter.items()):
        print(f"  {activity}: {count} clips ({count * clip_length} frames)")
        total_clips += count
    
    print(f"\n✅ Total clips created: {total_clips}")
    print(f"✅ Dataset saved to: {output_dir}")
    
    return clip_counter

# ============================
# Main execution
# ============================

if __name__ == "__main__":
    print("="*50)
    print("YOLO + CVAT → ResNet Dataset Generator")
    print("="*50)
    
    print("\nStep 1: Parsing CVAT annotations...")
    frame_labels = parse_cvat_xml(CVAT_XML)
    
    if len(frame_labels) == 0:
        print("❌ ERROR: No labels found in CVAT XML!")
        exit(1)
    
    print("\nStep 2: Extracting clips with YOLO...")
    clip_stats = extract_clips_with_yolo(
        VIDEO_PATH, 
        YOLO_MODEL, 
        frame_labels,
        OUTPUT_DIR,
        clip_length=CLIP_LENGTH,
        stride=STRIDE,
        min_confidence=MIN_CONFIDENCE,
        crop_size=CROP_SIZE
    )
    
    if sum(clip_stats.values()) == 0:
        print("\n⚠️  WARNING: No clips were created!")
        print("Check if:")
        print("  1. YOLO model is detecting excavators")
        print("  2. CVAT labels match video frames")
        print("  3. Confidence threshold is not too high")
    else:
        print("\n" + "="*50)
        print("✅ SUCCESS! Dataset is ready for ResNet training")
        print("="*50)
        print("\nNext steps:")
        print("1. Review the generated dataset")
        print("2. Update ACTIVITY_NAMES in your ResNet training script")
        print("3. Run the ResNet training script")
        print(f"4. Your model will be saved as: best_activity_model.pth")
import os
import cv2
import csv
from pathlib import Path
from lxml import etree
from ultralytics import YOLO
from tqdm import tqdm

# ============================
# Configuration
# ============================

# Paths
VIDEO_PATH = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Day_1.mp4"
CVAT_XML = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\cvat_annotations.xml"
YOLO_MODEL = r"C:\Users\shubh\Desktop\Research work with AVIK\Test\yolo_excavator_custom\weights\best.pt"
OUTPUT_DIR = r"C:\Users\shubh\Desktop\Research work with AVIK\Test\dataset"

# ResNet parameters
CLIP_LENGTH = 16  # number of frames per clip
STRIDE = 8  # how many frames to skip between clips (lower = more clips)
MIN_CONFIDENCE = 0.5  # minimum YOLO confidence to accept detection

# ============================
# Parse CVAT XML to get frame->activity mapping
# ============================

def parse_cvat_xml(xml_path):
    """
    Returns: dict mapping frame_number -> activity_label
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    frame_labels = {}
    
    # Parse tracks
    for track in root.findall(".//track"):
        label = track.get("label")
        
        # Get all boxes in this track
        boxes = track.findall("box")
        if len(boxes) < 2:
            continue
            
        # Find start and end frames
        start_frame = None
        end_frame = None
        
        for box in boxes:
            frame_num = int(box.get("frame"))
            outside = int(box.get("outside", "0"))
            
            if outside == 0:  # track is active
                if start_frame is None:
                    start_frame = frame_num
            else:  # track ends
                end_frame = frame_num
                break
        
        # If no explicit end, use last box
        if end_frame is None and len(boxes) > 0:
            end_frame = int(boxes[-1].get("frame"))
        
        # Fill in all frames in this range
        if start_frame is not None and end_frame is not None:
            for frame in range(start_frame, end_frame):
                frame_labels[frame] = label
    
    return frame_labels

# ============================
# Run YOLO and extract crops
# ============================

def extract_clips_with_yolo(video_path, yolo_model_path, frame_labels, output_dir, 
                            clip_length=16, stride=8, min_conf=0.5):
    """
    Process video with YOLO, crop excavator, organize by activity into clips
    """
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(yolo_model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directories for each activity
    activities = set(frame_labels.values())
    for activity in activities:
        os.makedirs(os.path.join(output_dir, activity.lower()), exist_ok=True)
    
    print(f"Processing {total_frames} frames...")
    print(f"Found activities: {activities}")
    
    # Store frames temporarily for creating clips
    current_clip_frames = []
    current_activity = None
    clip_counter = {}  # counter per activity
    
    frame_num = 0
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get activity label for this frame
            activity = frame_labels.get(frame_num)
            
            if activity is None:
                frame_num += 1
                pbar.update(1)
                continue
            
            # Run YOLO detection
            results = model(frame, imgsz=480, verbose=False)
            
            if len(results[0].boxes) > 0:
                # Get best detection
                best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                conf = float(best_box.conf[0])
                
                if conf >= min_conf:
                    # Crop excavator region
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                    cropped = frame[y1:y2, x1:x2]
                    
                    # Resize to standard size
                    if cropped.size > 0:
                        cropped = cv2.resize(cropped, (112, 112))
                        
                        # Add to current clip
                        current_clip_frames.append((cropped, activity))
                        
                        # If we have enough frames for a clip
                        if len(current_clip_frames) >= clip_length:
                            # Check if all frames in clip have same activity
                            clip_activities = [a for _, a in current_clip_frames[-clip_length:]]
                            if len(set(clip_activities)) == 1:  # all same activity
                                # Save this clip
                                activity_name = clip_activities[0].lower()
                                
                                if activity_name not in clip_counter:
                                    clip_counter[activity_name] = 0
                                
                                clip_dir = os.path.join(output_dir, activity_name, 
                                                       f"clip_{clip_counter[activity_name]:05d}")
                                os.makedirs(clip_dir, exist_ok=True)
                                
                                # Save frames
                                for i, (crop_frame, _) in enumerate(current_clip_frames[-clip_length:]):
                                    frame_path = os.path.join(clip_dir, f"frame_{i:03d}.jpg")
                                    cv2.imwrite(frame_path, crop_frame)
                                
                                clip_counter[activity_name] += 1
                                
                                # Slide window by stride
                                current_clip_frames = current_clip_frames[stride:]
            
            frame_num += 1
            pbar.update(1)
    
    cap.release()
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    for activity, count in sorted(clip_counter.items()):
        print(f"{activity}: {count} clips")
    
    total_clips = sum(clip_counter.values())
    print(f"\nTotal clips created: {total_clips}")
    
    return clip_counter

# ============================
# Main execution
# ============================

if __name__ == "__main__":
    print("Step 1: Parsing CVAT annotations...")
    frame_labels = parse_cvat_xml(CVAT_XML)
    print(f"Found labels for {len(frame_labels)} frames")
    
    print("\nStep 2: Extracting clips with YOLO...")
    clip_stats = extract_clips_with_yolo(
        VIDEO_PATH, 
        YOLO_MODEL, 
        frame_labels,
        OUTPUT_DIR,
        clip_length=CLIP_LENGTH,
        stride=STRIDE,
        min_confidence=MIN_CONFIDENCE
    )
    
    print("\n✅ Dataset creation complete!")
    print(f"Dataset saved to: {OUTPUT_DIR}")
    print("\nYou can now train your ResNet model using the existing training script.")