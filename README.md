# Excavator Activity Recognition System

A comprehensive computer vision pipeline for detecting excavators in construction site videos and recognizing their activities using deep learning models.

## 📋 Overview

This system processes construction site video footage to:
1. Detect excavators using custom-trained YOLOv8
2. Track excavators across frames using DeepSORT
3. Detect idling behavior through motion analysis
4. Recognize activities (Digging, Loading, Swinging, Dumping, Travelling, Idling) using 3D ResNet
5. Generate annotated videos and comprehensive activity reports

## 🎯 Key Features

- **Custom Excavator Detection**: Fine-tuned YOLOv8 model for accurate excavator detection
- **Multi-Object Tracking**: DeepSORT-based tracking for consistent excavator identification
- **Idling Detection**: Signal processing-based algorithm for automatic idling detection
- **Activity Recognition**: 3D ResNet-50 for temporal activity classification
- **CVAT Integration**: Export/import annotations for manual labeling and validation
- **Comprehensive Reports**: CSV outputs with frame-level and segment-level activity data

## 🏗️ System Architecture

```
Video Input
    ↓
[1] YOLOv8 Detection → CSV detections
    ↓
[2] DeepSORT Tracking → Track IDs + Cropped frames
    ↓
[3] Idling Detection → Idling segments (optional)
    ↓
[4] Manual Labeling via CVAT (optional)
    ↓
[5] Dataset Generation → Frame clips organized by activity
    ↓
[6] 3D ResNet Training → Activity recognition model
    ↓
[7] Activity Recognition → Final predictions + Timeline
    ↓
Output: Annotated Videos + Activity Reports
```

## 📁 Project Structure

```
excavator-activity-recognition/
├── 1_train_yolo.py              # Train custom YOLOv8 model
├── 2_detect_yolo.py             # Run detection and save to CSV
├── 3_track_deepsort.py          # Track excavators and extract crops
├── 4_detect_idling.py           # Detect idling behavior
├── 5_cvat_excel_to_xml.py       # Convert manual labels to CVAT format
├── 6_generate_dataset.py        # Create training dataset from annotations
├── 7_train_resnet.py            # Train 3D ResNet activity classifier
├── 8_recognize_activities.py    # Run activity recognition on tracks
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install deep-sort-realtime
pip install scipy pandas matplotlib
pip install lxml openpyxl
pip install tqdm
```

### Installation

```bash
git clone https://github.com/yourusername/excavator-activity-recognition.git
cd excavator-activity-recognition
pip install -r requirements.txt
```

## 📖 Usage Guide

### Step 1: Train Custom YOLOv8 Model

```python
# 1_train_yolo.py
python 1_train_yolo.py
```

**Configuration:**
- Place your annotated dataset in YOLOv8 format
- Update `data_yaml` path to your `data.yaml` file
- Adjust epochs, batch size, and image size as needed

**Output:**
- `best.pt` - Trained model weights
- Training metrics and validation results

---

### Step 2: Detect Excavators

```python
# 2_detect_yolo.py
python 2_detect_yolo.py
```

**Configuration:**
```python
VIDEO_PATH = "path/to/your/video.mp4"
MODEL_PATH = "path/to/best.pt"
CSV_PATH = "output/detections.csv"
```

**Output:**
- `detections.csv` - Frame-by-frame bounding box coordinates
- Annotated video with detection boxes

---

### Step 3: Track Excavators

```python
# 3_track_deepsort.py
python 3_track_deepsort.py
```

**Configuration:**
```python
VIDEO_PATH = "path/to/video.mp4"
CSV_PATH = "detections.csv"
SAVE_DIR = "output/tracks"
CLIP_LENGTH = 16  # frames per clip for ResNet
```

**Output:**
- Individual folders per track: `track_1/`, `track_2/`, etc.
- `track_metadata.csv` - Track statistics
- `Track_Output.csv` - Frame-level tracking data
- Annotated video with track IDs

---

### Step 4: Detect Idling (Optional)

```python
# 4_detect_idling.py
python 4_detect_idling.py
```

**Configuration:**
```python
TRACK_CSV = "Track_Output.csv"
FPS = 59
WINDOW = 40  # detection window size
DIST_THRESHOLD = 0.2  # movement threshold
```

**Output:**
- `Idling_segments.csv` - Detected idling periods
- Visualization plots for each track

---

### Step 5: Manual Labeling (Optional)

If you want to manually label activities:

1. **Label your video using Excel/Sheets:**
   ```
   Time        | Activity
   ------------|----------
   00:00-00:15 | Digging
   00:15-00:30 | Swinging
   ```

2. **Convert to CVAT XML:**
   ```python
   python 5_cvat_excel_to_xml.py
   ```

3. **Import XML to CVAT for review/editing**

---

### Step 6: Generate Training Dataset

```python
# 6_generate_dataset.py
python 6_generate_dataset.py
```

**Configuration:**
```python
VIDEO_PATH = "path/to/video.mp4"
CVAT_XML = "annotations.xml"
YOLO_MODEL = "best.pt"
OUTPUT_DIR = "dataset/"
CLIP_LENGTH = 16
STRIDE = 8  # 50% overlap
```

**Output:**
```
dataset/
├── digging/
│   ├── clip_00000/
│   │   ├── frame_000.jpg
│   │   ├── frame_001.jpg
│   │   └── ...
│   └── clip_00001/
├── loading/
├── swinging/
└── ...
```

---

### Step 7: Train Activity Recognition Model

```python
# 7_train_resnet.py
python 7_train_resnet.py
```

**Configuration:**
```python
DATASET_DIR = "dataset/"
NUM_CLASSES = 6
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
```

**Activities:**
- Digging
- Loading
- Swinging
- Dumping
- Travelling
- Idling

**Output:**
- `best_activity_model.pth` - Trained model
- Training loss and accuracy metrics

---

### Step 8: Run Activity Recognition

```python
# 8_recognize_activities.py
python 8_recognize_activities.py
```

**Configuration:**
```python
SAVE_DIR = "tracks/"
MODEL_PATH = "best_activity_model.pth"
FPS = 59
CLIP_LENGTH = 16
STRIDE = 16
VOTING_SECONDS = 2.0  # temporal smoothing
```

**Output:**
- `Activity_Output.csv` - Frame-by-frame predictions
- `Activity_Visual.csv` - Activity segments timeline

---

## 📊 Output Files

### Detection CSV (`detections.csv`)
```csv
frame,x1,y1,x2,y2,confidence,class
1,245,180,567,432,0.95,0
2,248,182,570,435,0.94,0
```

### Tracking CSV (`Track_Output.csv`)
```csv
frame,track_id,x1,y1,x2,y2
1,1,245,180,567,432
2,1,248,182,570,435
```

### Idling Segments (`Idling_segments.csv`)
```csv
track_id,start_frame,end_frame,duration_sec,start_time_hms,end_time_hms
1,100,350,4.237,00:00:01.695,00:00:05.932
```

### Activity Output (`Activity_Output.csv`)
```csv
track_id,frame,activity_label,activity_name,raw_prediction
1,1,0,digging,0
1,2,0,digging,0
```

### Activity Timeline (`Activity_Visual.csv`)
```csv
track_id,activity,start_frame,end_frame,duration_sec,start_time_sec,end_time_sec
1,digging,1,150,2.54,0.02,2.56
1,swinging,151,300,2.54,2.56,5.10
```

## ⚙️ Configuration Parameters

### Detection & Tracking
- `FPS`: Video frame rate (default: 59)
- `max_age`: DeepSORT max frames to keep track (default: 35)
- `n_init`: Frames before track confirmation (default: 4)

### Idling Detection
- `WINDOW`: Sliding window size in frames (default: 40)
- `DIST_THRESHOLD`: Movement threshold in pixels (default: 0.2)
- `AREA_THRESHOLD_PERCENT`: Bounding box area variation (default: 0.5%)
- `MIN_IDLE_DURATION_SEC`: Minimum idling duration (default: 3.0s)

### Activity Recognition
- `CLIP_LENGTH`: Frames per clip (default: 16)
- `STRIDE`: Frames to skip between clips (default: 8-16)
- `CROP_SIZE`: Spatial input size (default: 112×112)
- `VOTING_SECONDS`: Temporal smoothing window (default: 2.0s)

## 🎓 Model Architecture

### YOLOv8n
- Pretrained on COCO, fine-tuned for excavators
- Input: 480×480
- Output: Bounding boxes + confidence scores

### 3D ResNet-50
- Architecture: 3D convolutional backbone
- Input: (3, 16, 112, 112) - RGB clips
- Output: 6-class activity predictions
- Temporal receptive field: ~0.27 seconds @ 59 FPS

## 📈 Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for faster processing
2. **Batch Processing**: Increase batch size if GPU memory allows
3. **Frame Skipping**: Process every Nth frame for faster inference
4. **Clip Overlap**: Use smaller stride (e.g., 4-8) for better temporal coverage
5. **Temporal Smoothing**: Adjust `VOTING_SECONDS` to reduce prediction jitter

## 🐛 Troubleshooting

### Low Detection Accuracy
- Train YOLOv8 with more diverse excavator images
- Adjust confidence threshold (default: 0.5)
- Use larger input size (e.g., 640×640)

### Track ID Switching
- Increase DeepSORT `max_age` parameter
- Reduce occlusions in video footage
- Use higher quality object crops

### Poor Activity Recognition
- Ensure dataset has balanced class distribution
- Increase `CLIP_LENGTH` for longer temporal context
- Add data augmentation during training
- Use deeper model (ResNet-101, ResNet-152)

## 📝 Citation

If you use this work, please cite:

```bibtex
@software{excavator_activity_recognition,
  author = {Your Name},
  title = {Excavator Activity Recognition System},
  year = {2025},
  url = {https://github.com/yourusername/excavator-activity-recognition}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or support, please open an issue or contact [your-email@example.com](mailto:your-email@example.com)

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- DeepSORT implementation by nwojke
- 3D ResNet architecture inspired by "Learning Spatiotemporal Features with 3D Convolutional Networks"
- CVAT annotation tool

---

**Note**: Update all file paths in the scripts to match your directory structure before running.
