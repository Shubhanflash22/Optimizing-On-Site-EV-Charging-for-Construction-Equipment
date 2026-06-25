# 🚜 Excavator Detection Pipeline using YOLO & Roboflow

A complete end-to-end workflow to train a **custom excavator detector** using **raw construction site videos**, **Roboflow annotation**, and **YOLOv8** training + inference.

---

## 📌 Overview

This project builds an automated pipeline to:

1. **Extract frames** from long construction videos
2. **Annotate** the “Excavator” class using Roboflow
3. **Train a custom YOLO model**
4. **Run inference** to detect excavators frame-by-frame
5. **Export results** as annotated video + CSV bounding-box logs

---

## 🏗️ Project Pipeline

```
Raw Video (.mp4)
      ↓
Step 1 — Frame Extraction (1 frame every 60 sec)
      ↓
Step 2 — Roboflow Annotation & Dataset Versioning
      ↓
Step 3 — Custom YOLO Training (Ultralytics)
      ↓
Step 4 — Inference on Full Video (best.pt)
      ↓
CSV logs + Annotated (YOLO.mp4)
```

---

# 🔹 Step 1 — Extract Frames From Video

The raw construction videos were converted into **frames at 1 frame/minute**.

### Why?

* Reduces dataset size
* Still captures enough excavator positions across time
* Makes annotation manageable

### Key Features of the extraction script:

* Extracts frames only if the folder is empty
* Automatically names frames `frame_000000.jpg`, …
* Does optional OCR to extract timestamps (not essential for YOLO training)
* Saves frames into:

```
frames_<video_name>_<date>/
```

➡️ **Script referenced:** `Excavator Detection – Version 1 (Discarded).py`
(Contains full frame extraction + OCR code)

---

# 🔹 Step 2 — Annotate the Dataset Using Roboflow

All extracted frames were uploaded to **Roboflow**.

### Steps performed:

1. **Created a new project** → "Excavator Detection"
2. **Created a custom class** → `excavator`
3. Manually annotated bounding boxes around excavators
4. Performed a **Train/Valid/Test split** (default 70/20/10)
5. Generated a **dataset version**
6. Exported it as:

```
Format → YOLOv8
```

This includes:

```
data.yaml
train/images, train/labels
valid/images, valid/labels
test/images, test/labels
```

---

# 🔹 Step 3 — Train a Custom YOLOv8 Model

Training was done using **Ultralytics YOLOv8n** finetuned on the Roboflow dataset.

### Key Training Settings:

* Base model: `yolov8n.pt` (lightweight, fast)
* Epochs: `50`
* Image size: `480`
* Batch size: `16`
* GPU auto-detected
* Project folder created automatically:

```
…/Test/yolo_excavator_custom/
```

### Minimal training code:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=480,
    batch=16,
    name="yolo_excavator_custom",
    project="path/to/project"
)
```

### Output of training:

```
yolo_excavator_custom/
   └── weights/
         ├── best.pt   <-- Best model (used for inference)
         └── last.pt
```

---

# 🔹 Step 4 — Run Inference on Full Construction Video

The trained model (`best.pt`) was used to:

* Detect excavators **frame-by-frame**
* Save **bounding box coordinates** into a CSV file
* Create an **annotated output video**

### What the inference script does:

* Loads YOLO model
* Reads video frame-by-frame
* Runs detection with `model(frame, imgsz=480)`
* Selects the **highest-confidence** box
* Draws bounding box (green rectangle)
* Writes to:

  * Annotated video: `YOLO.mp4`
  * CSV log: `detections.csv`

### CSV fields:

```
frame, x1, y1, x2, y2, confidence, class
```

### Minimal inference code:

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
results = model(frame)

# draw and save detections...
```

---

# 📁 Project Structure

```
project/
│
├── frames_<video_name>_<date>/         # Extracted training frames
├── roboflow_dataset/                    # Exported YOLOv8 dataset
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
│
├── yolo_excavator_custom/               # Training output
│   └── weights/
│       ├── best.pt
│       └── last.pt
│
├── output/
│   ├── YOLO.mp4                         # Annotated video
│   └── detections.csv                   # Bounding-box logs
```

---

# 🛠️ Requirements

* Python 3.8+
* ultralytics
* opencv-python
* torch (with CUDA if available)
* roboflow (for uploads/automation, optional)
* easyocr (used only in frame extraction script)

Install:

```bash
pip install ultralytics opencv-python torch easyocr
```

---

# 📈 Future Improvements

* Add multiprocessing for faster inference
* Add DeepSORT tracking for multi-excavator tracking
* Train YOLOv8m or YOLO11 for stronger accuracy
* Train with multi-angle excavator images for robustness
* Automate frame extraction + upload directly to Roboflow API

---

# ✅ Conclusion

This workflow provides a **robust, reproducible pipeline** for training a custom excavator detector using YOLO and Roboflow — starting from raw construction footage all the way to structured CSV outputs and annotated videos.

Absolutely, hombre — here is a **plain-text version** of the README with **zero formatting**, perfect for pasting directly into Notepad.

---

Excavator Detection Pipeline using YOLO and Roboflow

Overview
This project builds an automated pipeline to:

1. Extract frames from long construction videos
2. Annotate the “Excavator” class using Roboflow
3. Train a custom YOLO model
4. Run inference to detect excavators frame by frame
5. Export results as an annotated video and a CSV file

Pipeline
Raw Video -> Frame Extraction -> Roboflow Annotation -> YOLO Training -> Inference -> CSV and Output Video

Step 1 - Extract Frames from Video
Frames were extracted from the raw construction video at a rate of one frame per minute to reduce dataset size.
Frames were saved into a dedicated folder for each video. This step made annotation easier and reduced redundancy.

Step 2 - Dataset Annotation using Roboflow
The extracted frames were uploaded to Roboflow.
A custom class named “excavator” was created.
Bounding boxes were manually drawn around excavators.
A train, validation, and test split was created.
Dataset was exported in YOLOv8 format, which included data.yaml, image folders, and label folders.

Step 3 - Custom YOLO Model Training
The YOLOv8n pretrained model was used as the base model.
Training was done for 50 epochs with image size 480 and batch size 16.
Training automatically saved the best model as best.pt inside the “weights” folder.

Step 4 - Running Inference on Full Video
The trained best.pt model was loaded.
Each frame of the video was passed into the YOLO model.
The highest confidence bounding box for an excavator was selected.
Coordinates were saved into a CSV file with the format:
frame, x1, y1, x2, y2, confidence, class
An annotated video was also generated with bounding boxes drawn around detected excavators.

Project Structure
frames_<video_name>_<date>/
roboflow_dataset/
yolo_excavator_custom/
output/
YOLO.mp4
detections.csv

Requirements
Python 3.8 or later
ultralytics
opencv-python
torch
roboflow (optional)
easyocr (optional)

Future Improvements
Add DeepSORT tracking for multi-excavator tracking
Train with a larger YOLO model for better accuracy
Automate the full pipeline from video to output
Improve timestamp extraction and sync with detections

Conclusion
This workflow creates a complete pipeline from raw construction videos to a trained excavator detection model, finishing with structured CSV logs and annotated videos using YOLO and Roboflow.

---

If you want a version with line breaks adjusted, indentation removed, or single-line spacing, just tell me.

