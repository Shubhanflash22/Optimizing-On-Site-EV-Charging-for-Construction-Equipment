# Excavator Activity Recognition System

A comprehensive, end-to-end computer vision pipeline for detecting excavators in construction site videos and recognizing their activities using deep learning. The system is designed to support **on-site productivity monitoring and EV charging optimization** for construction equipment by identifying when machines are working, idling, or travelling.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Getting Started](#5-getting-started)
6. [Detailed Pipeline: Step-by-Step](#6-detailed-pipeline-step-by-step)
   - [Step 0 (Discarded): OCR-Based Timestamp Detection](#step-0-discarded-ocr-based-timestamp-detection)
   - [Step 1: Custom YOLOv8 Model Training](#step-1-custom-yolov8-model-training)
   - [Step 2: Excavator Detection with YOLO](#step-2-excavator-detection-with-yolo)
   - [Step 3: Multi-Object Tracking with DeepSORT](#step-3-multi-object-tracking-with-deepsort)
   - [Step 4: Idling Detection](#step-4-idling-detection)
   - [Step 5 (Optional): Manual Labeling via Excel](#step-5-optional-manual-labeling-via-excel)
   - [Step 6: CVAT Annotation — Excel to XML Conversion](#step-6-cvat-annotation--excel-to-xml-conversion)
   - [Step 7: Dataset Generation from CVAT Annotations](#step-7-dataset-generation-from-cvat-annotations)
   - [Step 8: Custom 3D ResNet Model Training](#step-8-custom-3d-resnet-model-training)
   - [Step 9: Activity Recognition (Inference)](#step-9-activity-recognition-inference)
   - [Step 10: Validation & Hyperparameter Optimization](#step-10-validation--hyperparameter-optimization)
7. [Output Files](#7-output-files)
8. [Model Architecture](#8-model-architecture)
9. [Configuration Parameters](#9-configuration-parameters)
10. [Activity Classes](#10-activity-classes)
11. [Productivity & Cycle Time Calculation](#11-productivity--cycle-time-calculation)
12. [Performance Results](#12-performance-results)
13. [NRP / Kubernetes Cluster Training](#13-nrp--kubernetes-cluster-training)
14. [Performance Tips](#14-performance-tips)
15. [Troubleshooting](#15-troubleshooting)
16. [References & Related Papers](#16-references--related-papers)
17. [Citation](#17-citation)
18. [License](#18-license)

---

## 1. Project Overview

This system processes construction site video footage (originally collected across **12+ recording days** from October 2025 to February 2026) to deliver a full activity recognition and productivity reporting pipeline. The project was developed in collaboration with **Avik** at UC San Diego as part of research into optimizing on-site EV charging for construction equipment by understanding machine utilization patterns.

The pipeline:
1. **Detects** excavators in video using a custom-trained YOLOv8 model
2. **Tracks** them across frames using DeepSORT to maintain consistent identities
3. **Detects idling** via a physics-based signal processing algorithm
4. **Recognizes activities** (Digging, Loading, Swinging, Travelling, Idling) using a 3D ResNet (R3D-18) model fine-tuned from Kinetics-400 weights
5. **Calculates productivity** metrics including cycle times and LCY/hr output
6. **Validates** predictions against manually labelled ground truth with a grid-search optimization pipeline

---

## 2. Key Features

- **Custom Excavator Detection**: YOLOv8n fine-tuned on a Roboflow-annotated dataset (single class: `Excavator`) for accurate, single-frame detection at 480×480 resolution
- **Multi-Object Tracking**: DeepSORT-based tracking (`max_age=35`, `n_init=4`) for consistent excavator IDs across frames, with per-track cropped frame exports at 112×112
- **Physics-Based Idling Detection**: Signal processing algorithm using Savitzky–Golay filtering, rolling median smoothing, and a sliding window over bounding box centroid displacement and area variance
- **Temporal Activity Recognition**: PyTorch R3D-18 (3D CNN) fine-tuned from Kinetics-400, operating on 16-frame clips at 25 FPS, producing 5-class activity predictions
- **Hybrid AI + Physics Override**: During inference, the idling detection logic overrides AI predictions for frames where the physics model classifies the machine as stationary
- **Finite State Machine (FSM) Post-processing**: Enforces valid activity transition sequences and minimum dwell times to produce physically plausible output
- **Majority Voting Smoothing**: Temporal smoothing over a configurable window (default 2 seconds) to reduce per-frame jitter
- **CVAT Integration**: Tools to convert Excel-based manual labels to CVAT-compatible XML annotations, enabling labelling review and editing in the CVAT web tool
- **Multi-Video Dataset Builder**: Batch clips extracted from 12 videos simultaneously across multiple recording days, resampled to 25 FPS with YOLO crop on each frame
- **Comprehensive Validation**: Grid search across 6 hyperparameters (~9,000+ combinations) comparing predictions against ground truth Excel sheets
- **Productivity Reports**: Automatic cycle detection, cycle time statistics, and LCY/hr productivity estimates (Paper Eq. 3)

---

## 3. System Architecture

```
Raw Construction Video(s)
         │
         ▼
[1] YOLOv8 Detection
    ├── Custom model (best.pt) trained on Roboflow dataset
    ├── imgsz=480, highest-confidence box selected per frame
    └── Output: detections.csv (frame, x1, y1, x2, y2, conf, class)
         │
         ▼
[2] DeepSORT Tracking
    ├── max_age=35, n_init=4
    ├── Crops each confirmed track to 112×112 and saves to track_N/ folder
    └── Output: Track_Output.csv + track_metadata.csv + annotated video
         │
         ▼
[3] Idling Detection (Physics-Based)
    ├── Savitzky–Golay + rolling median smoothing of centroid & area signals
    ├── Sliding window (40 frames) checking std(dist) < 0.2 px & std(ΔArea)
    └── Output: Idling_segments.csv + per-track visualization plots
         │
         ▼
[4] Manual Labeling (Parallel Path)
    ├── Excel spreadsheets with time-range → activity labels per day
    ├── CVAT Excel→XML converter produces CVAT-importable annotation files
    └── CVAT used for label review, correction, and export
         │
         ▼
[5] Dataset Generation
    ├── Resample video to 25 FPS (round-nearest)
    ├── For each resampled frame, run YOLO and crop excavator to 112×112
    ├── Buffer 16-frame windows; emit clips with stride=3
    └── Output: Dataset_Resnet_3/ organized by activity/clip_NNNNN/
         │
         ▼
[6] 3D ResNet Training (R3D-18 / Kinetics-400)
    ├── Fine-tune layer4 + fc only (frozen backbone)
    ├── Mixed precision (AMP), WeightedRandomSampler for class balance
    ├── Augmentation: horizontal flip, channel shift, affine shear
    └── Output: resnet3d_best_kinetics_2.pth + training_history.json
         │
         ▼
[7] Inference + Hybrid Post-Processing
    ├── Resample video to TARGET_FPS (25)
    ├── YOLO crop → sliding 16-frame clip → R3D-18 → raw predictions
    ├── Physics idling override (travelling → idling where physics says idle)
    ├── Majority voting (2s window = 50 frames at 25 FPS)
    └── Optional: FSM cleaning (dwell enforcement + transition validation)
         │
         ▼
[8] Validation & Grid Search
    ├── Runs inference ONCE, caches raw predictions to Excel
    ├── Grid-searches 6 hyperparameters (~9,000+ combinations)
    └── Outputs: confusion matrix, timeline comparison, master CSV report
         │
         ▼
Output: Activity Timeline + Cycle Analysis + Productivity (LCY/hr)
```

---

## 4. Repository Structure

```
Optimizing-On-Site-EV-Charging-for-Construction-Equipment/
│
├── Codes/                                    ← All Python scripts (primary source)
│   ├── 1.Exacavator Detection - Version 1 (Discarded).py
│   ├── 2.Custom Yolo model training.py
│   ├── 3.Step 1 - YOLO.py
│   ├── 4.3d rcnn model.py
│   ├── 5.Step 2 - Deep Sort.py
│   ├── 6.Step 3 - Idling.py
│   ├── 7.CVAT (excel to xml) for one video.py
│   ├── 7.CVAT (excel to xml) for multiple videos.py
│   ├── 7a.Step 7a - Batch_cvat_pipeline.py
│   ├── 8.Creating clips from CVAT Annotations from one video.py
│   ├── 8.Creating clips from CVAT Annotations from multiple videos.py
│   ├── 9.Custom resnet model training.py
│   ├── 10.Step 4 - Resnet.py
│   ├── 11.Validation of Pipeline.py
│   └── Old Discarded codes/
│
├── 1.Exacavator Detection - Version 1 (Discarded)/
│   ├── frames_Day_2_2025-10-21/             ← Extracted frames (1/min) for OCR
│   ├── frames_Day_3_2025-10-22/
│   ├── frames_Day_4_2025-10-23/
│   ├── minute_by_minute_activities_all.xlsx
│   └── tesseract-ocr-w64-setup-5.5.0.20241111.exe
│
├── 2.Custom Yolo model training/
│   ├── data.yaml                            ← Roboflow dataset config (1 class: Excavator)
│   ├── yolov8n.pt                           ← Base model weights
│   ├── yolo_excavator_custom/               ← Training output (weights/best.pt)
│   ├── Readme for Training best.pt.txt
│   ├── README.dataset.txt
│   └── README.roboflow.txt
│
├── 3.Step 1 - YOLO/
│   ├── best.pt                              ← Final trained YOLO model
│   └── Test.csv                             ← Sample detection output CSV
│
├── 5.Step 2 - Deep Sort/
│   └── Track_Output.csv                     ← Tracking output with track IDs
│
├── 6.Step 3 - Idling/
│   ├── Idling_segments.csv
│   ├── Track_Output.csv
│   └── track_1.png                          ← Idling visualization plot
│
├── 7.CVAT (excel to xml)/
│   ├── Tasks.xlsx / Tasks_all.xlsx          ← Master label spreadsheets
│   ├── output_cvat.xml / output_cvat.json
│   ├── Tasks_Day_2_Oct_21/ … Tasks_Day_12_Feb_13/
│   ├── CEV-Analysis/
│   └── How to install and use CVAT.docx
│
├── 8.Creating clips from CVAT Annotations/
│   ├── annotations.xml                      ← CVAT export (40 MB+)
│   └── best.pt                              ← YOLO model copy for clip creation
│
├── 9.Custom resnet model training/
│   ├── Multi Layer model/
│   ├── dataset_mean.npy                     ← Per-channel mean for normalization
│   └── dataset_std.npy                      ← Per-channel std for normalization
│
├── 10.Step 4 - Resnet/
│   ├── frame_predictions.csv               ← Frame-level activity predictions
│   ├── activity_timeline.png               ← Visual timeline plot
│   ├── cycles.json                         ← Detected work cycles
│   ├── summary.json                        ← Aggregate stats (productivity, confidence)
│   └── training_history.json              ← Loss/accuracy curves per epoch
│
├── 11.Step 5 - Validation/
│   ├── confusion_matrix_Day2.png
│   ├── frame_comparison_Day2.csv           ← Frame-level GT vs prediction
│   ├── segment_report_Day2.csv
│   └── timeline_Day2.png
│
├── Run from step 8 to 11/
│   ├── V1 Full video/
│   └── V2 Validation only/
│
├── NRP Stuff/                               ← Kubernetes/Nautilus cluster configs
│   ├── Using NRP.txt
│   ├── Storage Instructions.txt
│   ├── NRP Setup Guide.docx
│   ├── resnet3d-train.yaml
│   ├── resnet3d-train_job.yaml
│   ├── shubhan-any-gpu.yaml
│   ├── nrp-dataset-pvc.yaml
│   ├── dataset-uploader.yaml
│   ├── pvc-inspector.yaml
│   ├── datascience_CV.yml
│   └── The Tmux stuff for remote desktop.pdf
│
├── Misc/
│   ├── Requirements.txt
│   ├── smartgrid.yml                        ← Conda environment spec
│   ├── Excavator_Pipeline_README_Complete.docx
│   ├── Excavator_Project_Overview.docx
│   └── Tasks_all.xlsx
│
├── Papers and references/                   ← Academic references
│   ├── Cho Latif Sharafat Seo.pdf          ← Primary paper (3D ResNet method)
│   ├── Chen Zhu Hammad.pdf
│   ├── Fard Heydarian Niebles.pdf
│   ├── Ghelmani Torabi Hammad Chen.pdf
│   ├── Hong Song Hong Kim Jeong.pdf
│   ├── Akın Muhammed Ali Beyazıt Jan Kleissl Yuanyuan Shi.pdf
│   └── References.txt
│
├── Update PPTs/
├── LICENSE
└── README.md
```

---

## 5. Getting Started

### Prerequisites

- Python 3.9+ (tested with Python 3.9 via `smartgrid` conda env)
- CUDA-capable GPU strongly recommended for training and inference
- [CVAT](https://www.cvat.ai/) account (for annotation review, optional)
- [Roboflow](https://roboflow.com/) account (for initial YOLO dataset, one-time)

### Installation

```bash
git clone https://github.com/Shubhanflash22/Optimizing-On-Site-EV-Charging-for-Construction-Equipment.git
cd Optimizing-On-Site-EV-Charging-for-Construction-Equipment
```

#### Option A — pip (from `Misc/Requirements.txt`)

```bash
pip install -r Misc/Requirements.txt
```

#### Option B — Conda (from `Misc/smartgrid.yml`)

```bash
conda env create -f Misc/smartgrid.yml
conda activate smartgrid
```

### Full Requirements

```
# Core
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
openpyxl>=3.1.0

# Detection & tracking
ultralytics>=8.0.0
deep-sort-realtime>=1.3.2

# Annotation tooling
lxml>=4.9.0

# Signal processing
scipy>=1.10.0

# Visualization & reporting
matplotlib>=3.7.0
scikit-learn>=1.3.0

# Utilities
tqdm>=4.65.0
pillow>=10.0.0

# Optional (legacy OCR script only)
easyocr>=1.7.0
xlwings>=0.30.0
```

---

## 6. Detailed Pipeline: Step-by-Step

All scripts live in the `Codes/` folder. Run them in numeric order for a new video.

---

### Step 0 (Discarded): OCR-Based Timestamp Detection

**Script:** `Codes/1.Exacavator Detection - Version 1 (Discarded).py`

This was the **first approach** to extracting activity timestamps — it attempted to use **OCR (EasyOCR)** to read the on-screen clock visible in the construction video footage, extracting timestamps at 1 frame per minute to produce a `minute_by_minute_activities_all.xlsx`.

**Why it was discarded:**
- OCR accuracy was inconsistent across different lighting conditions and video angles
- The timestamp-only approach could not distinguish between activity types (just when the machine was present)
- Replaced by CVAT-based manual annotation with activity labels

**What it did:**
1. Extracted 1 frame per minute from each video day
2. Auto-detected optimal OCR rotation (0°/90°/180°/270°) and scale (1.0/0.75/0.5) using sample frames
3. Ran EasyOCR on each frame to extract `HH:MM` timestamp
4. Fallback: if OCR fails, increments previous timestamp by 1 minute
5. Saved all results to multi-sheet Excel (`minute_by_minute_activities_all.xlsx`)

```python
# Key configuration (now archived)
fps_extract = 1/60         # 1 frame per minute
rotations   = [0, 1, 2, 3] # 0°, 90°, 180°, 270°
scales      = [1.0, 0.75, 0.5]
```

---

### Step 1: Custom YOLOv8 Model Training

**Script:** `Codes/2.Custom Yolo model training.py`
**Assets:** `2.Custom Yolo model training/`

A custom YOLOv8n detector was trained on a Roboflow-annotated dataset to detect the single class `Excavator`.

#### Dataset (Roboflow)
- **Workspace:** `excavatar-research-project`
- **Project:** `object-detection-jct8v` (version 2)
- **License:** CC BY 4.0
- **Split:** 70% train / 20% valid / 10% test (Roboflow default)
- **Annotation:** Bounding boxes manually drawn around excavators in frames extracted at 1 frame/minute

#### Training Configuration

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")       # Lightweight nano model as base
model.train(
    data   = "data.yaml",
    epochs = 50,
    imgsz  = 480,
    batch  = 16,
    name   = "yolo_excavator_custom",
    project= "path/to/project"
)
```

| Parameter   | Value         |
|-------------|---------------|
| Base model  | yolov8n.pt    |
| Epochs      | 50            |
| Image size  | 480×480       |
| Batch size  | 16            |
| Classes     | 1 (Excavator) |

**Output:**
- `yolo_excavator_custom/weights/best.pt` — best model by validation mAP (used in all subsequent steps)
- `yolo_excavator_custom/weights/last.pt` — final checkpoint

---

### Step 2: Excavator Detection with YOLO

**Script:** `Codes/3.Step 1 - YOLO.py`
**Output folder:** `3.Step 1 - YOLO/`

Runs the trained `best.pt` on the full construction video frame-by-frame. Only the **highest-confidence** bounding box is kept per frame (since only one excavator is expected per scene).

```bash
python "Codes/3.Step 1 - YOLO.py"
```

**Configuration (edit at top of script):**

```python
model      = YOLO(r"path/to/best.pt")
video_path = r"path/to/video.mp4"
out_path   = r"path/to/output-YOLO.mp4"
csv_path   = r"path/to/detections.csv"
```

**Output:**
- `detections.csv` — frame-level bounding box coordinates
- Annotated video with green bounding boxes and confidence scores

```csv
frame, x1,  y1,  x2,  y2,  confidence, class
1,     245, 180, 567, 432, 0.95,       0
2,     248, 182, 570, 435, 0.94,       0
```

---

### Step 3: Multi-Object Tracking with DeepSORT

**Script:** `Codes/5.Step 2 - Deep Sort.py`
**Output folder:** `5.Step 2 - Deep Sort/`

Assigns persistent track IDs to detections across frames using DeepSORT. Frames are saved as 112×112 crops organized per track — this folder structure directly feeds into the ResNet inference step.

```bash
python "Codes/5.Step 2 - Deep Sort.py"
```

**Configuration:**

```python
VIDEO_PATH       = r"path/to/video.mp4"
CSV_PATH         = r"path/to/detections.csv"     # from Step 2
OUTPUT_PATH      = r"path/to/DEEPSORT.mp4"
TRACK_OUTPUT_CSV = r"path/to/Track_Output.csv"
SAVE_DIR         = r"path/to/Tubes/"             # per-track crop folders
CLIP_LENGTH      = 16                            # frames per ResNet clip
tracker          = DeepSort(max_age=35, n_init=4)
```

| Parameter | Value | Meaning                                    |
|-----------|-------|--------------------------------------------|
| max_age   | 35    | Frames a track can be missed before deletion |
| n_init    | 4     | Frames before a new track is confirmed     |
| crop size | 112×112 | Saved ROI size (matches ResNet input)   |

**Output:**
- `Track_Output.csv` — `frame, track_id, x1, y1, x2, y2`
- `Tubes/track_1/000001.jpg`, `Tubes/track_2/...` — per-track frame crops
- `track_metadata.csv` — `track_id, total_frames, frame_folder`
- Annotated video with track ID labels

---

### Step 4: Idling Detection

**Script:** `Codes/6.Step 3 - Idling.py`
**Output folder:** `6.Step 3 - Idling/`

A **physics-based** idling detector that analyses the centroid movement and bounding box area of each tracked excavator over time. If both are statistically stable within a sliding window, the period is classified as idling.

```bash
python "Codes/6.Step 3 - Idling.py"
```

**Algorithm:**

1. For each track, compute centroid `(cx, cy)` and bounding box area from `Track_Output.csv`
2. Smooth signals using **Savitzky–Golay filter** (window=11, polyorder=2) followed by **rolling median** (window=5)
3. Slide a window of 40 frames; within each window compute:
   - `dist = sqrt(Δcx² + Δcy²)` — frame-to-frame centroid movement
   - `dA = |Δarea|` — frame-to-frame area change
4. If `std(dist) < DIST_THRESHOLD (0.2 px)` AND `std(dA) < AREA_THRESHOLD (0.5% of mean area)` → mark window as **Idling**
5. **Post-processing:** merge gaps < 1 s, drop segments < 3 s

**Configuration:**

```python
FPS                   = 59       # source video FPS (pre-resampling)
WINDOW                = 40       # sliding window size in frames
DIST_THRESHOLD        = 0.2      # centroid movement std threshold (px)
AREA_THRESHOLD_PERCENT= 0.5      # bounding box area std as % of mean
MERGE_GAP_SEC         = 1.0      # merge idling segments < 1 s apart
MIN_IDLE_DURATION_SEC = 3.0      # drop idling segments < 3 s
```

**Output:**
- `Idling_segments.csv`

```csv
track_id, start_frame, end_frame, duration_sec, start_time_sec, end_time_sec, start_time_hms, end_time_hms
1,         100,         350,        4.237,         1.695,          5.932,         00:00:01.695, 00:00:05.932
```

- Per-track PNG plots showing raw vs. smoothed movement, area signal, and idle mask

---

### Step 5 (Optional): Manual Labeling via Excel

Before generating the training dataset, each video was manually labelled in Excel spreadsheets with the following format:

```
Time        | Activity
------------|----------
00:00-00:15 | Digging
00:15-00:30 | Swinging
00:30-01:00 | Loading
```

Spreadsheets are stored in `7.CVAT (excel to xml)/`:
- `Tasks_all.xlsx` — master multi-sheet file with all recording days
- `Tasks_all_updated.xlsx` — revised version with corrections
- Individual day files: `Tasks_Day_2_Oct_21/`, `Tasks_Day_3_Oct_22/`, ..., `Tasks_Day_12_Feb_13/`

These Excel files also serve as **ground truth** for the validation step.

---

### Step 6: CVAT Annotation — Excel to XML Conversion

**Scripts:** `Codes/7.CVAT (excel to xml) for one video.py` and `Codes/7.CVAT (excel to xml) for multiple videos.py`
**Assets:** `7.CVAT (excel to xml)/`

Converts the Excel time-range labels into CVAT-compatible XML annotation files (CVAT 1.1 format) that can be imported directly into CVAT for review, correction, and export.

```bash
python "Codes/7.CVAT (excel to xml) for multiple videos.py"
```

**Configuration:**

```python
EXCEL_PATH       = r"path/to/Labelling.xlsx"
VIDEO_FOLDER     = r"path/to/cvat_data/videos"
DEFAULT_ACTIVITY = "Idling"     # fills gaps between labelled segments
```

**What it does:**
1. Reads each sheet (one sheet = one video clip)
2. Parses time ranges (e.g., `"1:30 - 2:45"` → seconds)
3. Fills unlabelled gaps between segments with `DEFAULT_ACTIVITY` (Idling)
4. Converts seconds → frame numbers using video FPS
5. Generates CVAT XML with `<track>` elements, each with `<box>` keyframes (start + end with `outside="1"`)
6. Saves one `*_raw.xml` per video sheet

**Output:** `output_cvat.xml` (ready for CVAT import)

**CVAT workflow:**
1. Create task in CVAT and upload the video
2. Import the generated XML via `Actions → Upload annotations`
3. Review and correct any errors using the CVAT web UI
4. Export corrected annotations (CVAT XML 1.1 format)

---

### Step 7: Dataset Generation from CVAT Annotations

**Scripts:** `Codes/8.Creating clips from CVAT Annotations from multiple videos.py`
**Input assets:** `8.Creating clips from CVAT Annotations/annotations.xml` (40 MB)

Builds the training dataset for the 3D ResNet by extracting 16-frame clips from labelled video segments, applying YOLO cropping to focus on the excavator.

```bash
python "Codes/8.Creating clips from CVAT Annotations from multiple videos.py"
```

**Configuration:**

```python
VIDEO_XML_PAIRS = [
    ("Day_2.mp4", "annotations_day2.xml"),
    ("Day_3.mp4", "annotations_day3.xml"),
    # ... 12 video pairs total (Day 2 through Day 12)
]
YOLO_MODEL  = "best.pt"
OUTPUT_DIR  = "Dataset_Resnet_3/"
CLIP_LENGTH = 16       # frames per clip (paper: 16)
TARGET_FPS  = 25       # resample to 25 FPS (paper: 25)
CLIP_STRIDE = 3        # step between clip start positions (81% overlap)
MIN_CONFIDENCE = 0.5   # min YOLO detection confidence to include frame
CROP_SIZE   = 112      # spatial resolution (paper: 112×112)
```

**Processing:**
1. **Resample** video from original FPS (e.g., 59 FPS) to 25 FPS using round-nearest indexing (identical to inference)
2. For each resampled frame, **run YOLO** and crop the highest-confidence detection to 112×112
3. Build a frame buffer tagged with activity labels from CVAT XML
4. Emit 16-frame clips where **all 16 frames** share the same activity label and have valid detections
5. Save clips as `activity_name/clip_NNNNN/frame_000.jpg … frame_015.jpg`

**Output structure:**

```
Dataset_Resnet_3/
├── digging/
│   ├── clip_00000/
│   │   ├── frame_000.jpg
│   │   ├── frame_001.jpg
│   │   └── ... (frame_015.jpg)
│   └── clip_00001/
├── loading/
├── swinging/
├── travelling/
└── idling/
```

> **Note:** Dataset was built across 12 videos from recording Days 2–12, covering October 2025 and February 2026.

---

### Step 8: Custom 3D ResNet Model Training

**Script:** `Codes/9.Custom resnet model training.py`
**Output folder:** `9.Custom resnet model training/`
**Run on:** NRP (Nautilus) Kubernetes cluster with GPU

Trains the activity recognition model. The implementation closely follows the methodology in *Cho et al.* with documented deviations.

```bash
python "Codes/9.Custom resnet model training.py"
```

**Training Configuration:**

| Parameter        | Value    | Paper Match? |
|-----------------|----------|--------------|
| Architecture    | R3D-18 (torchvision) | ✅ 3D ResNet |
| Pre-training    | Kinetics-400 weights | ✅ Kinetics |
| Input shape     | (3, 16, 112, 112) | ✅ 16×112×112 |
| Target FPS      | 25       | ✅           |
| Batch size      | 16       | ✅           |
| Learning rate   | 1e-3     | ✅           |
| Optimizer       | Adam     | ➖ (not specified) |
| Epochs          | 20       | ➖ (not specified) |
| Classes         | 5        | ⚠️ Paper has 3 |
| LR scheduler    | ReduceLROnPlateau | ➖ (not specified) |
| Mixed precision | AMP (autocast) | ➖ (not in paper) |

**Fine-tuning strategy:**
- All backbone layers frozen initially
- Only `layer4` and `fc` (classifier) are unfrozen for training
- ~18% of total parameters are trainable

**Data augmentation (per paper Section 4.5):**
- Horizontal flip (50% probability) ✅
- Channel shift: random ±0.1 per-channel brightness ✅
- Affine shear: random factor [-0.15, 0.15], 70% probability ✅
- Spatial: resize to 128×128, then random crop to 112×112 (training) or center crop (validation)

**Class balancing:**
- `WeightedRandomSampler` used to handle class imbalance
- 80/20 train/val split at the **video level** (not frame level, to avoid data leakage)

**Normalization:**
- Dataset-specific mean and std computed from 200 training samples
- Saved as `dataset_mean.npy` and `dataset_std.npy` (must match inference)

**Output:**
- `resnet3d_best_kinetics_2.pth` — full checkpoint containing:
  - `model_state_dict`
  - `optimizer_state_dict`
  - `scheduler_state_dict`
  - `activity_names` list
  - `config` dict (`clip_length`, `crop_size`, `target_fps`, `num_classes`, etc.)
  - Best epoch metrics
- `training_history.json` — per-epoch loss, accuracy, precision, recall

---

### Step 9: Activity Recognition (Inference)

**Script:** `Codes/10.Step 4 - Resnet.py`
**Output folder:** `10.Step 4 - Resnet/`
**Run on:** NRP cluster (paths begin with `/mnt/nvme1/avik_shubhan/`)

Runs the full hybrid inference pipeline on a new video.

```bash
python "Codes/10.Step 4 - Resnet.py"
```

**Configuration:**

```python
MODEL_PATH = Path("resnet3d_best_kinetics_2.pth")
VIDEO_PATH = Path("Day_3.mp4")
OUTPUT_DIR = Path("day_3_pred/")
```

**Inference pipeline:**

1. **Load model config** from checkpoint (`clip_length`, `crop_size`, `target_fps`, `activity_names`)
2. **Resample video** to `TARGET_FPS` (25) using round-nearest frame selection and `cap.grab()` for efficient skipping
3. **Per-frame:** run YOLO → crop excavator to 112×112 → RGB convert → buffer into sliding 16-frame window
4. **AI inference:** normalize clip with training stats → R3D-18 → softmax → top prediction + confidence
5. **Physics override:** compute idling mask from bbox history (same algorithm as Step 4); override `travelling` predictions → `idling` where mask is True
6. **Majority voting:** 50-frame sliding window (= 2 s at 25 FPS) smoothing
7. **Cycle detection:** find digging start events; each pair of consecutive digging starts = one work cycle
8. **Productivity:** `cycles_per_hr × bucket_payload_lcy`

**Output:**
- `frame_predictions.csv` — `Frame, Time_s, Activity, Confidence`
- `cycles.json` — per-cycle breakdown with activity percentages
- `summary.json` — aggregate stats
- `activity_timeline.png` — visual timeline plot

---

### Step 10: Validation & Hyperparameter Optimization

**Script:** `Codes/11.Validation of Pipeline.py`
**Output folder:** `11.Step 5 - Validation/`

A grid-search pipeline that runs AI inference **once**, caches raw predictions, then exhaustively tests all combinations of post-processing hyperparameters against manually labelled ground truth.

```bash
python "Codes/11.Validation of Pipeline.py"
```

**Ground truth loading:**
- Reads `Tasks.xlsx` (same Excel format as Step 5)
- Parses time ranges → frame indices
- Compares per-frame GT label vs. predicted label

**Hyperparameter grid:**

| Parameter                | Values Tested                               |
|--------------------------|---------------------------------------------|
| `min_activity_duration_s`| 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0 |
| `dist_threshold`         | 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6 |
| `idle_window`            | 20, 24, 30, 36, 40, 48, 56, 64, 80          |
| `fsm_min_dwell_seconds`  | 1.0, 1.5, 2.0, 2.5, 3.0                    |
| `enable_fsm`             | True, False                                 |
| `override_travelling_only`| True, False                                |

**Finite State Machine (FSM) — Activity Transitions:**

The FSM enforces physically valid activity sequences:

```
travelling  →  idling, swinging
idling      →  travelling, digging, swinging
digging     →  loading, swinging
loading     →  swinging
swinging    →  digging, loading, idling, travelling
```

Transitions outside this table are rejected unless model confidence > 0.95 (override threshold).

**Inference caching:**
- On first run, heavy AI inference is run and saved to `raw_inference_cache.xlsx`
- Subsequent grid-search runs load from cache, skipping the expensive model forward pass

**Per-run outputs (inside `optimization_runs/Run_NNN/`):**
- `predictions.csv`
- `Run_NNN_DayX_timeline.png` — side-by-side GT vs. prediction timeline
- `Run_NNN_confusion.png` — normalized confusion matrix

**Master report:** `master_optimization_report.csv` — all hyperparameter combinations with accuracy%, cycles/hr, and productivity

---

## 7. Output Files

### Detection CSV (`detections.csv`)

```csv
frame, x1,  y1,  x2,  y2,  confidence, class
1,     245, 180, 567, 432, 0.95,       0
2,     248, 182, 570, 435, 0.94,       0
```

### Tracking CSV (`Track_Output.csv`)

```csv
frame, track_id, x1,  y1,  x2,  y2
1,     1,        245, 180, 567, 432
2,     1,        248, 182, 570, 435
```

### Track Metadata (`track_metadata.csv`)

```csv
track_id, total_frames, frame_folder
1,        1423,         /path/to/Tubes/track_1
2,        87,           /path/to/Tubes/track_2
```

### Idling Segments (`Idling_segments.csv`)

```csv
track_id, start_frame, end_frame, duration_sec, start_time_sec, end_time_sec, start_time_hms,  end_time_hms
1,         100,         350,        4.237,         1.695,          5.932,         00:00:01.695, 00:00:05.932
```

### Frame Predictions (`frame_predictions.csv`)

```csv
Frame, Time_s, Activity,  Confidence
0,     0.000,  digging,   0.9821
1,     0.040,  digging,   0.9754
50,    2.000,  swinging,  0.8833
```

### Cycles (`cycles.json`)

```json
[
  {
    "cycle_number": 1,
    "start_frame": 0,
    "end_frame": 994,
    "duration_frames": 994,
    "duration_seconds": 39.76,
    "activity_counts": {"digging": 312, "swinging": 420, "loading": 262},
    "activity_percentages": {"digging": 31.4, "swinging": 42.2, "loading": 26.4}
  }
]
```

### Summary (`summary.json`)

```json
{
  "total_frames": 83012,
  "duration_seconds": 3320.48,
  "fps": 25,
  "activity_distribution": {
    "digging":    28.99,
    "swinging":   26.49,
    "travelling": 18.82,
    "idling":     18.65,
    "loading":     7.05
  },
  "num_cycles": 77,
  "avg_cycle_time_seconds": 39.75,
  "productivity_lcy_per_hour": 135.84,
  "avg_confidence": 0.9413
}
```

---

## 8. Model Architecture

### YOLOv8n (Detection)

| Property       | Value                   |
|----------------|-------------------------|
| Architecture   | YOLOv8 Nano             |
| Base weights   | `yolov8n.pt` (COCO)     |
| Fine-tuned on  | Roboflow excavator dataset |
| Input size     | 480×480                 |
| Classes        | 1 (`Excavator`)          |
| Epochs trained | 50                      |
| Strategy       | Full fine-tune          |

### R3D-18 (Activity Recognition)

| Property              | Value                        |
|-----------------------|------------------------------|
| Architecture          | 3D ResNet-18 (`r3d_18`)      |
| Pre-trained on        | Kinetics-400                 |
| Input shape           | (1, 3, 16, 112, 112)         |
| Frozen layers         | `stem`, `layer1`, `layer2`, `layer3` |
| Trained layers        | `layer4`, `fc`               |
| Output classes        | 5                            |
| Target FPS            | 25                           |
| Temporal receptive field | 16 frames = 0.64 s at 25 FPS |
| Normalization         | Dataset-specific mean/std     |

---

## 9. Configuration Parameters

### Detection & Tracking (Steps 2–3)

| Parameter     | Default | Description                                  |
|---------------|---------|----------------------------------------------|
| `imgsz`       | 480     | YOLO inference image size                    |
| `max_age`     | 35      | DeepSORT: frames before a lost track is deleted |
| `n_init`      | 4       | DeepSORT: frames before a new track is confirmed |
| `CLIP_LENGTH` | 16      | Frames saved per track segment               |

### Idling Detection (Step 4)

| Parameter              | Default | Description                                    |
|------------------------|---------|------------------------------------------------|
| `FPS`                  | 59      | Source video FPS (pre-resampling)              |
| `WINDOW`               | 40      | Sliding window size in frames                  |
| `DIST_THRESHOLD`       | 0.2     | Max std of centroid displacement (px)          |
| `AREA_THRESHOLD_PERCENT` | 0.5   | Max std of area change as % of mean area       |
| `SAVGOL_WINDOW`        | 11      | Savitzky–Golay filter window (must be odd)     |
| `SAVGOL_POLYORDER`     | 2       | Savitzky–Golay polynomial order                |
| `ROLL_MEDIAN_WINDOW`   | 5       | Rolling median window for spike suppression    |
| `MERGE_GAP_SEC`        | 1.0     | Merge idling segments closer than this (s)     |
| `MIN_IDLE_DURATION_SEC`| 3.0     | Drop idling segments shorter than this (s)     |

### Dataset Generation (Step 7)

| Parameter       | Default | Description                                    |
|----------------|---------|------------------------------------------------|
| `CLIP_LENGTH`  | 16      | Frames per training clip (paper: 16)           |
| `TARGET_FPS`   | 25      | Resample target FPS (paper: 25)                |
| `CLIP_STRIDE`  | 3       | Step between clip start positions (81% overlap)|
| `MIN_CONFIDENCE` | 0.5   | Min YOLO confidence to include a frame         |
| `CROP_SIZE`    | 112     | Spatial resolution of saved crops (paper: 112) |

### ResNet Training (Step 8)

| Parameter        | Default | Description                              |
|-----------------|---------|------------------------------------------|
| `CLIP_LENGTH`   | 16      | Frames per clip (paper: 16)              |
| `CROP_SIZE`     | 112     | Spatial size in px (paper: 112)          |
| `TARGET_FPS`    | 25      | Training FPS (paper: 25)                 |
| `BATCH_SIZE`    | 16      | Training batch size (paper: 16)          |
| `LEARNING_RATE` | 1e-3    | Adam LR (paper: 0.001)                   |
| `NUM_EPOCHS`    | 20      | Training epochs                          |
| `NUM_CLASSES`   | 5       | Activity classes                         |
| `STRIDE`        | 1       | Frame sampling stride within a clip      |

### Activity Recognition / Post-Processing (Steps 9–10)

| Parameter                 | Default | Description                                 |
|--------------------------|---------|---------------------------------------------|
| `TARGET_FPS`             | 25      | Inference resampling FPS                    |
| `CLIP_LENGTH`            | 16      | Frames per inference window                 |
| `min_activity_duration_s`| 2.0     | Majority voting window duration in seconds  |
| `IDLE_WINDOW`            | 40      | Physics idling: sliding window frames       |
| `DIST_THRESHOLD`         | 0.2     | Physics idling: centroid movement threshold |
| `OVERRIDE_CONF_THRESHOLD`| 0.95    | FSM: high-confidence override of transitions |
| `bucket_payload_lcy`     | 1.5     | Bucket payload for productivity calculation |

---

## 10. Activity Classes

The model recognizes **5 activity classes**. The paper this work is based on (*Cho et al.*) uses 3 classes (digging, loading, swinging); this implementation extends to 5 for more comprehensive monitoring.

| Class       | Label Index | Description                                            |
|-------------|-------------|--------------------------------------------------------|
| `digging`   | 0           | Bucket lowered into the ground, excavating material   |
| `idling`    | 1           | Machine stationary; engine running but no work        |
| `loading`   | 2           | Bucket filled, being lifted before swing              |
| `swinging`  | 3           | Upper structure rotating to dump or return position   |
| `travelling`| 4           | Machine moving from one location to another           |

**Typical activity distribution** (from Day 3 inference, `summary.json`):

| Activity    | % of Time |
|-------------|-----------|
| Digging     | 28.99%    |
| Swinging    | 26.49%    |
| Travelling  | 18.82%    |
| Idling      | 18.65%    |
| Loading     | 7.05%     |

---

## 11. Productivity & Cycle Time Calculation

Based on **Paper Section 4.5, Equation 3**:

```
Productivity (LCY/hr) = Cycles_per_hour × Bucket_payload (LCY)
```

A **work cycle** is defined as the period from one digging start to the next digging start. Cycle components typically include:
1. **Digging** — bucket excavation
2. **Loading** — bucket lift (sometimes merged with digging)
3. **Swinging** (loaded) — rotation to dump zone
4. **Swinging** (empty) — rotation back to dig zone

**Results from Day 3 video:**

| Metric                | Value              |
|-----------------------|--------------------|
| Total duration        | 3,320 s (55.3 min) |
| Work cycles detected  | 77                 |
| Avg cycle time        | 39.75 s            |
| Cycles per hour       | ~90.6              |
| Bucket payload        | 1.5 LCY            |
| **Productivity**      | **135.84 LCY/hr**  |
| Avg model confidence  | 94.1%              |

---

## 12. Performance Results

> All results below come from the full end-to-end run documented in
> `Run from step 8 to 11/V2 Validation only/pipeline_steps_8_9_10_summary.txt`
> and `Run from step 8 to 11/V1 Full video/optimization_runs/`.

---

### 12.1 Dataset Summary (Step 8)

| Metric | Value |
|---|---|
| Source videos | 11 (Day 2, Day 3, Day 4\_1, TC\_00011–00021) |
| Total 16-frame clips | **306,780** |
| Training clips | 261,281 (85%) |
| Validation clips | 45,499 (15%) |
| Clip duration | 0.64 s @ 25 FPS |
| Spatial resolution | 112×112 px (YOLO-cropped) |
| Train/val split | By video group — no frame leakage |

**Dataset class distribution:**

| Activity | Train clips | Val clips |
|---|---|---|
| Digging | 32,257 | 5,721 |
| Idling | 126,180 | 22,242 |
| Loading | 10,743 | 1,788 |
| Swinging | 62,252 | 10,485 |
| Travelling | 29,849 | 5,263 |
| **Total** | **261,281** | **45,499** |

Normalization stats computed from 200 training clips:
- Mean: `[0.2725, 0.3182, 0.3059]`
- Std: `[0.1956, 0.1834, 0.1846]`

---

### 12.2 Training Results (Step 9 — R3D-18)

**Configuration:** R3D-18, Kinetics-400 pretrained, `layer4 + fc` unfrozen (75.1% trainable params), 20 epochs, Adam @ LR=0.001, mixed precision (AMP), batch size 16.

**Training curve (selected epochs):**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Avg Precision | Avg Recall |
|---|---|---|---|---|---|---|
| 1 | 0.4100 | 84.95% | 0.4458 | 84.44% | 76.1% | 80.6% |
| 4 | 0.1149 | 95.93% | **0.3498** | **89.68%** | 84.1% | 83.4% |
| 9 | 0.0443 | 98.44% | 0.3936 | 90.13% | 83.0% | 86.1% |
| 15 | 0.0215 | 99.22% | 0.3928 | 90.96% | 84.9% | 85.9% |
| 18 | 0.0162 | 99.40% | 0.4118 | 91.22% | 85.0% | 86.8% |
| 20 | 0.0151 | 99.45% | 0.4357 | 91.38% | 86.4% | 85.2% |

> ⚠️ Best val loss was at **epoch 4** (0.3498, 89.68% acc). Training loss continued dropping to 0.015 while val loss plateaued/rose to 0.40–0.48, indicating overfitting. Early stopping around epoch 4–9 would likely generalize better.

**Final epoch (20) per-class precision / recall:**

| Activity | Precision | Recall |
|---|---|---|
| Digging | 87.3% | 90.3% |
| Idling | 97.0% | 97.3% |
| Loading | 73.9% | 67.5% |
| Swinging | 86.3% | 87.6% |
| Travelling | 87.7% | 83.1% |
| **Average** | **86.4%** | **85.2%** |

**Best checkpoint:** epoch 4 → val accuracy **89.68%** (saved as `resnet3d_best_kinetics_2.pth`)

---

### 12.3 Hyperparameter Optimization Results (V1 — Grid Search)

The grid search tested **~9,000+ parameter combinations** over the full video, with results in `Run from step 8 to 11/V1 Full video/optimization_runs/all_param_results.csv`.

**Best configuration (`PARAM_SET_9063`) — Overall accuracy 88.81%:**

```json
{
  "min_activity_duration_s": 2.0,
  "dist_threshold": 0.05,
  "idle_window": 36,
  "fsm_min_dwell_seconds": 1.0,
  "enable_fsm": false,
  "override_travelling_only": true
}
```

**Per-class accuracy at best parameters:**

| Activity | Accuracy |
|---|---|
| Digging | 87.82% |
| Idling | 98.64% |
| Loading | 81.78% |
| Swinging | 74.60% |
| Travelling | 80.94% |
| **Overall** | **88.81%** |

---

### 12.4 Full Validation Results (V2 — 11 Videos, Best Params Locked)

Using the best parameters from the grid search, continuous inference was run on the held-out validation frames across all 11 source videos.

**Per-video accuracy:**

| Video | Val Frames | Correct | Accuracy |
|---|---|---|---|
| Day\_2 | 13,470 | 10,660 | 79.14% |
| Day\_3 | 39,207 | 32,877 | 83.85% |
| Day\_4\_1 | 2,432 | 2,009 | 82.61% |
| TC\_00011 | 5,937 | 5,604 | 94.39% |
| TC\_00012 | 25,932 | 22,513 | 86.82% |
| TC\_00013 | 365 | 365 | 100.00% |
| TC\_00014 | 16,600 | 15,273 | 92.01% |
| TC\_00015 | 1,413 | 1,298 | 91.86% |
| TC\_00016 | 23,512 | 22,602 | 96.13% |
| TC\_00019 | 21,340 | 19,050 | 89.27% |
| TC\_00021 | 18,893 | 16,431 | 86.97% |
| **Overall** | **169,101** | **148,682** | **87.92%** |

**Overall per-class classification report (169,101 val frames):**

| Activity | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Digging | 0.851 | 0.879 | 0.865 | 21,193 |
| Idling | 0.918 | 0.981 | 0.949 | 80,675 |
| Loading | 0.640 | 0.728 | 0.681 | 7,457 |
| Swinging | 0.899 | 0.724 | 0.802 | 39,853 |
| Travelling | 0.814 | 0.833 | 0.823 | 19,923 |
| **Macro avg** | **0.824** | **0.829** | **0.824** | 169,101 |
| **Weighted avg** | **0.881** | **0.879** | **0.877** | 169,101 |

**Confusion Matrix (rows = ground truth, cols = predicted):**

```
               digging   idling   loading   swinging   travelling
  digging       18,631      768       572        957          265
  idling           201   79,171       104        501          698
  loading          960      210     5,427        794           66
  swinging       1,752    4,229     2,240     28,862        2,770
  travelling       347    1,843       133      1,009       16,591
```

Visual confusion matrix heatmaps and GT-vs-prediction timelines for all 11 videos are saved in:
`Run from step 8 to 11/V2 Validation only/Plots/` (24 plots: `heatmap_*.png` + `timeline_*.png`)

---

### 12.5 Comparison to Paper (*Cho et al.* — 3 classes)

| Metric | Paper (3 classes) | This Implementation (5 classes) |
|---|---|---|
| Architecture | Custom 3D ResNet | R3D-18 (torchvision) |
| Pre-training | Kinetics-400 | Kinetics-400 ✅ |
| Input | 16×112×112 @ 25 FPS | 16×112×112 @ 25 FPS ✅ |
| Avg accuracy | 87.6% | **87.92%** (val set) / **89.68%** (best epoch) |
| Digging precision | 95% | 85.1% |
| Swinging precision | 86% | 89.9% |
| Loading precision | 84% | 64.0% |

> **Key observations:**
> - Overall accuracy **matches and slightly exceeds** the paper (87.92% vs 87.6%) despite the harder 5-class problem
> - **Idling** is the strongest class (98.1% recall) — physics-based override and high support both help
> - **Loading** is the weakest class everywhere (64% precision, 73% recall) — least training data (10,743 clips vs 126,180 for idling), and visually similar to digging/swinging transitions
> - Post-processing (idling physics + 2s majority voting, FSM disabled) is roughly accuracy-neutral relative to raw classifier output
> - Per-video accuracy varies widely (79–100%), with older field days (Day\_2) being harder — likely different camera angles or equipment

---

## 13. NRP / Kubernetes Cluster Training

Model training (Step 8) and inference (Steps 9–10) were performed on the **National Research Platform (NRP / Nautilus)** Kubernetes cluster due to the compute requirements of 3D CNN training.

All Kubernetes configs are in `NRP Stuff/`.

### Quick Start — NRP Workflow

```powershell
# 1. Set context
kubectl config use-context nautilus
kubectl get pods -n shi-group-ece-advanced

# 2. Create persistent storage (once)
kubectl apply -f nrp-dataset-pvc.yaml      # 50 GB PVC

# 3. Upload dataset (once)
kubectl apply -f dataset-uploader.yaml
kubectl cp ./Dataset_Resnet_3 dataset-uploader:/data/dataset
kubectl delete pod dataset-uploader

# 4. Launch training pod
kubectl apply -f resnet3d-train.yaml       # or resnet3d-train_job.yaml

# 5. Monitor training
kubectl exec -n shi-group-ece-advanced -it resnet3d-train -- tail -f /data/models/train.log
kubectl exec -n shi-group-ece-advanced resnet3d-train -- grep Epoch /data/models/train.log

# 6. Copy results back
kubectl cp shi-group-ece-advanced/resnet3d-train:/data/models/resnet3d_best_kinetics_2.pth .

# 7. Delete pod when done
kubectl delete pod resnet3d-train -n shi-group-ece-advanced
```

### Persistent Storage Layout on NRP

```
/mnt/nvme1/avik_shubhan/resnet3d/
├── Dataset_Resnet_3/        ← Training dataset (uploaded once)
├── resnet3d_best_kinetics_2.pth
├── best.pt                  ← YOLO model
├── dataset_mean.npy
├── dataset_std.npy
├── Day_2.mp4 … Day_3.mp4   ← Source videos
└── day_3_pred/              ← Inference outputs
```

> ⚠️ **Important:** Everything outside `/mydata` (or your PVC mount) is **lost when the pod is terminated**. Always save outputs to the mounted PVC path.

### Copying Scripts to Cluster

```powershell
kubectl cp "Codes/9.Custom resnet model training.py" `
    shi-group-ece-advanced/resnet3d-train:/data/scripts/train.py
```

---

## 14. Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available. The inference and training scripts auto-detect `torch.device("cuda")`. Run `nvidia-smi` inside the pod to confirm.
2. **Batch Processing**: For training, increase `BATCH_SIZE` if GPU VRAM allows (the script uses mixed-precision AMP which significantly reduces memory usage).
3. **Inference Caching**: The validation script (`Step 11`) automatically caches raw AI predictions to `raw_inference_cache.xlsx` after the first run. This lets you iterate on post-processing hyperparameters without re-running the heavy model.
4. **Clip Overlap (Dataset)**: The `CLIP_STRIDE=3` in dataset generation gives 81% overlap — good dataset size without redundancy. Reduce to 1 for maximum data, or increase to 8 for faster but sparser dataset creation.
5. **Temporal Smoothing**: Adjust `min_activity_duration_s` (majority voting window) to reduce jitter. Values between 1.5–2.5 s typically work best.
6. **FSM Tuning**: Enable the FSM (`enable_fsm=True`) if output contains physically impossible sequences (e.g., Digging → Travelling → Loading). The FSM `min_dwell_seconds` controls how long a state must persist before a transition is allowed.
7. **Frame Skipping**: The inference generator uses `cap.grab()` for non-target frames — this skips decoding entirely and is significantly faster than `cap.read()` for high-FPS source videos.

---

## 15. Troubleshooting

### Low Detection Accuracy
- Train YOLOv8 with more diverse excavator images (different angles, lighting, distances)
- Adjust confidence threshold (default: 0.5 for dataset generation, none for inference)
- Try a larger YOLO model (`yolov8s.pt`, `yolov8m.pt`)
- Increase input image size to `640×640` for finer detail

### Track ID Switching
- Increase DeepSORT `max_age` parameter (try 50–100)
- Increase `n_init` to require more frames before track confirmation
- Reduce occlusions in video footage (camera placement)

### Poor Activity Recognition
- Ensure dataset has balanced class distribution (use `WeightedRandomSampler` — already implemented)
- Increase `CLIP_LENGTH` for longer temporal context (try 24 or 32 frames)
- Add more data augmentation (temporal jitter, color jitter)
- Unfreeze more ResNet layers (try `layer3` + `layer4` + `fc`)
- Try `r3d_18` → `r3d_50` for a larger model capacity

### Normalization Mismatch
- The `dataset_mean.npy` and `dataset_std.npy` files **must match** between training (Step 8), clip creation (Step 7), and inference (Steps 9–10). If you retrain, recompute and redistribute these files.

### CVAT XML Import Fails
- Ensure CVAT task frame count matches the video used for annotation
- Check that the XML version header is `1.1`
- Verify no `start_frame >= end_frame` segments (the script caps these)

### NRP Pod Stuck / Not Running
- Check resource requests in YAML (GPU availability varies)
- Run `kubectl describe pod <podname> -n shi-group-ece-advanced` to see scheduling events
- Try `shubhan-any-gpu.yaml` which requests any available GPU type

---

## 16. References & Related Papers

All PDFs are in the `Papers and references/` folder.

| Authors | Description |
|---------|-------------|
| **Cho, Latif, Sharafat, Seo** | Primary paper — 3D ResNet for excavator activity recognition. Defines the 3-class model, 16×112×112 input, 25 FPS, Kinetics-400 fine-tuning, cycle time methodology |
| **Chen, Zhu, Hammad** | Construction equipment activity monitoring |
| **Fard, Heydarian, Niebles** | Video-based construction worker and equipment analysis |
| **Ghelmani, Torabi, Hammad, Chen** | Site equipment monitoring pipeline |
| **Hong, Song, Hong, Kim, Jeong** | Deep learning for construction site monitoring |
| **Akın, Muhammed Ali, Beyazıt, Jan Kleissl, Yuanyuan Shi** | EV charging optimization for on-site equipment |
| **Power Flow Security Maximization** | Grid-level EV integration study |

---

## 17. Citation

If you use this work, please cite:

```bibtex
@software{excavator_activity_recognition,
  author = {Shubhan Mital},
  title  = {Excavator Activity Recognition System},
  year   = {2025},
  url    = {https://github.com/Shubhanflash22/Optimizing-On-Site-EV-Charging-for-Construction-Equipment}
}
```

Primary methodology paper:

```bibtex
@article{cho2021excavator,
  author  = {Cho, Y. K. and Latif, E. and Sharafat, A. and Seo, J.},
  title   = {Automated activity recognition of excavators using 3D convolutional neural networks},
  journal = {[See Papers and references/ for full citation]},
  year    = {2021}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The Roboflow training dataset is licensed under **CC BY 4.0** (see `2.Custom Yolo model training/README.roboflow.txt`).
