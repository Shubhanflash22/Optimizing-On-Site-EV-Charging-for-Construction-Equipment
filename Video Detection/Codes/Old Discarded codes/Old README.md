# Excavator Activity Recognition Pipeline

This repository implements a complete end‑to‑end **excavator activity
recognition pipeline** --- from training a custom YOLO detector to Deep
SORT tracking, idling detection, CVAT‑based clip generation, 3D ResNet
model training, and final activity classification on tracked excavator
tubes.

------------------------------------------------------------------------

## 📌 Pipeline Overview

1.  **Train custom YOLOv8 excavator detector**\
    Script: `2.Custom-Yolo-model-training.py`

    -   Trains YOLOv8n using a Roboflow-style dataset.
    -   Outputs `best.pt` weights stored under the chosen project path.
    -   Used later for accurate excavator detection.

2.  **Run YOLO inference to extract per-frame detections**\
    Script: `3.Step-1-YOLO.py`

    -   Loads YOLO weights and performs inference on video frames.
    -   Saves bounding box detections per frame.

    *Note:* `1.Exacavator-Detection-Version-1-Discarded.py` is an early
    prototype kept for reference.

3.  **Track excavators using Deep SORT**\
    Script: `5.Step-2-Deep-Sort.py`

    -   Uses YOLO detections + Deep SORT to maintain consistent
        excavator IDs.
    -   Produces:
        -   `Track_Output.csv` (per-frame track info)
        -   Per-track frame folders (excavator tubes)
        -   `track_metadata.csv` summarizing tracks and folders

4.  **Detect idling activity**\
    Script: `6.Step-3-Idling.py`

    -   Computes motion and bounding box area statistics for each track.
    -   Applies smoothing + sliding-window classification to detect
        idling.
    -   Outputs:
        -   `Idling_segments.csv`
        -   Optional diagnostic plots per track

5.  **Convert CVAT Excel annotations to XML**\
    Script: `7.CVAT-excel-to-xml.py`

    -   Converts CVAT-exported spreadsheets into structured XML for
        downstream clip generation.

6.  **Generate training clips from CVAT annotations**\
    Script: `8.Creating-clips-from-CVAT-Annotations.py`

    -   Reads CVAT XML + raw videos.
    -   Cuts labeled segments into uniform 3D clips (e.g., 16 frames).
    -   Saves clips in class‑specific folders:
        -   `digging/`
        -   `loading/`
        -   `swinging/`

7.  **Train 3D ResNet model for activity classification**\
    Script: `9.Custom-resnet-model-training.py`

    -   Defines a `ResNet3D` backbone with 3D bottleneck blocks.
    -   Trains on generated clips.
    -   Outputs:
        -   `best_activity_model.pth`

8.  **Run activity recognition on track-wise frame tubes**\
    Script: `10.Step-4-Resnet.py`

    -   Loads `best_activity_model.pth`.
    -   Reads `track_metadata.csv` and track frame folders.
    -   Builds sliding-window clips over each track.
    -   Predicts frame-wise activities:\
        **digging, loading, swinging**
    -   Applies smoothing + segmentation to produce readable activity
        timelines.
    -   Outputs:
        -   `Activity_Output.csv` (frame-wise predictions)
        -   `Activity_Visual.csv` (segment-level activity timeline)

------------------------------------------------------------------------

## 🧩 Detailed Description of Each Stage

### 1. **YOLO Training**

-   Initializes YOLOv8n with pretrained weights (`yolov8n.pt`).
-   Trains on excavator dataset using configurable hyperparameters
    (epochs, batch size, image size).
-   Produces high-quality excavator detections for the tracking stage.

------------------------------------------------------------------------

### 2. **Detection & Tracking**

#### Detection (`3.Step-1-YOLO.py`)

-   Inference on video frames.
-   Outputs bounding boxes, class IDs, and scores.

#### Tracking (`5.Step-2-Deep-Sort.py`)

-   Deep SORT matches detections frame-to-frame.
-   Exports track folders containing only the excavator for that track.
-   `Track_Output.csv` includes:
    -   `track_id`
    -   `frame`
    -   `x1, y1, x2, y2`
    -   confidence / class columns (if included)

------------------------------------------------------------------------

### 3. **Idling Analysis**

-   Uses motion (center displacement) + scale (bounding box area)
    changes.
-   Smoothing via:
    -   Savitzky--Golay filter
    -   Rolling median
-   Detects low-variation windows → idling.
-   Exports:
    -   Start/end frames
    -   Start/end time (sec & H:M:S)
    -   Duration

------------------------------------------------------------------------

### 4. **CVAT → XML Conversion**

-   Converts annotated activity segments from Excel to XML.
-   Organizes attributes like:
    -   `track_id`
    -   `start_frame`, `end_frame`
    -   `activity_label`

------------------------------------------------------------------------

### 5. **Clip Generation**

-   Extracts fixed-length clips aligned with activity timestamps.
-   Produces datasets suitable for 3D CNNs.

------------------------------------------------------------------------

### 6. **3D ResNet Model Training**

-   Preprocessing:
    -   Resize images to `CROP_SIZE`
    -   Normalize
-   Clips shape: `C × T × H × W`
-   Trains classifier for:
    -   Digging
    -   Loading
    -   Swinging

------------------------------------------------------------------------

### 7. **Final Activity Recognition**

-   Loads all frames for each track.
-   Creates sliding window segments:
    -   Clip length = `CLIP_LENGTH`
    -   Stride = `STRIDE`
-   Predicts frame-level activities.
-   Smooths predictions using temporal voting.
-   Outputs:
    -   `Activity_Output.csv`: Continuous per-frame labels
    -   `Activity_Visual.csv`: Segmented human-readable summary

------------------------------------------------------------------------

## 📁 Directory / File Overview

  File                                          Description
  --------------------------------------------- ---------------------------------
  `2.Custom-Yolo-model-training.py`             Train custom YOLO model
  `3.Step-1-YOLO.py`                            Run YOLO inference
  `5.Step-2-Deep-Sort.py`                       Deep SORT multi-object tracking
  `6.Step-3-Idling.py`                          Idling detection
  `7.CVAT-excel-to-xml.py`                      Convert CVAT Excel → XML
  `8.Creating-clips-from-CVAT-Annotations.py`   Extract labeled clips
  `9.Custom-resnet-model-training.py`           Train 3D ResNet
  `10.Step-4-Resnet.py`                         Final activity recognition

------------------------------------------------------------------------

## 🔗 Outputs Summary

### Detection & Tracking

-   `Track_Output.csv`
-   Track frame folders
-   `track_metadata.csv`

### Idling

-   `Idling_segments.csv`
-   Optional plots

### Activity Recognition

-   `Activity_Output.csv`
-   `Activity_Visual.csv`

------------------------------------------------------------------------

## 🏁 End-to-End Flow Diagram

    YOLO Training → Detection → Deep SORT Tracking → Idling Detection
           ↓                     ↓
      Clip Generation (CVAT) → 3D ResNet Training → Final Activity Classification

This pipeline transforms raw construction-site video into interpretable
excavator activity timelines with state-of-the-art deep learning
methods.

------------------------------------------------------------------------

## 📜 Citation Notes

This README summarizes functionality based on scripts included in the
repository and their intended usage.
