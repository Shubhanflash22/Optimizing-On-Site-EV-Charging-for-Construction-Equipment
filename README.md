# Optimizing On-Site EV Charging for Construction Equipment

A two-sided research project developed in collaboration with Avik at UC San Diego:

1. **A computer-vision pipeline** that watches construction excavators (Construction
   Equipment Vehicles, or CEVs) on jobsite video and classifies what they're doing —
   digging, loading, swinging, travelling, or idling — frame by frame, at 25 FPS, in
   16-frame temporal windows, across 12+ recording days spanning October 2025 through
   February 2026 and three different site materials (soil, concrete, sand).
2. **An optimization and control stack** that schedules a fleet of Mobile Charging
   Stations (MCS — battery packs on towed electric pickups) to keep those CEVs charged
   and working, at minimum cost, subject to real physical and operational constraints:
   battery capacity, charging/discharging rates, time-of-use electricity pricing,
   demand charges, travel time between sites, and uncertainty in exactly how much
   power each activity actually draws.

The two sides are not independent case studies bolted together — they form one
pipeline. The CV side's whole purpose is to characterize *how CEVs actually behave* on
a real jobsite, so that the optimization side isn't scheduling around a textbook
assumption but around a statistically fitted, uncertainty-quantified model of real
machine power draw. Section 1 below walks through exactly how the handoff works.

This file is the map and the master overview for the entire repository. Each of the
four project areas has its own dramatically more detailed README (with formal math,
line-by-line constraint audits, and full appendices) — this file goes deep enough that
you should be able to understand *what every piece does and why it exists* without
opening any of them, and tells you exactly which one to open next for the level of
detail you actually need.

---

## Contents

1. [How the four parts fit together](#1-how-the-four-parts-fit-together)
2. [Repository map](#2-repository-map)
3. [Video Detection — excavator activity recognition](#3-video-detection--excavator-activity-recognition)
4. [Bayesian Regression — activity power fitting](#4-bayesian-regression--activity-power-fitting)
5. [MPC — original baseline (Avik, Scenario 1)](#5-mpc--original-baseline-avik-scenario-1)
6. [MPC_Shubhan — Approach 1 (certainty-equivalent MPC)](#6-mpc_shubhan--approach-1-certainty-equivalent-mpc)
7. [MPC_Shubhan — Approach 2 (stochastic, scenario-based MPC)](#7-mpc_shubhan--approach-2-stochastic-scenario-based-mpc)
8. [Known issues across the repo](#8-known-issues-across-the-repo)
9. [Getting started, end to end](#9-getting-started-end-to-end)
10. [Citation](#10-citation)
11. [License](#11-license)

---

## 1. How the four parts fit together

```
   Video Detection                Bayesian Regression              MPC / MPC_Shubhan
   ────────────────                ────────────────────              ─────────────────
   Watches raw jobsite     ──►     Regresses per-activity    ──►     Schedules MCS routing/
   video, classifies each          power draw (kW) from real         charging + CEV work to
   frame as Digging /              task logs + battery SoC           minimize energy, demand-
   Loading / Swinging /            drop. Produces μ (mean) and       charge, and missed-work
   Travelling / Idling.            σ (uncertainty) per activity.     cost, subject to physical
                                                                      and operational constraints.
```

### The handoff, precisely

The Video Detection pipeline's end product, per video, is a frame-level activity
timeline (`frame_predictions.csv`) plus a derived cycle/productivity report
(`cycles.json`, `summary.json`) — a machine-generated record of exactly how long a CEV
spent digging, loading, swinging, travelling, or idling, second by second, for an
entire recorded shift.

The Bayesian Regression scripts need exactly that shape of data as input — a
time-stamped activity log paired with the CEV's battery State-of-Charge (SoC) over the
same period — to solve for each activity's power draw via `A z = b`, where `A` is
hours-per-activity per SoC-drop "equation" and `z` is the unknown per-activity power
vector. **Today the regression scripts consume manually-labelled Excel task logs
directly** (the same `Tasks_*.xlsx` files that also serve as CV validation ground
truth), not the CV pipeline's automated CSV output — but the two are the same shape of
data, and the natural next step is feeding `frame_predictions.csv` in directly to scale
labelling up past what 12 manually-annotated days can cover.

The regression's fitted output — a posterior mean `μ` and standard deviation `σ` per
activity, in kW — is not a side artifact. It is *literally* the `p_digging`,
`p_loading_swinging`, `p_traveling` parameters (and their uncertainty) that every
downstream MILP reads out of `parameters.csv` and plans against. Trace it end to end:
`Tasks_energy_loading_swinging_bayesian.py` writes posterior mean/std → `parameters.csv`
→ `2_DataLoader.jl`'s `load_all_data` → `est.mu` / `est.sigma` inside every controller's
`build_window_model` → the MILP's work-power equation `P_work = Σₐ pₐ·u` (Eq. 5e in the
formal model). Every cost number reported anywhere in the MPC side of this repo
ultimately traces back to a coefficient fitted from real field video data.

### The optimization lineage

The optimization side itself is not one model — it's three generations, each an
extension of the last, all solving structurally the same physical problem:

```
MPC/Scenario 1  (Avik's original baseline)
     │  one-shot, single-scenario MILP, solved once — no re-planning, no feedback
     ▼
MPC_Shubhan/Approach 1  (certainty-equivalent MPC)
     │  adds closed-loop re-planning every 15 minutes (two horizon flavors: Shrinking
     │  and Receding), the Approach 0 vs Approach 1 comparison methodology with a
     │  shared random-number pool, and full batch/sweep tooling (RUN_ALL.jl)
     ▼
MPC_Shubhan/Approach 2  (stochastic, scenario-based MPC)
        adds multi-scenario, non-anticipative optimization at every re-solve — instead
        of planning against one mean power estimate, plans against a small sampled set
        of possible futures simultaneously, coupled so the immediately-applied action
        can't depend on which future turns out to be real. Same physics, same data,
        same comparison methodology as Approach 1 — see §7's full breakdown of the delta.
```

Understanding this lineage matters for reading the codebase: constraint numbering and
naming conventions throughout `MPC_Shubhan`'s documentation (e.g. "Eq. 13",
"`work_per_travel = 4 (Avik)`") refer back to `MPC/Scenario 1`'s original
`MCS_OPTIMAL_v4_real.jl` — that file is the ground truth for "what did the original
model say," and everything downstream is documented relative to it.

---

## 2. Repository map

```
Optimizing-On-Site-EV-Charging-for-Construction-Equipment/
│
├── README.md                          ← this file
│
├── Video Detection/                   ← CV pipeline (excavator activity recognition)
│   ├── README_CV.md                   ← full pipeline documentation (18 sections)
│   ├── LICENSE
│   ├── Codes/                         ← all pipeline scripts, numbered in run order
│   │   ├── 1.Exacavator Detection - Version 1 (Discarded).py
│   │   ├── 2.Custom Yolo model training.py
│   │   ├── 3.Step 1 - YOLO.py
│   │   ├── 4.3d rcnn model.py
│   │   ├── 5.Step 2 - Deep Sort.py
│   │   ├── 6.Step 3 - Idling.py
│   │   ├── 7.CVAT (excel to xml) for one video.py
│   │   ├── 7.CVAT (excel to xml) for multiple videos.py
│   │   ├── 7a.Step 7a - Batch_cvat_pipeline.py
│   │   ├── 8.Creating clips from CVAT Annotations from {one,multiple} video(s).py
│   │   ├── 9.Custom resnet model training.py
│   │   ├── 10.Step 4 - Resnet.py
│   │   ├── 11.Validation of Pipeline.py
│   │   └── Old Discarded codes/       ← earlier validation/comparison script iterations
│   ├── 1.Exacavator Detection - Version 1 (Discarded)/   ← OCR approach, abandoned
│   │   └── frames_Day_{2,3,4}_2025-10-{21,22,23}/   ← 1-frame-per-minute extractions
│   ├── 2.Custom Yolo model training/  ← YOLOv8n training run
│   │   ├── data.yaml, yolov8n.pt      ← Roboflow dataset config + base weights
│   │   └── yolo_excavator_custom/weights/{best,last}.pt
│   ├── 3.Step 1 - YOLO/               ← detection output (detections.csv)
│   ├── 5.Step 2 - Deep Sort/          ← tracking output (Track_Output.csv)
│   ├── 6.Step 3 - Idling/             ← physics-based idling detector output
│   ├── 7.CVAT (excel to xml)/         ← manual labels, per recording day
│   │   ├── Tasks_all.xlsx, Tasks_all_updated.xlsx  ← master multi-sheet label files
│   │   └── Tasks_Day_{2..12}_{month}/  ← 12 per-day folders, Oct 2025 - Feb 2026,
│   │                                     each with battery-SoC logs alongside labels
│   ├── 8.Creating clips from CVAT Annotations/
│   ├── 9.Custom resnet model training/
│   │   ├── dataset_{mean,std}.npy     ← normalization stats
│   │   └── Multi Layer model/         ← unfreeze-depth ablation runs (5 configs)
│   ├── 10.Step 4 - Resnet/            ← inference output
│   ├── 11.Step 5 - Validation/        ← grid-search validation output
│   ├── Run from step 8 to 11/
│   │   ├── V1 Full video/             ← full-video inference + grid search (9000+ runs)
│   │   ├── V2 Validation only/        ← held-out validation across 11 videos
│   │   └── V3 Ablation Study/         ← architecture ablation (undocumented in README_CV.md)
│   ├── NRP Stuff/                     ← Kubernetes/Nautilus cluster configs
│   ├── Papers and references/         ← 7 source papers
│   └── Update PPTs/
│
├── Bayesian Regression/                ← activity power-draw fitting
│   ├── README.md
│   ├── Tasks_energy_loading_swinging.py            ← point-estimate (weighted least squares)
│   ├── Tasks_energy_loading_swinging_bayesian.py   ← full Bayesian posterior (PyMC/MCMC)
│   └── {Oct,Feb}_*_Tasks_*.xlsx       ← 23 per-day task logs, Oct 2025 - Feb 2026
│
├── MPC/                                ← Avik's original baseline
│   ├── README.md
│   └── Scenario 1/
│       ├── mcs_optimization_main_v4_real.jl      ← entry point
│       ├── helper functions/
│       │   ├── MCS_OPTIMAL_v4_real.jl            ← the MCSOptimizer module (the model)
│       │   └── DataLoader_v4_real.jl             ← the DataLoader module
│       └── simple_dataset/csv_files/  ← example single-CEV, single-MCS, single-site input
│           ├── ev_data.csv, mcs_data.csv, place.csv, parameters.csv
│           └── time_data.csv, travel_time.csv, work_flexible.csv
│
└── MPC_Shubhan/                        ← extended MPC work (this repo's main contribution)
    ├── Optimization.pdf                ← the paper-level MILP formulation
    ├── Understanding_Deterministic_vs_Stochastic_MPC.md
    ├── Approaches 1 and 2.pptx
    │
    ├── Approach 1/                     ← certainty-equivalent MPC
    │   ├── README.md                   ← master overview (12 sections + 5-level appendix)
    │   ├── RUN_ALL.jl                  ← one-click batch runner, 7 stages
    │   ├── batch_logs/                 ← one log per stage
    │   ├── Shrinking_Horizon/          ← single-day controller
    │   │   ├── code/                   0_Regression … 6_Shrinking_Horizon_main.jl,
    │   │   │                           run_soe_sweep.jl, Old Code - To not delete/
    │   │   ├── data/{input,synthetic}_data/   the 7 CSVs (real + hardcoded mirror)
    │   │   ├── docs/                   README.md, math_model.tex,
    │   │   │                           constraints_code_vs_model.txt
    │   │   └── output/{input,synthetic}/, output/input_testing/  ← 10-point SOE sweep
    │   ├── Receding_Horizon/           ← multi-day controller — same layout as above
    │   └── Comparison/                 ← 3-way comparison driver
    │       ├── Code/                   7_Comparison_main.jl, 8_ComparisonOutput.jl,
    │       │                           README.md, Input data similarity check.py
    │       ├── Input/                  auto-generated by the driver each run
    │       └── Output/
    │
    └── Approach 2/                     ← stochastic, scenario-based MPC
        ├── README.md                   ← same structure as Approach 1's, + §13 delta
        ├── RUN_ALL.jl
        ├── Shrinking_Horizon/          ← + code/2b_ScenarioSampler.jl
        ├── Receding_Horizon/           ← + code/2b_ScenarioSampler.jl
        └── Comparison/
```


---

## 3. Video Detection — excavator activity recognition

**Full docs:** `Video Detection/README_CV.md` (18 sections — this summary condenses all
of them; go there for exact code snippets, every configuration parameter, and the full
troubleshooting guide).

A comprehensive, end-to-end computer vision pipeline for detecting excavators in
construction site video and recognizing their activities using deep learning. Built to
support on-site productivity monitoring and EV charging optimization by identifying
when machines are working, idling, or travelling.

### 3.1 System architecture

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
    ├── Crops each confirmed track to 112x112 and saves to track_N/ folder
    └── Output: Track_Output.csv + track_metadata.csv + annotated video
         │
         ▼
[3] Idling Detection (Physics-Based)
    ├── Savitzky-Golay + rolling median smoothing of centroid & area signals
    ├── Sliding window (40 frames) checking std(dist) < 0.2 px & std(dArea)
    └── Output: Idling_segments.csv + per-track visualization plots
         │
         ▼
[4] Manual Labeling (Parallel Path)
    ├── Excel spreadsheets with time-range -> activity labels per day
    ├── CVAT Excel->XML converter produces CVAT-importable annotation files
    └── CVAT used for label review, correction, and export
         │
         ▼
[5] Dataset Generation
    ├── Resample video to 25 FPS (round-nearest)
    ├── For each resampled frame, run YOLO and crop excavator to 112x112
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
    ├── YOLO crop -> sliding 16-frame clip -> R3D-18 -> raw predictions
    ├── Physics idling override (travelling -> idling where physics says idle)
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

### 3.2 Key features

- **Custom excavator detection** — YOLOv8n fine-tuned on a Roboflow-annotated dataset
  (single class: `Excavator`) for accurate single-frame detection at 480x480.
- **Multi-object tracking** — DeepSORT (`max_age=35`, `n_init=4`) for consistent
  excavator IDs across frames, with per-track cropped frame exports at 112x112.
- **Physics-based idling detection** — signal-processing algorithm using
  Savitzky-Golay filtering, rolling-median smoothing, and a sliding window over
  bounding-box centroid displacement and area variance.
- **Temporal activity recognition** — PyTorch R3D-18 (3D CNN) fine-tuned from
  Kinetics-400, operating on 16-frame clips at 25 FPS, producing 5-class predictions.
- **Hybrid AI + physics override** — during inference, the idling-detection logic
  overrides AI predictions for frames the physics model classifies as stationary.
- **Finite State Machine (FSM) post-processing** — enforces valid activity transition
  sequences and minimum dwell times.
- **Majority voting smoothing** — temporal smoothing over a configurable window
  (default 2 seconds) to reduce per-frame jitter.
- **CVAT integration** — tools to convert Excel-based manual labels to CVAT-compatible
  XML annotations.
- **Multi-video dataset builder** — batch clips extracted from 12 videos simultaneously
  across multiple recording days, resampled to 25 FPS with YOLO crop on each frame.
- **Comprehensive validation** — grid search across 6 hyperparameters (~9,000+
  combinations) comparing predictions against ground-truth Excel sheets.
- **Productivity reports** — automatic cycle detection, cycle-time statistics, and
  LCY/hr productivity estimates (paper Eq. 3).

### 3.3 The pipeline, step by step

**Step 0 (discarded) — OCR-based timestamp detection.** The first approach attempted
to use OCR (EasyOCR) to read the on-screen clock in the footage, extracting timestamps
at 1 frame/minute. Discarded: OCR accuracy was inconsistent across lighting/angle, and
a timestamp alone can't distinguish activity types. Replaced by CVAT-based manual
annotation with actual activity labels. What it did: extracted 1 frame/minute per
video day, auto-detected optimal OCR rotation (0/90/180/270°) and scale
(1.0/0.75/0.5) from sample frames, ran EasyOCR per frame, fell back to incrementing
the previous timestamp on OCR failure, and saved everything to a multi-sheet Excel.

**Step 1 — Custom YOLOv8 model training.** A custom YOLOv8n detector trained on a
Roboflow-annotated dataset to detect the single class `Excavator`. Dataset: workspace
`excavatar-research-project`, project `object-detection-jct8v` (v2), CC BY 4.0
license, 70/20/10 train/valid/test split (Roboflow default), bounding boxes drawn
around excavators in frames extracted at 1 frame/minute. Training config: base model
`yolov8n.pt`, 50 epochs, image size 480x480, batch size 16, 1 class. Output:
`yolo_excavator_custom/weights/{best,last}.pt` — `best.pt` (highest validation mAP) is
used in every subsequent step.

**Step 2 — Excavator detection with YOLO.** Runs the trained `best.pt` on the full
video frame-by-frame; only the highest-confidence box is kept per frame (one excavator
expected per scene). Output: `detections.csv` (frame, x1, y1, x2, y2, confidence,
class) plus an annotated video with bounding boxes and confidence scores.

**Step 3 — Multi-object tracking with DeepSORT.** Assigns persistent track IDs across
frames. Frames saved as 112x112 crops organized per track — this structure feeds
directly into the ResNet inference step. Config: `max_age=35` (frames a track can be
missed before deletion), `n_init=4` (frames before a new track is confirmed),
`CLIP_LENGTH=16` (frames per ResNet clip). Output: `Track_Output.csv`
(frame/track_id/x1/y1/x2/y2), `Tubes/track_N/*.jpg` per-track crops,
`track_metadata.csv`, annotated video with track ID labels.

**Step 4 — Idling detection.** A physics-based detector analyzing centroid movement
and bounding-box area over time; if both are statistically stable within a sliding
window, the period is classified idling. Algorithm: (1) compute centroid `(cx, cy)`
and box area per frame; (2) smooth with Savitzky-Golay (window=11, polyorder=2) then
rolling median (window=5); (3) slide a 40-frame window computing
`dist = sqrt(Δcx² + Δcy²)` and `dA = |Δarea|`; (4) if `std(dist) < 0.2px` AND
`std(dA) < 0.5%` of mean area, mark the window Idling; (5) post-process: merge gaps
<1s, drop segments <3s. Output: `Idling_segments.csv` (track_id, start/end frame,
duration, start/end time in seconds and HH:MM:SS.mmm) plus per-track PNG plots of raw
vs. smoothed signals and the idle mask.

**Step 5 (optional) — Manual labeling via Excel.** Before dataset generation, each
video was manually labelled in Excel with `Time` -> `Activity` rows (e.g.
`00:00-00:15 | Digging`). Stored in `7.CVAT (excel to xml)/`: `Tasks_all.xlsx` (master
multi-sheet file, all days), `Tasks_all_updated.xlsx` (corrected version), and
individual per-day folders `Tasks_Day_2_Oct_21/` through `Tasks_Day_12_Feb_13/`. These
files double as ground truth for the validation step.

**Step 6 — CVAT annotation (Excel to XML).** Converts the Excel time-range labels into
CVAT-compatible XML (CVAT 1.1 format) importable into CVAT for review and export.
Config: `EXCEL_PATH`, `VIDEO_FOLDER`, `DEFAULT_ACTIVITY="Idling"` (fills gaps between
labelled segments). Process: reads each sheet (one sheet = one video clip), parses
time ranges (`"1:30 - 2:45"` -> seconds), fills unlabelled gaps with the default
activity, converts seconds -> frame numbers via video FPS, generates CVAT XML with
`<track>` elements and `<box>` keyframes, saves one `*_raw.xml` per sheet. CVAT
workflow: create task + upload video, import the XML via Actions -> Upload
annotations, review/correct in the CVAT web UI, export corrected annotations.

**Step 7 — Dataset generation from CVAT annotations.** Builds the R3D-18 training
dataset by extracting 16-frame clips from labelled segments, YOLO-cropped to focus on
the excavator. Config: `VIDEO_XML_PAIRS` (12 video/annotation pairs, Day 2 through Day
12), `YOLO_MODEL="best.pt"`, `OUTPUT_DIR="Dataset_Resnet_3/"`, `CLIP_LENGTH=16` (paper:
16), `TARGET_FPS=25` (paper: 25), `CLIP_STRIDE=3` (81% overlap between clips),
`MIN_CONFIDENCE=0.5`, `CROP_SIZE=112` (paper: 112x112). Processing: resample video to
25 FPS via round-nearest indexing (identical method to inference); per resampled
frame, run YOLO and crop the highest-confidence detection; buffer a frame stream
tagged with CVAT activity labels; emit 16-frame clips where all 16 frames share one
activity label and have valid detections; save as
`activity_name/clip_NNNNN/frame_000.jpg ... frame_015.jpg`. Dataset spans 12 videos
from Days 2-12, October 2025 and February 2026.

**Step 8 — Custom 3D ResNet model training.** Trains the activity recognition model,
following Cho et al.'s methodology with documented deviations. Run on the NRP
(Nautilus) Kubernetes cluster with GPU. Configuration:

| Parameter | Value | Paper Match? |
|---|---|---|
| Architecture | R3D-18 (torchvision) | matches (3D ResNet) |
| Pre-training | Kinetics-400 weights | matches |
| Input shape | (3, 16, 112, 112) | matches |
| Target FPS | 25 | matches |
| Batch size | 16 | matches |
| Learning rate | 1e-3 | matches |
| Optimizer | Adam | not specified in paper |
| Epochs | 20 | not specified in paper |
| Classes | 5 | paper has 3 |
| LR scheduler | ReduceLROnPlateau | not specified in paper |
| Mixed precision | AMP (autocast) | not in paper |

Fine-tuning: all backbone layers frozen initially; only `layer4` and `fc` unfrozen
(~18% of total parameters trainable). Data augmentation (per paper §4.5): horizontal
flip (50%), channel shift (random ±0.1 per-channel brightness), affine shear (random
factor [-0.15, 0.15], 70% probability), spatial resize to 128x128 then random crop to
112x112 (train) / center crop (val). Class balancing via `WeightedRandomSampler`, 80/20
train/val split **at the video level** (not frame level, to avoid data leakage).
Normalization: dataset-specific mean/std computed from 200 training clips, saved as
`dataset_{mean,std}.npy` — must match at inference. Output:
`resnet3d_best_kinetics_2.pth` (model/optimizer/scheduler state dicts, activity names,
config dict, best-epoch metrics) plus `training_history.json`.

**Step 9 — Activity recognition (inference).** Runs the full hybrid pipeline on a new
video, on the NRP cluster (paths under `/mnt/nvme1/avik_shubhan/`). Pipeline: (1) load
model config from checkpoint; (2) resample video to 25 FPS via round-nearest + `cap.grab()`
for efficient skipping; (3) per frame — YOLO -> crop 112x112 -> RGB -> buffer into a
sliding 16-frame window; (4) normalize with training stats -> R3D-18 -> softmax -> top
prediction + confidence; (5) physics override — compute idling mask from bbox history
(same algorithm as Step 4), override `travelling` predictions to `idling` where the
mask says stationary; (6) majority voting over a 50-frame (2s @ 25 FPS) sliding window;
(7) cycle detection — each pair of consecutive digging-start events = one work cycle;
(8) productivity = `cycles_per_hr x bucket_payload_lcy`. Output: `frame_predictions.csv`
(Frame, Time_s, Activity, Confidence), `cycles.json` (per-cycle activity breakdown),
`summary.json` (aggregate stats), `activity_timeline.png`.

**Step 10 — Validation & hyperparameter optimization.** A grid-search pipeline: runs
AI inference **once**, caches raw predictions, then exhaustively tests every
combination of post-processing hyperparameters against manually labelled ground
truth. Ground truth: `Tasks.xlsx` (Step 5 format), time ranges parsed to frame
indices, per-frame GT vs. prediction comparison. Grid (6 parameters):

| Parameter | Values tested |
|---|---|
| `min_activity_duration_s` | 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0 |
| `dist_threshold` | 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6 |
| `idle_window` | 20, 24, 30, 36, 40, 48, 56, 64, 80 |
| `fsm_min_dwell_seconds` | 1.0, 1.5, 2.0, 2.5, 3.0 |
| `enable_fsm` | True, False |
| `override_travelling_only` | True, False |

FSM activity-transition table (transitions outside this table are rejected unless
model confidence > 0.95, the override threshold):

```
travelling  ->  idling, swinging
idling      ->  travelling, digging, swinging
digging     ->  loading, swinging
loading     ->  swinging
swinging    ->  digging, loading, idling, travelling
```

Inference caching: first run does the heavy AI pass and saves to
`raw_inference_cache.xlsx`; subsequent grid-search runs load from cache, skipping the
model forward pass entirely. Per-run output (`optimization_runs/Run_NNN/`):
predictions.csv, timeline PNG (GT vs. prediction), confusion-matrix PNG. Master
report: `master_optimization_report.csv` — every combination's accuracy%, cycles/hr,
and productivity.

### 3.4 Model architecture

**YOLOv8n (detection):** YOLOv8 Nano, base weights `yolov8n.pt` (COCO), fine-tuned on
the Roboflow excavator dataset, 480x480 input, 1 class (`Excavator`), 50 epochs, full
fine-tune strategy.

**R3D-18 (activity recognition):** 3D ResNet-18 (`r3d_18`), pretrained on Kinetics-400,
input shape `(1, 3, 16, 112, 112)`, frozen `stem`/`layer1`/`layer2`/`layer3`, trained
`layer4`/`fc`, 5 output classes, target FPS 25, temporal receptive field 16 frames =
0.64s @ 25 FPS, dataset-specific mean/std normalization.

### 3.5 Activity classes

| Class | Index | Description |
|---|---|---|
| `digging` | 0 | Bucket lowered into the ground, excavating material |
| `idling` | 1 | Machine stationary; engine running but no work |
| `loading` | 2 | Bucket filled, being lifted before swing |
| `swinging` | 3 | Upper structure rotating to dump or return position |
| `travelling` | 4 | Machine moving from one location to another |

Typical activity distribution (Day 3 inference, `summary.json`): Digging 28.99%,
Swinging 26.49%, Travelling 18.82%, Idling 18.65%, Loading 7.05%.

### 3.6 Productivity & cycle-time calculation

Per paper §4.5 Eq. 3: `Productivity (LCY/hr) = Cycles_per_hour x Bucket_payload (LCY)`.
A work cycle is defined as the period from one digging start to the next digging
start, typically composed of digging (excavation), loading (bucket lift, sometimes
merged with digging), swinging loaded (rotation to dump zone), and swinging empty
(rotation back to dig zone). Day 3 results: 3,320s (55.3 min) total duration, 77
cycles detected, 39.75s average cycle time, ~90.6 cycles/hour, 1.5 LCY bucket payload
-> **135.84 LCY/hr productivity**, 94.1% average model confidence.

### 3.7 Full performance results

**Dataset summary (Step 8):** 11 source videos (Day 2, Day 3, Day 4_1, TC_00011 through
TC_00021), **306,780** total 16-frame clips, 261,281 training (85%) / 45,499
validation (15%), 0.64s clip duration @ 25 FPS, 112x112 YOLO-cropped, split by video
group (no frame leakage). Class distribution:

| Activity | Train clips | Val clips |
|---|---|---|
| Digging | 32,257 | 5,721 |
| Idling | 126,180 | 22,242 |
| Loading | 10,743 | 1,788 |
| Swinging | 62,252 | 10,485 |
| Travelling | 29,849 | 5,263 |
| **Total** | **261,281** | **45,499** |

Normalization stats (200 training clips): mean `[0.2725, 0.3182, 0.3059]`, std
`[0.1956, 0.1834, 0.1846]`.

**Training results (R3D-18):** 20 epochs, Adam @ LR=0.001, AMP, batch 16. Selected
epochs:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Avg Precision | Avg Recall |
|---|---|---|---|---|---|---|
| 1 | 0.4100 | 84.95% | 0.4458 | 84.44% | 76.1% | 80.6% |
| 4 | 0.1149 | 95.93% | **0.3498** | **89.68%** | 84.1% | 83.4% |
| 9 | 0.0443 | 98.44% | 0.3936 | 90.13% | 83.0% | 86.1% |
| 15 | 0.0215 | 99.22% | 0.3928 | 90.96% | 84.9% | 85.9% |
| 18 | 0.0162 | 99.40% | 0.4118 | 91.22% | 85.0% | 86.8% |
| 20 | 0.0151 | 99.45% | 0.4357 | 91.38% | 86.4% | 85.2% |

**Best val loss was at epoch 4** (0.3498, 89.68% acc) — training loss kept dropping to
0.015 while val loss plateaued/rose to 0.40-0.48 (overfitting past epoch ~4-9). Best
checkpoint saved as `resnet3d_best_kinetics_2.pth`. Final-epoch (20) per-class metrics:

| Activity | Precision | Recall |
|---|---|---|
| Digging | 87.3% | 90.3% |
| Idling | 97.0% | 97.3% |
| Loading | 73.9% | 67.5% |
| Swinging | 86.3% | 87.6% |
| Travelling | 87.7% | 83.1% |
| **Average** | **86.4%** | **85.2%** |

**Hyperparameter optimization (V1 — grid search, ~9,000+ combinations):** Best
configuration `PARAM_SET_9063`, 88.81% overall accuracy:
`min_activity_duration_s=2.0, dist_threshold=0.05, idle_window=36,
fsm_min_dwell_seconds=1.0, enable_fsm=false, override_travelling_only=true`.

| Activity | Accuracy |
|---|---|
| Digging | 87.82% |
| Idling | 98.64% |
| Loading | 81.78% |
| Swinging | 74.60% |
| Travelling | 80.94% |
| **Overall** | **88.81%** |

**Full validation (V2 — 11 videos, best params locked):**

| Video | Val Frames | Correct | Accuracy |
|---|---|---|---|
| Day_2 | 13,470 | 10,660 | 79.14% |
| Day_3 | 39,207 | 32,877 | 83.85% |
| Day_4_1 | 2,432 | 2,009 | 82.61% |
| TC_00011 | 5,937 | 5,604 | 94.39% |
| TC_00012 | 25,932 | 22,513 | 86.82% |
| TC_00013 | 365 | 365 | 100.00% |
| TC_00014 | 16,600 | 15,273 | 92.01% |
| TC_00015 | 1,413 | 1,298 | 91.86% |
| TC_00016 | 23,512 | 22,602 | 96.13% |
| TC_00019 | 21,340 | 19,050 | 89.27% |
| TC_00021 | 18,893 | 16,431 | 86.97% |
| **Overall** | **169,101** | **148,682** | **87.92%** |

Per-class classification report (169,101 val frames):

| Activity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Digging | 0.851 | 0.879 | 0.865 | 21,193 |
| Idling | 0.918 | 0.981 | 0.949 | 80,675 |
| Loading | 0.640 | 0.728 | 0.681 | 7,457 |
| Swinging | 0.899 | 0.724 | 0.802 | 39,853 |
| Travelling | 0.814 | 0.833 | 0.823 | 19,923 |
| **Macro avg** | **0.824** | **0.829** | **0.824** | 169,101 |
| **Weighted avg** | **0.881** | **0.879** | **0.877** | 169,101 |

Confusion matrix (rows = ground truth, cols = predicted):

```
             digging   idling   loading   swinging   travelling
digging       18,631      768       572        957          265
idling           201   79,171       104        501          698
loading          960      210     5,427        794           66
swinging       1,752    4,229     2,240     28,862        2,770
travelling       347    1,843       133      1,009       16,591
```

**Comparison to source paper (Cho et al., 3-class):**

| Metric | Paper (3 classes) | This Implementation (5 classes) |
|---|---|---|
| Architecture | Custom 3D ResNet | R3D-18 (torchvision) |
| Pre-training | Kinetics-400 | Kinetics-400 (match) |
| Input | 16x112x112 @ 25 FPS | 16x112x112 @ 25 FPS (match) |
| Avg accuracy | 87.6% | **87.92%** (val) / **89.68%** (best epoch) |
| Digging precision | 95% | 85.1% |
| Swinging precision | 86% | 89.9% |
| Loading precision | 84% | 64.0% |

Key observations: overall accuracy **matches and slightly exceeds** the paper (87.92%
vs 87.6%) despite the harder 5-class problem. Idling is the strongest class (98.1%
recall — physics override + high support both help). Loading is the weakest class
everywhere (64% precision, 73% recall — least training data of any class, and
visually similar to digging/swinging transitions). Post-processing (idling physics +
2s majority voting, FSM disabled) is roughly accuracy-neutral relative to raw
classifier output. Per-video accuracy varies widely (79-100%), with older field days
(Day_2) harder — likely different camera angles or equipment.

**A third run variant not yet in the main results writeup:**
`Run from step 8 to 11/V3 Ablation Study/` — a full architecture ablation sweeping
r3d_18 vs. r2plus1d_18, temporal stride 1 vs. 2, self-supervised pretraining on/off,
progressive vs. full layer unfreezing, and label smoothing 0.05 vs. 0.15 (trial_4
through trial_6, with per-config training-history JSONs and aggregated comparison
CSVs). This exists on disk but isn't folded into README_CV.md's §12 results section.

### 3.8 Compute — NRP/Kubernetes cluster

Training (Step 8) and inference (Steps 9-10) run on the UCSD **National Research
Platform (NRP/Nautilus)** Kubernetes cluster. Configs in `Video Detection/NRP Stuff/`.
Quick-start workflow:

```bash
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

# 6. Copy results back
kubectl cp shi-group-ece-advanced/resnet3d-train:/data/models/resnet3d_best_kinetics_2.pth .

# 7. Delete pod when done
kubectl delete pod resnet3d-train -n shi-group-ece-advanced
```

Persistent storage layout: `/mnt/nvme1/avik_shubhan/resnet3d/` holds
`Dataset_Resnet_3/`, the model checkpoint, `best.pt`, normalization stats, source
videos, and per-video inference output folders. **Everything outside the mounted PVC
is lost when the pod terminates.**

### 3.9 Requirements

```
opencv-python>=4.8.0, torch>=2.0.0, torchvision>=0.15.0, numpy>=1.24.0,
pandas>=2.0.0, openpyxl>=3.1.0            # core
ultralytics>=8.0.0, deep-sort-realtime>=1.3.2   # detection & tracking
lxml>=4.9.0                                # annotation tooling
scipy>=1.10.0                              # signal processing
matplotlib>=3.7.0, scikit-learn>=1.3.0     # visualization & reporting
tqdm>=4.65.0, pillow>=10.0.0               # utilities
easyocr>=1.7.0, xlwings>=0.30.0            # optional, legacy OCR script only
```

Install via `pip install -r Video Detection/Misc/Requirements.txt` or
`conda env create -f Video Detection/Misc/smartgrid.yml`.

### 3.10 Performance tips & troubleshooting (condensed)

Ensure CUDA is detected (`torch.device("cuda")`, verify with `nvidia-smi`). Increase
`BATCH_SIZE` if VRAM allows (AMP already reduces memory pressure). `CLIP_STRIDE=3`
(dataset generation) gives 81% clip overlap; reduce to 1 for max data, increase to 8
for faster/sparser generation. Tune `min_activity_duration_s` (majority-vote window)
in the 1.5-2.5s range to reduce jitter without over-smoothing. Enable the FSM if raw
output shows physically impossible sequences. The inference frame generator uses
`cap.grab()` to skip non-target frames without decoding, which matters a lot on
high-FPS source video. For low detection accuracy: more diverse training images, try
`yolov8s`/`yolov8m`, or 640x640 input. For track-ID switching: raise `max_age`
(50-100) and `n_init`. For poor activity recognition: verify class balance,
lengthen `CLIP_LENGTH`, add temporal/color jitter augmentation, unfreeze more layers,
or try `r3d_50`. `dataset_mean.npy`/`dataset_std.npy` **must match** across training,
clip creation, and inference — recompute and redistribute if you retrain.

### 3.11 References

Source papers in `Video Detection/Papers and references/`: **Cho, Latif, Sharafat,
Seo** (primary — 3D ResNet for excavator activity recognition, defines the 3-class
model, 16x112x112 @ 25 FPS, Kinetics-400 fine-tuning, cycle-time methodology); Chen,
Zhu, Hammad (construction equipment activity monitoring); Fard, Heydarian, Niebles
(video-based construction worker/equipment analysis); Ghelmani, Torabi, Hammad, Chen
(site equipment monitoring pipeline); Hong, Song, Hong, Kim, Jeong (deep learning for
construction site monitoring); Akın, Beyazıt, Kleissl, Shi (EV charging optimization
for on-site equipment); a grid-level EV integration study (Power Flow Security
Maximization).

---

## 4. Bayesian Regression — activity power fitting

**Full docs:** `Bayesian Regression/README.md`

Fits per-activity power draw (kW) for the CEV from field task logs, by regressing
observed battery State-of-Charge (SoC) drop against time spent on each activity. The
fitted values are the `mu`/`sigma` values consumed by every MPC/MILP model downstream.

### 4.1 The core method (shared by both scripts)

For each task-log Excel sheet (one sheet per recording day/site):

1. **Read** `Start time (actual)`, `End time (actual)`, `Activity`, `SoC` columns.
2. **Bucket rows** by accumulating activity time until cumulative `|ΔSoC|` from the
   bucket's anchor reaches `MIN_DELTA_SOC` (default 3%), then emit one equation for
   the whole bucket and start a fresh bucket. Cumulative-bucket attribution preserves
   total energy: sub-threshold drops merge into the next equation along with their
   activity time, so no SoC drop is discarded. `MIN_DELTA_SOC=1` reproduces
   one-equation-per-step. Larger buckets trade equation count for signal-to-noise (1%
   steps carry ~50% relative quantization noise; 2% steps carry ~25%).
3. **Stack** all equations from all days into `A z = b`: `A` = hours per activity per
   equation, `b` = energy drop in kWh per equation (`-ΔSoC · battery_cap / 100`), `z` =
   unknown per-activity power (kW).
4. **Solve** for `z` — weighted least squares (point-estimate script) or full Bayesian
   posterior (Bayesian script) — with idling power fixed to 0 in both.

Two operating modes, controlled by a `grading` flag: `grading="False"` folds
Grading 1/2 into Digging/Loading+Swinging (5 activities: Digging, Loading+Swinging,
Travelling, Idling, Mixing); `grading="True"` keeps them separate (7 activities).

### 4.2 Point-estimate script (`Tasks_energy_loading_swinging.py`)

Solves `min ||sqrt(W)·(Az - b)||² + reg·||z||²` via `cvxpy` with the MOSEK solver,
`z >= 0`. `W` = per-equation weight matrix, selectable via `WEIGHT_SCHEME`:

| Scheme | Formula | Effect |
|---|---|---|
| `uniform` | `w_i = 1` | no weighting (default) |
| `linear` | `w_i = |b_i|` | sharp; ~6x weight for 6% vs 1% ΔSoC |
| `bounded_median` | `w_i = min(|b_i|/median(|b|), 1)` | capped above median; only sub-median equations penalized |
| `quadratic` | `w_i = b_i²` | very sharp; ~36x for 6% vs 1% |

Weights are normalized to mean 1 so the regularization term stays comparable across
schemes without retuning `reg_param`.

**Uncertainty estimation:** 200 repeated random 80/20 train/test splits, plus a
parallel repeated 5-fold CV (5 folds x 200 repeats = 1000 fits, giving every equation
exactly 200 test-fold appearances vs. the ShuffleSplit's binomial coverage). Reports
per-split MAE/RMSE/MAPE/NMAE, and per-coefficient mean ± SD with a 95% empirical
interval (2.5th/97.5th percentile across splits). **Headline coefficients** come from
one final fit on all `m` equations — the repeated splits are for uncertainty
estimation only, not the reported point estimate. Diagnostic outputs also include:
activity time-share of the full dataset, a correlation matrix/heatmap between
activities (co-occurrence), an observed-vs-median-predicted scatter with 95% bands
aggregated over all held-out appearances, and per-coefficient violin plots across the
200 splits with the full-data fit marked as a reference line.

### 4.3 Bayesian script (`Tasks_energy_loading_swinging_bayesian.py`)

Same equation-building pipeline, but fits `z` (and residual noise `sigma`) as a full
posterior via PyMC/MCMC: NUTS sampler, 4 chains x 2000 draws, 2000 tuning steps,
`target_accept=0.9`. Configurable priors per activity (`x_prior_config`) — supports
Normal, truncated Normal, LogNormal, Gamma, and Exponential distributions on each
`z_i` (via `build_prior_x`), plus a configurable prior on the noise scale `sigma`
(Half-Normal, Exponential, or Half-Student-t, via `build_prior_sigma`). Reports
posterior means/intervals via `arviz` instead of split-based point estimates, and
computes predictive-interval coverage on held-out equations
(`predictive_interval_coverage_from_trace`). **This is the version whose output
(posterior mean + std per activity) feeds the MPC's stochastic `mu`/`sigma` inputs** —
the point-estimate script's output is a single number per activity with no native
uncertainty quantification suitable for a stochastic controller; the Bayesian script's
posterior std is.

### 4.4 Inputs

23 per-day task-log Excel files spanning October 2025 - February 2026, across three
site materials:

| Period | Site material | Files |
|---|---|---|
| Oct 21-23, 2025 | Soil | `Oct_21_Tasks_1`, `Oct_22_Tasks_1..5`, `Oct_23_Tasks_1` |
| Feb 02-03, 2026 | Soil | `Feb_02_Tasks_1..3`, `Feb_03_Tasks_1..2` |
| Feb 04, 2026 | Concrete | `Feb_04_Tasks_1..3` |
| Feb 11, 2026 | Concrete | `Feb_11_Tasks_2..3` |
| Feb 11-13, 2026 | Sand | `Feb_11_Tasks_1`, `Feb_12_Tasks_1..4`, `Feb_13_Tasks_1` |

**Known gap:** both scripts currently hardcode local file paths
(`/Users/avikghosh/Desktop/CEV-Analysis/Analysis/...xlsx`) to these files. They need
to be parameterized/updated before running on another machine. Both scripts also
auto-relaunch themselves under a specific conda env (`CEV_MCS`) via `os.execv` if not
already running in it, and clear IPython/terminal state at the start of each run.
Battery capacity is hardcoded as `Battery_cap = 14.8` (kWh), matching the CEV's
`SOE_max` used throughout the MPC side.

### 4.5 Requirements

`pandas`, `numpy`, `matplotlib`, `seaborn`, `cvxpy` (+ MOSEK license), `scikit-learn`,
`scipy`. The Bayesian script additionally needs `pymc`, `arviz`, `xarray`.

---

## 5. MPC — original baseline (Avik, Scenario 1)

**Full docs:** `MPC/README.md`

Avik's original single-scenario MILP jointly scheduling MCS routing/charging and CEV
work — the reference implementation everything in `MPC_Shubhan` extends and is
documented relative to. Solves **one full-day MILP, once**, with no re-planning loop —
the plan made at the start of the day is the plan that gets carried out.

### 5.1 Files

| File | Role |
|---|---|
| `mcs_optimization_main_v4_real.jl` | Entry point — loads data, builds the time grid, solves, exports plots/CSVs |
| `helper functions/MCS_OPTIMAL_v4_real.jl` | The `MCSOptimizer` module — full JuMP model and all plotting/export helpers |
| `helper functions/DataLoader_v4_real.jl` | The `DataLoader` module — reads the 7 input CSVs |
| `simple_dataset/csv_files/*.csv` | Example single-CEV, single-MCS, single-site input dataset |

### 5.2 Running it

```julia
julia mcs_optimization_main_v4_real.jl                    # defaults to simple_dataset/
julia mcs_optimization_main_v4_real.jl <dataset_folder>    # any folder with a csv_files/ subdir
julia mcs_optimization_main_v4_real.jl --all               # run every dataset folder in the cwd
```

`optimizer_choice` at the top of the main script selects which model variant to
include (`"OPTIMAL"` = the full model documented here; `"B1"`-`"B4"` reference other
variant files not included in this repo — likely earlier/alternative formulations).
Solver time limit defaults to 1 hour (`time_limit_sec`); `Inf` for unlimited.

### 5.3 Model summary

Single JuMP/HiGHS MILP over the whole scheduling horizon (`T` boundary indices, `K`
interval indices, `delta_T`-hour intervals — default 15 min). Decision variables:

- **MCS side:** `P_ch_MCS`/`P_dch_MCS` (charge/discharge power per node),
  `P_ch_tot`/`P_dch_tot`/`L_trv_tot` (aggregates), `z` (presence), `x` (departure
  along an edge), `y_trv` (in-transit indicator), `beta_arr`/`beta_dep`
  (arrival/departure events), `g_ch` (grid-charging active), `SOE_MCS`.
- **CEV side:** `P_MCS_CEV` (per-plug transfer), `P_work`, `s_miss_work` (missed-work
  slack), `u[E,N,B,K]` (binary activity selection), `rho` (MCS-to-CEV connection),
  `mu` (CEV charging-ready indicator), `SOE_CEV`.
- **Shared:** `P_peak_NC`, `P_peak_OP` (non-coincident and on-peak demand trackers).

**Objective:** grid energy cost + monetized carbon + non-coincident demand charge +
on-peak demand charge (4-9 PM window) + missed-work penalty (`rho_miss`) + MCS travel
labor cost (`rho_labor`).

**Constraints present:** charging/discharging capacity and plug-count limits, CEV
work-power capacity, SOE dynamics and bounds, **exact** terminal SOE equality
(`SOE[·,last] == SOE_ini` for *both* MCS and CEV — not a floor; the CEV side is later
relaxed to a floor in `MPC_Shubhan` because an exact equality can become unrecoverable
once a stochastic plant lets a CEV drift high, since it can't discharge below what
work already consumed), MCS routing/presence/arrival-departure bookkeeping (including
travel-time delay via `tau_trv`), cumulative digging-before-loading **precedence**
(`scale=2`: cumulative loading can't outpace `scale x` cumulative digging), and the
two-sided **travel-pacing** band (`work_per_travel=4`: `4V <= W <= 4V+4` on cumulative
CEV travel `V` vs. cumulative useful work `W`).

**Not present in this baseline** (both added later in `MPC_Shubhan`): the operator
rest-rule constraint (limiting continuous work without a break) and the closed-loop
MPC re-solve loop itself — this version plans the entire day in one shot and never
reacts to realized deviations mid-day. There is also no `pacing_tol` numerical
tolerance on the travel-pacing floor (that fix only became necessary once
`MPC_Shubhan` introduced energy-based work-duration capping — see §8).

### 5.4 Inputs (`simple_dataset/csv_files/`)

| File | Contents |
|---|---|
| `ev_data.csv` | Per-CEV `SOE_min/max/ini`, charge rate (`ch_rate`), work capacity (`work_cap`) |
| `mcs_data.csv` | Per-MCS `SOE_min/max/ini`, charge/discharge rates, plug count (`C_MCS_plug`), efficiency (`eta_ch_dch`) |
| `place.csv` | Per-site CEV assignment and required digging/loading hours |
| `parameters.csv` | Scalar model parameters (`k_trv`, `rho_miss`, `rho_labor`, `scale`, etc.) |
| `time_data.csv` | Per-interval carbon intensity and electricity price |
| `travel_time.csv` | Node-to-node MCS travel time matrix |
| `work_flexible.csv` | Per-interval CEV work-availability mask (0/1 per 15-min slot — e.g. encodes a lunch-break blackout window) |

### 5.5 Outputs

Per-run results directory (`<dataset>/results/`): cost/energy/CO2/KPI summary plots,
per-MCS power profile plots + CSVs, cumulative cost/CO2 timeseries, and a **parsed
HiGHS MIP-progress log** — `parse_highs_mip_log` reads the solver's live progress
table (BestBound/BestSol/Gap/Time rows) from a log file HiGHS mirrors its output to,
and turns it into a tidy DataFrame for convergence diagnostics — including handling
HiGHS's k/m/g/t magnitude-suffix abbreviations and `inf`/`-inf`/`Large` tokens.

### 5.6 Relationship to the rest of the repo

This is the reference implementation `MPC_Shubhan/Approach 1`'s `3_MCSModel.jl` /
`4_MPCLoop.jl` were built from. Constraint numbering and naming conventions in that
codebase's documentation (e.g. "Eq. 13", "`work_per_travel = 4 (Avik)`") refer back
directly to this file. See `MPC_Shubhan/Optimization.pdf` for the paper-level
formulation these constraints implement.

---

## 6. MPC_Shubhan — Approach 1 (certainty-equivalent MPC)

**Full docs:** `MPC_Shubhan/Approach 1/README.md` (12 sections + a 5-level appendix on
planning-vs-reality) — this section reproduces its full argument, condensed.

### 6.1 The whole thing in plain words

You have **one big battery on a truck** — the MCS. You have a couple of **electric
excavators** parked at construction sites. Excavators dig, which drains their
batteries, and they can't drive to a charger, so the truck has to drive to *them*.

Your job is to plan the truck's whole day — where it goes, when it plugs into the
grid, when it drives to a site, which excavator it charges and for how long — while
simultaneously planning what each excavator is *doing* every 15 minutes: digging,
loading, repositioning, or resting. You want required digging/loading finished,
nobody's battery flat or overfull, everyone back to starting charge by morning, and
the electricity bill as small as possible.

**Why this isn't just one calculation.** Two reasons. First, electricity isn't one
price — cheap at night, expensive in the late afternoon, plus a charge on your single
biggest spike of the month and again on your biggest spike during peak hours, so
*when* the truck charges matters enormously. Second, and this is the real point: **you
don't know exactly how much power digging takes.** It depends on soil, operator,
machine. You have a good estimate (~4.8 kW from the Bayesian Regression fit), but
reality varies around it, and small errors accumulate across a day.

**How the controller handles that.** It doesn't plan the day once and hope. It plans,
acts a little, looks, and re-plans: *every 15 minutes, work out the best possible plan
for the rest of the day, do only the first 15 minutes of it, measure what actually
happened to the batteries, throw the rest of the plan away, plan again from the real
state.* That's Model Predictive Control. The plan is constantly wrong, and it
constantly doesn't matter, because it's constantly being rebuilt from what actually
happened — 96 full plans built and discarded to produce one day of decisions. Each
plan is a MILP (mixed-integer program: yes/no decisions like "is the truck at this
site?" plus continuous ones like "how many kW?"), solved by HiGHS.

**The power estimate** comes from a one-time statistical fit (§4 above) done before
any of this starts, reading real recorded task data and working out, per activity, the
average power draw and how much it varies. The average is what the planner plans on;
the variation is what simulated reality wobbles by. The fit happens **once, before the
run**, and stays frozen — the controller does not re-learn the power model as the day
goes on (the machinery exists but is deliberately not called).

**The honest bit about "reality."** There's no actual truck. "Reality" is simulated,
and the code keeps two worlds strictly apart: the **planner** always believes digging
takes exactly the average; the **plant** (simulated reality) decides what actually
happened this interval. The plant has two settings — this is the switch you'll flip
most often (§6.3 below).

**Comparing strategies fairly.** Every run also computes a baseline: **Approach 0** —
plan the entire day once at 8am, then carry it out, never re-plan no matter what
happens — versus **Approach 1**, the real controller re-planning every 15 minutes.
Both face the **same** sequence of random numbers, drawn from one shared pool before
either starts, so any cost difference is down to strategy, not luck.

**Two flavours of controller: Shrinking Horizon** plans one day — at 8am it plans 24
hours ahead, by noon 20 hours ahead, near the end just the last interval; the window
*shrinks*. **Receding Horizon** handles several days — a fixed-width window slides
forward so it always sees roughly a day ahead, including into tomorrow; work is issued
as a per-day quota that rolls over if a day falls behind.

The **comparison driver** runs Approach 0, Shrinking, and Receding on identical input
from one shared random pool and draws all three on the same charts. **`RUN_ALL.jl`**
runs the whole lot — both controllers on both datasets, both sensitivity sweeps, and
the comparison — in one click.

### 6.2 The problem, formally

One MCS serves a fleet of CEVs fixed at construction sites. The MCS charges from
**grid connection nodes** and discharges to CEVs at **construction nodes**; it's towed
between them, costing time and battery. Per 15-minute interval, the controller
chooses: MCS grid-charging power per grid node, MCS discharge power to each specific
CEV, MCS routing (depart/in-transit/parked, with travel time and energy), each CEV's
activity (digging, loading+swinging, travelling/repositioning, or idle), and which
CEVs are plugged in and charging — subject to battery physics/limits, plug counts,
work quotas, digging-before-loading precedence, an operator rest rule, travel pacing,
and end-of-horizon energy targets. Objective: time-of-use energy cost + monetized
carbon + non-coincident demand charge + on-peak demand charge + missed-work penalty +
towing labour. The uncertainty is in the per-activity power draw — digging,
loading+swinging, travelling, and idling each pull a different, only approximately
known number of kW.

### 6.3 Sampled vs. mean mode — the switch you'll flip most often

The planner is unaffected by this switch — the MILP always plans on the mean `μ`.
What changes is what the *simulated plant* does when the plan is carried out.

| | `:sampled` | `:mean` |
|---|---|---|
| Realized power | next unused draw from the shared pool | pinned to the planning mean `μ` |
| Activity split within an interval | may be randomized (`multi_activity`) | the single planned activity, full interval |
| Realized vs. planned | drifts apart | **identical, exactly** |
| Consumes pool samples? | yes, advances the cursor | **no**, cursor untouched |
| Needs a seed to reproduce? | yes | no |

**Why `:mean` exists:** when the plan and reality agree perfectly, the cost you measure
at the end **is** the cost the optimizer promised at the start — there's nothing else
it could be. `:mean` gives a trustworthy reference cost with no randomness muddying
it — the clean reproduction of the deterministic single-shot model.

**What each choice measures.** Approach 1 is *always* `:sampled`; only Approach 0's
plant mode changes, which decides what a reported gap actually means:

```
  A0(:sampled) --> A1    =  value of re-planning against drift     (like-for-like plant)
  A0(:mean)    --> A1    =  drift AND re-planning together         (the net figure)
```

Running the batch both ways and differencing the two Approach 0 numbers isolates the
**cost of plan drift alone**, with no re-planning involved. Because a `:mean` run
consumes no samples, it can never perturb a `:sampled` run sharing the same pool.

**Where to set it:** `PLANT_MODE` at the top of `RUN_ALL.jl` (all stages);
`approach0_plant` kwarg on `run_scenario_1`/`run_comparison`; the `A0_PLANT` constant
in a standalone `run_soe_sweep.jl` (deferring to `MASTER_A0_PLANT` when the master
runner sets it).

**The diagnostic use.** Both `run_mpc` and `run_one_shot` accept `plant`, so a fully
deterministic closed loop is available for debugging — with `plant=:mean` on both
sides and `shrinking=true`, Approach 1 should reproduce Approach 0 exactly. If it
doesn't, the cause is a closed-loop seam or carry-in bug, not randomness. **This is
the first thing to run whenever the two approaches disagree by more than expected.**

### 6.4 Shrinking vs. Receding Horizon

Both are MPC in the general sense (re-solve, apply one interval, repeat); they differ
in how long the planning window is and whether it spans more than one day.

**Shrinking Horizon (single day):** plans the entire remaining day every step — window
`[now, next 08:00]`, so it shrinks from 96 intervals at 08:00 to 1 interval at 07:45
the next morning. One lumpsum work requirement per site. Overnight MCS recharge is
scheduled *inside the same MILP*, no separate phase. Every window reaches the day
boundary, so terminal targets always bind — simplest and fastest, the natural
baseline. A fixed-`H` lookahead exists behind `shrinking=false` but is **experimental**
— terminal rules are gated on reaching day-end, so under a fixed `H` they vanish from
all but the last `H` windows with nothing replacing them; don't use it for reported
results without adding a terminal cost first.

**Receding Horizon (multi-day):** plans a fixed-width sliding window — the rest of
today plus `lookahead_days` further day-blocks — so the plan always sees into
tomorrow. Work is a genuine per-day schedule (`dig_by_day`/`load_by_day`, optionally
from `work_by_day.csv`); unfinished work rolls over into the next day's backlog,
handled by the loop outside the MILP. Simulates the reported days *plus one dropped
buffer day*, purely so the last reported day still has a full day of lookahead — the
buffer day is fully simulated then discarded from all outputs, which matters for pool
sizing (§6.5). Terminal targets pin at the next day-start boundary, so each window
closes the current day's energy cycle. Like Shrinking, overnight recharge is inside
the same MILP.

| Aspect | Shrinking | Receding |
|---|---|---|
| Horizon | single day, window shrinks | multi-day, fixed-width window slides |
| Lookahead | none beyond today | `lookahead_days` further day-blocks |
| Work demand | one lumpsum per site | per-day schedule, rolls over via a backlog |
| Buffer day | none | one extra day simulated then dropped |
| Terminal targets | at day-end | at each next-day-start boundary |
| Overnight recharge | inside the MILP | inside the MILP |
| Entry point | `6_Shrinking_Horizon_main.jl` | `6_Receding_Horizon_main.jl` |

Shrinking for a clean single-day study or a fast baseline; Receding when work spans
multiple days, the schedule should anticipate tomorrow, or day-to-day carry-over of
work and energy matters.

### 6.5 The comparison driver

`Comparison/Code/7_Comparison_main.jl` runs **three** approaches on **one** dataset
from **one** shared random pool and overlays them on merged figures: **Approach 0**
(one-shot plan, executed open-loop — by default the Shrinking codebase's solver,
`approach0_source=:receding` picks the other), **Approach 1a** (Shrinking Horizon
closed loop), **Approach 1b** (Receding Horizon closed loop, run with `n_days=1` so
its single reported day is directly comparable to Shrinking's).

Three non-obvious design points: **Nothing is copied** — the driver `include`s both
codebases in place from their own folders, so edits are picked up on the next run and
no stale duplicate exists under `Comparison/`. **Both codebases are wrapped in their
own module** — they define identically-named submodules (`Common`, `DataLoader`,
`MCSModel`, `MPCLoop`, `Output`), so each is included inside a `RecedingApp` /
`ShrinkingApp` wrapper to stop one silently overwriting the other. **One shared pool
via a deliberate alias** — Julia dispatches on nominal types, so a pool built by one
app's `Common` would be *rejected* by the other app's `run_mpc` even with
field-for-field identical structs; the fix is that `ShrinkingApp` doesn't include its
own `1_Common.jl` and instead aliases `RecedingApp.Common` (safe — the two files
differ only by three multi-day helpers Shrinking never calls). **Pool sizing** —
`next_power!` *errors* when a cursor runs past the pre-drawn samples (it does not
wrap); since the Receding run always simulates a buffer day on top of its reported
days, the pool is sized `nK_day · (n_days_receding + 1) + 5` — sizing on one day alone
would leave roughly half the needed margin, and the failure would surface mid-run
after a long solve. Unconsumed samples cost nothing.

### 6.6 `RUN_ALL.jl` — the one-click runner

```julia
julia "C:\Users\shubh\Desktop\MPC\Approach 1\RUN_ALL.jl"
```

Runs seven stages in order:

| # | Stage | What it does |
|---|---|---|
| 1 | Shrinking — `:input` | real-CSV dataset, single day |
| 2 | Shrinking — `:synthetic` | built-in dataset, single day |
| 3 | Shrinking — sweep | 10 runs varying `SOE_CEV_ini`, writes `summary.html` |
| 4 | Receding — `:input` | real-CSV dataset, `N_DAYS_RECEDING` days |
| 5 | Receding — `:synthetic` | built-in dataset, `N_DAYS_RECEDING` days |
| 6 | Receding — sweep | 10 runs varying `SOE_CEV_ini`, writes `summary.html` |
| 7 | Comparison | 3-way, into `Comparison/Output/` |

Three edit points at the top of the file, in boxed comment banners: **(1) plant
mode** — `const PLANT_MODE = :sampled` propagates to all seven stages. **(2) which
stages to run** — comment out any `STAGES` line to skip it (the two sweeps are by far
the slowest — 10 full closed-loop runs each). **(3) shared settings** — `SEED`,
`TIME_LIMIT_SEC`, `MCMC_SAMPLES`, `H_SHRINKING`, `SHRINKING_MODE`, `N_DAYS_RECEDING`,
`RUN_REGRESSION`, `APPROACH0_SOURCE`, and paths. `RUN_REGRESSION=false` skips the
step-0 statistical fit and reuses the existing `parameters.csv` — much faster, and
necessary if the task `.xlsx` folder isn't present. Synthetic stages never run step 0.

**On `:synthetic`:** `DataLoader.load_data(:synthetic)` returns `build_default_data()`
— a scenario **hardcoded in `2_DataLoader.jl`**, reading no files. The CSVs under
`data/synthetic_data/` are a human-readable **mirror** of those same hardcoded values,
kept for inspection — not an input; nothing reads that folder. If you change the
synthetic scenario, edit `build_default_data()` *and* update the mirror CSVs to match
— a drifted mirror is worse than none.

**Why each stage gets its own Julia process:** both codebases define modules called
`Common`, `DataLoader`, `MCSModel`, `MPCLoop`, `Output`; loading them into one session
would have the second silently redefine the first's modules, making results depend on
load order. Separate processes give each stage a clean namespace, and mean a crashed
stage can't take the rest of the batch down. Console output streams live *and* is teed
to `batch_logs/<stage>.log`; a summary table of per-stage status and wall time prints
at the end; a failed stage is recorded and the batch continues.

### 6.7 File-by-file map

| File | Module | Role |
|---|---|---|
| `0_Regression.jl` | `Regression` | **Step 0.** Reads task `.xlsx` files, builds cumulative-ΔSoC energy-balance equations, fits `N(μ, σ)` per activity with NUTS, writes `p_*`/`sigma_*` into `parameters.csv`. Fail-soft if the folder is missing. `:input` only. |
| `1_Common.jl` | `Common` | Time/clock helpers, plot helpers, the Bayesian estimator type, and the `ActivityPowerPool` — the shared plant randomness. |
| `2_DataLoader.jl` | `DataLoader` | Loads the whole scenario into one immutable `NamedTuple d`. `:synthetic` (built-in) or `:input` (7 CSVs). |
| `3_MCSModel.jl` | `MCSModel` | `build_window_model(...)` — the window MILP, built and solved with JuMP + HiGHS. |
| `4_MPCLoop.jl` | `MPCLoop` | `run_mpc` (Approach 1, closed loop) and `run_one_shot` (Approach 0). Both share `apply_and_simulate!` for the apply/draw/advance step, so plant physics are identical and only the strategy differs. Both take the `plant` switch. |
| `5_Output.jl` | `Output` | Every artefact: figures, KPI and cost CSVs, solver diagnostics, replan grids, the Approach 0 vs 1 comparison report. |
| `6_<n>_main.jl` | — | Thin orchestrator: `run_scenario_1(; mode, …)`. Tees console to `run_log.txt`. Auto-runs unless `SCENARIO1_NO_AUTORUN` is set. |
| `run_soe_sweep.jl` | — | Sensitivity harness. Sweeps `SOE_CEV_ini` across 10 points at a fixed seed, keeps a subset of artefacts per run, writes `summary.html`. |

Plus in `Comparison/Code/`: `7_Comparison_main.jl` (driver), `8_ComparisonOutput.jl`
(merged 3-way figures and reports).

**The shared power pool.** `draw_activity_power_pool` pre-draws `n_samples` values per
`(entity, activity)` pair *before any approach runs*. Each approach gets its own
**cursor** into that pool, so Approach 0 and Approach 1 see identical numbers for
identical occurrence sequences without one consuming the other's. A draw is consumed
only when that pair actually occurs in an interval — idle has `σ=0` and never consumes
a slot. This is what makes the comparison a like-for-like measurement rather than two
different random days.

### 6.8 The model, briefly (equation numbers follow the source paper)

**(1) Objective** — energy + carbon + NC demand + OP demand + missed-work penalty +
towing labour. **(2)-(4) Power flow** — aggregation, charge only at grid nodes,
discharge only at sites, plug and acceptance caps, connection binaries. **(5) Work
coupling** — work power capped by availability, forced to zero while charging; work
power equals the chosen activity's constant draw; charge only while idle. **(6)-(7)
Energy** — travel energy, MCS and CEV SOE recursions with efficiency. **(8) Terminal &
bounds** — MCS exact energy-neutral equality, CEV **floor** at its start level (a
deliberate deviation: a CEV can't discharge, so an exact equality can become
unrecoverable once the stochastic plant lets one drift high), SOE ranges at every
boundary. **(9)-(11) Routing** — presence partition, arrivals/departures, flow
conservation, travel-time delay, park at a grid node by the day boundary. **(12) Work
scheduling** — one activity per CEV per interval; the work quota; digging-before-
loading precedence; the operator rest rule. **(13) Travel pacing** — a two-sided band
tying repositioning to productive work.

**On the work quota:** written as a **balance equality** — work done + shortfall slack
= requirement outstanding. The shortfall is soft and priced, but because the slack is
non-negative the equality *also* implies a hard upper cap — a CEV cannot do more work
than the requirement still outstanding. "No working ahead" comes out implicitly rather
than as a separate rule. (Earlier doc revisions described this as an inequality with
no cap; that was wrong — the source paper writes the equality too.)

**MPC adaptations** beyond a single-shot solve, in both controllers: closed-loop
carry-in of measured state; rolling rules (rest, precedence, pacing) **seeded from the
applied history** so a work-run can't leak across re-solves; carried-in demand peaks;
a generalized flow balance and carried-in start position for a window that begins
mid-route.

### 6.9 What each run writes

Into `<controller>/output/<synthetic|input>/`: numbered figures (PNG) and backing
CSVs (grid power, work profiles, MCS/CEV SOE, prices/emissions, MCS location
trajectory, per-MCS power); KPI and cost reports; per-window solver diagnostics and
worker schedule; **replan grids** — what was planned at each re-solve versus what was
ultimately done; `approach0_vs_approach1.html` — the two approaches' fully realized
outcomes side by side, self-labelled by Approach 0's plant mode, plus a
run-diagnostics table (plant mode, infeasible windows, energy-capped count, solve
time); `run_log.txt` — the full console output. Sweeps write into
`<controller>/output/input_testing/`, one subfolder per sweep point plus
`summary.html`. The comparison writes into `Comparison/Output/`, including
`approach0_vs_shrinking_vs_receding.html` and merged 3-way figures.

### 6.10 The five-level appendix — planning vs. reality

The single most important thing to understand about this codebase is the relationship
between **what the optimizer assumes** and **what the simulated world actually does**.
Everything else — the two approaches, the two plant modes, the whole comparison
methodology — falls out of it. One question, five levels of depth.

**The question:** do we optimize using the mean and then subtract sampled power from
the batteries, or do we optimize using the sampled power too? **The answer, in one
line:** we *always* optimize on the mean. What gets subtracted from the batteries is a
*sample*, except in `:mean` mode where that is also the mean.

**Master table:**

| | What the MILP optimizes on | How often it re-solves | What the plant subtracts from CEV batteries | Realized = planned? |
|---|---|---|---|---|
| Approach 0, `plant=:mean` | mean `μ` | once, at 08:00 (per day, in Receding) | mean `μ` | **Yes, exactly** |
| Approach 0, `plant=:sampled` | mean `μ` | once, at 08:00 (per day, in Receding) | a sample from the shared pool | No — drifts, nothing corrects it |
| Approach 1, Shrinking | mean `μ` | every 15 min; window `[now, next 08:00]`, shrinking | a sample from the shared pool | No — but every re-solve starts from the true measured state |
| Approach 1, Receding | mean `μ` | every 15 min; fixed-width window sliding into tomorrow | a sample from the shared pool | No — but every re-solve starts from the true measured state |

Notice what does **not** vary in the first column — the optimizer never sees a sample,
in any row. The MILP is byte-identical across all four; only the plant and the
re-solve cadence change.

**Level 1, for a ten-year-old.** You're packing lunch for a school trip. You *think*
everyone eats one sandwich, so you pack one each — that's the plan. Then the trip
happens; some kids eat one and a half, some eat half. Two ways to handle it: Approach
0 — pack at 8am and don't look in the bag again all day. Approach 1 — every fifteen
minutes, peek in the bag, count what's left, and re-do the plan. Mean mode is the
pretend world where every kid eats *exactly* one sandwich, just like you guessed —
useful because it tells you how good your plan was, without luck getting in the way.

**Level 2, for a curious adult.** The controller has one number for how much power
digging takes — about 4.8 kW — and builds the whole schedule around it, with no other
information. Reality wobbles: the simulator draws a slightly different number each
time (5.1, then 4.6, then 4.9), so by afternoon the real world has drifted from the
paper plan. Approach 0 built one schedule at 8am and follows it regardless; Approach 1
rebuilds every 15 minutes from the real battery levels, catching drift while it's
small. Mean mode switches the wobble off — reality obediently uses 4.8 kW every time,
giving a clean reference cost with no luck in it. Why not optimize on the sampled
numbers directly? Because those are the future — feeding them to the optimizer would
let it plan around wobbles that haven't happened yet, which no real controller could
do; the number it would report would be a fantasy.

**Level 3, for an engineer.** Two strictly separated worlds. **Controller side:**
`build_window_model` receives `pvec = est.mu`, a fixed 4-vector of per-activity
powers. Work power is pinned by Eq. 5e, `P_work = Σₐ pₐ·u`, so the MILP's battery
trajectory is entirely determined by `μ` — the controller has no access to `σ` and
none to the pool; structurally, it cannot see the disturbance. **Plant side:**
`apply_and_simulate!` applies interval `k₀` of the solved plan, then advances the true
state:
```julia
pt[a]      = use_mean ? pool.mu[a] : next_power!(pool, cursor, e, a)
work_true  = dot(a_real[e], p_true[e])
soe_cev[e] = clamp(soe_cev[e] + charged - work_true, SOE_min, SOE_max)
```
That ternary is the entire mode switch — everything else (which activity ran, how the
MCS moved, how the objective is accounted) is shared code. **The feedback loop:**
Approach 1 passes the measured `soe_cev`, `mcs_node`, `rem_dig`, `rem_load`, `hist`
back into the next `build_window_model` call; Approach 0 never does — it solves once
and replays. That's the *only* structural difference between them. **Fair
comparison:** both approaches draw from one `ActivityPowerPool` pre-generated before
either runs, each with its own cursor — identical occurrence sequences see identical
numbers, so the cost gap measures strategy, not luck. **Why `:mean` matters:** with
realized equal to planned, the measured end-of-day cost *is* the cost the optimizer
promised at 08:00 — nothing else it could be — so `A0(:mean)` gives a trustworthy
reference, and comparing it against `A0(:sampled)` isolates the cost of plan drift
with no re-planning involved.

**Level 4, for a control engineer.** This is **certainty-equivalent MPC**. The
stochastic program `min E[J(x,u,w)] s.t. x⁺=f(x,u,w), w~N(μ,σ)` is replaced by its
deterministic surrogate at `w = E[w] = μ`. The certainty-equivalence principle is
exact only for LQG; here the dynamics carry integer decisions and constraints are
hard, so the surrogate is a heuristic — a good one, the standard industrial choice,
but its optimality gap against the true stochastic optimum is not characterized.
Robustness comes from **feedback, not the model** — the MILP is nominal; what handles
the disturbance is re-solving from the measured state every 15 minutes and applying
only `u₀`. Textbook receding-horizon control. The two horizon variants differ in
terminal handling: **Shrinking** always reaches the day boundary, so terminal
constraints bind in every window — recursive feasibility is inherited from the
single-shot problem. **Receding** pins the same targets at the next day-start, so each
window closes the current day's energy cycle. Neither carries a terminal cost or value
function — the constraints do all the work, which is precisely why `shrinking=false`
is experimental (it drops terminal rules from all but the last `H` windows with
nothing replacing them, making the controller myopic). On the mode as an experimental
instrument: `A0(:mean) --> A0(:sampled)` = cost of open-loop plan drift;
`A0(:sampled) --> A1` = value of feedback. Running `run_mpc(plant=:mean)` with
`shrinking=true` gives the fourth cell of the 2x2 — it *must* reproduce `A0(:mean)` to
the MIP gap, since both solve the same problem from the same state on the same data.
Any residual gap is a defect in the closed-loop machinery (history seeds, carried-in
position, peak carry-in), not a plant effect — the diagnostic to reach for first
whenever the approaches disagree.

**Level 5, the honest caveats.** Four qualifications that matter for anything
published. **Only the CEV side is stochastic** — `soe_mcs` is advanced by the
*planned* value, not a simulated measurement; grid draw, travel loss, and hence every
cost term follow the plan exactly. The disturbance enters solely through CEV depletion
and reaches the objective only indirectly, through what the next re-solve decides
about it — the reported cost variance understates what a fully stochastic plant would
produce. **Energy accounting** — see §8 below for how this has changed since the
Approach 1 README's own text was written (it still describes a since-fixed "clamp
breaks energy conservation" issue). **Re-planning is not monotonically beneficial, and
this codebase demonstrates it** — each re-solve is an independent MILP; a small state
perturbation can flip a discrete routing binary, so the applied trajectory is a
concatenation of first-steps from 96 different optimization problems, not a coherent
plan. Nothing penalizes churn between successive solutions. On the synthetic dataset
this shows up as Approach 1 accumulating substantially more MCS transit than Approach
0 — the closed loop paying real money to chase disturbances a committed plan simply
absorbed. Whether that's genuine or a seam defect is exactly what the `:mean`
diagnostic settles, and per the README it was **not yet settled** as of that writing.
**A shared pool is a variance-reduction device, not a proof** — common random numbers
make a single-seed comparison far tighter than independent sampling, but one seed is
still one sample path; a cost gap smaller than the across-seed spread means nothing.
For any claim about the value of re-planning, sweep the seed and report a
distribution — the machinery already exists in `run_soe_sweep.jl`.

### 6.11 Setup

```julia
Pkg.add(["JuMP", "HiGHS", "Plots", "DataFrames", "CSV", "Turing", "XLSX"])
```
`XLSX` is needed only for the step-0 regression; set `RUN_REGRESSION=false` to skip it.

### 6.12 Where to read next

`Shrinking_Horizon/docs/README.md` (full single-day controller documentation),
`Receding_Horizon/docs/README.md` (full multi-day controller documentation),
`Comparison/Code/README.md` (the 3-way driver, module aliasing, pool sharing),
`<subfolder>/docs/math_model.tex` (the formal MILP — sets, variables, every equation),
`<subfolder>/docs/constraints_code_vs_model.txt` (line-by-line code-vs-model audit,
including the full list of known discrepancies and open items — this is where the
capping and pacing-tolerance fixes discussed in §8 are actually documented correctly).

---

## 7. MPC_Shubhan — Approach 2 (stochastic, scenario-based MPC)

**Full docs:** `MPC_Shubhan/Approach 2/README.md`; conceptual explainer:
`MPC_Shubhan/Understanding_Deterministic_vs_Stochastic_MPC.md` (recommended reading
*before* the file-by-file detail — five levels of depth plus a worked numerical
example of deterministic vs. stochastic MPC on the same hidden reality).

### 7.1 What's shared with Approach 1

Sections 1-12 of the Approach 2 README are the *same* architecture and physics as
Approach 1 (§6 above), reproduced with inline notes wherever a claim depends on which
controller is planning. The folder structure is identical (`RUN_ALL.jl`,
`Shrinking_Horizon/`, `Receding_Horizon/`, `Comparison/`, `batch_logs/`), the same 7
real CSVs are read under `:input`, the same `synthetic_data/` mirror convention
applies, the same module-aliasing and shared-pool tricks are used in the comparison
driver, and — critically — **the Approach 0 baseline is unchanged**: it's still the
one-shot 8am plan, executed open-loop, still certainty-equivalent, still the fixed
reference every headline comparison is measured against.

One inline distinction worth calling out explicitly: in Approach 1's sampled-vs-mean
discussion (§6.3 above), "the planner always plans on the mean `μ`" is true only for
Approach 0 and for Approach 1's own controllers. In Approach 2, this is no longer
true for the closed-loop side — **Approach 2 always plans on a sampled scenario set**
(§7.2 below), regardless of the plant mode switch. The plant-mode switch still exists
and still controls what the *simulated reality* does, but it no longer describes what
the *optimizer* itself sees, which is the entire point of the extension.

### 7.2 What actually changed — the stochastic extension, in full

**§13.1 — What changed, in one paragraph.** Approach 1's controllers plan every
re-solve on a single point estimate (the posterior mean `μ`). Approach 2's controllers
instead sample a small set of `S` scenarios from that same posterior at every
re-solve, and solve **one** MILP containing `S` linked copies of the future — coupled
by **non-anticipativity**: the action chosen for the interval about to be applied must
be identical across every scenario copy, because the controller doesn't yet know which
one is real. Everything else — the physics, the data schema, the two horizon flavors,
the comparison methodology, the plant/pool separation — is unchanged.

**§13.2 — What's new on disk.** Both `Shrinking_Horizon/code/` and
`Receding_Horizon/code/` gain one new file, **`2b_ScenarioSampler.jl`** — a small,
standalone module (`sample_scenarios`, `equal_weights`, `DEFAULT_N_SCENARIOS = 5`)
that both `3_MCSModel.jl` and `4_MPCLoop.jl` `using ..ScenarioSampler` and call — the
sampling logic lives in exactly one place, not inlined into either file.
`3_MCSModel.jl` gains a **second** model builder, `build_window_model_stochastic`,
alongside the original `build_window_model` (kept byte-for-byte for the Approach 0
baseline). `4_MPCLoop.jl`'s `run_mpc` (the closed-loop controller) now samples
scenarios and solves the stochastic model; `run_one_shot` (Approach 0) is untouched.
`1_Common.jl`, `2_DataLoader.jl`, `5_Output.jl`, `0_Regression.jl` are unchanged in
both codebases.

**§13.3 — What's new in the entry points.** A single new keyword, `n_scenarios`
(default `5`), threaded through every entry point: `run_scenario_1(mode=:input,
n_scenarios=10, ...)`, `run_soe_sweep.jl`'s `N_SCENARIOS` constant (overridable via
`MASTER_N_SCENARIOS`, mirroring how `MASTER_A0_PLANT` already works), and
`RUN_ALL.jl`'s `N_SCENARIOS` master setting (applies to every stage — both
controllers, both sweeps, and the comparison — exactly like `PLANT_MODE` already
does). `run_comparison` in `Comparison/Code/7_Comparison_main.jl` also takes
`n_scenarios` and passes it to both `ShrinkingApp.MPCLoop.run_mpc` and
`RecedingApp.MPCLoop.run_mpc`, which are now the scenario-based controllers — the
comparison driver's "Approach 1a/1b" columns are relabelled "Approach 2a/2b" for
clarity, since they're solving a materially different model now.

**§13.4 — The Approach 0 baseline is unchanged, on purpose.** Approach 0 stays
certainty-equivalent in every codebase and every driver in this tree — it's the fixed
reference every headline comparison is measured against, exactly as in Approach 1;
only now the closed-loop side of that comparison hedges against uncertainty instead of
ignoring it. `approach0_vs_approach1.html` (still that literal filename) and the 3-way
`approach0_vs_shrinking_vs_receding.html` are unchanged in format — same layout, same
columns — but the column that used to report Approach 1's results now reports Approach
2's, relabelled accordingly.

**§13.5 — A note on the Comparison driver's paths.** `Comparison/Code/7_Comparison_main.jl`
`include`s the two controller codebases *in place* from `Shrinking_Horizon/code/` and
`Receding_Horizon/code/`, resolved relative to its own file location (`@__DIR__`)
rather than a hardcoded machine path — the whole `Approach 2` folder is self-contained
and can be moved or copied as a unit. Both `Comparison`'s wrapper modules
(`RecedingApp`, `ShrinkingApp`) now also include `2b_ScenarioSampler.jl` alongside the
other five numbered files, so the comparison run draws scenarios exactly the same way
each standalone controller does.

### 7.3 Updated file-by-file map (Approach 2 deltas from §6.7)

| File | Module | What's different from Approach 1 |
|---|---|---|
| `2b_ScenarioSampler.jl` | `ScenarioSampler` | **New.** `sample_scenarios(mu, sd, n_scenarios; rng)`, `equal_weights(n_scenarios)`, `DEFAULT_N_SCENARIOS = 5`. Doesn't need the `Common`-style module-aliasing trick the comparison driver uses elsewhere. |
| `3_MCSModel.jl` | `MCSModel` | `build_window_model(...)` retained unchanged (Approach 0's baseline). **New:** `build_window_model_stochastic(...)` — the scenario-linked sibling that Approach 2's `run_mpc` actually solves every re-solve. |
| `4_MPCLoop.jl` | `MPCLoop` | `run_mpc` now samples scenarios and solves `build_window_model_stochastic` every step (via a thin `apply_and_simulate_stochastic!` wrapper reaching the same shared `apply_and_simulate!`). `run_one_shot` unchanged. |
| `6_<name>_main.jl` | — | `run_scenario_1(; mode, n_scenarios, …)` — same auto-run/tee behavior, plus the new keyword. |

### 7.4 Why this matters conceptually — non-anticipativity in one sentence

A deterministic (certainty-equivalent) controller plans as if it already knew the
future would be exactly `μ`. A scenario-based stochastic controller instead asks: "if
the true future power draw turns out to be any one of these `S` plausible scenarios,
what single action for *this* interval performs well across all of them?" — because
the very next action must be chosen *before* which scenario is real becomes known,
every scenario's copy of the plan is forced to agree on that first action
(non-anticipativity), while being free to diverge from there onward as each
hypothetical future unfolds differently. This is the standard formulation for
scenario-based stochastic programming under recourse, applied here to a receding-
horizon MILP rather than a single-shot LP.

---

## 8. Known issues across the repo

Collected from working through this codebase in depth, well past what either
`Approach 1` or `Approach 2`'s own README currently documents. Some of these correct
statements that appear (now stale) in the individual subfolder READMEs; some describe
gaps that exist in *no* documentation yet and are recorded here for the first time.

### 8.1 The energy-accounting fix (corrects §6.10 Level 5 and §10 of both Approach READMEs)

Both top-level Approach READMEs' "Known limits" sections, and the Level-5 caveats in
§6.10 above (reproduced faithfully from the source doc), state that the CEV SOE
balance is **"clamped"** — when a sampled over-draw would push a CEV below `SOE_min`,
the guard silently restores it, "energy from nowhere," and the reported SOE stops
being the integral of the reported power. This is invisible to the solver's
feasibility status, so a run could report zero infeasible windows while being
physically inconsistent. Events were counted as `n_clamped`.

**This has since been fixed**, and the fix is real and load-bearing, not cosmetic:
`apply_and_simulate!` now **caps the realized dig/load/travel duration by the energy
actually available** (`headroom = soe_cev + charged - SOE_min`) *before* crediting any
work to `rem_dig`/`rem_load` — only the affordable fraction of an interval's activity
is credited, with the remainder becoming idle. No energy is fabricated; the SOE
trajectory genuinely stays within bounds under normal operation, and the old `clamp()`
call remains only as a should-never-bind safety net. The diagnostic field was renamed
`n_clamped` -> `n_capped` throughout the codebase, printed messages, and HTML reports.
The corrected version of this writeup lives in each controller's
`docs/constraints_code_vs_model.txt` (entries D3 for Shrinking, A2 for Receding) and
`docs/README.md` — **but has not yet been propagated up into either top-level
`Approach 1/README.md` or `Approach 2/README.md`**, which is why §6.10 above still
reproduces the stale language faithfully (it was pulled directly from those files) — a
follow-up correction of those two files specifically is still open.

### 8.2 A second, related fix — not yet documented in any README at all

The energy-capping fix above has a subtle second-order consequence that is genuinely
undocumented anywhere in the repo prior to this writeup: the two-sided travel-pacing
band (Eq. 13, §6.8) checks cumulative work/travel counts with **exact arithmetic and
no tolerance**. The capping fix can leave a small fractional residue in
`cum_dig_e`/`cum_load_e` (e.g. one interval credited as `0.246h` instead of a clean
`0.25h`) — a residue that's energetically meaningless but can land the pacing band's
floor (`4V <= W`) and ceiling (`W <= 4V+4`) on opposite sides of an integer boundary
with no whole-interval solution reachable in between. Concretely: needing "between
11.996 and 12.246" buckets of work when work is only ever creditable in whole 0.25h
steps. This **permanently and spuriously blocks further CEV travel/work for the rest
of the day**, over a shortfall as small as one minute — not because the pacing rule's
intent was violated, but because its exact-arithmetic implementation has zero
tolerance for a rounding residue introduced by an entirely different part of the
model.

This was root-caused via a HiGHS **IIS (Irreducible Infeasible Set)** query on a
concrete failing case (a 10-point `SOE_CEV_ini` sweep's tenth point, where the CEV
starts the day already at its maximum charge), after a standalone repro script and a
constraint-by-constraint bisection harness (relaxing each candidate constraint one at
a time — terminal floor, SOE bounds, precedence, rest rule, each half of the pacing
band independently) failed to isolate it, because the actual conflict spans two
adjacent 15-minute intervals rather than a single constraint in isolation. **Fixed**
by adding a small numerical tolerance, `pacing_tol = 0.05` (interval-units, i.e. 5% of
one interval), to the floor side of the constraint only:
`work_per_travel * V <= W + pacing_tol`. This is large enough to absorb realistic
capping residue (~0.02 typical) while remaining far too small to be mistaken for a
free extra work unit or to meaningfully weaken the rule's actual intent. This fix and
its rationale are documented in each controller's `docs/constraints_code_vs_model.txt`
(new entries D11/A3) and `docs/math_model.tex`, but — like §8.1 — **has not yet been
mirrored into either top-level Approach README.**

### 8.3 The root README itself was stale before this rewrite

The repo's root `README.md`, before this rewrite, was scoped entirely to the CV
pipeline and used un-prefixed paths (`Codes/...` instead of `Video Detection/Codes/...`)
— it predated the repo being reorganized into the four top-level folders described in
§2. A near-identical, correctly-scoped copy already existed at
`Video Detection/README_CV.md` — meaning someone had already started the fix by
dropping a corrected copy in the right place, but never updated (or removed/redirected)
the stale root copy to match. This file replaces the root README with a proper landing
page linking out to all four project areas.

### 8.4 Structural notes worth knowing before touching the code

`MPC_Shubhan/Approach 1` and `Approach 2` each carry an `Old Code - To not delete/`
subfolder inside their controller `code/` directories — intentionally-preserved prior
versions, not part of the active codebase; anything that imports/includes code should
resolve to the numbered files one level up, never into that subfolder. Both `Approach`
trees also carry an "Imp result ... - Do not delete" style output snapshot folder
(`Shrinking_Horizon/Imp result shrinking horizon - Do not delete/`) preserving a
specific historical run's output outside the normal `output/` directory structure —
worth knowing exists so it isn't mistaken for stale/duplicate output to be cleaned up.
`Comparison/Code/` in Approach 1 also carries two near-duplicate similarity-check
scripts (`Input data similarity check.py` and `Input data similariy check.py` — note
the typo in the second filename) — likely one is the intended canonical version and
the other a leftover from a rename; worth confirming which before relying on either.

---

## 9. Getting started, end to end

To reproduce the whole pipeline from raw video to a scheduling result, in dependency
order:

1. **Video Detection** — run Steps 1-3 (YOLO training, detection, tracking) on your
   own video, or use the provided `best.pt` weights directly on new footage. Run Step
   4 (idling detection) and either label manually (Step 5) or trust the existing CVAT
   annotations for the 12 recorded days. Run Steps 7-9 to produce
   `frame_predictions.csv` / `cycles.json` for a new video, or use the existing
   validated outputs under `Run from step 8 to 11/`.
2. **Bayesian Regression** — point `Tasks_energy_loading_swinging_bayesian.py` at your
   task-log Excel files (update the hardcoded paths first — see §4.4) and run it to
   produce posterior mean/std per activity. Manually transcribe (or script) the
   printed coefficients into whichever `parameters.csv` you intend to use downstream.
3. **MPC** — start with `MPC/Scenario 1` if you want to see the original,
   single-shot, no-re-planning baseline behave on the `simple_dataset/` example. Then
   move to `MPC_Shubhan/Approach 1/RUN_ALL.jl` for the full closed-loop treatment —
   set `RUN_REGRESSION=false` first if you don't want to re-run step 0 with a fresh
   regression fit. Once comfortable with Approach 1's output, `MPC_Shubhan/Approach 2`
   runs the identical pipeline with `n_scenarios` scenario-based planning instead —
   read `Understanding_Deterministic_vs_Stochastic_MPC.md` first.

All three optimization codebases (`MPC/Scenario 1`, `Approach 1`, `Approach 2`)
require Julia with `JuMP`, `HiGHS`, `Plots`, `DataFrames`, `CSV`, `Turing`, `XLSX`. The
CV pipeline requires Python with the packages listed in §3.9. The regression scripts
require Python with the packages listed in §4.5, plus a MOSEK license for the
point-estimate script's solver.

---

## 10. Citation

```bibtex
@software{ev_charging_construction_equipment,
  author = {Shubhan Mital},
  title  = {Optimizing On-Site EV Charging for Construction Equipment},
  year   = {2025},
  url    = {https://github.com/Shubhanflash22/Optimizing-On-Site-EV-Charging-for-Construction-Equipment}
}
```

Primary methodology paper for the activity-recognition side:

```bibtex
@article{cho2021excavator,
  author  = {Cho, Y. K. and Latif, E. and Sharafat, A. and Seo, J.},
  title   = {Automated activity recognition of excavators using 3D convolutional neural networks},
  journal = {[See Video Detection/Papers and references/ for full citation]},
  year    = {2021}
}
```

Source papers for both halves of the project: `Video Detection/Papers and references/`
(activity recognition, primarily Cho, Latif, Sharafat, Seo, plus 6 related works — see
§3.11) and `MPC_Shubhan/Optimization.pdf` (the MILP formulation these constraints
implement, referenced throughout §5-7 by equation number).

---

## 11. License

MIT License — see [`LICENSE`](LICENSE). The Roboflow training dataset used in
`Video Detection/` is licensed separately under CC BY 4.0 (see
`Video Detection/2.Custom Yolo model training/README.roboflow.txt`).