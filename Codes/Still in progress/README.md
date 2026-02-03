# 3D ResNet Excavator Activity Recognition - Paper Implementation

This repository contains a **fully corrected and commented** implementation of the 3D ResNet activity recognition system described in the paper, with detailed explanations of what matches the paper and what assumptions were made.

## 📄 Paper Reference

**Section 4.4**: Activity Recognition with 3D ResNet  
**Section 4.5**: Productivity Calculation  
**Section 5.1**: Training and Testing

---

## 🎯 Key Corrections from Original Code

### 1. ✅ **Learning Rate** (CRITICAL FIX)
- **Paper**: 0.001
- **Original Code**: 0.0001 (1e-4)
- **Fixed Code**: 0.001 (1e-3)
- **Impact**: 10× higher learning rate as specified in paper

### 2. ✅ **Frame Rate** (CRITICAL FIX)
- **Paper**: "all video clips are fixed at 25 FPS for training"
- **Original Code**: Used stride=2 on unknown FPS (likely ~30 FPS effective)
- **Fixed Code**: Explicit TARGET_FPS=25 with stride=1
- **Impact**: Ensures temporal consistency with paper

### 3. ✅ **Data Augmentation** (ADDED MISSING TECHNIQUES)
- **Paper**: "flipping the video frames, shifting the image channels, and shearing the frame size"
- **Original Code**: Only had horizontal flipping + basic noise
- **Fixed Code**: Implements ALL THREE techniques:
  - Horizontal flipping ✓
  - Channel shifting (brightness/contrast variation) ✓
  - Shearing transformation ✓

### 4. ✅ **Spatial Augmentation** (IMPROVED)
- **Paper**: 112×112 input
- **Original Code**: Direct resize to 112×112 (no scale variation)
- **Fixed Code**: 
  - Training: Resize to 128×128 → Random crop to 112×112
  - Validation: Resize to 128×128 → Center crop to 112×112
- **Impact**: Provides scale augmentation, standard practice

### 5. ✅ **Majority Voting Post-Processing** (CRITICAL ADDITION)
- **Paper**: "each frame is labeled to indicate the excavator activity after correcting the recognition errors with majority voting"
- **Original Code**: NO post-processing
- **Fixed Code**: Implements temporal smoothing with majority voting
- **Impact**: Essential for achieving paper's reported accuracy

### 6. ✅ **Evaluation Metrics** (MATCHES PAPER)
- **Paper**: Reports precision and recall per class (Table 3)
- **Original Code**: Only accuracy
- **Fixed Code**: 
  - Per-class precision and recall
  - Confusion matrix
  - Matches paper's reporting format

### 7. ✅ **Cycle Time Calculation** (IMPLEMENTED)
- **Paper**: "The total time of one cycle is the difference between the start times of two adjacent digging activities"
- **Original Code**: Not implemented
- **Fixed Code**: Full cycle detection and timing

### 8. ✅ **Productivity Calculation** (IMPLEMENTED)
- **Paper Equation (3)**: Productivity = Cycles/hr × Bucket payload
- **Original Code**: Not implemented
- **Fixed Code**: Complete productivity calculation

---

## 📊 What MATCHES the Paper

| Aspect | Paper Specification | Implementation | Status |
|--------|-------------------|----------------|--------|
| **Model** | 3D ResNet | PyTorch R3D-18 | ✅ MATCHES |
| **Pre-training** | Kinetics-400 | Kinetics-400 weights | ✅ MATCHES |
| **Input Size** | 16 × 112 × 112 | 16 × 112 × 112 | ✅ MATCHES |
| **Frame Rate** | 25 FPS | 25 FPS (corrected) | ✅ MATCHES |
| **Batch Size** | 16 | 16 | ✅ MATCHES |
| **Learning Rate** | 0.001 | 0.001 (corrected) | ✅ MATCHES |
| **Augmentation** | Flip, shift, shear | All three (corrected) | ✅ MATCHES |
| **Post-processing** | Majority voting | Implemented | ✅ MATCHES |
| **Metrics** | Precision, Recall | Both computed | ✅ MATCHES |
| **Cycle Detection** | Digging-based | Implemented | ✅ MATCHES |
| **Productivity** | Equation (3) | Implemented | ✅ MATCHES |

---

## 🔬 Valid Assumptions (Not in Paper)

These are reasonable choices not explicitly specified in the paper:

### 1. **Optimizer: Adam**
- **Paper**: Not specified
- **Implementation**: Adam optimizer
- **Justification**: 
  - Industry standard for deep learning
  - Works well with fine-tuning
  - Better than SGD for most transfer learning scenarios

### 2. **Learning Rate Schedule: ReduceLROnPlateau**
- **Paper**: Not specified
- **Implementation**: Reduce LR when validation plateaus
- **Justification**: 
  - Helps convergence
  - Standard practice for fine-tuning
  - Prevents overfitting

### 3. **Mixed Precision Training**
- **Paper**: Not mentioned
- **Implementation**: Uses PyTorch AMP
- **Justification**: 
  - Faster training (no accuracy loss)
  - Reduces memory usage
  - Modern best practice

### 4. **Weighted Sampling for Class Balance**
- **Paper**: Not mentioned
- **Implementation**: WeightedRandomSampler
- **Justification**: 
  - Handles class imbalance
  - Improves learning of minority classes
  - Standard practice for imbalanced datasets

### 5. **Data Split: 80/20 Train/Val**
- **Paper**: Doesn't specify split ratio
- **Implementation**: 80% train, 20% validation
- **Justification**: 
  - Standard ratio
  - **CRITICAL**: Split by VIDEO (no leakage)
  - Ensures robust evaluation

### 6. **Extended to 5 Activity Classes**
- **Paper**: 3 classes (digging, loading, swinging)
- **Implementation**: 5 classes (+ idling, travelling)
- **Justification**: 
  - More comprehensive monitoring
  - Common in real-world scenarios
  - Framework easily scales

### 7. **Handling Short Videos**
- **Paper**: Doesn't specify
- **Implementation**: Pad with last frame
- **Justification**: 
  - More natural than looping
  - Maintains temporal coherence
  - Prevents artificial transitions

---

## 🗂️ File Structure

```
├── resnet3d_training_corrected.py      # Training script (fully commented)
├── resnet3d_inference_corrected.py     # Inference with majority voting
├── README.md                            # This file
└── requirements.txt                     # Python dependencies
```

---

## 🚀 Usage

### Training

```python
# 1. Set your paths in resnet3d_training_corrected.py
BASE_DIR = Path("/path/to/your/data")
DATASET_DIR = BASE_DIR / "Dataset_Resnet_2"

# 2. Ensure dataset structure:
# Dataset_Resnet_2/
#   ├── digging/
#   │   ├── video1/ (folder with frames)
#   │   │   ├── frame_0001.jpg
#   │   │   ├── frame_0002.jpg
#   │   │   └── ...
#   │   └── video2/
#   ├── loading/
#   ├── swinging/
#   ├── idling/
#   └── travelling/

# 3. Run training
python resnet3d_training_corrected.py
```

**Expected Output:**
- Best model: `resnet3d_best_corrected.pth`
- Training history: `training_history.json`
- Console output with per-class precision/recall (matches Paper Table 3 format)

### Inference

```python
# 1. Set paths in resnet3d_inference_corrected.py
VIDEO_PATH = Path("/path/to/test/video.mp4")
MODEL_PATH = Path("/path/to/resnet3d_best_corrected.pth")

# 2. Run inference
python resnet3d_inference_corrected.py
```

**Outputs:**
- `frame_predictions.csv` - Frame-by-frame predictions
- `cycles.json` - Detected work cycles
- `summary.json` - Statistics and productivity
- `activity_timeline.png` - Visualization

---

## 📈 Paper Results Comparison

### Paper's Reported Results (Section 5.1, Table 3)

| Activity | Precision | Recall |
|----------|-----------|--------|
| Digging  | 95%       | 86%    |
| Swinging | 86%       | 93%    |
| Loading  | 84%       | 80%    |
| **Average** | **88.3%** | **86.3%** |
| **Overall Accuracy** | **87.6%** |

**Note**: Paper tested on 3 classes. This implementation extends to 5 classes.

### Testing on Long Video

**Paper (Section 5.1)**: "the model was also applied on a 60.2 min video to recognize excavator's activities in the implementation stage and achieved the accuracy of 92.5%"

The inference script supports this type of long-form testing.

---

## 🔍 Code Commentary Style

Every section of code includes:

1. **PAPER**: Direct quotes from the paper
2. **MATCHES PAPER**: Confirms alignment with paper
3. **DEVIATION**: Where code differs from paper
4. **JUSTIFICATION**: Why the deviation is valid
5. **ASSUMPTION**: Reasonable choices not specified in paper

Example:
```python
# PAPER: "batch size of the model was set to 16"
BATCH_SIZE = 16  # MATCHES PAPER: Batch size of 16

# PAPER: Not specified
# ASSUMPTION: Adam optimizer
# JUSTIFICATION: Industry standard for fine-tuning
optimizer = optim.Adam(...)
```

---

## 🎓 Key Learnings & Best Practices

### 1. **Fine-tuning Strategy**
```python
# Freeze early layers, train deeper layers
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
```
**Why**: Preserves learned low-level features, prevents overfitting

### 2. **Temporal Consistency**
```python
# Majority voting smooths predictions
smoothed = apply_majority_voting(predictions, window_size=25)
```
**Why**: Excavators don't change activities instantly (Paper insight: "each activity lasts at least 2s")

### 3. **Data Leakage Prevention**
```python
# Split by VIDEO, not by frames
random.shuffle(all_videos)  # Shuffle videos
train_videos = all_videos[:split_idx]
```
**Why**: Prevents frames from same video in both train and val sets

### 4. **Class Imbalance**
```python
# Weighted sampling
sampler = WeightedRandomSampler(sample_weights, ...)
```
**Why**: Ensures all classes are learned equally, even if dataset is imbalanced

---

## 🛠️ Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📝 Paper Citation

If using this implementation, please cite the original paper:

```bibtex
@article{excavator_activity_recognition,
  title={Activity Recognition for Excavators using 3D ResNet},
  author={[Authors from your paper]},
  journal={[Journal Name]},
  year={[Year]},
  pages={[Pages]}
}
```

---

## ❓ FAQ

### Q: Why use R3D-18 instead of custom ResNet?
**A**: R3D-18 IS a 3D ResNet. It's PyTorch's implementation of the exact architecture described in the paper. Same concept, official implementation.

### Q: Why 5 classes instead of 3?
**A**: Extended for more comprehensive monitoring. The framework easily scales. You can use 3 classes by removing 'idling' and 'travelling' from ACTIVITY_NAMES.

### Q: What if my videos aren't 25 FPS?
**A**: The code resamples videos to 25 FPS automatically. This matches the paper's preprocessing.

### Q: How to improve results?
**A**: 
1. Collect more diverse training data
2. Increase training epochs if not converged
3. Adjust majority voting window size
4. Fine-tune more layers after initial training
5. Compute dataset-specific normalization stats

### Q: Why is accuracy different from paper?
**A**: 
- Paper: 3 classes, specific construction sites
- Your data: May have 5 classes, different conditions
- Paper's 87.6% is on their specific test set
- Compare on similar data for fair evaluation

---

## 🐛 Troubleshooting

### Low Accuracy
1. Check if frames are extracted at correct FPS
2. Verify data split (no leakage)
3. Ensure augmentation is applied
4. Check class balance in dataset
5. Try training longer or unfreezing more layers

### Out of Memory
1. Reduce BATCH_SIZE (e.g., from 16 to 8)
2. Reduce NUM_WORKERS
3. Use smaller CLIP_LENGTH (but paper uses 16)
4. Enable gradient checkpointing

### Slow Training
1. Increase NUM_WORKERS
2. Ensure GPU is being used (check DEVICE)
3. Reduce augmentation probability
4. Use mixed precision (already enabled)

---

## 🎯 Next Steps

1. **Train the model** on your dataset
2. **Evaluate** using the provided metrics
3. **Test on long videos** like the paper (60+ minutes)
4. **Compare results** with paper's Table 3
5. **Adjust hyperparameters** if needed
6. **Deploy for real-time monitoring** (optional)

---

## 📧 Support

For questions about the implementation:
1. Check the inline comments (every line is documented)
2. Review this README
3. Compare with paper sections 4.4, 4.5, and 5.1

---

## ⚖️ License

[Your License Here]

---

## 🙏 Acknowledgments

- Original paper authors for the methodology
- PyTorch team for torchvision models
- Kinetics-400 dataset creators

---

**Note**: This implementation prioritizes **correctness** and **alignment with the paper** over brevity. Every choice is documented and justified. Use the comments as a learning resource for understanding the paper's methodology.
