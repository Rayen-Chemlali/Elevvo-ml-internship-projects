# Traffic Sign Recognition — Project Report
## GTSRB Dataset | Task 8

---

## Overview

Deep learning pipeline to classify German traffic signs into 43 categories using convolutional neural networks. The project covers full exploratory analysis, image preprocessing, data augmentation, training and comparison of three architectures — Baseline CNN, Deep CNN with BatchNorm, and MobileNetV2 transfer learning.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign` |
| Train images | 39,209 |
| Test images | 12,630 |
| Classes | 43 traffic sign types |
| Image size (original) | 25×25 to 178×178 px (variable) |
| Image size (resized) | 32×32 (CNN) / 48×48 (MobileNetV2) |
| Missing values | None |

### Class Imbalance

| Property | Value |
|----------|-------|
| Smallest class | Class 0 — 210 samples |
| Largest class | Class 2 — 2,250 samples |
| Imbalance ratio | ~10x |
| Mean per class | 912 samples |

Classes 0–8 (speed limit signs) are visually similar and most prone to confusion.

---

## Key Observations from EDA

- Images vary widely in size (mean ~49×49 px) — resizing to a fixed size is mandatory
- Class distribution is moderately imbalanced (10x ratio) but manageable
- Speed limit signs (classes 0–8) share the same circular red-border shape, differing only by number
- Prohibition signs, warning triangles, and directional signs form visually distinct groups
- **Horizontal flip was disabled** in augmentation — traffic signs are direction-specific (left/right turn signs would become incorrect labels if flipped)

---

## Methodology

### Preprocessing
- Images loaded with OpenCV, converted BGR → RGB, resized to 32×32 (or 48×48 for MobileNetV2)
- Normalized to [0, 1] float32
- Stratified 80/20 train/val split → Train: 31,367 | Val: 7,842 | Test: 12,630

### Data Augmentation
Applied via `tf.data` pipeline (version-safe, no ImageDataGenerator):
- Random brightness ±20%
- Random contrast [0.8, 1.2]
- No horizontal flip (direction-specific signs)

### Pipeline
`tf.data.Dataset` with shuffling, augmentation, batching (64), and prefetching — avoids all generator-based bugs that caused val_accuracy=0 in earlier attempts.

---

## Model Architectures

### Baseline CNN (549K parameters)
- 2× Conv2D blocks (32→64 filters) + MaxPooling
- Dense 128 + Dropout 0.5
- Softmax output (43 classes)

### Deep CNN (1.2M parameters)
- 3× Conv blocks (32→64→128 filters) with BatchNormalization after each
- Progressive Dropout (0.25 → 0.25 → 0.25 → 0.5)
- Dense 512 + BatchNorm + Dropout
- Softmax output

### MobileNetV2 Transfer Learning
- Frozen ImageNet pretrained base (48×48 input)
- GlobalAveragePooling2D → Dense 256 → BatchNorm → Dropout 0.4
- Softmax output (43 classes)

---

## Model Comparison

| Model | Test Accuracy | Time (s) |
|-------|--------------|----------|
| **Deep CNN** | **98.26%** | 5,675 |
| Baseline CNN | 96.62% | 1,819 |
| MobileNetV2 | 54.37% | 2,346 |

**Deep CNN wins** with 98.26% — production-level accuracy on a 43-class problem. The Baseline CNN also performs strongly at 96.62%. MobileNetV2 underperforms significantly at 54.37% — the frozen ImageNet weights are not well-suited for the low-resolution (48×48), domain-specific GTSRB images without fine-tuning.

---

## Results — Best Model: Deep CNN (98.26%)

### Key Metrics
- **Test Accuracy**: 98.26%
- **Train Accuracy**: ~99%+ (converged)
- **Val Accuracy**: tracked via EarlyStopping with patience=7

### Confusion Analysis
- Speed limit signs (classes 0–8) are the most confused group — same shape, differ by number only
- Directional signs (classes 33–42) are well-separated — distinct shapes and colors
- Class 0 (smallest, 210 samples) is the most challenging due to low training data

---

## Key Findings

1. **Deep CNN outperforms MobileNetV2** — frozen ImageNet features are not optimal for small, domain-specific traffic sign images at 32–48px resolution
2. **BatchNormalization is the critical ingredient** — Deep CNN vs Baseline CNN: +1.64% accuracy
3. **tf.data pipeline fixes the val_accuracy=0 bug** — ImageDataGenerator with path-based flow caused silent failures in all previous attempts
4. **Horizontal flip must be disabled** — flipping directional signs corrupts the label
5. **MobileNetV2 needs fine-tuning** to reach its potential on this dataset — with unfrozen layers it could exceed 99%
6. **98.26% on 43 classes** is competitive with published GTSRB benchmarks (best reported ~99.6%)
7. **Class imbalance (10x)** has limited impact at this accuracy level — the model generalizes well even to minority classes

---

## Notebook Structure

```
traffic_sign_recognition.ipynb    # Full pipeline (Colab-ready, 14 cells)
split_notebook_task8.py           # Script to split into 6 sub-notebooks

notebooks/
├── 01_setup_eda.ipynb            # Imports, data loading, EDA
├── 02_preprocessing.ipynb        # Image loading, normalization, split
├── 03_baseline_cnn.ipynb         # Baseline CNN training
├── 04_deep_cnn.ipynb             # Deep CNN with BatchNorm
├── 05_transfer_learning.ipynb    # MobileNetV2 transfer learning
└── 06_evaluation_summary.ipynb   # Comparison, confusion matrix, summary
```

---

## Dependencies

```
tensorflow >= 2.10
opencv-python
kagglehub
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install:
```bash
pip install tensorflow opencv-python kagglehub pandas numpy matplotlib seaborn scikit-learn
```
