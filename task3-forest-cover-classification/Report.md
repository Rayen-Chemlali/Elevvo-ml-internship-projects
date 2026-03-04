# Forest Cover Type Classification — Project Report
## UCI Covertype Dataset | Task 3

---

## Overview

Multi-class classification pipeline to predict forest cover type from cartographic and environmental features. The project covers full exploratory analysis, preprocessing of a heavily imbalanced 580k-row dataset, comparison of five classifiers, and deep evaluation of the best model including feature importance and per-class performance.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — `uciml/forest-cover-type-dataset` |
| Total samples | 581,012 |
| Features | 54 (10 continuous + 4 wilderness area + 40 soil type binary flags) |
| Target classes | 7 forest cover types |
| Missing values | None |

### Class Distribution (heavily imbalanced)

| Class | Cover Type | Samples | % of total |
|-------|-----------|---------|------------|
| 1 | Spruce/Fir | 211,840 | 36.5% |
| 2 | Lodgepole Pine | 283,301 | 48.8% |
| 3 | Ponderosa Pine | 35,754 | 6.2% |
| 4 | Cottonwood/Willow | 2,747 | 0.5% |
| 5 | Aspen | 9,493 | 1.6% |
| 6 | Douglas Fir | 17,367 | 3.0% |
| 7 | Krummholz | 20,510 | 3.5% |

Classes 1 and 2 together account for **85.3%** of the dataset. Cottonwood/Willow is the most underrepresented class at 0.5% — a **103x imbalance ratio** vs Lodgepole Pine.

---

## Key Observations from EDA

- **Elevation** is the most discriminating single feature — each cover type occupies a distinct elevation band
- **Hillshade_9am and Hillshade_3pm** are strongly negatively correlated (-0.78) — expected from solar geometry
- **Horizontal and Vertical Distance to Hydrology** are moderately correlated (0.61)
- **Slope** is negatively correlated with hillshade values — steeper terrain reduces effective shade
- **Wilderness Areas 1 and 3** dominate (~87% of samples), Areas 2 and 4 are sparse
- **Soil Type 29** is by far the most common (116k samples), many soil types are near-zero

---

## Methodology

### Preprocessing
- Stratified sample: capped each class at **5,000 samples** (Cottonwood/Willow kept at 2,747) to manage Colab runtime while preserving class ratios
- Final sample: **32,747 rows**
- Applied **StandardScaler** to all 54 features
- 80/20 stratified train/test split → Train: 26,197 | Test: 6,550

### Why Sampling?
The full 580k-row dataset would require 5–10× more training time on Colab with marginal accuracy gains for tree-based models. Stratified sampling ensures all 7 classes are well represented.

---

## Model Comparison

| Model | Accuracy | Time (s) |
|-------|----------|----------|
| **Extra Trees** | **88.73%** | 7.79 |
| XGBoost | 88.60% | 10.11 |
| Random Forest | 88.58% | 8.60 |
| KNN | 81.11% | 0.00 |
| Logistic Regression | 69.62% | 5.16 |

**Extra Trees wins** — marginally better than XGBoost and Random Forest, and trains faster than both. The three tree ensemble methods cluster within 0.15% of each other, confirming the dataset is well-suited to ensemble tree approaches.

---

## Results — Best Model: Extra Trees (88.73%)

### Per-Class Accuracy

| Cover Type | Accuracy | Status |
|-----------|----------|--------|
| Spruce/Fir | 0.821 | OK |
| Lodgepole Pine | 0.743 | WEAK |
| Ponderosa Pine | 0.855 | GOOD |
| Cottonwood/Willow | 0.962 | EXCELLENT |
| Aspen | 0.971 | EXCELLENT |
| Douglas Fir | 0.917 | GOOD |
| Krummholz | 0.976 | EXCELLENT |

### Classification Report

| Cover Type | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| Spruce/Fir | 0.83 | 0.81 | 0.82 |
| Lodgepole Pine | 0.83 | 0.73 | 0.78 |
| Ponderosa Pine | 0.88 | 0.86 | 0.87 |
| Cottonwood/Willow | 0.93 | 0.97 | 0.95 |
| Aspen | 0.92 | 0.97 | 0.94 |
| Douglas Fir | 0.86 | 0.91 | 0.89 |
| Krummholz | 0.96 | 0.98 | 0.97 |
| **Overall** | **0.89** | **0.89** | **0.88** |

### Confusion Analysis
- **Spruce/Fir ↔ Lodgepole Pine** is the hardest pair — 15% mutual confusion. Both occupy overlapping elevation ranges and environmental conditions
- **Ponderosa Pine ↔ Douglas Fir** — 9% confusion, adjacent elevation bands
- **Cottonwood/Willow, Aspen, Krummholz** — near-perfect separation despite being minority classes, because each occupies a very specific ecological niche

---

## Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Elevation | 0.22 |
| 2 | Horizontal_Distance_To_Roadways | 0.10 |
| 3 | Horizontal_Distance_To_Fire_Points | 0.08 |
| 4 | Horizontal_Distance_To_Hydrology | 0.06 |
| 5 | Vertical_Distance_To_Hydrology | 0.06 |
| 6 | Hillshade_9am | 0.05 |
| 7 | Aspect | 0.05 |
| 8 | Hillshade_3pm | 0.05 |
| 9 | Hillshade_Noon | 0.05 |
| 10 | Slope | 0.04 |

Elevation alone accounts for **22% of total importance** — nearly 3× more than the second feature. All top 10 are continuous features. Soil type and wilderness area binary flags contribute individually at low levels.

---

## Key Findings

1. **Elevation is the dominant feature** — forest cover type is primarily determined by altitude
2. **Distance features matter** — roads, fire points, and hydrology together account for ~24% of importance
3. **Spruce/Fir and Lodgepole Pine are nearly inseparable** — they share overlapping environmental niches and produce 15% mutual confusion
4. **Minority classes outperform majority classes** — Cottonwood/Willow achieves 96.2% accuracy despite being the smallest class, because it occupies a very specific low-elevation niche near water
5. **Extra Trees is faster and slightly better** than Random Forest — recommended default for this dataset type
6. **Logistic Regression fails** at 69.6% — the decision boundary between cover types is highly non-linear
7. **Full dataset training** would likely push accuracy to ~92–94% based on published benchmarks

---

## Notebook Structure

```
forest_cover_classification.ipynb    # Full pipeline (Colab-ready)
split_notebook_task3.py              # Script to split into 5 sub-notebooks

notebooks/
├── 01_setup_data.ipynb              # Data loading & class distribution
├── 02_eda.ipynb                     # Feature distributions, correlations, boxplots
├── 03_preprocessing.ipynb           # Stratified sampling, scaling, train/test split
├── 04_model_training.ipynb          # 5 models trained and compared
└── 05_evaluation_summary.ipynb      # Confusion matrices, feature importance, summary
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
kagglehub
```

Install all:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub
```
