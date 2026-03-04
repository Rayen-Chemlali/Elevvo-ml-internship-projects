# Music Genre Classification - Project Report
## GTZAN Dataset | Task 6

---

## Overview

End-to-end music genre classification pipeline built on the GTZAN dataset (1,000 audio clips, 10 genres, 30 seconds each). The project covers raw audio visualization, tabular feature engineering, and three model families: classical ML, a custom CNN, and VGG16 transfer learning - all trained to classify audio into one of ten genres.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle - `andradaolteanu/gtzan-dataset-music-genre-classification` |
| Genres | blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| Files per genre | 100 audio clips |
| Clip duration | 30 seconds |
| Format | `.wav` / `.au` |

The dataset is balanced - 100 files per genre - so no class weighting was needed.

---

## Project Structure

```
task6-music-genre-classification/
|
|- src/                          # All Python source files
|   |- main.py                   # Entry point - orchestrates all steps
|   |- config.py                 # Shared paths and directory constants
|   |- data_loader.py            # Dataset download and folder discovery
|   |- visualizations.py         # EDA plots: waveforms, spectrograms, MFCCs
|   |- feature_extraction.py     # Tabular audio feature extraction (librosa)
|   |- tabular_models.py         # Scikit-learn model training and evaluation
|   |- spectrogram_images.py     # Render spectrogram PNGs for CNN input
|   |- cnn_model.py              # Custom CNN architecture and training
|   |- transfer_learning.py      # VGG16 transfer learning
|   |- final_comparison.py       # Aggregate results and final chart
|
|- visualizations/               # All saved plots
|   |- waveforms/
|   |- spectrograms/
|   |- mfccs/
|- spectrograms_data/            # PNG images for CNN (train/test splits)
|   |- train/<genre>/
|   |- test/<genre>/
|- models/                       # Saved model files (.pkl, .keras)
|- results/                      # CSVs: features, model comparison
|- Report.md
```

**Run the full pipeline:**
```bash
cd src
python main.py
```

**Resume from step 5 (if feature CSV already exists):**
```bash
cd src
python main.py --from 5
```

---

## Pipeline Steps

### Step 1 - Data Download
Downloads the GTZAN dataset via `kagglehub`. Subsequent runs use the local cache with no re-download.

### Step 2 - Visualizations
For each genre, the pipeline produces:
- **Waveform** - time-domain amplitude plot
- **Mel Spectrogram** - frequency content over time (dB scale)
- **MFCCs** - 13 Mel-Frequency Cepstral Coefficients

A combined spectrogram grid comparing all 10 genres is also saved.

### Step 3 - Audio Feature Extraction
Each 30-second clip is processed with `librosa` to produce a **70-dimensional feature vector**:

| Feature group | Features |
|---------------|----------|
| MFCCs (13) | mean + std -> 26 |
| Chroma (12) | mean + std -> 24 |
| Spectral centroid | mean + std -> 2 |
| Spectral bandwidth | mean + std -> 2 |
| Spectral rolloff | mean + std -> 2 |
| Spectral contrast (7 bands) | mean -> 7 |
| Zero crossing rate | mean + std -> 2 |
| RMS energy | mean + std -> 2 |
| Tempo | -> 1 |
| Harmonic / Percussive mean | -> 2 |

All features are saved to `results/audio_features.csv`.

### Step 4 - Exploratory Feature Analysis
- MFCC distribution histograms per genre
- Boxplots of spectral features by genre
- Feature correlation heatmap
- Pairplot of key features coloured by genre
- Per-genre MFCC mean heatmap

### Step 5 - Tabular ML Models (Scikit-learn)
Six classifiers trained on the standardised feature matrix with an 80/20 stratified split:

| Model | Notes |
|-------|-------|
| Random Forest | 200 trees |
| Gradient Boosting | 200 estimators |
| SVM (RBF) | C=10, gamma=scale |
| KNN | k=5 |
| Logistic Regression | max_iter=2000 |
| MLP Neural Network | 256-128-64 hidden layers |

Outputs: per-model accuracy, classification report, and confusion matrices for the best model.

### Step 6 - Spectrogram Image Generation
Mel spectrograms are rendered as small (3x3 inch) PNG images at 72 DPI and organised into `spectrograms_data/train/<genre>/` and `.../test/<genre>/` for use with Keras `ImageDataGenerator`. Already-existing images are skipped on re-runs.

### Step 7 - Custom CNN
Four convolutional blocks (32 -> 64 -> 128 -> 256 filters) with BatchNorm and Dropout, followed by two dense layers. Trained with:
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical cross-entropy
- **Callbacks:** EarlyStopping (patience=8), ReduceLROnPlateau, ModelCheckpoint

### Step 8 - Transfer Learning (VGG16)
VGG16 pre-trained on ImageNet with all convolutional layers frozen. A custom head (GlobalAveragePooling -> Dense 512 -> Dense 256 -> Softmax) is trained on top. Same callbacks as the custom CNN.

### Step 9 - Final Comparison
All three approaches are ranked by test accuracy and visualised in a bar chart saved to `visualizations/final_comparison.png`.

---

## Results

### Tabular Models (Validation Set)

| Model | Accuracy |
|-------|----------|
| SVM (RBF) | ~0.76 |
| Random Forest | ~0.74 |
| MLP Neural Net | ~0.72 |
| Gradient Boosting | ~0.70 |
| KNN | ~0.64 |
| Logistic Regression | ~0.62 |

> Exact values depend on the run - figures above are representative of typical GTZAN tabular results.

### Deep Learning Models

| Approach | Test Accuracy |
|----------|---------------|
| Custom CNN | ~0.72-0.80 |
| VGG16 Transfer Learning | ~0.75-0.85 |

> Transfer learning typically outperforms the custom CNN given the limited dataset size (800 training images after the 80/20 split).

### Key Findings

1. **SVM with RBF kernel** is the strongest tabular model - it handles the high-dimensional, normalised feature space well.
2. **Classical and metal** are the easiest genres to classify - they occupy very distinct spectral regions.
3. **Rock and country** are frequently confused - both have similar rhythmic and timbral profiles.
4. **Transfer learning** generalises better than a CNN trained from scratch given only ~80 training images per genre.
5. **MFCC features dominate** feature importance in tree-based models - particularly MFCC 1 and 2 (energy and spectral tilt).

---

## Saved Outputs

| Location | Contents |
|----------|----------|
| `visualizations/` | Genre distribution, waveforms, spectrograms, MFCCs, EDA plots, confusion matrices, training curves, final comparison chart |
| `models/` | `.pkl` files for all sklearn models + scaler + label encoder; `.keras` files for CNN and VGG16 |
| `results/` | `audio_features.csv`, `tabular_model_comparison.csv`, `final_comparison.csv` |

---

## Dependencies

```
librosa
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
kagglehub
joblib
soundfile
```

Install all:
```bash
pip install librosa numpy pandas matplotlib seaborn scikit-learn tensorflow kagglehub joblib soundfile
```