# Machine Learning Internship Projects
### Elevvo Internship - February & March 2026
**Made by Rayen Chemlali**

---

> This repository contains all the work completed during my ML internship at **Elevvo** (February–March 2026).
> It covers 5 tasks spanning unsupervised learning, classification, time series forecasting, computer vision, and audio classification.
> A big thank you to the **Elevvo team** for the opportunity, the structure, and the support throughout this program.

---

## Repository Structure

```
├── task2-customer-segmentation/
├── task3-forest-cover-classification/
├── task6-music-genre-classification/
├── task7-walmart-sales-forecasting/
├── task8-traffic-sign-recognition/
└── README.md
```

---

## Tasks Overview

### Task 2 - Customer Segmentation
**Platform:** Google Colab | **Type:** Unsupervised Learning

Segmented 200 mall customers into 5 behavioral groups using K-Means clustering on Annual Income and Spending Score. Tested K-Means, Agglomerative Clustering, and DBSCAN. Optimal K=5 selected via elbow method, silhouette score, and Ward dendrogram - all three methods in agreement.

| Best Model | Silhouette | Davies-Bouldin | Clusters |
|-----------|-----------|----------------|---------|
| KMeans K=5 | 0.5547 | 0.5722 | 5 |

**5 Segments identified:** Average Customers, High Income High Spenders, Low Income High Spenders, High Income Low Spenders, Low Income Low Spenders.

→ See [`task2-customer-segmentation/Report.md`](task2-customer-segmentation/Report.md) and [`customer_segmentation.ipynb`](task2-customer-segmentation/customer_segmentation.ipynb) for full details.

---

### Task 3 - Forest Cover Type Classification
**Platform:** Google Colab | **Type:** Multi-class Classification

Predicted 7 forest cover types from 54 cartographic features (elevation, slope, hydrology distances, soil types) on the UCI Covertype dataset (581k rows). Used stratified sampling to manage class imbalance (103x ratio) and trained 5 classifiers.

| Model | Accuracy |
|-------|----------|
| **Extra Trees** | **88.73%** |
| XGBoost | 88.60% |
| Random Forest | 88.58% |
| KNN | 81.11% |
| Logistic Regression | 69.62% |

Elevation alone accounts for 22% of feature importance - by far the strongest predictor.

→ See [`task3-forest-cover-classification/Report.md`](task3-forest-cover-classification/Report.md) and [`forest_cover_classification.ipynb`](task3-forest-cover-classification/forest_cover_classification.ipynb) for full details.

---

### Task 6 - Music Genre Classification
**Platform:** Local Python (run locally to test a full modular pipeline) | **Type:** Audio Classification

Classified 1,000 audio clips (10 genres, GTZAN dataset) using three approaches: classical ML on extracted audio features, a custom CNN on mel spectrograms, and VGG16 transfer learning. This task was intentionally run **locally** as a way to test building and running a full modular Python pipeline (not a notebook) with proper module separation.

| Approach | Best Accuracy |
|---------|--------------|
| SVM (RBF) - tabular features | ~76% |
| Custom CNN - spectrograms | ~72–80% |
| VGG16 Transfer Learning | ~75–85% |

The pipeline is fully modular - run everything with a single `python main.py` command.

→ See [`task6-music-genre-classification/Report.md`](task6-music-genre-classification/Report.md) and the source modules for full details.

---

### Task 7 - Walmart Sales Forecasting
**Platform:** Google Colab | **Type:** Time Series Forecasting

Built a weekly sales forecasting pipeline on 421k rows of Walmart store/department data. Engineered lag features (1, 4, 8, 52 weeks), rolling statistics, holiday flags, and store encodings. Compared Prophet (trend-level) vs XGBoost (store/department-level).

| Model | RMSE | R² |
|-------|------|----|
| Prophet | $1,413,464 | 0.078 |
| **XGBoost** | **$2,157** | **0.9904** |

XGBoost achieved **99.06% R²** on the test set with minimal overfitting (train→test R² drop of only 0.006).

→ See [`task7-walmart-sales-forecasting/Report.md`](task7-walmart-sales-forecasting/Report.md) and the notebooks for full details.

---

### Task 8 - Traffic Sign Recognition
**Platform:** Google Colab | **Type:** Computer Vision / Deep Learning

Classified 43 types of German traffic signs (GTSRB dataset, 39,209 training images) using CNNs. Trained a Baseline CNN, a Deep CNN with BatchNorm, and MobileNetV2 transfer learning. Augmentation applied via `tf.data` pipeline (brightness + contrast - horizontal flip disabled since traffic signs are direction-specific).

| Model | Test Accuracy |
|-------|--------------|
| **Deep CNN** | **98.26%** |
| Baseline CNN | 96.62% |
| MobileNetV2 (frozen) | 54.37% |

Deep CNN reached **98.26%** - competitive with published GTSRB benchmarks. MobileNetV2 underperforms with frozen weights on low-resolution domain-specific images.

→ See [`task8-traffic-sign-recognition/Report.md`](task8-traffic-sign-recognition/Report.md) and [`notebooks/traffic_sign_recognition.ipynb`](task8-traffic-sign-recognition/notebooks/traffic_sign_recognition.ipynb) for full details.

---

## How to Run

### Colab tasks (2, 3, 7, 8)
Each task folder contains a `.ipynb` file - open it directly in Google Colab and run all cells. Datasets are downloaded automatically via `kagglehub`.

### Local task (6)
```bash
cd task6-music-genre-classification
pip install -r requirements.txt
python main.py

# Resume from a specific step (e.g. if features already extracted)
python main.py --from 5
```

---

## Results Summary

| Task | Problem | Best Model | Score |
|------|---------|-----------|-------|
| Task 2 | Customer Segmentation | KMeans K=5 | Silhouette 0.5547 |
| Task 3 | Forest Cover Classification | Extra Trees | 88.73% accuracy |
| Task 6 | Music Genre Classification | VGG16 Transfer Learning | ~75–85% accuracy |
| Task 7 | Walmart Sales Forecasting | XGBoost | R² 0.9904 |
| Task 8 | Traffic Sign Recognition | Deep CNN | 98.26% accuracy |

---

## Acknowledgements

A sincere thank you to **Elevvo** for organizing this internship program and providing a structured, real-world oriented set of tasks. The program pushed me to work across the full ML spectrum - from unsupervised clustering to deep learning for computer vision - and to write production-quality, well-documented code.

---

*Made by **Rayen Chemlali** - Elevvo ML Internship, February–March 2026*