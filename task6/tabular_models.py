"""
tabular_models.py
Trains and evaluates multiple scikit-learn classifiers on extracted audio features.
Saves models, comparison CSV, and confusion matrix plots.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from config import VIS_DIR, MODELS_DIR, RESULTS_DIR


MODELS = {
    'Random Forest':      RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':  GradientBoostingClassifier(n_estimators=200, random_state=42),
    'SVM (RBF)':          SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    'KNN':                KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression':LogisticRegression(max_iter=2000, random_state=42),
    'MLP Neural Net':     MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42),
}


def run_tabular_models(df):
    """
    Prepare data, train all classifiers, evaluate on a held-out test split,
    and return results dict plus the best model name and accuracy.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Tabular model training (scikit-learn)")
    print("=" * 70)

    X = df.drop(['genre', 'filename'], axis=1, errors='ignore')
    y = df['genre']

    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Classes: {list(le.classes_)}")

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(le,     os.path.join(MODELS_DIR, 'label_encoder.pkl'))

    results = {}

    for name, model in MODELS.items():
        print(f"\n  {'─' * 50}")
        print(f"  {name}")
        print(f"  {'─' * 50}")

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        results[name] = {'accuracy': acc, 'time': elapsed, 'y_pred': y_pred}
        print(f"  Accuracy: {acc:.4f}  |  Time: {elapsed:.2f}s")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(model, os.path.join(MODELS_DIR, f'{safe_name}.pkl'))

    # Summary table
    summary = pd.DataFrame([
        {'Model': k, 'Accuracy': v['accuracy'], 'Time (s)': round(v['time'], 2)}
        for k, v in results.items()
    ]).sort_values('Accuracy', ascending=False)

    print("\n  " + "=" * 50)
    print("  TABULAR MODEL SUMMARY")
    print("  " + "=" * 50)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(RESULTS_DIR, 'tabular_model_comparison.csv'), index=False)

    _plot_model_comparison(summary)

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_pred = results[best_name]['y_pred']
    best_acc  = results[best_name]['accuracy']

    _plot_confusion_matrices(y_test, best_pred, le.classes_, best_name, best_acc)

    return results, le, best_name, best_acc


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_model_comparison(summary):
    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(summary)))
    bars = plt.barh(summary['Model'], summary['Accuracy'], color=colors)
    plt.xlabel('Accuracy', fontsize=12)
    plt.title('Tabular model comparison', fontsize=14)
    plt.xlim(0, 1)
    for bar, acc in zip(bars, summary['Accuracy']):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{acc:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'tabular_model_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved: tabular_model_comparison.png")


def _plot_confusion_matrices(y_test, y_pred, class_names, model_name, acc):
    cm = confusion_matrix(y_test, y_pred)

    # Raw counts
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.title(f'Confusion matrix — {model_name} (Acc={acc:.3f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'best_tabular_confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Normalised
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.title(f'Normalised confusion matrix — {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'best_tabular_confusion_matrix_norm.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  saved: confusion matrices (best model: {model_name})")
