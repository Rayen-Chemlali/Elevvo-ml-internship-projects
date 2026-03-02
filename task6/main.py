"""
main.py
Entry point for the GTZAN Music Genre Classification pipeline.

Run modes
---------
  python main.py              — full pipeline (steps 1-9)
  python main.py --from 5     — resume from step 5 (features CSV must exist)

Steps
-----
  1  Download dataset
  2  Visualizations (waveforms, spectrograms, MFCCs)
  3  Audio feature extraction → audio_features.csv
  4  Exploratory feature analysis
  5  Tabular ML models (Random Forest, SVM, KNN, etc.)
  6  Generate spectrogram images for CNN
  7  Custom CNN on spectrogram images
  8  Transfer Learning with VGG16
  9  Final comparison of all approaches
"""

import os
import sys
import time
import subprocess
import warnings

warnings.filterwarnings('ignore')

# ── Dependency check / auto-install ──────────────────────────────────────────
_REQUIRED = {
    'matplotlib': 'matplotlib',
    'seaborn':    'seaborn',
    'numpy':      'numpy',
    'pandas':     'pandas',
    'sklearn':    'scikit-learn',
    'librosa':    'librosa',
    'soundfile':  'soundfile',
    'joblib':     'joblib',
    'kagglehub':  'kagglehub',
    'tensorflow': 'tensorflow',
}

print("Checking dependencies...")
for _mod, _pkg in _REQUIRED.items():
    try:
        __import__(_mod)
        print(f"  {_mod}")
    except ImportError:
        print(f"  Installing {_pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', _pkg, '-q'])
        print(f"  {_pkg} installed")
print("  All dependencies ready.\n")

# ── Project imports ───────────────────────────────────────────────────────────
import pandas as pd

from config           import RESULTS_DIR, create_dirs
from data_loader      import download_dataset
from visualizations   import run_all_visualizations, run_eda
from feature_extraction  import run_extraction
from tabular_models   import run_tabular_models
from spectrogram_images  import generate_spectrogram_images
from cnn_model        import run_cnn
from transfer_learning   import run_transfer_learning
from final_comparison    import run_final_comparison


def _parse_start_step():
    """Return the step number to start from (default 1)."""
    if '--from' in sys.argv:
        idx = sys.argv.index('--from')
        try:
            return int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass
    return 1


def main():
    create_dirs()

    start_step = _parse_start_step()
    t_start    = time.time()

    print("=" * 70)
    print("   TASK 6: MUSIC GENRE CLASSIFICATION — GTZAN DATASET")
    print("=" * 70)
    if start_step > 1:
        print(f"  Resuming from step {start_step}")

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if start_step <= 1:
        audio_path, genres, genre_counts, csv_path = download_dataset()
    else:
        # Retrieve from cache without re-downloading
        import kagglehub
        _path = kagglehub.dataset_download(
            "andradaolteanu/gtzan-dataset-music-genre-classification"
        )
        audio_path = None
        for _root, _dirs, _ in os.walk(_path):
            if 'genres_original' in _dirs:
                audio_path = os.path.join(_root, 'genres_original')
                break
        if audio_path is None:
            for _root, _dirs, _ in os.walk(_path):
                if 'genres' in _dirs:
                    audio_path = os.path.join(_root, 'genres')
                    break
        if audio_path is None:
            print("ERROR: audio folder not found in kagglehub cache.")
            sys.exit(1)
        genres      = sorted([g for g in os.listdir(audio_path)
                               if os.path.isdir(os.path.join(audio_path, g))])
        genre_counts = {g: len([f for f in os.listdir(os.path.join(audio_path, g))
                                  if f.endswith(('.wav', '.au'))]) for g in genres}
        print(f"  Audio path : {audio_path}")
        print(f"  Genres     : {genres}")

    # ── Step 2: Visualizations ────────────────────────────────────────────────
    if start_step <= 2:
        run_all_visualizations(audio_path, genres, genre_counts)

    # ── Step 3: Feature extraction ────────────────────────────────────────────
    if start_step <= 3:
        df, feature_names = run_extraction(audio_path, genres)
    else:
        csv_path = os.path.join(RESULTS_DIR, 'audio_features.csv')
        if not os.path.exists(csv_path):
            print(f"ERROR: {csv_path} not found. Run steps 1-3 first.")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        print(f"  Feature CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Step 4: EDA ───────────────────────────────────────────────────────────
    if start_step <= 4:
        run_eda(df)

    # ── Step 5: Tabular models ────────────────────────────────────────────────
    tabular_results, le, best_tabular_name, best_tabular_acc = run_tabular_models(df)

    # ── Step 6: Spectrogram images ────────────────────────────────────────────
    generate_spectrogram_images(audio_path, genres)

    # ── Step 7: CNN ───────────────────────────────────────────────────────────
    cnn_acc, cnn_model = run_cnn(genres)

    # ── Step 8: Transfer Learning ─────────────────────────────────────────────
    transfer_acc = run_transfer_learning(genres)

    # ── Step 9: Final comparison ──────────────────────────────────────────────
    run_final_comparison(
        tabular_results, best_tabular_name, best_tabular_acc,
        cnn_acc, transfer_acc,
    )

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    main()
