"""
data_loader.py
Downloads the GTZAN dataset via kagglehub and locates the audio folder and
optional pre-extracted feature CSV.
"""

import os
import sys
import kagglehub


DATASET_SLUG = "andradaolteanu/gtzan-dataset-music-genre-classification"


def download_dataset():
    """Download (or retrieve from cache) the GTZAN dataset and return key paths."""
    print("=" * 70)
    print("STEP 1: Downloading GTZAN dataset")
    print("=" * 70)

    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Dataset path: {path}")

    audio_path = _find_audio_folder(path)
    if audio_path is None:
        _print_structure(path)
        sys.exit(1)

    print(f"Audio folder found: {audio_path}")

    genres = sorted([
        g for g in os.listdir(audio_path)
        if os.path.isdir(os.path.join(audio_path, g))
    ])

    genre_counts = {}
    for genre in genres:
        gp = os.path.join(audio_path, genre)
        cnt = len([f for f in os.listdir(gp) if f.endswith(('.wav', '.au'))])
        genre_counts[genre] = cnt

    print(f"\nGenres ({len(genres)}): {genres}")
    for g, c in genre_counts.items():
        print(f"  {g}: {c} files")

    csv_path = _find_feature_csv(path)
    if csv_path:
        print(f"\nPre-extracted feature CSV: {csv_path}")

    return audio_path, genres, genre_counts, csv_path


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_audio_folder(base_path):
    """Search for genres_original or genres subfolder."""
    for candidate in ('genres_original', 'genres'):
        for root, dirs, _ in os.walk(base_path):
            if candidate in dirs:
                return os.path.join(root, candidate)
    return None


def _find_feature_csv(base_path):
    """Return path of the first relevant feature CSV found."""
    csv_path = None
    for root, _, files in os.walk(base_path):
        for f in files:
            if f == 'features_30_sec.csv':
                return os.path.join(root, f)
            if f.endswith('.csv') and 'feature' in f.lower() and csv_path is None:
                csv_path = os.path.join(root, f)
            if f == 'features_3_sec.csv' and csv_path is None:
                csv_path = os.path.join(root, f)
    return csv_path


def _print_structure(path, max_depth=3):
    print("Dataset structure:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        if level >= max_depth:
            continue
        indent = '  ' * level
        print(f'{indent}{os.path.basename(root)}/')
        for f in files[:3]:
            print(f'{indent}  {f}')
        if len(files) > 3:
            print(f'{indent}  ... +{len(files) - 3} files')
