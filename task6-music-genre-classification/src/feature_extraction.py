"""
feature_extraction.py
Extracts tabular audio features (MFCCs, chroma, spectral, rhythmic)
from raw .wav / .au files and saves the result to a CSV.
"""

import os
import numpy as np
import pandas as pd
import librosa

from config import RESULTS_DIR


FEATURE_NAMES = (
    [f'mfcc{i}_mean' for i in range(1, 14)] +
    [f'mfcc{i}_std'  for i in range(1, 14)] +
    [f'chroma{i}_mean' for i in range(1, 13)] +
    [f'chroma{i}_std'  for i in range(1, 13)] +
    ['spectral_centroid_mean', 'spectral_centroid_std',
     'spectral_bandwidth_mean', 'spectral_bandwidth_std',
     'spectral_rolloff_mean',   'spectral_rolloff_std'] +
    [f'spectral_contrast{i}' for i in range(1, 8)] +
    ['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std',
     'tempo', 'harmony', 'perceptr']
)


def extract_features(file_path, duration=30):
    """
    Load an audio file and compute a fixed-length feature vector.
    Returns a numpy array of shape (len(FEATURE_NAMES),) or None on failure.
    """
    try:
        y, sr = librosa.load(file_path, duration=duration)

        mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        sc   = librosa.feature.spectral_centroid(y=y, sr=sr)
        sb   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        sro  = librosa.feature.spectral_rolloff(y=y, sr=sr)
        scon = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr  = librosa.feature.zero_crossing_rate(y)
        rms  = librosa.feature.rms(y=y)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)

        harmony   = np.mean(librosa.effects.harmonic(y))
        perceptr  = np.mean(librosa.effects.percussive(y))

        feature_vector = np.concatenate([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),        # 26
            np.mean(chroma, axis=1), np.std(chroma, axis=1),      # 24
            [np.mean(sc),   np.std(sc)],                           # 2
            [np.mean(sb),   np.std(sb)],                           # 2
            [np.mean(sro),  np.std(sro)],                          # 2
            np.mean(scon, axis=1),                                 # 7
            [np.mean(zcr),  np.std(zcr)],                          # 2
            [np.mean(rms),  np.std(rms)],                          # 2
            [tempo_val],                                            # 1
            [harmony, perceptr],                                    # 2
        ])
        return feature_vector

    except Exception:
        return None


def run_extraction(audio_path, genres):
    """
    Iterate over all genre folders, extract features for every file,
    and save the resulting dataframe to results/audio_features.csv.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Audio feature extraction")
    print("=" * 70)
    print(f"  Features per file: {len(FEATURE_NAMES)}")

    data, labels, filenames, errors = [], [], [], []

    for genre in genres:
        gp    = os.path.join(audio_path, genre)
        files = sorted([f for f in os.listdir(gp) if f.endswith(('.wav', '.au'))])
        print(f"\n  Genre: {genre} ({len(files)} files)")

        for i, fname in enumerate(files):
            feats = extract_features(os.path.join(gp, fname))
            if feats is not None and len(feats) == len(FEATURE_NAMES):
                data.append(feats)
                labels.append(genre)
                filenames.append(fname)
            else:
                errors.append((genre, fname))

            if (i + 1) % 25 == 0 or (i + 1) == len(files):
                print(f"    Processed: {i + 1}/{len(files)}")

    print(f"\n  Extraction complete — success: {len(data)}, errors: {len(errors)}")
    for g, f in errors[:5]:
        print(f"    - {g}/{f}")

    df = pd.DataFrame(data, columns=FEATURE_NAMES)
    df['genre']    = labels
    df['filename'] = filenames

    out_path = os.path.join(RESULTS_DIR, 'audio_features.csv')
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    return df, FEATURE_NAMES
