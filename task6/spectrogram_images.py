"""
spectrogram_images.py
Converts raw audio files into mel spectrogram PNG images organised into
train/ and test/ sub-folders for use with Keras ImageDataGenerator.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display

from sklearn.model_selection import train_test_split

from config import SPEC_DIR


def generate_spectrogram_images(audio_path, genres):
    """
    For each genre, split files 80/20, render mel spectrograms as small PNGs,
    and save them under spectrograms_data/train/<genre>/ and .../test/<genre>/.
    Already-existing images are skipped.
    """
    print("\n" + "=" * 70)
    print("STEP 6: Generating spectrogram images for CNN")
    print("=" * 70)

    # Create output directories
    for split in ('train', 'test'):
        for genre in genres:
            os.makedirs(os.path.join(SPEC_DIR, split, genre), exist_ok=True)

    total_train = total_test = 0

    for genre in genres:
        gp    = os.path.join(audio_path, genre)
        files = sorted([f for f in os.listdir(gp) if f.endswith(('.wav', '.au'))])

        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

        for flist, split in [(train_files, 'train'), (test_files, 'test')]:
            for fname in flist:
                src  = os.path.join(gp, fname)
                base = os.path.splitext(fname)[0]
                dst  = os.path.join(SPEC_DIR, split, genre, f'{base}.png')

                if os.path.exists(dst):
                    if split == 'train':
                        total_train += 1
                    else:
                        total_test += 1
                    continue

                try:
                    y, sr = librosa.load(src, duration=30)
                    S_dB  = librosa.power_to_db(
                        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128),
                        ref=np.max
                    )
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.axis('off')
                    librosa.display.specshow(S_dB, sr=sr, ax=ax)
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.savefig(dst, bbox_inches='tight', pad_inches=0, dpi=72)
                    plt.close()

                    if split == 'train':
                        total_train += 1
                    else:
                        total_test += 1
                except Exception:
                    pass

        print(f"  {genre}")

    print(f"\n  Images generated — train: {total_train}, test: {total_test}")
