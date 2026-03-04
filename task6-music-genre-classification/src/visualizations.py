"""
visualizations.py
EDA visualizations: genre distribution, waveforms, mel spectrograms, MFCCs,
and comparative plots across all genres.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

from config import VIS_DIR, VIS_WAVEFORMS, VIS_SPECTROGRAMS, VIS_MFCCS


def plot_genre_distribution(genre_counts):
    """Bar chart of file counts per genre."""
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(genre_counts)))
    bars = plt.bar(genre_counts.keys(), genre_counts.values(), color=colors)
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Number of files', fontsize=12)
    plt.title('Audio file distribution by genre', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, genre_counts.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'genre_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved: genre_distribution.png")


def plot_per_genre(audio_path, genres):
    """Waveform, mel spectrogram, and MFCC plot for one sample per genre."""
    print("\n  Generating per-genre visualizations...")
    for genre in genres:
        gp = os.path.join(audio_path, genre)
        files = sorted([f for f in os.listdir(gp) if f.endswith(('.wav', '.au'))])
        if not files:
            continue

        fpath = os.path.join(gp, files[0])
        try:
            y, sr = librosa.load(fpath, duration=30)

            # Waveform
            fig, ax = plt.subplots(figsize=(12, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.7)
            ax.set_title(f'Waveform - {genre}', fontsize=12)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_WAVEFORMS, f'{genre}_waveform.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

            # Mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(12, 5))
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title(f'Mel Spectrogram - {genre}', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_SPECTROGRAMS, f'{genre}_spectrogram.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            fig, ax = plt.subplots(figsize=(12, 5))
            img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
            fig.colorbar(img, ax=ax)
            ax.set_title(f'MFCCs - {genre}', fontsize=12)
            ax.set_ylabel('MFCC Coefficients')
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_MFCCS, f'{genre}_mfcc.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

            print(f"    {genre}")
        except Exception as e:
            print(f"    {genre} - error: {e}")


def plot_all_spectrograms(audio_path, genres):
    """Grid comparison of mel spectrograms for all genres."""
    ncols = 5
    nrows = (len(genres) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    for idx, genre in enumerate(genres):
        gp = os.path.join(audio_path, genre)
        files = sorted([f for f in os.listdir(gp) if f.endswith(('.wav', '.au'))])
        if files:
            try:
                y, sr = librosa.load(os.path.join(gp, files[0]), duration=30)
                S_dB = librosa.power_to_db(
                    librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max
                )
                librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axes[idx])
                axes[idx].set_title(genre.upper(), fontsize=10, fontweight='bold')
                axes[idx].set_xlabel('')
                axes[idx].set_ylabel('')
            except Exception:
                axes[idx].set_title(f'{genre} (error)')

    for idx in range(len(genres), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Mel spectrogram comparison by genre', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'all_genres_spectrograms_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved: all_genres_spectrograms_comparison.png")


def run_all_visualizations(audio_path, genres, genre_counts):
    print("\n" + "=" * 70)
    print("STEP 2: Visualizations")
    print("=" * 70)
    plot_genre_distribution(genre_counts)
    plot_per_genre(audio_path, genres)
    plot_all_spectrograms(audio_path, genres)


# ── EDA on extracted features ─────────────────────────────────────────────────

def run_eda(df):
    """Plots for the extracted tabular feature dataframe."""
    print("\n" + "=" * 70)
    print("STEP 4: Exploratory feature analysis")
    print("=" * 70)

    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")

    # MFCC distributions per genre
    mfcc_mean_cols = [c for c in df.columns if 'mfcc' in c and 'mean' in c]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(mfcc_mean_cols[:6]):
        for genre in df['genre'].unique():
            axes[i].hist(df[df['genre'] == genre][col], alpha=0.5, label=genre, bins=15)
        axes[i].set_title(col)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        if i == 0:
            axes[i].legend(fontsize=6, loc='upper right')
    plt.suptitle('MFCC distributions by genre', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'mfcc_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved: mfcc_distributions.png")

    # Boxplots for spectral features
    spectral_cols = ['spectral_centroid_mean', 'spectral_bandwidth_mean',
                     'spectral_rolloff_mean', 'tempo', 'zcr_mean', 'rms_mean']
    existing = [c for c in spectral_cols if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, col in enumerate(existing[:6]):
        df.boxplot(column=col, by='genre', ax=axes[idx])
        axes[idx].set_title(col)
        axes[idx].set_xlabel('')
        axes[idx].tick_params(axis='x', rotation=45)
    for idx in range(len(existing), 6):
        axes[idx].set_visible(False)
    plt.suptitle('Spectral features by genre', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'spectral_features_boxplot.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved: spectral_features_boxplot.png")

    # Correlation matrix
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(18, 15))
    sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.1,
                cbar_kws={'shrink': 0.6}, square=True)
    plt.title('Feature correlation matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'correlation_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved: correlation_matrix.png")

    # Pairplot
    subset_cols = ['mfcc1_mean', 'mfcc2_mean', 'spectral_centroid_mean', 'tempo', 'genre']
    existing_sub = [c for c in subset_cols if c in df.columns]
    if len(existing_sub) >= 3:
        sns.pairplot(df[existing_sub], hue='genre', diag_kind='kde',
                     plot_kws={'alpha': 0.5, 's': 20})
        plt.suptitle('Pairplot of selected features', y=1.02, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, 'pairplot_features.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  saved: pairplot_features.png")

    # MFCC mean heatmap per genre
    mfcc_all = [c for c in df.columns if 'mfcc' in c and 'mean' in c]
    genre_mfcc = df.groupby('genre')[mfcc_all].mean()
    plt.figure(figsize=(14, 8))
    sns.heatmap(genre_mfcc, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5)
    plt.title('Mean MFCCs per genre', fontsize=14)
    plt.ylabel('Genre')
    plt.xlabel('MFCC Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'mfcc_heatmap_by_genre.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved: mfcc_heatmap_by_genre.png")
