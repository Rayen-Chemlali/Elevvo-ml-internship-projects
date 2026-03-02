"""
config.py
Shared paths and constants for the GTZAN Music Genre Classification project.
"""

import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VIS_DIR     = os.path.join(BASE_DIR, 'visualizations')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SPEC_DIR    = os.path.join(BASE_DIR, 'spectrograms_data')

# Sub-folders for visualizations
VIS_WAVEFORMS    = os.path.join(VIS_DIR, 'waveforms')
VIS_SPECTROGRAMS = os.path.join(VIS_DIR, 'spectrograms')
VIS_MFCCS        = os.path.join(VIS_DIR, 'mfccs')

ALL_DIRS = [
    VIS_DIR, VIS_WAVEFORMS, VIS_SPECTROGRAMS,
    VIS_MFCCS, MODELS_DIR, RESULTS_DIR, SPEC_DIR,
]

def create_dirs():
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)
