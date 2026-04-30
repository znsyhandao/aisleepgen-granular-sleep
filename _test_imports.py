import sys, os, warnings, json, gc, numpy as np, mne
print('All basic imports OK', flush=True)
project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from features.spindle import detect_spindles
print('Feature imports OK', flush=True)

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
psg_files = [f for f in os.listdir(EDF_DIR) if f.endswith('-PSG.edf') and f.startswith('SC4')]
print(f'Found {len(psg_files)} PSG files', flush=True)

# Quick test first subject
SUB = 'SC4001'
psg_path = os.path.join(EDF_DIR, f'{SUB}E0-PSG.edf')
hyp_path = os.path.join(EDF_DIR, f'{SUB}EC-Hypnogram.edf')
print('Loading PSG...', flush=True)
raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
print(f'SFreq={raw.info["sfreq"]}, Channels={raw.ch_names}', flush=True)
print('SUCCESS', flush=True)
