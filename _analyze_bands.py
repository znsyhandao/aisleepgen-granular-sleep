import sys, os, json
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)
import numpy as np

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUB = 'SC4001'

import mne
raw = mne.io.read_raw_edf(os.path.join(EDF_DIR, f'{SUB}E0-PSG.edf'), preload=True, verbose=False)
sfreq = raw.info['sfreq']
eeg_ch = [c for c in raw.ch_names if 'EEG' in c.upper()][0]
data, _ = raw[eeg_ch]
data = data.flatten()

# 读取阶段
annot = mne.read_annotations(os.path.join(EDF_DIR, f'{SUB}EC-Hypnogram.edf'))
stage_map = {'W': 'W', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'N3', 'R': 'REM'}
stages = []
for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
    code = desc.strip()[-1].upper() if desc.strip() else '?'
    s = stage_map.get(code, '?')
    ne = max(1, int(round(dur / 30)))
    se = int(round(onset / 30))
    while len(stages) < se: stages.append('?')
    for _ in range(ne): stages.append(s)
stages = np.array(stages[:len(data)//int(30*sfreq)])

BANDS = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'sigma': (11,16), 'beta': (16,30)}
epoch_len = int(30 * sfreq)
n_epochs = len(stages)

features = {}
for name in BANDS: features[name] = np.zeros(n_epochs)

for i in range(n_epochs):
    ed = data[i*epoch_len:(i+1)*epoch_len]
    if len(ed) < epoch_len: continue
    freqs, psd = compute_epoch_spectrum(ed, sfreq)
    for name, band in BANDS.items():
        features[name][i] = band_power(freqs, psd, band)

# 按阶段统计
for fname in features:
    print(f'\n=== {fname}_power ===')
    print(f'  ALL:  mean={np.mean(features[fname]):.4e}  std={np.std(features[fname]):.4e}  min={np.min(features[fname]):.4e}  max={np.max(features[fname]):.4e}')
    for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
        vals = features[fname][stages == stage]
        if len(vals) > 0:
            print(f'  {stage:4s}: mean={np.mean(vals):.4e}  std={np.std(vals):.4e}  min={np.min(vals):.4e}  max={np.max(vals):.4e}  (n={len(vals)})')

# delta 分布直方图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 7))
band_names = list(BANDS.keys())
for idx, fname in enumerate(band_names):
    ax = axes[idx//3, idx%3]
    for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
        vals = features[fname][stages == stage]
        if len(vals) > 5:
            ax.hist(vals, bins=50, alpha=0.5, label=stage)
    ax.set_title(fname)
    ax.legend(fontsize=8)
    ax.set_xscale('log')

plt.tight_layout()
plt.savefig(r'D:\AISleepGen_GranularSleep\results\SC4001\band_stats.png', dpi=150)
plt.close()
print(f'\n分布图已保存')
