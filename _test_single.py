"""Quick test of single subject pipeline"""
import sys, os, warnings, gc
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)
warnings.filterwarnings('ignore')

import numpy as np
import mne

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from features.spindle import detect_spindles

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUB = 'SC4001'

print('=== Test SC4001 ===')
print('Step 1: Loading PSG...')
raw = mne.io.read_raw_edf(os.path.join(EDF_DIR, f'{SUB}E0-PSG.edf'), preload=True, verbose=False)
print(f'  sfreq={raw.info["sfreq"]}, channels={raw.ch_names}')

eeg_ch = [c for c in raw.ch_names if 'EEG' in c.upper()][0]
data, _ = raw[eeg_ch]
data = data.flatten()
print(f'  data shape={data.shape}')

print('Step 2: Loading Hypnogram...')
annot = mne.read_annotations(os.path.join(EDF_DIR, f'{SUB}EC-Hypnogram.edf'))
print(f'  annotations={len(annot)}')

STAGE_MAP = {'W':'W','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM'}
stages = []
for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
    code = desc.strip()[-1].upper() if desc.strip() else '?'
    s = STAGE_MAP.get(code, '?')
    ne = max(1, int(round(dur / 30)))
    se = int(round(onset / 30))
    while len(stages) < se: stages.append('?')
    for _ in range(ne): stages.append(s)
stages = np.array(stages)
print(f'  stages shape={stages.shape}')
print(f'  unique={np.unique(stages)}')

valid = stages != '?'
print(f'  valid epochs={valid.sum()}/{len(stages)}')

print('Step 3: Extracting features...')
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'sigma':(11,16),'beta':(16,30)}
FEATURE_NAMES = [f'{n}_power' for n in BANDS] + ['delta_theta_ratio','alpha_delta_ratio','spindle_density']
sfreq = raw.info['sfreq']
epoch_len = int(30 * sfreq)
n_epochs = len(stages)
n_feat = len(FEATURE_NAMES)
X = np.zeros((n_epochs, n_feat))

for i in range(n_epochs):
    ed = data[i*epoch_len:(i+1)*epoch_len]
    if len(ed) < epoch_len // 2: continue
    freqs, psd = compute_epoch_spectrum(ed, sfreq)
    col = 0
    for name, band in BANDS.items():
        X[i, col] = band_power(freqs, psd, band); col += 1
    dp = X[i,0]; tp = X[i,1]; ap = X[i,2]
    X[i, col] = dp/(tp+1e-15); col += 1
    X[i, col] = ap/(dp+1e-15); col += 1
    _, density, _ = detect_spindles(ed, sfreq, amplitude_threshold=2.0)
    X[i, col] = density

print(f'  X shape={X.shape}')
Xv = X[valid]; yv = stages[valid]
print(f'  valid X shape={Xv.shape}')

print('Step 4: Training LR...')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

n = len(Xv)
split = n // 2
X_train, y_train = Xv[:split], yv[:split]
X_test, y_test = Xv[split:], yv[split:]

scaler = StandardScaler()
Xt = scaler.fit_transform(X_train)
Xs = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000, C=10)
lr.fit(Xt, y_train)
y_pred = lr.predict(Xs)
acc = accuracy_score(y_test, y_pred)
print(f'  Accuracy: {acc:.4f}')

# No spindle
lr_ns = LogisticRegression(max_iter=2000, C=10)
lr_ns.fit(Xt[:,:7], y_train)
y_pred_ns = lr_ns.predict(Xs[:,:7])
acc_ns = accuracy_score(y_test, y_pred_ns)
print(f'  Accuracy (no spindle): {acc_ns:.4f}')
print('=== SUCCESS ===')
