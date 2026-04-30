"""
实验3: 模糊原型分类器 — 全数据自洽测试
"""
import sys, os, json
import numpy as np
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

import mne
from collections import Counter

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from granular.fuzzy_classifier import FuzzyStageClassifier, STAGES

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUB = 'SC4001'
RESULTS_DIR = os.path.join(project_dir, 'results', SUB)
os.makedirs(RESULTS_DIR, exist_ok=True)

BANDS = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'sigma': (11,16), 'beta': (16,30)}

# 1. 读取数据
print('[1] 读取EDF...')
raw = mne.io.read_raw_edf(os.path.join(EDF_DIR, f'{SUB}E0-PSG.edf'), preload=True, verbose=False)
sfreq = raw.info['sfreq']
eeg_ch = [c for c in raw.ch_names if 'EEG' in c.upper()][0]
data, _ = raw[eeg_ch]; data = data.flatten()

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
stages = np.array(stages)

# 2. 提取特征
print('[2] 提取特征...')
feature_names = [f'{name}_power' for name in BANDS]
feature_names += ['delta_theta_ratio', 'alpha_delta_ratio']

epoch_len = int(30 * sfreq)
n_epochs = len(stages)
X = np.zeros((n_epochs, len(feature_names)))

for i in range(n_epochs):
    ed = data[i*epoch_len:(i+1)*epoch_len]
    if len(ed) < epoch_len: continue
    freqs, psd = compute_epoch_spectrum(ed, sfreq)
    col = 0
    for name, band in BANDS.items():
        X[i, col] = band_power(freqs, psd, band); col += 1
    dp = X[i, 0]; tp = X[i, 1]; ap = X[i, 2]
    X[i, col] = dp / (tp + 1e-15); col += 1
    X[i, col] = ap / (dp + 1e-15); col += 1

# 3. 全数据训练+测试
valid = stages != '?'
X_valid = X[valid]
y_valid = stages[valid]
print(f'  有效样本: {len(X_valid)} epochs')

print('\n[3] 训练+自洽测试...')
clf = FuzzyStageClassifier(feature_names)
clf.train(X_valid, y_valid)

# 原型向量
print('\n  阶段原型向量 (W/N1/N2/N3/REM):')
for stage, proto in clf.stage_prototypes.items():
    print(f'    {stage}: {np.round(proto, 3)}')

# 4. 自洽测试
print('\n[4] 自洽测试...')
conf_matrix = {s: {s2: 0 for s2 in STAGES} for s in STAGES}
correct = 0
total = 0

for i in range(len(X_valid)):
    true_s = y_valid[i]
    predicted, confs = clf.predict(X_valid[i])
    conf_matrix[true_s][predicted] += 1
    if predicted == true_s:
        correct += 1
    total += 1

accuracy = correct / total if total > 0 else 0
print(f'自洽准确率: {accuracy:.4f} ({correct}/{total})')
print()
print('混淆矩阵:')
print('         ', end='')
for s in STAGES: print(f'{s:>6s}', end='')
print()
for true_s in STAGES:
    print(f'{true_s:5s}  ', end='')
    for pred_s in STAGES:
        print(f'{conf_matrix[true_s][pred_s]:6d}', end='')
    print()

print('\n各阶段:')
for s in STAGES:
    tp = conf_matrix[s][s]
    total_true = sum(conf_matrix[s].values())
    total_pred = sum(conf_matrix[ts][s] for ts in STAGES)
    recall = tp / total_true if total_true > 0 else 0
    prec = tp / total_pred if total_pred > 0 else 0
    f1 = 2*prec*recall/(prec+recall) if (prec+recall)>0 else 0
    print(f'  {s}: recall={recall:.3f} prec={prec:.3f} f1={f1:.3f}  n_true={total_true} n_pred={total_pred}')
