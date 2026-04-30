"""
实验3: 模糊原型分类器测试
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

# 3. 划分训练/测试 (50/50 时序划分)
valid = stages != '?'
X_valid = X[valid]
y_valid = stages[valid]
n = len(X_valid)
split = n // 2

X_train, y_train = X_valid[:split], y_valid[:split]
X_test, y_test = X_valid[split:], y_valid[split:]

print(f'  训练: {len(X_train)} epochs')
print(f'  测试: {len(X_test)} epochs')

# 4. 训练
print('\n[3] 训练模糊分类器...')
clf = FuzzyStageClassifier(feature_names)
clf.train(X_train, y_train)

# 查看原型向量
print('\n  阶段原型向量 (5维 → W/N1/N2/N3/REM):')
for stage, proto in clf.stage_prototypes.items():
    print(f'    {stage}: {np.round(proto, 4)}')

# 5. 测试
print('\n[4] 测试...')
conf_matrix = {s: {s2: 0 for s2 in STAGES} for s in STAGES}
correct = 0
total = 0

for i in range(len(X_test)):
    true_s = y_test[i]
    predicted, confs = clf.predict(X_test[i])
    
    conf_matrix[true_s][predicted] += 1
    if predicted == true_s:
        correct += 1
    total += 1

accuracy = correct / total if total > 0 else 0
print(f'\n准确率: {accuracy:.4f} ({correct}/{total})')

print('\n混淆矩阵:')
print('         ', end='')
for s in STAGES: print(f'{s:>6s}', end='')
print()
for true_s in STAGES:
    print(f'{true_s:5s}  ', end='')
    for pred_s in STAGES:
        n = conf_matrix[true_s][pred_s]
        print(f'{n:6d}', end='')
    print()

# 每阶段召回率
print('\n各阶段召回率:')
for s in STAGES:
    true_n = conf_matrix[s][s]
    total_s = sum(conf_matrix[s].values())
    recall = true_n / total_s if total_s > 0 else 0
    print(f'  {s}: {recall:.4f} ({true_n}/{total_s})')

# 每阶段精确率
print('\n各阶段精确率:')
for s in STAGES:
    pred_n = sum(conf_matrix[ts][s] for ts in STAGES)
    tp = conf_matrix[s][s]
    prec = tp / pred_n if pred_n > 0 else 0
    print(f'  {s}: {prec:.4f} ({tp}/{pred_n})')

print('\n完成!')
