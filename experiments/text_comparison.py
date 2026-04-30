"""
快速对比实验: 模糊原型 vs 简单ML基线

问题: N1精确率12%（模糊原型），是特征不够还是模型不够？
解答: 用相同特征跑逻辑回归/随机森林做对比
"""
import sys, os
import numpy as np
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

import mne

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from granular.fuzzy_classifier import FuzzyStageClassifier, STAGES

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUB = 'SC4001'

BANDS = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'sigma': (11,16), 'beta': (16,30)}

# 1. 读取
print('[1] 读取...')
raw = mne.io.read_raw_edf(os.path.join(EDF_DIR, f'{SUB}E0-PSG.edf'), preload=True, verbose=False)
sfreq = raw.info['sfreq']
eeg_ch = [c for c in raw.ch_names if 'EEG' in c.upper()][0]
data, _ = raw[eeg_ch]; data = data.flatten()

annot = mne.read_annotations(os.path.join(EDF_DIR, f'{SUB}EC-Hypnogram.edf'))
stage_map = {'W':'W','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM'}
stages = []
for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
    code = desc.strip()[-1].upper() if desc.strip() else '?'
    s = stage_map.get(code, '?')
    ne = max(1, int(round(dur / 30)))
    se = int(round(onset / 30))
    while len(stages) < se: stages.append('?')
    for _ in range(ne): stages.append(s)
stages = np.array(stages)

# 2. 特征
print('[2] 特征提取...')
feature_names = [f'{n}_power' for n in BANDS] + ['delta_theta_ratio', 'alpha_delta_ratio']
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
    dp = X[i,0]; tp = X[i,1]; ap = X[i,2]
    X[i, col] = dp/(tp+1e-15); col += 1
    X[i, col] = ap/(dp+1e-15); col += 1

# 3. 清洗+切分
valid = stages != '?'
Xv = X[valid]; yv = stages[valid]
n = len(Xv); split = n // 2
X_train, y_train = Xv[:split], yv[:split]
X_test, y_test = Xv[split:], yv[split:]

print(f'  训练: {len(X_train)} 测试: {len(X_test)}')

# === Baseline 1: 逻辑回归 (z-score) ===
print('\n[3] 逻辑回归基线...')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xt_scaled = scaler.fit_transform(X_train)
Xs_scaled = scaler.transform(X_test)

clf_lr = LogisticRegression(max_iter=1000, C=10)
clf_lr.fit(Xt_scaled, y_train)
y_pred_lr = clf_lr.predict(Xs_scaled)

from sklearn.metrics import classification_report, confusion_matrix
print('\n逻辑回归报告:')
print(classification_report(y_test, y_pred_lr, labels=['W','N1','N2','N3','REM'], digits=4))

cm_lr = confusion_matrix(y_test, y_pred_lr, labels=['W','N1','N2','N3','REM'])
print('混淆矩阵:')
print('        ', end='')
for s in ['W','N1','N2','N3','REM']: print(f'{s:>6s}', end='')
print()
for ti, ts in enumerate(['W','N1','N2','N3','REM']):
    print(f'{ts:5s}  ', end='')
    for c in cm_lr[ti]: print(f'{c:6d}', end='')
    print()

# === Baseline 2: 随机森林 ===
print('\n[4] 随机森林基线...')
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
print('\n随机森林报告:')
print(classification_report(y_test, y_pred_rf, labels=['W','N1','N2','N3','REM'], digits=4))

# === 特征重要性 ===
print('\n特征重要性 (RF):')
for name, imp in sorted(zip(feature_names, clf_rf.feature_importances_), key=lambda x: -x[1]):
    print(f'  {name}: {imp:.4f}')

# === 模糊原型分类器 ===
print('\n[5] 模糊原型分类器...')
clf_fp = FuzzyStageClassifier(feature_names)
clf_fp.train(X_train, y_train)

correct = 0; total = 0
y_pred_fp = []
for i in range(len(X_test)):
    predicted, _ = clf_fp.predict(X_test[i])
    y_pred_fp.append(predicted)
    if predicted == y_test[i]: correct += 1
    total += 1

from sklearn.metrics import accuracy_score
print(f'\n模糊原型准确率: {accuracy_score(y_test, y_pred_fp):.4f}')
print('\n模糊原型报告:')
print(classification_report(y_test, y_pred_fp, labels=['W','N1','N2','N3','REM'], digits=4))

cm_fp = confusion_matrix(y_test, y_pred_fp, labels=['W','N1','N2','N3','REM'])
print('混淆矩阵:')
print('        ', end='')
for s in ['W','N1','N2','N3','REM']: print(f'{s:>6s}', end='')
print()
for ti, ts in enumerate(['W','N1','N2','N3','REM']):
    print(f'{ts:5s}  ', end='')
    for c in cm_fp[ti]: print(f'{c:6d}', end='')
    print()

# === 结论 ===
print('\n' + '='*60)
print('结论: 相同特征下，模糊原型 vs ML基线')
print(f'  逻辑回归: {accuracy_score(y_test, y_pred_lr):.4f}')
print(f'  随机森林: {accuracy_score(y_test, y_pred_rf):.4f}')
print(f'  模糊原型: {accuracy_score(y_test, y_pred_fp):.4f}')
print('='*60)
