"""
测试可解释分类器 v5 + 信息粒平滑
"""
import sys, os
import numpy as np
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

import mne
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from granular.explainable_classifier import ExplainableFuzzyClassifier, STAGES

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUB = 'SC4001'

BANDS = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'sigma': (11,16), 'beta': (16,30)}

# 1. 读取
print('[1] 读取...')
raw = mne.io.read_raw_edf(os.path.join(EDF_DIR, f'{SUB}E0-PSG.edf'), preload=True, verbose=False)
sfreq = raw.info['sfreq']
eeg_ch = [c for c in raw.ch_names if 'EEG' in c.upper()][0]; data, _ = raw[eeg_ch]; data = data.flatten()

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

# 3. 切分
valid = stages != '?'
Xv = X[valid]; yv = stages[valid]
n = len(Xv); split = n // 2
X_train, y_train = Xv[:split], yv[:split]
X_test, y_test = Xv[split:], yv[split:]

# 4. 训练+预测
print('\n[3] 可解释分类器...')
clf = ExplainableFuzzyClassifier()
clf.fit(X_train, y_train, feature_names)

# 无平滑
y_pred_raw = clf.predict(X_test, smooth_window=1)
# 3-epoch 平滑
y_pred_smooth = clf.predict(X_test, smooth_window=3)
# 5-epoch 平滑
y_pred_smooth5 = clf.predict(X_test, smooth_window=5)

print('\n原始（无平滑）:')
print(classification_report(y_test, y_pred_raw, labels=STAGES, digits=4))

print('\n3-epoch 平滑:')
print(classification_report(y_test, y_pred_smooth, labels=STAGES, digits=4))

print('\n5-epoch 平滑:')
print(classification_report(y_test, y_pred_smooth5, labels=STAGES, digits=4))

# 5. 对比 baseline LR
print('\n[4] 对比 LR z-score 基线...')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
Xs_scaled = scaler.transform(X_test)
clf_lr = LogisticRegression(max_iter=2000, C=10)
clf_lr.fit(scaler.transform(X_train), y_train)
y_pred_lr = clf_lr.predict(Xs_scaled)

print('\nLR z-score:')
print(classification_report(y_test, y_pred_lr, labels=STAGES, digits=4))

# 6. summary
print('\n' + '='*50)
print('准确率对比:')
print(f'  原始(无平滑): {accuracy_score(y_test, y_pred_raw):.4f}')
print(f'  3-epoch平滑:  {accuracy_score(y_test, y_pred_smooth):.4f}')
print(f'  5-epoch平滑:  {accuracy_score(y_test, y_pred_smooth5):.4f}')
print(f'  LR基线:       {accuracy_score(y_test, y_pred_lr):.4f}')
print('='*50)

# 7. 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)

# 真实阶段
ax = axes[0]
for si, (s, c) in enumerate([('W','#ff6b6b'),('N1','#ffd93d'),('N2','#6bcb77'),('N3','#4d96ff'),('REM','#9b59b6')]):
    idx = np.where(y_test == s)[0]
    ax.scatter(idx, [si]*len(idx), c=c, s=2, alpha=0.6)
ax.set_yticks(range(5)); ax.set_yticklabels(STAGES)
ax.set_ylabel('真实')

# 预测（原始）
ax = axes[1]
y_nums = [STAGES.index(s) for s in y_pred_raw]
ax.plot(y_nums, 'b-', lw=0.4, alpha=0.7)
ax.set_yticks(range(5)); ax.set_yticklabels(STAGES)
ax.set_ylabel('原始预测')

# 预测（平稳）
ax = axes[2]
y_nums_s = [STAGES.index(s) for s in y_pred_smooth]
ax.plot(y_nums_s, 'g-', lw=0.4, alpha=0.7)
ax.set_yticks(range(5)); ax.set_yticklabels(STAGES)
ax.set_ylabel('3平滑')

# LR
ax = axes[3]
y_nums_lr = [STAGES.index(s) for s in y_pred_lr]
ax.plot(y_nums_lr, 'r-', lw=0.4, alpha=0.7)
ax.set_yticks(range(5)); ax.set_yticklabels(STAGES)
ax.set_ylabel('LR基线')
ax.set_xlabel('Epoch (30s)')

plt.tight_layout()
plt.savefig(os.path.join(project_dir, 'results', SUB, 'v5_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'\n可视化: results/SC4001/v5_comparison.png')

# 8. 样本规则推理
print('\n[5] 样本规则推理示例:')
example_idx = [1050, 1250, 1450, 1650, 1850]
for ei in example_idx:
    if ei >= len(X_test): continue
    probs = clf.predict_proba_single(X_test[ei])
    probs_str = ', '.join([f'{STAGES[si]}:{probs[si]:.3f}' for si in range(5)])
    print(f'  Epoch {ei}: true={y_test[ei]} → {probs_str}')
