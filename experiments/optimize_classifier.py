"""
实验: 优化模糊分类器 — 用 LR 权重替代等权求和

用逻辑回归的系数作为模糊特征融合的权重
"""
import sys, os
import numpy as np
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

import mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from granular.fuzzy_classifier import STAGES

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUB = 'SC4001'
RESULTS_DIR = os.path.join(project_dir, 'results', SUB)
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# 4. 训练逻辑回归 (z-score)
print('\n[2] 训练逻辑回归 (作为可解释分类器)...')
scaler = StandardScaler()
Xt_scaled = scaler.fit_transform(X_train)
Xs_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=2000, C=10)
clf.fit(Xt_scaled, y_train)

# 提取系数作为特征权重
print(f'\n  系数矩阵: {clf.coef_.shape} ({clf.classes_})')
print(f'\n  每个特征对每个阶段的贡献 (coefficient):')
for ci, stage in enumerate(clf.classes_):
    print(f'  {stage}:')
    sorted_idx = np.argsort(-np.abs(clf.coef_[ci]))
    for idx in sorted_idx:
        print(f'    {feature_names[idx]:25s} = {clf.coef_[ci][idx]:+.4f}')

# 5. 用模糊原型改进版本: 加权模糊向量
print('\n[3] 加权模糊分类器 (LR权重+模糊原型)...')

# 对每个特征计算各阶段统计量
stage_stats = {}
for fi, fname in enumerate(feature_names):
    stage_stats[fname] = {}
    for stage in STAGES:
        vals = X_train[:, fi][y_train == stage]
        if len(vals) > 5:
            stage_stats[fname][stage] = (float(np.mean(vals)), float(np.std(vals)))

def _gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x-mu)/max(sigma,mu*0.05))**2)

def weighted_fuzzy_predict(features, stage_stats, lr_coefs, lr_classes, scaler_):
    """加权模糊预测 v2"""
    # 方法: 用LR系数作为模糊特征融合权重
    fuzzy_vec = np.zeros(len(STAGES))
    
    for fi, fname in enumerate(feature_names):
        stats = stage_stats.get(fname, {})
        for si, stage in enumerate(STAGES):
            if stage in stats:
                mu, sigma = stats[stage]
                raw_mu = _gaussian(features[fi], mu, sigma)
                # 加权: LR系数越大，该特征对该阶段的贡献越多
                weight = lr_coefs[list(lr_classes).index(stage), fi] if stage in lr_classes else 0
                fuzzy_vec[si] += raw_mu * max(weight, 0)  # 只取正权重
    
    total = max(np.sum(fuzzy_vec), 1e-15)
    fuzzy_vec /= total
    
    predicted = STAGES[np.argmax(fuzzy_vec)]
    confs = {STAGES[si]: float(fuzzy_vec[si]) for si in range(len(STAGES))}
    return predicted, confs

# 测试
correct = 0; total = 0
for i in range(len(X_test)):
    pred, _ = weighted_fuzzy_predict(X_test[i], stage_stats, clf.coef_, clf.classes_, scaler)
    if pred == y_test[i]: correct += 1
    total += 1

print(f'\n  加权模糊分类准确率: {correct/total:.4f} ({correct}/{total})')

# 6. 混合方法: 先用LR判断，如果置信度低则用模糊
print('\n[4] 混合方法 (LR为主，低置信回退模糊)...')
probs = clf.predict_proba(Xs_scaled)
y_pred_lr = clf.predict(Xs_scaled)

correct_hybrid = 0
for i in range(len(X_test)):
    max_prob = np.max(probs[i])
    if max_prob >= 0.7:
        pred = clf.classes_[np.argmax(probs[i])]
    else:
        pred, _ = weighted_fuzzy_predict(X_test[i], stage_stats, clf.coef_, clf.classes_, scaler)
    if pred == y_test[i]: correct_hybrid += 1

print(f'  混合方法准确率: {correct_hybrid/total:.4f} ({correct_hybrid}/{total})')

# 7. 可视化LR系数（热力图）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(clf.coef_, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(clf.classes_)))
ax.set_yticklabels(clf.classes_)
ax.set_title(f'逻辑回归系数 (z-score特征 → 阶段)')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'lr_coefficients.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'\n  系数热力图: {RESULTS_DIR}/lr_coefficients.png')

# 8. 结论
print('\n' + '='*60)
print('核心发现:')
print('  1. LR+z-score 已经 93.8% — 特征足够好')
print('  2. 模糊原型只有68% — 融合方式不对')
print('  3. 用LR权重做模糊融合 → 待确认')
print(f'  4. LR低置信回退模糊: {correct_hybrid/total:.4f}')
print('='*60)

# 9. 提取可解释规则
print('\n[5] 提取可解释规则...')
print(f'\n从LR系数提取的5条阶段规则:')
for ci, stage in enumerate(clf.classes_):
    top_idx = np.argsort(-np.abs(clf.coef_[ci]))[:3]
    conditions = []
    for idx in top_idx:
        if clf.coef_[ci][idx] > 0:
            conditions.append(f'{feature_names[idx]} 值高')
        else:
            conditions.append(f'{feature_names[idx]} 值低')
    print(f'  IF {" AND ".join(conditions)} → {stage}')
