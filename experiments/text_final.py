"""
最终测试: 信息粒增强型睡眠分期

输出:
- 准确率 + 混淆矩阵
- 信息粒细胞
- 过渡期标记
- 睡眠分析报告
"""
import sys, os
import numpy as np
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

import mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from granular.granular_stager import GranularSleepStager, STAGES

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
valid = stages != '?'; Xv = X[valid]; yv = stages[valid]
n = len(Xv); split = n // 2
X_train, y_train = Xv[:split], yv[:split]
X_test, y_test = Xv[split:], yv[split:]

# 4. 训练
print('\n[3] 训练...')
scaler = StandardScaler().fit(X_train)
Xt_s = scaler.transform(X_train)
Xs_s = scaler.transform(X_test)

base_clf = LogisticRegression(max_iter=2000, C=10)
base_clf.fit(Xt_s, y_train)

stager = GranularSleepStager(base_clf)
stager.fit(X_train, y_train, scaler)

# 5. 预测
print('[4] 预测 + 信息粒...')
y_pred, confs, granules = stager.predict(X_test, smooth_window=3, min_granule_epochs=2)
probs = stager.predict_proba(X_test)
ambiguous = stager.detect_ambiguous_regions(probs, threshold=0.4)

print(f'\n准确率: {accuracy_score(y_test, y_pred):.4f}')
print('\n分类报告:')
print(classification_report(y_test, y_pred, labels=STAGES, digits=4))

print('\n混淆矩阵:')
cm = confusion_matrix(y_test, y_pred, labels=STAGES)
print('        ', end='')
for s in STAGES: print(f'{s:>6s}', end='')
print()
for ti, ts in enumerate(STAGES):
    print(f'{ts:5s}  ', end='')
    for c in cm[ti]: print(f'{c:6d}', end='')
    print()

# 6. 信息粒细胞
print(f'\n[5] 信息粒 ({len(granules)} granules):')
for g in granules[:20]:
    marker = ' ⚠' if g['confidence'] < 0.7 else ''
    print(f'  [{g["start_minute"]:.0f}-{g["end_minute"]:.0f}min] {g["stage"]} {g["duration_minutes"]:.0f}min  conf={g["confidence"]:.3f}{marker}')

print(f'  ... {len(granules)-20} more granules omitted')

# 7. 歧义区统计
print(f'\n[6] 歧义区: {np.sum(ambiguous)} epochs ({np.mean(ambiguous)*100:.1f}%)')
ambiguous_by_stage = {}
for i, (pred, amb) in enumerate(zip(y_pred, ambiguous)):
    if amb:
        true_s = y_test[i]
        if true_s not in ambiguous_by_stage:
            ambiguous_by_stage[true_s] = 0
        ambiguous_by_stage[true_s] += 1

for s in STAGES:
    n_amb = ambiguous_by_stage.get(s, 0)
    n_total = np.sum(y_test == s)
    if n_total > 0:
        print(f'  {s}: {n_amb}/{n_total} = {n_amb/n_total*100:.1f}%')

# 8. 睡眠报告（最后8小时 = 960 epochs）
print(f'\n[7] 睡眠报告 (测试集):')
n_epochs_test = len(y_test)
if n_epochs_test > 0:
    sleep_mask = y_pred != 'W'
    sleep_pct = np.mean(sleep_mask) * 100
    
    total_hours = n_epochs_test * 0.5 / 60
    print(f'  Total: {total_hours:.1f}h, {n_epochs_test} epochs')
    print(f'  Sleep: {sleep_pct:.1f}% (⌀{total_hours * sleep_pct / 100:.1f}h)')
    
    for s in ['N1', 'N2', 'N3', 'REM']:
        pct = np.mean(y_pred == s) * 100
        print(f'    {s}: {pct:.1f}% ({pct * total_hours / 100:.1f}h)')

# 9. 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)

# 真实
ax = axes[0]
for si, (s, c) in enumerate([('W','#ff6b6b'),('N1','#ffd93d'),('N2','#6bcb77'),('N3','#4d96ff'),('REM','#9b59b6')]):
    idx = np.where(y_test == s)[0]
    ax.scatter(idx, [si]*len(idx), c=c, s=1, alpha=0.6)
ax.set_yticks(range(5)); ax.set_yticklabels(STAGES); ax.set_ylabel('真实')

# 预测
ax = axes[1]
y_nums = [STAGES.index(s) for s in y_pred]
ax.plot(y_nums, 'b-', lw=0.3, alpha=0.7)
ax.fill_between(range(len(ambiguous)), 0, 5, where=ambiguous, alpha=0.1, color='red', label='歧义区')
ax.set_yticks(range(5)); ax.set_yticklabels(STAGES); ax.set_ylabel('预测'); ax.legend()

# 置信度
ax = axes[2]
ax.fill_between(range(n_epochs_test), 0, confs, alpha=0.4, color='green')
ax.plot(confs, 'g-', lw=0.3)
ax.axhline(0.7, color='orange', ls='--', alpha=0.5, label='低置信阈值')
ax.set_ylabel('置信度'); ax.set_ylim(0, 1.05); ax.legend()

# 信息粒
ax = axes[3]
for g in granules:
    if g['end_epoch'] >= n_epochs_test: continue
    si = STAGES.index(g['stage'])
    ax.barh(si, g['n_epochs'], left=g['start_epoch'], 
            height=0.8, color=['#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#9b59b6'][si],
            alpha=0.7)
ax.set_yticks(range(5)); ax.set_yticklabels(STAGES); ax.set_ylabel('信息粒')
ax.set_xlabel(f'Epoch (x30s, {n_epochs_test*30/3600:.1f}h)')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'granular_stager.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'\n  可视化: results/SC4001/granular_stager.png')

# 10. 推 GitHub
print('\n[8] 推送到 GitHub...')
import subprocess
os.chdir(project_dir)
subprocess.run(['git', 'add', '-A'], capture_output=True)
subprocess.run(['git', 'commit', '-m', 'feat: 信息粒增强型睡眠分期器 (LR+平滑+约束+信息粒)'], capture_output=True)
result = subprocess.run(['git', 'push', '-u', 'origin', 'main'], capture_output=True, text=True)
print(f'  GitPush: {result.stdout.strip()[:200]} {result.stderr.strip()[:200]}')
print('完成!')
