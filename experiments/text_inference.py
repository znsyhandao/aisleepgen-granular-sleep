"""
实验2: 模糊规则推理 — 用数据驱动模糊集做睡眠阶段分类

测试 DataDrivenFuzzySets + SleepStageFuzzyRules
验证每个阶段的模糊规则一致性
"""

import sys, os, json
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from granular.fuzzy_sets_v2 import DataDrivenFuzzySets, SleepStageFuzzyRules

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUB = 'SC4001'
RESULTS_DIR = os.path.join(project_dir, 'results', SUB)
os.makedirs(RESULTS_DIR, exist_ok=True)

BANDS = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'sigma': (11,16), 'beta': (16,30)}

# 1. 读取数据
print('[1] 读取数据...')
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
epoch_len = int(30 * sfreq)
feature_names = [f'{name}_power' for name in BANDS]
feature_names += ['delta_theta_ratio', 'alpha_delta_ratio']

n_epochs = len(stages)
feature_matrix = np.zeros((n_epochs, len(feature_names)))

for i in range(n_epochs):
    ed = data[i*epoch_len:(i+1)*epoch_len]
    if len(ed) < epoch_len: continue
    freqs, psd = compute_epoch_spectrum(ed, sfreq)
    col = 0
    for name, band in BANDS.items():
        feature_matrix[i, col] = band_power(freqs, psd, band); col += 1
    dp = feature_matrix[i, 0]; tp = feature_matrix[i, 1]; ap = feature_matrix[i, 2]
    feature_matrix[i, col] = dp / (tp + 1e-15); col += 1
    feature_matrix[i, col] = ap / (dp + 1e-15); col += 1

# 3. 计算各阶段的统计量（训练模糊集）
print('[3] 训练模糊集...')
stage_stats = {}
for fi, fname in enumerate(feature_names):
    stage_stats[fname] = {}
    for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
        vals = feature_matrix[:, fi][stages == stage]
        if len(vals) > 5:
            stage_stats[fname][stage] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

# 初始化数据驱动模糊集
fuzzy = DataDrivenFuzzySets(stage_stats)
rules = SleepStageFuzzyRules(feature_names)

# 4. 推理测试（全数据集）
print('[4] 模糊推理测试...')
results = []  # [(epoch, true_stage, predicted_stage, confidences), ...]
conf_matrix = {s: {s2: 0 for s2 in ['W','N1','N2','N3','REM']} for s in ['W','N1','N2','N3','REM']}
correct = 0
total = 0
correct_by_stage = Counter()
total_by_stage = Counter()

for i in range(n_epochs):
    true_s = stages[i]
    if true_s == '?':
        continue
    total_by_stage[true_s] += 1
    
    # 模糊化
    fuzzy_inputs = {}
    for fi, fname in enumerate(feature_names):
        fuzzy_inputs[fname] = fuzzy.fuzzify(fname, feature_matrix[i, fi])
    
    # 推理
    confs = rules.infer(fuzzy_inputs)
    predicted = max(confs, key=confs.get)
    
    conf_matrix[true_s][predicted] += 1
    if predicted == true_s:
        correct += 1
        correct_by_stage[true_s] += 1
    
    results.append({
        'epoch': i, 'true': true_s, 'predicted': predicted,
        'confs': {k: round(v, 4) for k, v in confs.items()}
    })

# 5. 结果
total_valid = sum(total_by_stage.values())
accuracy = correct / total_valid if total_valid > 0 else 0
print(f'\n总准确率: {accuracy:.4f} ({correct}/{total_valid})')

print('\n各阶段准确率:')
for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
    ok = correct_by_stage[stage]
    all_s = total_by_stage[stage]
    rate = ok/all_s if all_s > 0 else 0
    print(f'  {stage}: {rate:.4f} ({ok}/{all_s})')

print('\n混淆矩阵:')
print('         ', end='')
for s in ['W','N1','N2','N3','REM']: print(f'{s:>6s}', end='')
print()
for true_s in ['W','N1','N2','N3','REM']:
    print(f'{true_s:5s}  ', end='')
    for pred_s in ['W','N1','N2','N3','REM']:
        print(f'{conf_matrix[true_s][pred_s]:6d}', end='')
    print()

# 6. 可视化
fig, axes = plt.subplots(2, 1, figsize=(18, 8))

ax = axes[0]
for stage, color in [('W','#ff6b6b'),('N1','#ffd93d'),('N2','#6bcb77'),('N3','#4d96ff'),('REM','#9b59b6')]:
    epochs_idx = [i for i in range(n_epochs) if results[i]['true'] == stage]
    ax.scatter(epochs_idx, [stage]*len(epochs_idx), c=color, s=2, alpha=0.5, label=f'{stage} (true)')
    correct_idx = [i for i in epochs_idx if results[i]['true'] == results[i]['predicted']]
    ax.scatter(correct_idx, [-1]*len(correct_idx), c='green', s=1, alpha=0.3)

ax.set_ylim(-2, 6)
ax.set_yticks([-1, 0, 1, 2, 3, 4])
ax.set_yticklabels(['正确', '', 'N1', 'N2', 'N3', 'REM'])

ax = axes[1]
for i in range(min(200, n_epochs)):
    r = results[i]
    for stage, c in [('W','r'),('N1','y'),('N2','g'),('N3','b'),('REM','m')]:
        conf = r['confs'].get(stage, 0)
        if conf > 0:
            ax.bar(i, conf, width=0.8, color=c, alpha=conf*0.8, bottom=0)
ax.set_ylabel('置信度')
ax.set_xlabel(f'Epoch (前200个)')
ax.legend(['W','N1','N2','N3','REM'], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fuzzy_inference.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7. 示例
print('\n=== 推理示例 ===')
for i in [1050, 1250, 1450, 1650, 1850, 2050]:
    r = results[i] if i < len(results) else results[-1]
    conf_str = ', '.join([f'{k}={v:.3f}' for k,v in sorted(r['confs'].items(), key=lambda x:-x[1])[:3]])
    print(f'  Epoch {r["epoch"]}: true={r["true"]}, pred={r["predicted"]} [{conf_str}]')

print(f'\n可视化: {RESULTS_DIR}/fuzzy_inference.png')
print('完成!')
