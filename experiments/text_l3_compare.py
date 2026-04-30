"""
L3 方案对比测试 - SC4001单被试
对比: 原始LR vs 方案A(N1 Boost) vs 方案B(过渡检测) vs 方案C(混合)
"""
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from granular.granular_stager import GranularSleepStager, STAGES

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUB = 'SC4001'
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'sigma':(11,16),'beta':(16,30)}
feature_names = [f'{n}_power' for n in BANDS] + ['delta_theta_ratio','alpha_delta_ratio']

# 1. 读取数据
print('[1] 读取...')
raw = mne.io.read_raw_edf(os.path.join(EDF_DIR,f'{SUB}E0-PSG.edf'), preload=True, verbose=False)
sfreq = raw.info['sfreq']
eeg_ch = [c for c in raw.ch_names if 'EEG' in c.upper()][0]
data_arr, _ = raw[eeg_ch]; data_arr = data_arr.flatten()
annot = mne.read_annotations(os.path.join(EDF_DIR,f'{SUB}EC-Hypnogram.edf'))
stage_map = {'W':'W','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM'}
stages = []
for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
    code = desc.strip()[-1].upper() if desc.strip() else '?'
    s = stage_map.get(code, '?')
    ne = max(1, int(round(dur/30)))
    se = int(round(onset/30))
    while len(stages) < se: stages.append('?')
    for _ in range(ne): stages.append(s)
stages = np.array(stages)

# 2. 特征
print('[2] 特征...')
epoch_len = int(30*sfreq)
n_epochs = len(stages)
X = np.zeros((n_epochs, len(feature_names)))
for i in range(n_epochs):
    ed = data_arr[i*epoch_len:(i+1)*epoch_len]
    if len(ed) < epoch_len: continue
    freqs, psd = compute_epoch_spectrum(ed, sfreq)
    col = 0
    for name, band in BANDS.items():
        X[i,col] = band_power(freqs, psd, band); col += 1
    dp, tp, ap = X[i,0], X[i,1], X[i,2]
    X[i,col] = dp/(tp+1e-15); col += 1
    X[i,col] = ap/(dp+1e-15); col += 1

# 3. 切分
valid = stages != '?'; Xv = X[valid]; yv = stages[valid]
n = len(Xv); split = n//2
X_train, y_train = Xv[:split], yv[:split]
X_test, y_test = Xv[split:], yv[split:]
scaler = StandardScaler().fit(X_train)
print(f'  训练: {len(X_train)} epochs, 测试: {len(X_test)} epochs')

# N1在测试集中的数量
n1_test = np.sum(y_test == 'N1')
print(f'  测试集N1: {n1_test} epochs')

# 4. 训练基分类器
print('[3] 训练LR...')
clf = LogisticRegression(max_iter=2000, C=10)
clf.fit(scaler.transform(X_train), y_train)

# 5. 四种方案预测
print('[4] 对比预测...')

results = {}
# 原始LR
from granular.explainable_classifier import ExplainableFuzzyClassifier
from granular.transition_detector import post_process

# ---- 原始LR ----
stager = GranularSleepStager(clf)
stager.fit(X_train, y_train, scaler)
y_base, _, _ = stager.predict(X_test, smooth_window=3, min_granule_epochs=2)
n1_recall_base = recall_score(y_test, y_base, labels=['N1'], average=None)[0] if n1_test>0 else 0
results['原始LR'] = {'pred': y_base, 'n1_recall': n1_recall_base}
print(f'  原始LR N1 recall: {n1_recall_base:.4f}')

# ---- 方案A: N1过渡区Boost ----
# 重新运行ExplainableFuzzyClassifier（已有方案A的改动）
try:
    efc = ExplainableFuzzyClassifier()
    X_test_scaled = scaler.transform(X_test)
    efc.fit(X_train, y_train, feature_names)
    y_a = efc.predict(X_test_scaled, smooth_window=3)
    n1_a = recall_score(y_test, y_a, labels=['N1'], average=None)[0] if n1_test>0 else 0
    results['方案A(Boost)'] = {'pred': y_a, 'n1_recall': n1_a}
    print(f'  方案A N1 recall: {n1_a:.4f}')
except Exception as e:
    print(f'  方案A失败: {e}')
    results['方案A(Boost)'] = {'pred': y_base, 'n1_recall': 0}

# ---- 方案B: 过渡检测器 ----
try:
    y_b = post_process(y_base.copy(), X_test, None, 
                       delta_theta_idx=feature_names.index('delta_theta_ratio'),
                       beta_idx=feature_names.index('beta_power'))
    n1_b = recall_score(y_test, y_b, labels=['N1'], average=None)[0] if n1_test>0 else 0
    results['方案B(过渡检测)'] = {'pred': y_b, 'n1_recall': n1_b}
    print(f'  方案B N1 recall: {n1_b:.4f}')
except Exception as e:
    print(f'  方案B失败: {e}')
    results['方案B(过渡检测)'] = {'pred': y_base, 'n1_recall': 0}

# ---- 方案C: 混合（方案A + 方案B组合） ----
try:
    y_c = efc.predict(X_test_scaled, smooth_window=3)
    y_c = post_process(y_c, X_test, None,
                       delta_theta_idx=feature_names.index('delta_theta_ratio'),
                       beta_idx=feature_names.index('beta_power'))
    n1_c = recall_score(y_test, y_c, labels=['N1'], average=None)[0] if n1_test>0 else 0
    results['方案C(混合)'] = {'pred': y_c, 'n1_recall': n1_c}
    print(f'  方案C N1 recall: {n1_c:.4f}')
except Exception as e:
    print(f'  方案C失败: {e}')
    results['方案C(混合)'] = {'pred': y_base, 'n1_recall': 0}

# 6. 对比结果
print(f'\n{"="*50}')
print(f'N1 Recall 对比 (SC4001)')
print(f'{"="*50}')
print(f'{"方案":20s} {"N1 Recall":>10s} {"提升":>10s}')
print(f'{"-"*40}')
base_r = results.get('原始LR', {}).get('n1_recall', 0)
for name, r in results.items():
    gain = r['n1_recall'] - base_r if name != '原始LR' else 0
    arrow_val = ' UP' if gain > 0.01 else (' DOWN' if gain < -0.01 else ' ->')
    n1r = r['n1_recall']
    print(f'{name:20s} {n1r:10.4f} {gain:+7.4f}{arrow_val}')

print(f'\n{"="*50}')
# N1数量变化
for name, r in results.items():
    n1_count = np.sum(r['pred'] == 'N1')
    true_n1 = np.sum(y_test == 'N1')
    print(f'{name:20s}: 预测N1={n1_count:4d} (真实N1={true_n1})')

print(f'\n[OK] 测试完成')
