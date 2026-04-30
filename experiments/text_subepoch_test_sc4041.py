"""子epoch N1测试 — 用SC4041（N1=166个epoch）"""
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from granular.granular_stager import GranularSleepStager, STAGES, set_subepoch_data
from granular.transition_detector import post_process

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUB = 'SC4041'
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'sigma':(11,16),'beta':(16,30)}
feature_names = [f'{n}_power' for n in BANDS] + ['delta_theta_ratio','alpha_delta_ratio']

print(f'[1] 读取{SUB}...')
raw = mne.io.read_raw_edf(os.path.join(EDF_DIR,f'{SUB}E0-PSG.edf'), preload=True)
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
print(f'  总epoch: {len(stages)}, N1: {np.sum(stages=="N1")}, 采样率: {sfreq}Hz')

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

valid = stages != '?'; Xv = X[valid]; yv = stages[valid]; dv = data_arr[:len(stages)*epoch_len]
n = len(Xv); split = n//2
X_train, y_train = Xv[:split], yv[:split]
X_test, y_test = Xv[split:], yv[split:]
scaler = StandardScaler().fit(X_train)
n1_test = np.sum(y_test == 'N1')
print(f'  训练:{len(X_train)} 测试:{len(X_test)} N1测试:{n1_test}')

print('[3] 训练LR...')
clf = LogisticRegression(max_iter=2000, C=10)
clf.fit(scaler.transform(X_train), y_train)

# 基线
stager = GranularSleepStager(clf)
stager.fit(X_train, y_train, scaler)
y_base, _, _ = stager.predict(X_test, smooth_window=3, min_granule_epochs=2)
n1_base = recall_score(y_test, y_base, labels=['N1'], average=None)[0]
print(f'  基线 N1: {n1_base:.4f} ({np.sum(y_base=="N1")} pred)')

# 过渡检测
y_b = post_process(y_base.copy(), X_test, None, feature_names.index('delta_theta_ratio'), feature_names.index('beta_power'))
n1_b = recall_score(y_test, y_b, labels=['N1'], average=None)[0]
print(f'  +过渡检测 N1: {n1_b:.4f} ({np.sum(y_b=="N1")} pred)')

# 子epoch
stager2 = GranularSleepStager(clf)
stager2.fit(X_train, y_train, scaler)
set_subepoch_data(dv[split*epoch_len:], sfreq)
y_sub, _, _ = stager2.predict(X_test, smooth_window=3, min_granule_epochs=2)
n1_sub = recall_score(y_test, y_sub, labels=['N1'], average=None)[0]
print(f'  +子epoch N1: {n1_sub:.4f} ({np.sum(y_sub=="N1")} pred)')

# 全部
y_all = post_process(y_sub.copy(), X_test, None, feature_names.index('delta_theta_ratio'), feature_names.index('beta_power'))
n1_all = recall_score(y_test, y_all, labels=['N1'], average=None)[0]
print(f'  全部组合 N1: {n1_all:.4f} ({np.sum(y_all=="N1")} pred)')

print(f'\n{"="*40}')
for name, pred, rec in [('基线', y_base, n1_base), ('+过渡', y_b, n1_b), ('+子epoch', y_sub, n1_sub), ('全部', y_all, n1_all)]:
    print(f'{name:12s}: recall={rec:.4f}  N1_pred={np.sum(pred=="N1")}')
