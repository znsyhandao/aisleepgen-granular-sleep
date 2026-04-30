"""可变粒度N1检测对比测试 — SC4041"""
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
from granular.variable_granularity import adaptive_staging

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUB = 'SC4041'
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'sigma':(11,16),'beta':(16,30)}
feature_names = [f'{n}_power' for n in BANDS] + ['delta_theta_ratio','alpha_delta_ratio']

print(f'[1] {SUB}...')
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

clf = LogisticRegression(max_iter=2000, C=10)
clf.fit(scaler.transform(X_train), y_train)

# 1. 基线
stager = GranularSleepStager(clf)
stager.fit(X_train, y_train, scaler)
y_base, y_conf, _ = stager.predict(X_test, smooth_window=3, min_granule_epochs=2)
n1_base = recall_score(y_test, y_base, labels=['N1'], average=None)[0]
print(f'  基线 N1: {n1_base:.4f} ({np.sum(y_base=="N1")} pred)')

# 2. 可变粒度（直接调adaptive_staging，不通过granular_stager）
try:
    test_data = dv[split*epoch_len : split*epoch_len + len(y_test)*epoch_len]
    # 确保数据够
    needed = len(y_test) * epoch_len
    test_seg = dv[split*epoch_len : split*epoch_len + needed]
    
    y_fine, fine_gs = adaptive_staging(test_seg, sfreq, y_base.copy(), y_conf, X_test)
    n1_fine = recall_score(y_test, y_fine, labels=['N1'], average=None)[0]
    print(f'  可变粒度 N1: {n1_fine:.4f} ({np.sum(y_fine=="N1")} pred)')
    n1_in_gs = sum(1 for g in fine_gs if g['stage']=='N1')
    print(f'  信息粒: {len(fine_gs)}个 (其中N1粒={n1_in_gs})')
except Exception as e:
    print(f'  可变粒度失败: {e}')
    import traceback; traceback.print_exc()
    n1_fine = 0
    y_fine = y_base

# 3. 可变粒度 + 过渡检测
y_both = post_process(y_fine, X_test, None,
                      feature_names.index('delta_theta_ratio'),
                      feature_names.index('beta_power'))
n1_both = recall_score(y_test, y_both, labels=['N1'], average=None)[0]
print(f'  可变+过渡 N1: {n1_both:.4f} ({np.sum(y_both=="N1")} pred)')

# 总结
print(f'\n{"="*40}')
for name, pr, rec in [('基线', y_base, n1_base), ('可变粒度', y_fine, n1_fine), ('可变+过渡', y_both, n1_both)]:
    print(f'{name:12s}: recall={rec:.4f}  N1_pred={np.sum(pr=="N1")}')
