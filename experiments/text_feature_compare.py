"""SC4131 vs SC4001 特征空间对比"""
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
from granular.granular_stager import GranularSleepStager

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'sigma':(11,16),'beta':(16,30)}
feature_names = [f'{n}_power' for n in BANDS] + ['delta_theta_ratio','alpha_delta_ratio']
stage_map = {'W':'W','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM'}

def process_sub(sub):
    raw = mne.io.read_raw_edf(os.path.join(EDF_DIR,f'{sub}E0-PSG.edf'), preload=True)
    sfreq = raw.info['sfreq']
    eeg_ch = [c for c in raw.ch_names if 'EEG' in c.upper()][0]
    data_arr, _ = raw[eeg_ch]; data_arr = data_arr.flatten()
    annot = mne.read_annotations(os.path.join(EDF_DIR,f'{sub}EC-Hypnogram.edf'))
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
    return X, stages, sfreq

print('='*60)
print('SC4131 (N1 recall 72.2%) vs SC4001 (N1 recall 2.4%)')
print('='*60)

# SC4131
X1, y1, sr1 = process_sub('SC4131')
v1 = y1 != '?'; X1v, y1v = X1[v1], y1[v1]
splt1 = len(X1v)//2
X1_train, y1_train = X1v[:splt1], y1v[:splt1]
X1_test, y1_test = X1v[splt1:], y1v[splt1:]
sc1 = StandardScaler().fit(X1_train)
X1t = sc1.transform(X1_test)

# SC4001
X2, y2, sr2 = process_sub('SC4001')
v2 = y2 != '?'; X2v, y2v = X2[v2], y2[v2]
splt2 = len(X2v)//2
X2_train, y2_train = X2v[:splt2], y2v[:splt2]
X2_test, y2_test = X2v[splt2:], y2v[splt2:]
sc2 = StandardScaler().fit(X2_train)
X2t = sc2.transform(X2_test)

# SC4601 (0% recall 增加了SC4601)
try:
    X3, y3, sr3 = process_sub('SC4601')
    v3 = y3 != '?'; X3v, y3v = X3[v3], y3[v3]
    splt3 = len(X3v)//2
    X3_train, y3_train = X3v[:splt3], y3v[:splt3]
    X3_test, y3_test = X3v[splt3:], y3v[splt3:]
    sc3 = StandardScaler().fit(X3_train)
    X3t = sc3.transform(X3_test)
except:
    X3, y3, X3t, y3_test = None, None, None, None

# 测试集N1的特征分布
for sub_name, X_test, y_test, Xt, label in [
    ('SC4131', X1_test, y1_test, X1t, 'SC4131(72%)'),
    ('SC4001', X2_test, y2_test, X2t, 'SC4001(2.4%)'),
    ('SC4601', X3_test if X3 is not None else None, y3_test if X3 is not None else None, X3t if X3 is not None else None, 'SC4601(0%)')
]:
    if X_test is None:
        continue
    n1_mask = y_test == 'N1'
    w_mask = y_test == 'W'
    n2_mask = y_test == 'N2'
    
    print(f'\n{label}:')
    print(f'  测试集: N1={n1_mask.sum()} W={w_mask.sum()} N2={n2_mask.sum()}')
    
    # 特征均值对比
    for feat_name, feat_idx in [('beta_power', 4), ('delta_theta_ratio', 5), ('alpha_delta_ratio', 6), ('sigma_power', 3), ('delta_power', 0)]:
        idx = feature_names.index(feat_name)
        # 在z-score空间
        n1_z = Xt[n1_mask, idx] if n1_mask.sum() > 0 else np.array([0])
        w_z = Xt[w_mask, idx]
        n2_z = Xt[n2_mask, idx]
        
        if len(n1_z) > 0:
            print(f'  {feat_name:20s}: N1_z={np.mean(n1_z):+5.2f}+-{np.std(n1_z):.2f}  W_z={np.mean(w_z):+5.2f}+-{np.std(w_z):.2f}  N2_z={np.mean(n2_z):+5.2f}+-{np.std(n2_z):.2f}')
            # 可分性指标 = (N1_z - W_z)的正交距离
            sep = abs(np.mean(n1_z) - np.mean(w_z)) / (np.std(n1_z)+np.std(w_z)+1e-15)
            sep2 = abs(np.mean(n1_z) - np.mean(n2_z)) / (np.std(n1_z)+np.std(n2_z)+1e-15)
            print(f'    N1-W区分度={sep:.3f}  N1-N2区分度={sep2:.3f}')
    
    # 多特征组合：LR + N1标签的准确率
    clf = LogisticRegression(max_iter=2000, C=10)
    clf.fit(Xt, y_test)  # 直接在测试集上做分类（看理论上限）
    yp = clf.predict(Xt)
    n1_rec = recall_score(y_test, yp, labels=['N1'], average=None)[0]
    acc = np.mean(yp == y_test)
    print(f'  理论N1上限(全数据集): {n1_rec:.3f}  ACC: {acc:.3f}')
