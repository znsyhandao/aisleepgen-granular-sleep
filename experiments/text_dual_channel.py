"""第二个EEG通道对比 — SC4001的C4-A1 vs Fpz-Cz"""
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'sigma':(11,16),'beta':(16,30)}
feature_names = [f'{n}_power' for n in BANDS] + ['delta_theta_ratio','alpha_delta_ratio']
stage_map = {'W':'W','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM'}

for sub in ['SC4001', 'SC4131']:
    raw = mne.io.read_raw_edf(os.path.join(EDF_DIR,f'{sub}E0-PSG.edf'), preload=True)
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    print(f'\n[{sub}] 通道: {ch_names}')
    
    # 找两个EEG通道
    eeg_chs = [c for c in ch_names if 'EEG' in c.upper()]
    print(f'  EEG通道: {eeg_chs}')
    
    if len(eeg_chs) < 2:
        print(f'  只有{eeg_chs}, 跳过')
        continue
    
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
    
    # 每个通道单独提取特征
    for ch in eeg_chs[:2]:
        data_arr, _ = raw[ch]; data_arr = data_arr.flatten()
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
        
        valid = stages != '?'
        Xv = X[valid]; yv = stages[valid]
        n = len(Xv); split = n//2
        X_train, y_train = Xv[:split], yv[:split]
        X_test, y_test = Xv[split:], yv[split:]
        
        scaler = StandardScaler().fit(X_train)
        Xt = scaler.transform(X_test)
        
        # LR on this channel
        clf = LogisticRegression(max_iter=2000, C=10)
        clf.fit(scaler.transform(X_train), y_train)
        yp = clf.predict(Xt)
        
        n1_rec = recall_score(y_test, yp, labels=['N1'], average=None)[0]
        n1_prec = classification_report(y_test, yp, output_dict=True, zero_division=0).get('N1',{}).get('precision',0)
        acc = np.mean(yp == y_test)
        n1_test = np.sum(y_test == 'N1')
        n1_pred = np.sum(yp == 'N1')
        
        # 理论N1上限（同通道全数据训练+测试）
        clf2 = LogisticRegression(max_iter=2000, C=10)
        clf2.fit(Xt, y_test)
        yp2 = clf2.predict(Xt)
        n1_upper = recall_score(y_test, yp2, labels=['N1'], average=None)[0]
        
        print(f'  {ch:15s}: N1_rec={n1_rec:.3f} N1_prec={n1_prec:.3f} ACC={acc:.3f} (N1_pred={n1_pred}/{n1_test}) 理论上限={n1_upper:.3f}')
    
    # 双通道合并（简单拼接特征）
    print(f'  双通道合并: ', end='')
    Xs = []
    for ch in eeg_chs[:2]:
        data_arr, _ = raw[ch]; data_arr = data_arr.flatten()
        Xc = np.zeros((len(stages), 7))  # 5频带+2比值
        for i in range(min(len(stages), len(data_arr)//int(30*sfreq))):
            ed = data_arr[i*int(30*sfreq):(i+1)*int(30*sfreq)]
            if len(ed) < int(30*sfreq)*0.5: continue
            freqs, psd = compute_epoch_spectrum(ed, sfreq)
            for j, (name, band) in enumerate(BANDS.items()):
                Xc[i,j] = band_power(freqs, psd, band)
            dp, tp, ap = Xc[i,0], Xc[i,1], Xc[i,2]
            Xc[i,5] = dp/(tp+1e-15)
            Xc[i,6] = ap/(dp+1e-15)
        Xs.append(Xc)
    
    X_dual = np.concatenate(Xs, axis=1)  # 14 features
    Xvd = X_dual[valid]
    split = len(Xvd)//2
    Xd_train, yd_train = Xvd[:split], yv[:split]
    Xd_test, yd_test = Xvd[split:], yv[split:]
    sc = StandardScaler().fit(Xd_train)
    
    clf_d = LogisticRegression(max_iter=2000, C=10)
    clf_d.fit(sc.transform(Xd_train), yd_train)
    yp_d = clf_d.predict(sc.transform(Xd_test))
    
    n1_rec_d = recall_score(yd_test, yp_d, labels=['N1'], average=None)[0]
    n1_prec_d = classification_report(yd_test, yp_d, output_dict=True, zero_division=0).get('N1',{}).get('precision',0)
    n1_test_d = np.sum(yd_test == 'N1')
    n1_pred_d = np.sum(yp_d == 'N1')
    print(f'N1_rec={n1_rec_d:.3f} N1_prec={n1_prec_d:.3f} (pred={n1_pred_d}/{n1_test})')
