"""双通道45人抽样验证 — 10人快速预检"""
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, classification_report

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'sigma':(11,16),'beta':(16,30)}
stage_map = {'W':'W','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM'}
feature_cols = 7  # 5 bands + 2 ratios

# 选N1多的受试者 + N1少的各5个
subs = ['SC4041', 'SC4131', 'SC4102', 'SC4101', 'SC4182',  # 好的
        'SC4001', 'SC4002', 'SC4601', 'SC4401', 'SC4161']  # 差的

results = []
for sub in subs:
    print(f'[{sub}] ', end='', flush=True)
    try:
        raw = mne.io.read_raw_edf(os.path.join(EDF_DIR,f'{sub}E0-PSG.edf'), preload=True)
        sfreq = raw.info['sfreq']
        eeg_chs = [c for c in raw.ch_names if 'EEG' in c.upper()]
        if len(eeg_chs) < 2:
            print(f'仅{eeg_chs}通道, 跳过')
            continue
        
        # 取前2个EEG
        ch1, ch2 = eeg_chs[0], eeg_chs[1]
        
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
        valid = stages != '?'
        yv = stages[valid]
        
        n1_total = np.sum(yv == 'N1')
        if n1_total < 10:
            print(f'N1={n1_total}跳过')
            continue
        
        # 双通道特征
        epoch_len = int(30*sfreq)
        X_dual = np.zeros((len(stages), feature_cols * 2))
        for ci, ch in enumerate([ch1, ch2]):
            data_arr, _ = raw[ch]; data_arr = data_arr.flatten()
            for i in range(len(stages)):
                ed = data_arr[i*epoch_len:(i+1)*epoch_len]
                if len(ed) < epoch_len * 0.7: continue
                freqs, psd = compute_epoch_spectrum(ed, sfreq)
                for j, (name, band) in enumerate(BANDS.items()):
                    X_dual[i, ci*feature_cols + j] = band_power(freqs, psd, band)
                dp, tp, ap = X_dual[i, ci*feature_cols], X_dual[i, ci*feature_cols+1], X_dual[i, ci*feature_cols+2]
                X_dual[i, ci*feature_cols+5] = dp/(tp+1e-15)
                X_dual[i, ci*feature_cols+6] = ap/(dp+1e-15)
        
        # 单通道对比
        sub_results = {'subject': sub, 'n1_total': int(n1_total)}
        for ch_name, ci in [(ch1, 0), (ch2, 1), ('dual', None)]:
            if ci is not None:
                X_single = X_dual[:, ci*feature_cols:(ci+1)*feature_cols]
            else:
                X_single = X_dual
            
            Xv = X_single[valid]
            n = len(Xv); split = n//2
            X_train, y_train = Xv[:split], yv[:split]
            X_test, y_test = Xv[split:], yv[split:]
            scaler = StandardScaler().fit(X_train)
            clf = LogisticRegression(max_iter=2000, C=10)
            clf.fit(scaler.transform(X_train), y_train)
            yp = clf.predict(scaler.transform(X_test))
            
            n1_rec = recall_score(y_test, yp, labels=['N1'], average=None)[0]
            n1_prec = classification_report(y_test, yp, output_dict=True, zero_division=0).get('N1',{}).get('precision',0)
            n1_test = np.sum(y_test == 'N1')
            n1_pred = np.sum(yp == 'N1')
            
            label = ch_name.replace('EEG ', '')
            sub_results[f'n1_rec_{label}'] = round(n1_rec, 4)
            sub_results[f'n1_prec_{label}'] = round(n1_prec, 4)
            sub_results[f'n1_test'] = int(n1_test)
            sub_results[f'n1_pred_{label}'] = int(n1_pred)
        
        results.append(sub_results)
        print(f'N1={n1_total} single={sub_results["n1_rec_Fpz-Cz"]:.3f} dual={sub_results["n1_rec_dual"]:.3f}')
    except Exception as e:
        print(f'失败: {e}')

print(f'\n{"="*60}')
print(f'双通道N1召回率对比 (10人抽样)')
print(f'{"="*60}')
print(f'{"受试者":8s} {"N1总数":>6s} {"Fpz-Cz":>10s} {"Pz-Oz":>10s} {"双通道":>10s} {"提升":>8s}')
print(f'{"-"*52}')
for r in sorted(results, key=lambda x: -max(x.get('n1_rec_dual',0), x.get('n1_rec_Fpz-Cz',0))):
    single = r.get('n1_rec_Fpz-Cz', 0)
    dual = r.get('n1_rec_dual', 0)
    gain = dual - single
    arrow = '↑' if gain > 0.02 else ('↓' if gain < -0.02 else '→')
    print(f'{r["subject"]:8s} {r["n1_total"]:6d} {single:10.3f} {r.get("n1_rec_Pz-Oz",0):10.3f} {dual:10.3f} {gain:+7.3f}{arrow}')

sings = [r.get('n1_rec_Fpz-Cz',0) for r in results]
duals = [r.get('n1_rec_dual',0) for r in results]
print(f'{"-"*52}')
print(f'{"平均":8s} {"":>6s} {np.mean(sings):10.3f} {"":>10s} {np.mean(duals):10.3f} {np.mean(duals)-np.mean(sings):+7.3f}')
print(f'\n[DONE]')
