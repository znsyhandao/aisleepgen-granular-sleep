"""45个EC标注受试者N1 recall批量评测"""
import sys, os, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, classification_report

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from granular.granular_stager import GranularSleepStager, STAGES
from granular.transition_detector import post_process

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'sigma':(11,16),'beta':(16,30)}
feature_names = [f'{n}_power' for n in BANDS] + ['delta_theta_ratio','alpha_delta_ratio']
stage_map = {'W':'W','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM'}

# 获取所有EC受试者
hyp_files = [f for f in os.listdir(EDF_DIR) if f.endswith('EC-Hypnogram.edf')]
subs = sorted(set(f.split('EC')[0] for f in hyp_files))
print(f'总受试者: {len(subs)}')

results = []

for sub in subs:
    print(f'\n[{sub}]')
    try:
        # 读取
        psg_path = os.path.join(EDF_DIR, f'{sub}E0-PSG.edf')
        hyp_path = os.path.join(EDF_DIR, f'{sub}EC-Hypnogram.edf')
        if not os.path.exists(psg_path) or not os.path.exists(hyp_path):
            print(f'  文件缺失')
            continue
        
        raw = mne.io.read_raw_edf(psg_path, preload=True)
        sfreq = raw.info['sfreq']
        eeg_ch = [c for c in raw.ch_names if 'EEG' in c.upper()]
        if not eeg_ch:
            print(f'  无EEG通道')
            continue
        data_arr, _ = raw[eeg_ch[0]]; data_arr = data_arr.flatten()
        
        annot = mne.read_annotations(hyp_path)
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
        
        n1_total = np.sum(stages[valid] == 'N1')
        if n1_total < 10:
            print(f'  N1仅{n1_total}, 跳过')
            continue
        
        # 特征
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
        
        # 切分
        Xv = X[valid]; yv = stages[valid]
        n = len(Xv); split = n//2
        if split < 50:
            print(f'  数据不足')
            continue
        X_train, y_train = Xv[:split], yv[:split]
        X_test, y_test = Xv[split:], yv[split:]
        scaler = StandardScaler().fit(X_train)
        
        # 训练
        clf = LogisticRegression(max_iter=2000, C=10)
        clf.fit(scaler.transform(X_train), y_train)
        
        # 预测
        stager = GranularSleepStager(clf)
        stager.fit(X_train, y_train, scaler)
        y_base, _, _ = stager.predict(X_test, smooth_window=3, min_granule_epochs=2)
        
        # 过渡检测
        y_b = post_process(y_base.copy(), X_test, None,
                          feature_names.index('delta_theta_ratio'),
                          feature_names.index('beta_power'))
        
        # 全类accuracy + N1 recall
        acc_base = np.mean(y_base == y_test)
        acc_b = np.mean(y_b == y_test)
        n1_test = np.sum(y_test == 'N1')
        n1_base = recall_score(y_test, y_base, labels=['N1'], average=None)[0] if n1_test > 0 else 0
        n1_b = recall_score(y_test, y_b, labels=['N1'], average=None)[0] if n1_test > 0 else 0
        
        # 各阶段recall
        report_base = classification_report(y_test, y_base, labels=STAGES, output_dict=True, zero_division=0)
        
        results.append({
            'subject': sub,
            'n_epochs': len(Xv), 'n1_total': n1_total, 'n1_test': n1_test,
            'acc_base': round(acc_base, 4), 'acc_post': round(acc_b, 4),
            'n1_recall_base': round(n1_base, 4), 'n1_recall_post': round(n1_b, 4),
            'recall_W': round(report_base.get('W',{}).get('recall',0), 4),
            'recall_N1': round(report_base.get('N1',{}).get('recall',0), 4),
            'recall_N2': round(report_base.get('N2',{}).get('recall',0), 4),
            'recall_N3': round(report_base.get('N3',{}).get('recall',0), 4),
            'recall_REM': round(report_base.get('REM',{}).get('recall',0), 4),
            'prec_N1': round(report_base.get('N1',{}).get('precision',0), 4),
            'f1_N1': round(report_base.get('N1',{}).get('f1-score',0), 4),
        })
        print(f'  N1_rec={n1_base:.3f}(+{n1_b-n1_base:+.3f}) N1_prec={results[-1]["prec_N1"]:.3f} acc={acc_base:.3f}')
        
    except Exception as e:
        print(f'  失败: {e}')
        continue

# 汇总
print(f'\n\n{"="*60}')
print(f'45个EC受试者 N1 Recall 汇总')
print(f'{"="*60}')
print(f'完成: {len(results)}/{len(subs)}')
print()

n1_recalls = [r['n1_recall_base'] for r in results if r['n1_test'] > 0]
n1_posts = [r['n1_recall_post'] for r in results if r['n1_test'] > 0]
n1_precs = [r['prec_N1'] for r in results]
n1_f1s = [r['f1_N1'] for r in results]
accs = [r['acc_base'] for r in results]

print(f'N1 Recall: mean={np.mean(n1_recalls):.4f} median={np.median(n1_recalls):.4f} '
      f'min={min(n1_recalls):.4f} max={max(n1_recalls):.4f}')
print(f'N1 Recall(后处理): mean={np.mean(n1_posts):.4f} median={np.median(n1_posts):.4f}')
print(f'N1 Precision: mean={np.mean(n1_precs):.4f} median={np.median(n1_precs):.4f}')
print(f'N1 F1: mean={np.mean(n1_f1s):.4f} median={np.median(n1_f1s):.4f}')
print(f'Accuracy: mean={np.mean(accs):.4f} median={np.median(accs):.4f}')

# N1<10%的受试者
bad = [r for r in results if r['n1_recall_base'] < 0.1]
if bad:
    print(f'\nN1 recall < 10% ({len(bad)}人):')
    for r in bad:
        print(f'  {r["subject"]}: N1={r["n1_test"]} recall={r["n1_recall_base"]:.3f} N1_prec={r["prec_N1"]:.3f}')

# N1>50%的受试者
good = [r for r in results if r['n1_recall_base'] > 0.5]
print(f'\nN1 recall > 50% ({len(good)}人/{len(results)}):')
for r in sorted(good, key=lambda x: -x['n1_recall_base'])[:10]:
    print(f'  {r["subject"]}: N1={r["n1_test"]} recall={r["n1_recall_base"]:.3f} prec={r["prec_N1"]:.3f}')

# 保存
with open(os.path.join(project_dir, 'results', 'n1_recall_45.json'), 'w') as f:
    json.dump({
        'n': len(results),
        'n1_recall_mean': round(float(np.mean(n1_recalls)), 4),
        'n1_recall_median': round(float(np.median(n1_recalls)), 4),
        'n1_prec_mean': round(float(np.mean(n1_precs)), 4),
        'n1_f1_mean': round(float(np.mean(n1_f1s)), 4),
        'acc_mean': round(float(np.mean(accs)), 4),
        'details': results
    }, f, indent=2)
print(f'\n结果保存: results/n1_recall_45.json')
print(f'\n[DONE]')
