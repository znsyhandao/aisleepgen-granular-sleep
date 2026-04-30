"""对比真实睡眠结构 vs 模型预测的睡眠结构"""
import sys, os, json
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import mne

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from granular.sleep_metrics import build_sleep_hypnogram, sleep_quality_score

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
stage_map = {'W':'W','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM'}

# 选10个受试者做对比
subjects = [f.replace('E0-PSG.edf','') for f in os.listdir(EDF_DIR) 
            if f.endswith('E0-PSG.edf') and f.startswith('SC4')][:10]

print(f'对比10个受试者的真实vs预测睡眠结构\n', flush=True)

for subj in subjects:
    try:
        # 读真实
        hypno_path = os.path.join(EDF_DIR, f'{subj}EC-Hypnogram.edf')
        if not os.path.exists(hypno_path): continue
        annot = mne.read_annotations(hypno_path)
        st = []
        for on, dr, desc in zip(annot.onset, annot.duration, annot.description):
            code = desc.strip()[-1].upper() if desc.strip() else '?'
            s = stage_map.get(code,'?')
            ne = max(1,int(round(dr/30)))
            se = int(round(on/30))
            while len(st) < se: st.append('?')
            for _ in range(ne): st.append(s)
        st = np.array(st)
        mask = st != '?'; st_true = st[mask]
        
        # 读预测
        rpath = os.path.join(project_dir, 'results', subj, 'results.json')
        ppath = os.path.join(project_dir, 'results', subj, 'y_pred.npy')
        if not os.path.exists(rpath) or not os.path.exists(ppath): continue
        r = json.load(open(rpath))
        y_pred = np.load(ppath)
        
        # 两份都用第2半（测试集）
        n = len(st_true)
        split = n // 2
        
        # 真实
        hypno_true = build_sleep_hypnogram(st_true[split:], np.ones(len(st_true[split:])))
        score_true = sleep_quality_score(hypno_true)
        
        # 预测
        hypno_pred = build_sleep_hypnogram(y_pred, np.ones(len(y_pred)))
        score_pred = sleep_quality_score(hypno_pred)
        
        print(f'{subj}:', flush=True)
        s_t = hypno_true['summary']
        s_p = hypno_pred['summary']
        print(f'  真实: 效={s_t["sleep_efficiency_pct"]:.0f}%  N3={s_t["stage_pct"].get("N3",0):.1f}%  REM={s_t["stage_pct"].get("REM",0):.1f}%  WASO={s_t["waso_min"]:.0f}min  评分={score_true["total_score"]:.0f}', flush=True)
        print(f'  预测: 效={s_p["sleep_efficiency_pct"]:.0f}%  N3={s_p["stage_pct"].get("N3",0):.1f}%  REM={s_p["stage_pct"].get("REM",0):.1f}%  WASO={s_p["waso_min"]:.0f}min  评分={score_pred["total_score"]:.0f}', flush=True)
        print(f'  acc={r["accuracy"]:.3f}', flush=True)
    except Exception as e:
        print(f'{subj}: {str(e)[:80]}', flush=True)
