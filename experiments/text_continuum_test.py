"""连续谱追踪 vs 传统LR对比 — SC4001"""
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
from granular.continuum_tracker import compute_continuum_trajectory, continuum_staging, epoch_stages_from_continuum

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
stage_map = {'W':'W','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM'}

for SUB in ['SC4001', 'SC4041']:
    print(f'\n[{SUB}]')
    raw = mne.io.read_raw_edf(os.path.join(EDF_DIR,f'{SUB}E0-PSG.edf'), preload=True)
    sfreq = raw.info['sfreq']
    eeg_ch = [c for c in raw.ch_names if 'EEG' in c.upper()][0]
    data_arr, _ = raw[eeg_ch]; data_arr = data_arr.flatten()
    annot = mne.read_annotations(os.path.join(EDF_DIR,f'{SUB}EC-Hypnogram.edf'))
    stages = []
    for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
        code = desc.strip()[-1].upper() if desc.strip() else '?'
        s = stage_map.get(code, '?')
        ne = max(1, int(round(dur/30)))
        se = int(round(onset/30))
        while len(stages) < se: stages.append('?')
        for _ in range(ne): stages.append(s)
    stages = np.array(stages)
    valid = stages != '?'; yv = stages[valid]
    n = len(yv); split = n//2
    y_test = yv[split:]
    
    # 测试集数据
    epoch_len = int(30 * sfreq)
    test_start = split * epoch_len
    test_end = test_start + len(y_test) * epoch_len + epoch_len  # 多一点
    test_data = data_arr[test_start:min(test_end, len(data_arr))]
    
    # LR基线
    n1_test = np.sum(y_test == 'N1')
    
    # 连续谱追踪
    try:
        sec_stages, traj, n1_prog, dists = continuum_staging(test_data, sfreq)
        # 聚合到30秒epoch
        y_cont = epoch_stages_from_continuum(sec_stages, n1_prog)
        
        # 对齐长度
        min_len = min(len(y_cont), len(y_test))
        y_cont = y_cont[:min_len]
        yt = y_test[:min_len]
        
        n1_cont = recall_score(yt, y_cont, labels=['N1'], average=None)[0] if n1_test > 0 else 0
        print(f'  N1:真实={n1_test} 连续谱N1预测={np.sum(y_cont=="N1")} recall={n1_cont:.4f}')
        
        # 轨迹的N1进度分布
        n1_progress_in_n1 = n1_prog[:min_len*30][y_cont[:min_len] == 'N1']
        print(f'  N1进度(只在N1预测区): mean={np.mean(n1_progress_in_n1):.3f}+-{np.std(n1_progress_in_n1):.3f}')
        
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'  连续谱失败: {e}')
