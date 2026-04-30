"""突变检测可视化 — 看看频谱突变和N1的相关性"""
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from granular.variable_granularity import detect_transition_regions

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
stage_map = {'W':'W','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM'}

for SUB in ['SC4131', 'SC4001', 'SC4041', 'SC4601']:
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
    
    # 取测试集(后半段)
    valid = stages != '?'
    n = np.sum(valid)
    split = n // 2
    y_test = stages[valid][split:]
    n1_idx = np.where(y_test == 'N1')[0]
    
    # 只分析测试集的数据
    epoch_len = int(30 * sfreq)
    test_start = split * epoch_len
    test_end = test_start + len(y_test) * epoch_len
    test_data = data_arr[test_start:test_end]
    
    # 突变检测
    change = detect_transition_regions(test_data, sfreq)
    
    # 计算每个30秒epoch的平均突变率
    epochs_change = np.zeros(len(y_test))
    for ei in range(len(y_test)):
        start_s = ei * 30
        end_s = min(start_s + 30, len(change))
        if start_s < len(change):
            epochs_change[ei] = np.mean(change[start_s:end_s])
    
    # 统计: N1 epoch的平均突变率 vs 非N1
    n1_change = epochs_change[n1_idx] if len(n1_idx) > 0 else np.array([0])
    non_n1 = np.delete(epochs_change, n1_idx) if len(n1_idx) > 0 else epochs_change
    non_n1_change = non_n1
    
    print(f'{SUB}:')
    print(f'  测试集: {len(y_test)} epochs, N1={len(n1_idx)}')
    print(f'  N1平均突变率: {np.mean(n1_change):.4f}+-{np.std(n1_change):.4f}')
    print(f'  非N1平均突变率: {np.mean(non_n1_change):.4f}+-{np.std(non_n1_change):.4f}')
    
    # 看N1 epoch中突变率>0.3的比例
    if len(n1_change) > 0:
        high_change = np.mean(n1_change > 0.3)
        print(f'  N1中显著突变(>0.3): {high_change:.1%}')
    
    # 看非N1中突变率>0.3的比例
    false_pos = np.mean(non_n1_change > 0.3) if len(non_n1_change) > 0 else 0
    print(f'  非N1中显著突变: {false_pos:.1%}')
    print()
