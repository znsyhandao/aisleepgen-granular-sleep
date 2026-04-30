"""
实验1: 单个受试者的粒计算睡眠阶段识别

流程序列:
1. 读取 EDF (PSG + Hypnogram) via MNE
2. 提取频谱特征 (纯numpy) 
3. 构建信息粒
4. 计算粒关系
5. 检测过渡期
6. 可视化
"""

import sys, os, json
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal as scipy_signal

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from granular.info_granules import GranuleBuilder

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
SUBJECT = 'SC4001'
RESULTS_DIR = os.path.join(project_dir, 'results', SUBJECT)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = datetime.now()

    print(f'粒计算睡眠阶段识别 — {SUBJECT}')
    print(f'时间: {t0.strftime("%H:%M:%S")}')
    print('=' * 50)

    # 1. 读取 EDF
    print('\n[1/5] 读取 EDF...')
    psg_path = os.path.join(EDF_DIR, f'{SUBJECT}E0-PSG.edf')
    hypno_path = os.path.join(EDF_DIR, f'{SUBJECT}EC-Hypnogram.edf')
    
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    print(f'  采样率: {sfreq:.0f}Hz, 通道: {raw.info["nchan"]}')
    print(f'  时长: {raw.times[-1]/3600:.1f}h')
    
    # 找EEG通道
    ch_names = [ch.upper() for ch in raw.ch_names]
    eeg_ch = None
    for ch_raw, ch_up in zip(raw.ch_names, ch_names):
        if 'EEG' in ch_up or 'FPZ' in ch_up or 'CZ' in ch_up:
            eeg_ch = ch_raw
            break
    if not eeg_ch:
        eeg_ch = raw.ch_names[0]
    print(f'  EEG通道: {eeg_ch}')
    
    data, _ = raw[eeg_ch]
    data = data.flatten()
    
    # 读取 Hypnogram
    stages = _parse_hypnogram(hypno_path, sfreq, len(data))
    if not stages:
        print('  Hypnogram解析失败!')
        return
    n_epochs = len(stages)
    print(f'  Epoch数: {n_epochs} ({n_epochs*30/60:.0f}min)')
    
    # 阶段分布
    unique, counts = np.unique(stages, return_counts=True)
    for s, c in zip(unique, counts):
        print(f'    {s}: {c} ({c*30/60:.1f}min)')
    
    # 2. 提取频谱特征
    print('\n[2/5] 提取频谱特征...')
    
    FREQ_BANDS = {
        'delta': (0.5, 4.0), 'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0), 'sigma': (11.0, 16.0),
        'beta': (16.0, 30.0),
    }
    
    epoch_len = int(30 * sfreq)
    feature_names = [f'{name}_power' for name in FREQ_BANDS]
    feature_names += ['delta_theta_ratio', 'alpha_delta_ratio', 'sef_50', 'sef_95']
    
    feature_matrix = np.zeros((n_epochs, len(feature_names)))
    
    for i in range(n_epochs):
        start = i * epoch_len
        end = start + epoch_len
        if end > len(data):
            feature_matrix = feature_matrix[:i]
            stages = stages[:i]
            n_epochs = i
            break
        
        epoch_data = data[start:end]
        freqs, psd = compute_epoch_spectrum(epoch_data, sfreq)
        
        col = 0
        for name, band in FREQ_BANDS.items():
            feature_matrix[i, col] = band_power(freqs, psd, band)
            col += 1
        
        dp = feature_matrix[i, 0]  # delta
        tp = feature_matrix[i, 1]  # theta
        ap = feature_matrix[i, 2]  # alpha
        feature_matrix[i, col] = dp / (tp + 1e-10); col += 1
        feature_matrix[i, col] = ap / (dp + 1e-10); col += 1
        
        # SEF
        cum = np.cumsum(psd)
        tot = cum[-1]
        if tot > 0:
            idx50 = np.searchsorted(cum, tot * 0.50)
            idx95 = np.searchsorted(cum, tot * 0.95)
            feature_matrix[i, col] = freqs[min(idx50, len(freqs)-1)]; col += 1
            feature_matrix[i, col] = freqs[min(idx95, len(freqs)-1)]; col += 1
        else:
            col += 2
    
    print(f'  特征矩阵: {feature_matrix.shape}')
    print(f'  特征: {feature_names}')
    
    # 3. 构建信息粒
    print('\n[3/5] 构建信息粒...')
    builder = GranuleBuilder(feature_names=feature_names)
    granules = builder.build_granules(feature_matrix, stages)
    relations = builder.compute_relations(granules)
    transitions = builder.detect_transition_regions(granules, relations)
    
    print(f'  信息粒: {len(granules)}')
    print(f'  粒关系: {len(relations)}')
    print(f'  过渡区: {len(transitions)}')
    
    # 4. 分析结果
    print('\n[4/5] 分析...')
    
    # 阶段间粒相似度
    stage_pairs = [(r.similarity, granules[r.i].stage, granules[r.j].stage) for r in relations]
    same_stage_sim = [s for s, si, sj in stage_pairs if si == sj]
    diff_stage_sim = [s for s, si, sj in stage_pairs if si != sj]
    
    if same_stage_sim:
        print(f'  同阶段平均相似度: {np.mean(same_stage_sim):.3f} ({len(same_stage_sim)}对)')
    if diff_stage_sim:
        print(f'  不同阶段平均相似度: {np.mean(diff_stage_sim):.3f} ({len(diff_stage_sim)}对)')
    
    # 保存结果
    results = {
        'subject': SUBJECT,
        'n_epochs': n_epochs,
        'stages': stages,
        'mean_sim_same': float(np.mean(same_stage_sim)) if same_stage_sim else 0,
        'mean_sim_diff': float(np.mean(diff_stage_sim)) if diff_stage_sim else 0,
        'n_transitions': len(transitions),
        'transition_epochs': sum(e-s for s,e in transitions),
    }
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 5. 可视化
    print('\n[5/5] 生成可视化...')
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    
    # 阶段图
    ax = axes[0]
    stage_order = ['W', 'N1', 'N2', 'N3', 'REM']
    stage_colors = {'W': '#ff6b6b', 'N1': '#ffd93d', 'N2': '#6bcb77', 'N3': '#4d96ff', 'REM': '#9b59b6', '?': '#ccc'}
    y = [stage_order.index(s) if s in stage_order else -1 for s in stages]
    ax.fill_between(range(n_epochs), y, 0, alpha=0.3, color='#4d96ff')
    ax.plot(y, 'b-', linewidth=0.5, alpha=0.7)
    ax.set_yticks(range(len(stage_order)))
    ax.set_yticklabels(stage_order)
    ax.set_ylabel('阶段')
    ax.set_title(f'{SUBJECT} — 粒计算睡眠阶段分析', fontsize=14)
    ax.grid(alpha=0.3)
    
    # 粒相似度
    ax = axes[1]
    sims = [r.similarity for r in relations]
    is_trans = [r.is_transition for r in relations]
    ax.plot(sims, 'g-', alpha=0.7, lw=0.8)
    for i, t in enumerate(is_trans):
        if t: ax.axvline(i, color='red', alpha=0.08, lw=0.5)
    ax.axhline(y=0.6, color='orange', ls='--', alpha=0.5, label='阈值')
    ax.set_ylabel('粒相似度')
    ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(alpha=0.3)
    
    # 包含度
    ax = axes[2]
    incs = [r.inclusion for r in relations]
    ax.plot(incs, 'purple', alpha=0.7, lw=0.8)
    ax.set_ylabel('包含度'); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    
    # Delta功率
    ax = axes[3]
    delta_idx = feature_names.index('delta_power')
    delta = feature_matrix[:, delta_idx]
    ax.fill_between(range(n_epochs), delta, alpha=0.4, color='#4d96ff')
    ax.plot(delta, 'b-', lw=0.5)
    ax.set_ylabel('Delta功率'); ax.grid(alpha=0.3)
    
    # Alpha/Delta
    ax = axes[4]
    ad_idx = feature_names.index('alpha_delta_ratio')
    ad = feature_matrix[:, ad_idx]
    ax.fill_between(range(n_epochs), np.clip(ad, 0, ad.max()), alpha=0.4, color='#ff6b6b')
    ax.plot(ad, 'r-', lw=0.5)
    ax.set_ylabel('Alpha/Delta')
    ax.set_xlabel(f'Epoch (x30s, total {n_epochs*30/3600:.1f}h)')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'granular_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  可视化: {RESULTS_DIR}/granular_analysis.png')
    
    # 示例粒
    print('\n  示例粒 (epoch 100):')
    g = granules[min(100, n_epochs-1)]
    print(f'    stage={g.stage}')
    for fname, fuzzy in g.fuzzy_repr.items():
        for label, mu in fuzzy:
            print(f'    {fname} = {label} (mu={mu:.3f})')
    
    print(f'\n{"="*50}')
    print(f'完成! {datetime.now().strftime("%H:%M:%S")} ({(datetime.now()-t0).seconds}s)')


def _parse_hypnogram(hypno_path, sfreq, n_samples):
    """解析 Hypnogram EDF — 用 MNE 的 annotations 机制"""
    try:
        annot = mne.read_annotations(hypno_path)
    except:
        return None
    
    stage_map = {'W': 'W', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'N3', 'R': 'REM'}
    epoch_dur = 30
    
    stages = []
    for onset, duration, desc in zip(annot.onset, annot.duration, annot.description):
        code = desc.strip()[-1].upper() if desc.strip() else '?'
        stage = stage_map.get(code, '?')
        
        n_epochs = max(1, int(round(duration / epoch_dur)))
        start_epoch = int(round(onset / epoch_dur))
        
        while len(stages) < start_epoch:
            stages.append('?')
        
        for _ in range(n_epochs):
            stages.append(stage)
    
    return stages


if __name__ == '__main__':
    main()
