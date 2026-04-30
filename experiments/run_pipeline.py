"""
实验1: 单个受试者的粒计算睡眠阶段识别

流程:
1. 读取 EDF (PSG + Hypnogram)
2. 提取频谱特征
3. 构建信息粒
4. 计算粒关系
5. 检测过渡期
6. 可视化: 粒规则 + 过渡期
"""

import os, sys, json
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# 项目模块
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from features.spectral import extract_multi_channel_features, extract_epoch_features
from granular.info_granules import GranuleBuilder
from granular.fuzzy_sets import fuzzify, FEATURE_FUZZY_MAP


# 配置
EDF_DIR = r"D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette"
TEST_SUBJECT = "SC4001"  # 第一个受试者
RESULTS_DIR = os.path.join(project_dir, 'results', TEST_SUBJECT)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"粒计算睡眠阶段识别 — 实验1: {TEST_SUBJECT}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: 读取 EDF
    print(f"\n[1/5] 读取 EDF 数据...")
    psg_path = os.path.join(EDF_DIR, f"{TEST_SUBJECT}E0-PSG.edf")
    hypno_path = os.path.join(EDF_DIR, f"{TEST_SUBJECT}EC-Hypnogram.edf")
    
    if not os.path.exists(psg_path):
        print(f"  ❌ PSG 文件不存在: {psg_path}")
        # 尝试其他后缀
        for f in os.listdir(EDF_DIR):
            if TEST_SUBJECT in f and 'PSG' in f:
                psg_path = os.path.join(EDF_DIR, f)
                break
    
    if not os.path.exists(hypno_path):
        print(f"  ❌ Hypnogram 文件不存在，尝试查找...")
        for f in os.listdir(EDF_DIR):
            if TEST_SUBJECT in f and 'Hypnogram' in f:
                hypno_path = os.path.join(EDF_DIR, f)
                break
    
    print(f"  PSG: {os.path.basename(psg_path)}")
    print(f"  Hypno: {os.path.basename(hypno_path)}")
    
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
    print(f"  通道数: {raw.info['nchan']}, 采样率: {raw.info['sfreq']} Hz")
    print(f"  时长: {raw.times[-1]/3600:.1f} 小时")
    
    # 加载 Hypnogram
    hypno = mne.read_annotations(hypno_path)
    
    # Step 2: 提取频谱特征
    print(f"\n[2/5] 提取频谱特征...")
    
    # 找出 EEG 通道
    eeg_chs = [ch for ch in raw.ch_names if 'EEG' in ch or 'eeg' in ch]
    if not eeg_chs:
        # Fpz-Cz/Pz-Oz 是标准通道
        eeg_chs = [ch for ch in raw.ch_names if ch in ['Fpz-Cz', 'Pz-Oz', 'Fz-Cz', 'Cz-Oz']]
    if not eeg_chs:
        eeg_chs = [raw.ch_names[0]]
    
    print(f"  使用通道: {eeg_chs[0]}")
    
    # 加载数据用于特征提取
    raw.load_data(verbose=False)
    feature_matrix, feature_names = extract_multi_channel_features(raw, eeg_chs)
    n_epochs = feature_matrix.shape[0]
    print(f"  特征矩阵: {feature_matrix.shape}")
    print(f"  特征: {feature_names[:6]}...")
    
    # Step 3: 解析 Hypnogram -> 阶段标签
    print(f"\n[3/5] 解析睡眠阶段标签...")
    
    stage_map = {
        'W': 'W', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'N3', 'R': 'REM', '?': '?'
    }
    
    epoch_duration = 30  # 秒
    sfreq = raw.info['sfreq']
    samples_per_epoch = int(epoch_duration * sfreq)
    
    stages = []
    for onset in hypno.onset:
        dur = hypno.duration[list(hypno.onset).index(onset)]
        sym = hypno.description[list(hypno.onset).index(onset)]
        # 处理可能的多字符描述
        stage = sym[0] if sym else '?'
        stage = stage_map.get(stage, '?')
        n_epochs_for_this = max(1, int(round(dur / epoch_duration)))
        stages.extend([stage] * n_epochs_for_this)
    
    stages = stages[:n_epochs]  # 截断到特征矩阵长度
    print(f"  阶段标签数: {len(stages)}")
    unique, counts = np.unique(stages, return_counts=True)
    for s, c in zip(unique, counts):
        print(f"    {s}: {c} epochs ({c*30/60:.1f} min)")
    
    # Step 4: 构建信息粒
    print(f"\n[4/5] 构建信息粒...")
    
    builder = GranuleBuilder(feature_names=feature_names)
    granules = builder.build_granules(feature_matrix, stages)
    relations = builder.compute_relations(granules)
    transitions = builder.detect_transition_regions(granules, relations)
    
    print(f"  信息粒数: {len(granules)}")
    print(f"  粒关系数: {len(relations)}")
    print(f"  过渡区域: {len(transitions)} 处")
    
    # 统计过渡区域
    total_transition_epochs = sum(end - start for start, end in transitions)
    print(f"  过渡期总epoch数: {total_transition_epochs} ({total_transition_epochs*30/60:.1f} min)")
    
    # 保存结果
    results = {
        'subject': TEST_SUBJECT,
        'n_epochs': n_epochs,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'stage_distribution': {s: int(c) for s, c in zip(unique, counts)},
        'n_transitions': len(transitions),
        'transition_epochs': total_transition_epochs,
        'transition_minutes': total_transition_epochs * 30 / 60,
        'mean_similarity': float(np.mean([r.similarity for r in relations])),
        'transition_accuracy': None,  # TODO: 与专家标注对比
    }
    
    with open(os.path.join(RESULTS_DIR, 'granular_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Step 5: 可视化
    print(f"\n[5/5] 生成可视化...")
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    
    # 1. 睡眠阶段图
    ax = axes[0]
    stage_colors = {'W': '#ff6b6b', 'N1': '#ffd93d', 'N2': '#6bcb77', 'N3': '#4d96ff', 'REM': '#9b59b6', '?': '#cccccc'}
    stage_order = ['W', 'N1', 'N2', 'N3', 'REM']
    stage_y = {s: i for i, s in enumerate(stage_order)}
    
    y = [stage_y.get(s, -1) for s in stages]
    ax.fill_between(range(len(stages)), y, 0, alpha=0.3, color='#4d96ff')
    ax.plot(y, 'b-', linewidth=0.5, alpha=0.7)
    ax.set_yticks(range(len(stage_order)))
    ax.set_yticklabels(stage_order)
    ax.set_ylabel('阶段')
    ax.set_title(f'{TEST_SUBJECT} — 粒计算睡眠阶段分析', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 2. 粒相似度
    ax = axes[1]
    sims = [r.similarity for r in relations]
    trans_mask = [r.is_transition for r in relations]
    
    ax.plot(sims, 'g-', alpha=0.7, linewidth=0.8)
    # 标记过渡期
    for i, is_trans in enumerate(trans_mask):
        if is_trans:
            ax.axvline(x=i, color='red', alpha=0.1, linewidth=0.5)
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='相似度阈值')
    ax.set_ylabel('粒相似度')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 包含度
    ax = axes[2]
    incs = [r.inclusion for r in relations]
    ax.plot(incs, 'purple', alpha=0.7, linewidth=0.8)
    ax.set_ylabel('包含度')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 4. Delta 功率 (粗粒特征)
    ax = axes[3]
    delta_idx = feature_names.index('delta_power') if 'delta_power' in feature_names else 0
    delta_power = feature_matrix[:, delta_idx]
    ax.fill_between(range(len(delta_power)), delta_power, alpha=0.4, color='#4d96ff')
    ax.plot(delta_power, 'b-', linewidth=0.5)
    ax.set_ylabel('Delta 功率')
    ax.grid(True, alpha=0.3)
    
    # 5. Alpha/Delta 比值
    ax = axes[4]
    ad_ratio = feature_matrix[:, feature_names.index('alpha_delta_ratio')] if 'alpha_delta_ratio' in feature_names else np.zeros(n_epochs)
    ax.fill_between(range(len(ad_ratio)), ad_ratio, alpha=0.4, color='#ff6b6b')
    ax.plot(ad_ratio, 'r-', linewidth=0.5)
    ax.set_ylabel('Alpha/Delta')
    ax.set_xlabel('Epoch (×30s)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    vis_path = os.path.join(RESULTS_DIR, 'granular_analysis.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  可视化保存: {vis_path}")
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ 实验完成!")
    print(f"  结果目录: {RESULTS_DIR}")
    print(f"  平均粒相似度: {results['mean_similarity']:.3f}")
    print(f"  过渡期: {results['transition_minutes']:.1f} 分钟 ({results['transition_epochs']} epochs)")
    
    # 打印一个示例粒的模糊表示
    print("\n示例粒 (epoch 100):")
    g = granules[min(100, len(granules)-1)]
    print(f"  真实阶段: {g.stage}")
    print(f"  模糊表示:")
    for fname, fuzzy_vals in g.fuzzy_repr.items():
        for label, mu in fuzzy_vals:
            print(f"    {fname} = {label} (μ={mu:.3f})")


if __name__ == '__main__':
    main()
