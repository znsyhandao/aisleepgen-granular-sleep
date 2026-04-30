"""
睡眠分析可视化报告 — 可直接用于小程序

输出: 一张综合睡眠报告图 (hypnogram + 评分 + 信息粒)
"""
import sys, os
import numpy as np
import warnings; warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from granular.sleep_metrics import build_sleep_hypnogram, sleep_quality_score, info_epoch_to_granules
from granular.granular_stager import STAGES

# === 配色 ===
STAGE_COLORS = {
    'W':   '#FF6B6B',  # 红色 - 觉醒
    'N1':  '#FFD93D',  # 黄色 - 浅睡过渡
    'N2':  '#6BCB77',  # 绿色 - 浅睡
    'N3':  '#4D96FF',  # 蓝色 - 深睡
    'REM': '#9B59B6',  # 紫色 - 快速眼动
}
STAGE_NAMES = {'W': '觉醒', 'N1': '浅睡N1', 'N2': '浅睡N2', 'N3': '深睡N3', 'REM': 'REM'}

def generate_sleep_report(pred_stages, confidences, output_path, title='睡眠分析报告'):
    """
    生成睡眠分析可视化报告
    
    Args:
        pred_stages: 预测的阶段数组 (n_epochs,)
        confidences: 每 epoch 置信度 (n_epochs,)
        output_path: 保存路径
        title: 报告标题
    """
    n = len(pred_stages)
    total_min = n * 0.5
    total_hours = total_min / 60
    
    # 构建睡眠结构
    hypno = build_sleep_hypnogram(pred_stages, confidences)
    score = sleep_quality_score(hypno)
    
    # 信息粒
    granules_raw = info_epoch_to_granules(pred_stages, confidences)
    compressed = hypno['granules']
    s = hypno['summary']
    
    # === 创建画布 ===
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    
    # 网格布局
    gs = fig.add_gridspec(3, 4, height_ratios=[1.5, 3, 1.2], hspace=0.3, wspace=0.3)
    
    # ========== 标题行 ==========
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor('#1a1a2e')
    ax_title.axis('off')
    
    ax_title.text(0.02, 0.85, title, fontsize=22, color='white', fontweight='bold', fontfamily='sans-serif')
    
    # 评分大号显示
    score_color = '#2ecc71' if score['total_score'] >= 70 else '#f39c12' if score['total_score'] >= 50 else '#e74c3c'
    ax_title.text(0.98, 0.80, f'{score["total_score"]:.0f}', fontsize=48, color=score_color, 
                  fontweight='bold', ha='right', va='center')
    ax_title.text(0.98, 0.40, f'/{score["grade"]}', fontsize=14, color='#888', ha='right', va='center')
    
    # 基本信息
    info_text = f'总时长: {total_hours:.1f}h  |  睡眠: {total_hours * s["sleep_efficiency_pct"]/100:.1f}h  |  WASO: {s["waso_min"]:.0f}min  |  信息粒: {len(compressed)}个'
    ax_title.text(0.02, 0.35, info_text, fontsize=11, color='#aaa', fontfamily='monospace')
    
    # ========== 脑电图 Hypnogram ==========
    ax_hyp = fig.add_subplot(gs[1, :])
    ax_hyp.set_facecolor('#16213e')
    ax_hyp.set_title('睡眠脑电图 (Hypnogram)', fontsize=13, color='#ddd', pad=8)
    
    # 用信息粒绘制
    y_vals = np.full(n, -1)
    for i, stage in enumerate(pred_stages):
        if stage in STAGES:
            y_vals[i] = STAGES.index(stage)
    
    ax_hyp.fill_between(range(n), y_vals, -0.5, alpha=0.15, color='#4d96ff')
    ax_hyp.plot(y_vals, 'w-', lw=0.5, alpha=0.7)
    
    # 在背景色上覆置信度
    ax_hyp2 = ax_hyp.twinx()
    ax_hyp2.fill_between(range(n), 0, confidences, alpha=0.15, color='#2ecc71')
    ax_hyp2.plot(confidences, 'g-', lw=0.3, alpha=0.4)
    ax_hyp2.set_ylim(0, 1)
    ax_hyp2.set_ylabel('置信度', fontsize=9, color='#2ecc71')
    ax_hyp2.tick_params(colors='#2ecc71', labelsize=8)
    
    # 过渡期标记
    transition_epochs = [i for i in range(n) if confidences[i] < 0.5]
    if transition_epochs:
        ax_hyp.scatter(transition_epochs, y_vals[transition_epochs], 
                       c='red', s=3, alpha=0.3, marker='|')
    
    ax_hyp.set_yticks(range(5))
    ax_hyp.set_yticklabels([STAGE_NAMES[s] for s in STAGES], fontsize=9, color='#bbb')
    ax_hyp.set_ylim(-1, 5)
    ax_hyp.set_xlabel('时间 (小时)', fontsize=10, color='#aaa')
    
    # 时间标尺
    hour_ticks = np.arange(0, n+1, int(60/0.5))  # 每小时
    ax_hyp.set_xticks(hour_ticks)
    ax_hyp.set_xticklabels([f'{i*0.5/60:.0f}h' for i in hour_ticks if i < n], fontsize=8, color='#999')
    ax_hyp.tick_params(colors='#888')
    ax_hyp.grid(axis='y', alpha=0.1, color='white')
    
    # 图例
    legend_patches = [mpatches.Patch(color=c, label=f'{s}') for s, c in STAGE_COLORS.items()]
    ax_hyp.legend(handles=legend_patches, loc='upper right', fontsize=8, 
                  framealpha=0.3, facecolor='#1a1a2e', labelcolor='white')
    
    # ========== 评分五维雷达图 ==========
    ax_radar = fig.add_subplot(gs[2, 0], projection='polar')
    ax_radar.set_facecolor('#16213e')
    
    dims = score['dimensions']
    categories = ['睡眠效率', '深睡比例', 'REM比例', '连续性', '入睡速度']
    values = [min(dims[k]/30*100 if k=='sleep_efficiency' else dims[k]/20*100, 100) for k in ['sleep_efficiency','deep_sleep_N3','REM','continuity','sleep_latency']]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    ax_radar.plot(angles, values, color=score_color, lw=2, alpha=0.8)
    ax_radar.fill(angles, values, color=score_color, alpha=0.15)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=8, color='#bbb')
    ax_radar.set_ylim(0, 100)
    ax_radar.set_yticks([25, 50, 75])
    ax_radar.set_yticklabels(['25', '50', '75'], fontsize=6, color='#666')
    ax_radar.set_title('五维睡眠评分', fontsize=11, color='#ddd', pad=12)
    
    # ========== 睡眠结构摘要 ==========
    ax_summary = fig.add_subplot(gs[2, 1])
    ax_summary.set_facecolor('#16213e')
    ax_summary.axis('off')
    
    stage_data = [(s, s['duration_min']) for s in compressed]
    sleep_total = sum(d for s, d in stage_data if s != 'W')
    
    ax_summary.text(0.05, 0.95, '睡眠结构', fontsize=12, color='#ddd', fontweight='bold')
    
    y_pos = 0.80
    for stage in ['N3', 'N2', 'N1', 'REM']:
        dur = s['stage_min'].get(stage, 0)
        pct = s['stage_pct'].get(stage, 0)
        if dur > 0:
            bar_len = pct / 3  # scale
            ax_summary.barh(y_pos, bar_len, height=0.12, color=STAGE_COLORS[stage], alpha=0.8)
            ax_summary.text(bar_len + 0.5, y_pos, f'{STAGE_NAMES[stage]} {dur:.0f}min ({pct:.0f}%)', 
                           fontsize=8, color='#bbb', va='center')
        y_pos -= 0.17
    
    # WASO
    waso = s.get('waso_min', 0)
    waso_pct = s.get('stage_pct', {}).get('W', 0)
    ax_summary.barh(y_pos-0.1, waso_pct/3, height=0.12, color=STAGE_COLORS['W'], alpha=0.5)
    ax_summary.text(waso_pct/3 + 0.5, y_pos-0.1, f'{STAGE_NAMES["W"]} {waso:.0f}min ({waso_pct:.0f}%)',
                   fontsize=8, color='#888', va='center')
    
    # ========== 关键指标 ==========
    ax_metrics = fig.add_subplot(gs[2, 2])
    ax_metrics.set_facecolor('#16213e')
    ax_metrics.axis('off')
    
    metrics = [
        ('睡眠效率', f'{s["sleep_efficiency_pct"]:.0f}%', '#2ecc71'),
        ('睡眠潜伏期', f'{s.get("sleep_latency_min",0):.0f}min', '#f39c12'),
        ('WASO', f'{s.get("waso_min",0):.0f}min', '#e74c3c'),
        ('总睡眠', f'{total_hours-ag_hours:.1f}h' if (ag_hours:=s['stage_min'].get('W',0)*30/60) else '', '#4d96ff'),
        ('信息粒', f'{len(compressed)}个', '#9b59b6'),
        ('过渡期', f'{s.get("transition_min",0):.0f}min', '#ffd93d'),
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        ax_metrics.text(0.05, 0.90 - i*0.17, label, fontsize=9, color='#888')
        ax_metrics.text(0.95, 0.90 - i*0.17, value, fontsize=11, color=color, 
                       fontweight='bold', ha='right')
    
    # ========== 阶段分布饼图 ==========
    ax_pie = fig.add_subplot(gs[2, 3])
    ax_pie.set_facecolor('#16213e')
    
    pie_values = [s['stage_min'].get(st, 0) for st in STAGES]
    pie_colors = [STAGE_COLORS[st] for st in STAGES]
    
    wedges, texts, autotexts = ax_pie.pie(
        pie_values, labels=None, colors=pie_colors, startangle=90,
        autopct='', pctdistance=0.8, wedgeprops={'linewidth': 1, 'edgecolor': '#16213e'}
    )
    
    # 手动标签
    total_dur = sum(pie_values)
    for i, (stage, wedge) in enumerate(zip(STAGES, wedges)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        r = 1.2
        x = r * np.cos(np.deg2rad(angle))
        y = r * np.sin(np.deg2rad(angle))
        pct = pie_values[i] / total_dur * 100 if total_dur > 0 else 0
        ax_pie.text(x, y, f'{STAGE_NAMES[stage]}\n{pct:.0f}%', fontsize=7, 
                   color='#bbb', ha='center', va='center')
    
    ax_pie.set_title('阶段分布', fontsize=11, color='#ddd', pad=8)
    
    # === 保存 ===
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f'报告已保存: {output_path}', flush=True)
    return output_path


def generate_sample_report():
    """用 SC4001 测试集生成示例报告"""
    project_dir = r'D:\AISleepGen_GranularSleep'
    
    # 读之前跑好的结果
    import json
    rpath = os.path.join(project_dir, 'results', 'SC4001', 'results.json')
    ppath = os.path.join(project_dir, 'results', 'SC4001', 'y_pred.npy')
    tpath = os.path.join(project_dir, 'results', 'SC4001', 'y_test.npy')
    
    if not all(os.path.exists(p) for p in [rpath, ppath, tpath]):
        print('结果文件不存在，跳过')
        return
    
    y_pred = np.load(ppath)
    y_test = np.load(tpath)
    
    # 置信度 = 预测正确与否
    confs = np.array([0.95 if p == t else 0.45 for p, t in zip(y_pred, y_test)])
    
    out = os.path.join(project_dir, 'results', 'SC4001', 'sleep_report.png')
    generate_sleep_report(y_pred, confs, out, title='SC4001 睡眠分析报告')
    
    print('示例报告生成完成!', flush=True)


if __name__ == '__main__':
    generate_sample_report()
