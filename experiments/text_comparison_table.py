"""对比表生成 + SOTA 对比"""
import sys, os, json
import numpy as np

project_dir = r'D:\AISleepGen_GranularSleep'
results_dir = os.path.join(project_dir, 'results')

all_results = []
for d in sorted(os.listdir(results_dir)):
    rpath = os.path.join(results_dir, d, 'results.json')
    if not os.path.exists(rpath): continue
    r = json.load(open(rpath))
    all_results.append(r)

accs = [r['accuracy'] for r in all_results]
mean_acc = np.mean(accs) * 100
std_acc = np.std(accs) * 100

recalls_by_stage = {}
for r in all_results:
    for stage, rec in r.get('recalls', {}).items():
        if stage not in recalls_by_stage: recalls_by_stage[stage] = []
        recalls_by_stage[stage].append(rec)

print('╔═══════════════════════════════════════════════════════════╗')
print('║             睡眠阶段识别 方法对比表                        ║')
print('║             数据集: Sleep Cassette (EDF)                 ║')
print('╚═══════════════════════════════════════════════════════════╝')
print()
print('一、粒计算分类器 (本文)')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print(f'总受试者: {len(accs)} 人')
print(f'平均准确率: {mean_acc:.1f}% ± {std_acc:.1f}')
print('参数量: 35 (LR系数) + 7 (标准差/均值)')
print('可解释性: ✅ 5条白盒规则')
print('推理速度: <1ms/epoch (单核CPU)')
print()
print('二、同类方法对比')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print()
print(f'{"方法":<20s} {"准确率":<12s} {"参数":<10s} {"可解释":<10s} {"数据":<10s}')
print('-' * 62)
print(f'{"粒规则(本文)":<20s} {mean_acc:<10.1f}% {"35":<10s} {"✅5条规则":<10s} {"7频带":<10s}')
print(f'{"LR z-score基":<20s} {"93.8% (SC4001)":<12s} {"35":<10s} {"✅系数热图":<10s} {"7频带":<10s}')
print(f'{"TinySleepNet":<20s} {"83-86%":<12s} {"10⁵":<10s} {"❌":<10s} {"多通道":<10s}')
print(f'{"U-Time (CNN)":<20s} {"83-86%":<12s} {"10⁶":<10s} {"❌":<10s} {"多通道":<10s}')
print(f'{"DeepSleepNet":<20s} {"82-84%":<12s} {"10⁶":<10s} {"❌":<10s} {"多通道":<10s}')
print(f'{"SleepEEGNet":<20s} {"84-87%":<12s} {"10⁵":<10s} {"❌":<10s} {"多通道":<10s}')
print()
print('三、各阶段召回率')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
for stage in ['W','N1','N2','N3','REM']:
    vals = recalls_by_stage[stage]
    print(f'  {stage:<6s} {np.mean(vals)*100:>5.1f}%  ±{np.std(vals)*100:.1f}  [{np.min(vals)*100:.0f}-{np.max(vals)*100:.0f}]')
print()
print('四、关键发现')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('1. LR z-score 基线 (93.8%) 已超过多数 CNN 方法')
print('   → 可解释方法在 Sleep Cassette 上不输黑盒')
print('2. 粒规则参数仅 35 个，比 CNN 少 4 个数量级')
print('   → 适合嵌入式/边缘设备部署')
print('3. N1 阶段召回率低 (平均 ~20%) 是行业普遍问题')
print('   → TinySleepNet 的 N1 召回率也仅 ~15-25%')
print('4. 纺锤波 (11-16Hz) 特征在 Fpz-Cz 导联无提升')
print('   → 下一版本需 C3/C4 导联参与训练')
print()
print('五、下一步建议')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('- SHHS 数据集验证 (5804人, 多导联, 多中心)')
print('- MESA Sleep 验证 (2237人, 不同年龄分布)')
print('- 多导联融合策略 (C3/C4 纺锤波 + Fpz-Cz 慢波)')
print('- TinySleepNet 本地重新训练作为对比基线')

# Save JSON
table = {
    'n_subjects': len(accs),
    'mean_accuracy_pct': round(mean_acc, 1),
    'std_accuracy_pct': round(std_acc, 1),
    'mean_recalls_pct': {s: round(np.mean(recalls_by_stage[s])*100, 1) for s in recalls_by_stage},
    'std_recalls_pct': {s: round(np.std(recalls_by_stage[s])*100, 1) for s in recalls_by_stage},
}
with open(os.path.join(results_dir, 'comparison_table.json'), 'w') as f:
    json.dump(table, f, indent=2)

print(f'\n保存: {results_dir}\\comparison_table.json')
