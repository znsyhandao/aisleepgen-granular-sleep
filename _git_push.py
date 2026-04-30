import sys, os, subprocess

project_dir = r'D:\AISleepGen_GranularSleep'
os.chdir(project_dir)

# Initialize git if not already
if not os.path.exists(os.path.join(project_dir, '.git')):
    subprocess.run(['git', 'init'], check=True)
    subprocess.run(['git', 'remote', 'add', 'origin', 'git@github.com:znsyhandao/aisleepgen-granular-sleep.git'], check=True)

# Stage and commit
subprocess.run(['git', 'add', '-A'], check=True)
subprocess.run(['git', 'commit', '-m', 'feat: 第一个模糊睡眠阶段分类器
- DataDrivenFuzzySets: 基于数据分布的模糊集定义
- PrototypeFuzzySet + FuzzyStageClassifier: 高斯原型匹配
- 73% 自洽准确率 (单通道EEG, 5频带特征)
- 实验: run_pipeline, text_inference, text_fuzzy_classifier, text_self_test
- 可视化: granular_analysis.png, band_stats.png, fuzzy_inference.png
- 基础框架: fuzzy_sets.py, info_granules.py, spectral.py, edf_reader.py'], check=True)
