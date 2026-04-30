"""调试模糊规则引擎"""
import sys
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

import numpy as np
from granular.fuzzy_sets_v2 import DataDrivenFuzzySets, SleepStageFuzzyRules

# 构建模拟输入
feature_names = ['delta_power', 'theta_power', 'alpha_power', 'sigma_power', 
                 'beta_power', 'delta_theta_ratio', 'alpha_delta_ratio']

stage_stats = {
    'delta_power': {'W': {'mean': 3e-10, 'std': 1.4e-10}, 'N3': {'mean': 6.7e-10, 'std': 3e-10}, 
                    'N1': {'mean': 6.6e-11, 'std': 7e-11}, 'N2': {'mean': 1.7e-10, 'std': 7.7e-11},
                    'REM': {'mean': 4.2e-11, 'std': 1.8e-11}},
    'theta_power': {'W': {'mean': 3.3e-11, 'std': 1.4e-11}, 'N3': {'mean': 3.4e-11, 'std': 9.3e-12},
                    'N1': {'mean': 1.8e-11, 'std': 1.1e-11}, 'N2': {'mean': 2.2e-11, 'std': 1.0e-11},
                    'REM': {'mean': 1.6e-11, 'std': 9.9e-12}},
    'alpha_power': {'W': {'mean': 8.2e-12, 'std': 2.7e-12}, 'N3': {'mean': 9.4e-12, 'std': 2.8e-12},
                    'N1': {'mean': 5.0e-12, 'std': 1.9e-12}, 'N2': {'mean': 6.8e-12, 'std': 2.4e-12},
                    'REM': {'mean': 4.1e-12, 'std': 9.0e-13}},
    'sigma_power': {'W': {'mean': 4.7e-12, 'std': 1.9e-12}, 'N3': {'mean': 4.7e-12, 'std': 1.8e-12},
                    'N1': {'mean': 1.9e-12, 'std': 1.0e-12}, 'N2': {'mean': 5.5e-12, 'std': 2.6e-12},
                    'REM': {'mean': 1.4e-12, 'std': 4.0e-13}},
    'beta_power': {'W': {'mean': 1.2e-11, 'std': 5.6e-12}, 'N3': {'mean': 1.3e-12, 'std': 1.4e-12},
                   'N1': {'mean': 3.3e-12, 'std': 2.7e-12}, 'N2': {'mean': 1.5e-12, 'std': 8.6e-13},
                   'REM': {'mean': 1.8e-12, 'std': 8.0e-13}},
    'delta_theta_ratio': {'W': {'mean': 9.3, 'std': 2.5}, 'N3': {'mean': 19.9, 'std': 7.5},
                          'N1': {'mean': 3.9, 'std': 3.0}, 'N2': {'mean': 8.8, 'std': 3.9},
                          'REM': {'mean': 3.1, 'std': 1.3}},
    'alpha_delta_ratio': {'W': {'mean': 0.033, 'std': 0.015}, 'N3': {'mean': 0.016, 'std': 0.006},
                          'N1': {'mean': 0.105, 'std': 0.067}, 'N2': {'mean': 0.047, 'std': 0.020},
                          'REM': {'mean': 0.113, 'std': 0.039}},
}

fuzzy = DataDrivenFuzzySets(stage_stats)

# 检查各阶段的模糊化
test_values = {
    'W': {'delta_power': 3e-10, 'beta_power': 1.2e-11},
    'N3': {'delta_power': 6.7e-10, 'beta_power': 1.3e-12},
    'REM': {'delta_power': 4.2e-11, 'beta_power': 1.8e-12},
}

print("=== 模糊集阈值检查 ===")
for fname in feature_names:
    sets = fuzzy.fuzzy_sets.get(fname, [])
    print(f'\n{fname}:')
    for label, (a,b,c,d) in sets:
        print(f'  {label:10s}: ({a:.4e}, {b:.4e}, {c:.4e}, {d:.4e})')

print('\n\n=== 各阶段典型值模糊化 ===')
for stage, vals in test_values.items():
    print(f'\n{stage}:')
    for fname, val in vals.items():
        fuzz = fuzzy.fuzzify(fname, val)
        print(f'  {fname}={val:.4e}: {fuzz[:3]}')

# 检查规则引擎
print('\n\n=== 规则引擎检查 ===')
rules = SleepStageFuzzyRules(feature_names)
print(f'\n规则:')
for stage, stage_rules in rules.rules.items():
    print(f'  {stage}:')
    for r in stage_rules:
        print(f'    {r}')

# 测试推理
print('\n\n=== 完整推理测试 ===')
for stage, vals in test_values.items():
    fuzzy_inputs = {}
    for fname, val in vals.items():
        fuzzy_inputs[fname] = fuzzy.fuzzify(fname, val)
    confs = rules.infer(fuzzy_inputs)
    print(f'\n{stage}:')
    for s, c in sorted(confs.items(), key=lambda x: -x[1]):
        print(f'  {s}: {c:.4f}')
