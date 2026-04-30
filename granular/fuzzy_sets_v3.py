"""
数据驱动的模糊集定义 v3 - 简单直接版

对每个特征，根据各阶段均值分布定义：
- very_low: <= mean_lowest
- low: 在 mean_low ~ mean_lowmid
- medium: 在 mean_mid
- high: >= mean_high

使用三角形隶属度，避免梯形边界问题
"""

import numpy as np
from typing import Dict, List, Tuple


def _triangular_membership(x, a, b, c):
    """三角形隶属度函数"""
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a + 1e-15)
    if b < x < c:
        return (c - x) / (c - b + 1e-15)
    return 0.0


class SimpleFuzzySets:
    """
    基于各阶段均值的简单模糊集
    
    对每个特征 f:
        1. 计算各阶段均值, 排序
        2. 在均值所在位置构建三角形隶属度函数
        3. 在均值之间自动插值出 low/medium/high 的边界
    """
    
    def __init__(self, stage_stats: Dict):
        self.sets = {}  # {feature_name: {stage: (a, b, c)}, ...}
        
        for fname, stages in stage_stats.items():
            means = {s: d['mean'] for s, d in stages.items()}
            self.sets[fname] = self._build_sets(fname, means)
    
    def _build_sets(self, fname, means):
        """为每个阶段构建一个三角形隶属度三角形 (峰在均值处)"""
        all_vals = list(means.values())
        min_val = min(all_vals)
        max_val = max(all_vals)
        range_val = max_val - min_val if max_val > min_val else max_val * 0.5
        
        # 如果值跨几个数量级，用 log 再 tri
        if max_val > 0 and min_val > 0 and max_val / min_val > 100:
            log_means = {s: np.log(v) for s, v in means.items()}
            all_log = list(log_means.values())
            log_min, log_max = min(all_log), max(all_log)
            log_range = log_max - log_min if log_max > log_min else 1.0
            
            sets = {}
            for stage, lv in log_means.items():
                spread = log_range * 0.3
                sets[stage] = (np.exp(lv - spread), np.exp(lv), np.exp(lv + spread))
            return sets
        
        # 正常范围
        sets = {}
        for stage, v in means.items():
            spread = range_val * 0.3 + 1e-15
            sets[stage] = (v - spread, v, v + spread)
        
        return sets
    
    def fuzzify(self, fname: str, value: float) -> List[Tuple[str, float]]:
        """模糊化 - 返回 [(stage, membership), ...]"""
        result = []
        sets = self.sets.get(fname, {})
        for stage, (a, b, c) in sets.items():
            mu = _triangular_membership(value, a, b, c)
            if mu > 0:
                result.append((stage, mu))
        return sorted(result, key=lambda x: -x[1])


class SleepStageFuzzyRulesV2:
    """基于 AASM 的模糊推理规则 (v2 简单版)"""
    
    def __init__(self):
        # 每个阶段的频谱模式特征
        self.stage_patterns = {
            'W':   {'delta': 'lowest', 'beta': 'highest', 'alpha': 'mid'},
            'N1':  {'delta': 'low', 'theta': 'mid', 'alpha': 'low'},
            'N2':  {'delta': 'mid', 'sigma': 'mid', 'beta': 'low'},
            'N3':  {'delta': 'highest', 'beta': 'lowest', 'delta_theta': 'high'},
            'REM': {'delta': 'lowest', 'alpha': 'lowest', 'theta': 'low'},
        }
    
    def infer(self, fuzzy_inputs: Dict[str, List[Tuple]]) -> Dict[str, float]:
        """推理: 对每个阶段, 计算其频谱模式与当前模糊输入的匹配度"""
        # 先构建特征 -> 各阶段隶属度
        feat_stages = {}
        for fname, fuzzy_list in fuzzy_inputs.items():
            feat_stages[fname] = {stage: mu for stage, mu in fuzzy_list}
        
        confidences = {}
        for stage, patterns in self.stage_patterns.items():
            conf = 0.0
            count = 0
            for feat_key, expected_rank in patterns.items():
                fname = feat_key
                if fname == 'delta_theta':
                    fname = 'delta_theta_ratio'
                
                fuzzy_map = feat_stages.get(fname, {})
                # 要找的是与该阶段最接近的匹配度
                if expected_rank == 'highest':
                    mu = fuzzy_map.get(stage, 0.0)
                elif expected_rank == 'lowest':
                    mu = fuzzy_map.get(stage, 0.0)
                elif expected_rank == 'mid':
                    mu = fuzzy_map.get(stage, 0.0)
                elif expected_rank == 'low':
                    mu = fuzzy_map.get(stage, 0.0)
                elif expected_rank == 'high':
                    mu = fuzzy_map.get(stage, 0.0)
                else:
                    mu = 0.0
                conf += mu
                count += 1
            
            confidences[stage] = conf / max(count, 1)
        
        return confidences
