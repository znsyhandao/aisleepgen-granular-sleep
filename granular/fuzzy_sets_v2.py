"""
数据驱动的模糊集定义 - 根据实际数据统计设定阈值

对每个频带特征，根据各阶段的均值+标准差定义：
- low, medium, high 的隶属度函数
- 使用梯形/三角形隶属度，参数由 {W, N1, N2, N3, REM} 分布决定
"""

import numpy as np
from typing import Dict, Tuple, List


class DataDrivenFuzzySets:
    """
    基于数据分布的模糊集
    
    对每个特征，定义 3-4 个模糊集（very_low, low, medium, high）
    阈值根据各睡眠阶段的均值+标准差自动计算
    """
    
    def __init__(self, stage_stats: Dict[str, Dict[str, Dict[str, float]]]):
        """
        stage_stats: {
            'delta_power': {
                'W': {'mean': x, 'std': y},
                'N1': {'mean': x, 'std': y},
                ...
            },
            ...
        }
        """
        self.stage_stats = stage_stats
        self.fuzzy_sets = {}  # {feature_name: [(label, a, b, c, d), ...]}
        
        for fname, stages in stage_stats.items():
            self.fuzzy_sets[fname] = self._build_fuzzy_sets(fname, stages)
    
    def _build_fuzzy_sets(self, fname: str, stages: Dict) -> List:
        """构建模糊集"""
        all_means = [s['mean'] for s in stages.values()]
        all_stds = [s['std'] for s in stages.values()]
        
        overall_min = min(s['mean'] - 2*s['std'] for s in stages.values())
        overall_max = max(s['mean'] + 2*s['std'] for s in stages.values())
        
        # 用 log 空间处理功率值（跨 3 个数量级）
        log_means = np.log10(np.array(all_means) + 1e-20)
        log_std = np.mean(np.log10(np.array(all_stds) + 1e-20))
        
        p10 = np.percentile(log_means, 15)
        p50 = np.percentile(log_means, 50)
        p90 = np.percentile(log_means, 85)
        
        # 在 log 空间构建模糊集，然后转回线性
        def logspace_trapezoid(low, high, margin=0.3):
            """log 空间中构建梯形"""
            m = log_std * margin
            return (10**low, 10**(low+m), 10**(high-m), 10**high)
        
        sets = []
        
        # very_low: p10 以下
        vl_base = p10 - log_std * 2
        sets.append(('very_low', logspace_trapezoid(vl_base, p10 - log_std*0.5)))
        
        # low: 围绕 p10-p40
        sets.append(('low', logspace_trapezoid(p10 - log_std, p50 - log_std*0.3)))
        
        # medium: 围绕 p30-p70
        sets.append(('medium', logspace_trapezoid(p10 - log_std*0.5, p90 + log_std*0.5)))
        
        # high: p60 以上
        sets.append(('high', logspace_trapezoid(p50 + log_std*0.3, p90 + log_std*2)))
        
        return sets
    
    def fuzzify(self, fname: str, value: float) -> List[Tuple[str, float]]:
        """将一个数值映射到模糊集隶属度"""
        result = []
        for label, (a, b, c, d) in self.fuzzy_sets[fname]:
            mu = self._trapezoid_membership(value, a, b, c, d)
            if mu > 0:
                result.append((label, mu))
        return sorted(result, key=lambda x: -x[1])
    
    def _trapezoid_membership(self, x: float, a: float, b: float, c: float, d: float) -> float:
        """梯形隶属度函数"""
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if a < x < b:
            return (x - a) / (b - a)
        if c < x < d:
            return (d - x) / (d - c)
        return 0.0


class SleepStageFuzzyRules:
    """
    睡眠阶段模糊推理规则库
    基于 AASM 标准 + 频谱特征 -> 模糊 if-then 规则
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.rules = self._build_rules()
    
    def _build_rules(self):
        """构建模糊推理规则"""
        rules = {
            'W': [
                f"delta_power is very_low AND beta_power is not low",
                f"beta_power is high AND delta_power is very_low",
                f"beta_power is medium AND alpha_power is not very_low",
            ],
            'N1': [
                f"delta_power is low AND theta_power is not very_low AND beta_power is not high",
                f"theta_power is medium AND alpha_power is low AND delta_power is not high",
            ],
            'N2': [
                f"delta_power is medium AND sigma_power is not very_low AND beta_power is low",
                f"delta_power is medium AND beta_power is very_low AND alpha_power is low",
            ],
            'N3': [
                f"delta_power is high AND beta_power is very_low",
                f"delta_power is high AND sigma_power is low AND beta_power is low",
                f"delta_theta_ratio is high AND beta_power is low",
            ],
            'REM': [
                f"delta_power is very_low AND theta_power is very_low AND beta_power is very_low",
                f"alpha_power is very_low AND delta_power is very_low AND theta_power is low",
                f"theta_power is very_low AND delta_power is very_low AND alpha_power is very_low",
            ],
        }
        return rules
    
    def infer(self, fuzzy_inputs: Dict[str, List[Tuple]]) -> Dict[str, float]:
        """模糊推理：输入 fuzzy_inputs {特征名: [(标签, 隶属度)], ...} -> 各阶段置信度"""
        # 先构建特征名 -> {(标签, 隶属度)}
        feat_idx = {}
        for fname, fuzz in fuzzy_inputs.items():
            feat_idx[fname] = {label: mu for label, mu in fuzz}
        
        confidences = {}
        for stage, stage_rules in self.rules.items():
            max_conf = 0.0
            for rule_text in stage_rules:
                conf = self._eval_rule(rule_text, feat_idx)
                max_conf = max(max_conf, conf)
            confidences[stage] = max_conf
        
        return confidences
    
    def _eval_rule(self, rule_text: str, feat_idx: Dict) -> float:
        """评估一条模糊规则"""
        # 简单解析: "feat is label AND feat is label"
        parts = rule_text.split(' AND ')
        confs = []
        for part in parts:
            part = part.strip()
            if ' is not ' in part:
                feat_label, val = part.split(' is not ')
                feat, label = feat_label.strip(), val.strip()
                feature = feat.strip()
                fuzzy_labels = feat_idx.get(feature, {})
                mu = fuzzy_labels.get(label, 0.0)
                confs.append(1.0 - mu)  # NOT = 1 - mu
            elif ' is ' in part:
                feat, label = part.split(' is ')
                feature, label = feat.strip(), label.strip()
                fuzzy_labels = feat_idx.get(feature, {})
                mu = fuzzy_labels.get(label, 0.0)
                confs.append(mu)  # AND = min
        return min(confs) if confs else 0.0
