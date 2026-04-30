"""
模糊推理引擎 v4 — 基于模糊向量原型匹配

核心思路:
1. 每个特征值 fuzzify 到 5 个阶段的隶属度 → 5维模糊向量
2. 各阶段有原型向量（训练阶段均值）
3. 推理 = 余弦相似度匹配

后续可升级为:
- 加权特征融合
- 信息粒构建 (一段连续 epoch 的平均模糊向量)
- 模糊规则提取 (if delta_high and beta_low then N3)
"""

import numpy as np
from typing import Dict, List, Tuple

# 标准睡眠阶段顺序
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def _gaussian(x, mu, sigma):
    """高斯隶属度函数"""
    return np.exp(-0.5 * ((x - mu) / max(sigma, 1e-15)) ** 2)


def _triangular(x, a, b, c):
    """三角形隶属度"""
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a + 1e-15)
    return (c - x) / (c - b + 1e-15)


class PrototypeFuzzySet:
    """
    原型匹配模糊集
    
    对每个特征，用高斯隶属度定义 5 个原型（每个阶段一个峰）
    
    例如 delta_power:
        W 原型: mu=3.0e-10, sigma=1.4e-10 → 在 3.0e-10 附近最像 W
        N3原型: mu=6.7e-10, sigma=3.0e-10 → 在 6.7e-10 附近最像 N3
        ...
    
    fuzzify(feature, value) → [(W, mu_W), (N1, mu_N1), ...]
    """
    
    def __init__(self, feature_name: str, stage_means: Dict[str, float], stage_stds: Dict[str, float]):
        self.name = feature_name
        self.prototypes = {
            stage: {'mu': stage_means[stage], 'sigma': max(stage_stds[stage], stage_means[stage] * 0.05)}
            for stage in STAGES if stage in stage_means
        }
        self.stages = list(self.prototypes.keys())
    
    def fuzzify(self, value: float) -> List[Tuple[str, float]]:
        """将数值映射到各阶段隶属度"""
        result = []
        for stage, p in self.prototypes.items():
            mu = _gaussian(value, p['mu'], p['sigma'])
            if mu > 1e-10:
                result.append((stage, float(mu)))
        result.sort(key=lambda x: -x[1])
        return result


class FuzzyStageClassifier:
    """
    模糊睡眠阶段分类器
    
    训练: 从标注数据中学习各阶段的原型向量
    推理: 特征值 → 模糊向量 → 原型匹配
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.fuzzy_sets = {}  # {feature_name: PrototypeFuzzySet}
        self.stage_prototypes = {}  # {stage: [proto_mu_for_each_feature]}
        
    def train(self, feature_matrix: np.ndarray, stages: np.ndarray):
        """
        训练分类器
        
        Args:
            feature_matrix: (n_epochs, n_features)
            stages: (n_epochs,) 阶段标签
        """
        print('  训练原型模糊集...')
        
        # 1. 计算每个特征在各阶段的统计量
        for fi, fname in enumerate(self.feature_names):
            stage_means = {}
            stage_stds = {}
            for stage in STAGES:
                vals = feature_matrix[:, fi][stages == stage]
                if len(vals) > 5:
                    stage_means[stage] = float(np.mean(vals))
                    stage_stds[stage] = float(np.std(vals))
            
            self.fuzzy_sets[fname] = PrototypeFuzzySet(fname, stage_means, stage_stds)
        
        # 2. 计算各阶段的原型向量
        #    原型 = 该阶段所有样本的平均模糊向量
        print('  计算阶段原型向量...')
        all_vecs = np.zeros((len(feature_matrix), len(STAGES)))
        
        for i in range(len(feature_matrix)):
            vec = self._to_fuzzy_vector(feature_matrix[i])
            all_vecs[i] = vec
        
        self.stage_prototypes = {}
        for si, stage in enumerate(STAGES):
            mask = stages == stage
            if mask.sum() > 5:
                self.stage_prototypes[stage] = np.mean(all_vecs[mask], axis=0)
            else:
                self.stage_prototypes[stage] = np.zeros(len(STAGES))
        
        # 3. 归一化原型向量
        for stage in self.stage_prototypes:
            norm = np.linalg.norm(self.stage_prototypes[stage])
            if norm > 0:
                self.stage_prototypes[stage] /= norm
    
    def _to_fuzzy_vector(self, features: np.ndarray) -> np.ndarray:
        """特征向量 → 5维模糊向量 (stage 隶属度)"""
        fuzzy_vec = np.zeros(len(STAGES))
        
        for fi, fname in enumerate(self.feature_names):
            fuzz = self.fuzzy_sets[fname].fuzzify(features[fi])
            for stage, mu in fuzz:
                si = STAGES.index(stage)
                fuzzy_vec[si] += mu
        
        # 归一化
        total = np.sum(fuzzy_vec)
        if total > 0:
            fuzzy_vec /= total
        
        return fuzzy_vec
    
    def predict(self, features: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        预测一个样本的阶段
        
        Returns:
            stage: 预测阶段
            confidences: 各阶段置信度向量 (5维)
        """
        fuzzy_vec = self._to_fuzzy_vector(features)
        fuzzy_norm = fuzzy_vec / max(np.linalg.norm(fuzzy_vec), 1e-15)
        
        # 与各阶段原型做余弦相似度
        scores = {}
        for stage, proto in self.stage_prototypes.items():
            scores[stage] = float(np.dot(fuzzy_norm, proto))
        
        predicted = max(scores, key=scores.get)
        return predicted, scores
