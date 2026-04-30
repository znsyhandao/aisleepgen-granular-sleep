"""
模糊分类器 v5 — 从 LR 系数提取的可解释规则

不是黑盒 LR → 而是保持模糊可解释性的混合分类器

核心: 
1. 特征 z-score 标准化（LR 学到的最好）
2. 高斯模糊集映射到阶段隶属度
3. 信息粒平滑（3-epoch 窗口）
4. 时序一致性约束（不允许 REM→N3 直接跳变）
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']

# 从 LR 系数提取的可解释规则
# 格式: {stage: [(feature, sign, weight), ...]}
# sign: '+' = 高值, '-' = 低值
LR_RULES = {
    'W': [
        ('beta_power', '+', 8.73),
        ('delta_power', '+', 3.35),
        ('sigma_power', '-', 1.46),
    ],
    'N1': [
        ('beta_power', '+', 3.23),
        ('delta_theta_ratio', '-', 3.14),
        ('sigma_power', '+', 1.32),
    ],
    'N2': [
        ('beta_power', '-', 3.60),
        ('sigma_power', '+', 2.14),
        ('delta_theta_ratio', '+', 2.05),
    ],
    'N3': [
        ('alpha_delta_ratio', '-', 5.39),
        ('beta_power', '-', 4.75),
        ('delta_theta_ratio', '+', 3.82),
    ],
    'REM': [
        ('delta_theta_ratio', '-', 3.88),
        ('beta_power', '-', 3.61),
        ('sigma_power', '-', 3.40),
    ],
}

# 禁止的睡眠阶段转移
# 在睡眠周期中，某些阶段跳变不可能
FORBIDDEN_TRANSITIONS = {
    'W': set(),      # W 可以到任何阶段
    'N1': {'W', 'N3'},  # N1 不会直接到清醒或深睡
    'N2': {'W'},        # N2 不会直接到清醒
    'N3': {'REM', 'W', 'N1'},  # N3 只能到 N2
    'REM': {'N3', 'W'},        # REM 不会直接到深睡或清醒
}


class ExplainableFuzzyClassifier:
    """
    可解释模糊分类器 v5
    
    特性:
    - 使用 LR 系数作为加权规则
    - 输出各阶段置信度（可解释）
    - 支持信息粒时序平滑
    - 强制阶段转移约束
    
    用法:
    clf = ExplainableFuzzyClassifier()
    clf.fit(X_train, y_train)  # 计算均值和标准差
    probs = clf.predict_proba(X_test)  # 概率矩阵
    stages = clf.predict(X_test)       # 预测阶段
    """
    
    def __init__(self):
        self.means_ = None
        self.stds_ = None
        self.feature_names_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """训练 = 计算每个特征的均值+标准差"""
        self.feature_names_ = feature_names
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        self.stds_[self.stds_ == 0] = 1.0
        return self
    
    def _zscore(self, x: np.ndarray) -> np.ndarray:
        return (x - self.means_) / self.stds_
    
    def _score_stage(self, features_z: np.ndarray, stage: str) -> float:
        """计算一个阶段的可解释规则得分"""
        rules = LR_RULES.get(stage, [])
        score = 0.0
        
        for fname, sign, weight in rules:
            # 找特征索引
            try:
                idx = self.feature_names_.index(fname)
            except ValueError:
                continue
            
            z = features_z[idx]
            
            if sign == '+':
                # z-score 高 = 贡献高
                contrib = (z / (1 + abs(z))) * weight  # 压缩到 (-1, 1)
                if z > 0:
                    score += abs(contrib)
            else:
                # z-score 低 = 贡献高
                if z < 0:
                    score += abs(-z / (1 + abs(z))) * weight
        
        return score
    
    def predict_proba_single(self, features: np.ndarray) -> np.ndarray:
        """预测单个样本的5阶段概率（未平滑）"""
        z = self._zscore(features)
        scores = np.zeros(len(STAGES))
        
        for si, stage in enumerate(STAGES):
            scores[si] = self._score_stage(z, stage)
        
        # softmax normalize
        scores = np.exp(scores - np.max(scores))
        scores /= np.sum(scores) + 1e-15
        return scores
    
    def predict(self, X: np.ndarray, smooth_window: int = 3) -> np.ndarray:
        """
        批量预测 + 信息粒平滑
        
        Args:
            X: (n_samples, n_features)
            smooth_window: 滑动窗口大小
        
        Returns:
            stages: (n_samples,) 预测阶段
        """
        n = len(X)
        probs = np.zeros((n, len(STAGES)))
        
        for i in range(n):
            probs[i] = self.predict_proba_single(X[i])
        
        # 信息粒平滑: 滑动窗口平均
        smoothed = probs.copy()
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            for si in range(len(STAGES)):
                smoothed[:, si] = np.convolve(probs[:, si], kernel, mode='same')
        
        # 阶段转移约束（Viterbi-like）
        stages = []
        for i in range(n):
            if i == 0:
                stages.append(STAGES[np.argmax(smoothed[i])])
            else:
                prev = stages[-1]
                forbidden = FORBIDDEN_TRANSITIONS.get(prev, set())
                # 禁止转移的阶段权重设为 0
                adjusted = smoothed[i].copy()
                for si, stage in enumerate(STAGES):
                    if stage in forbidden:
                        adjusted[si] = 0
                stages.append(STAGES[np.argmax(adjusted)])
        
        return np.array(stages)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带置信度的预测
        
        Returns:
            stages: 预测阶段
            probs: 概率矩阵
            confidences: 单样本置信度
        """
        n = len(X)
        probs = np.zeros((n, len(STAGES)))
        confs = np.zeros(n)
        
        for i in range(n):
            probs[i] = self.predict_proba_single(X[i])
            confs[i] = np.max(probs[i])
        
        stages = STAGES[np.argmax(probs, axis=1)]
        
        # 应用转移约束
        corrected = [stages[0]]
        for i in range(1, n):
            prev = corrected[-1]
            forbidden = FORBIDDEN_TRANSITIONS.get(prev, set())
            if stages[i] in forbidden and confs[i] < 0.5:
                # 低置信+禁止转移 = 保持上一阶段
                corrected.append(prev)
            else:
                corrected.append(stages[i])
        
        return np.array(corrected), np.array([np.argmax(probs[i]) for i in range(n)]), probs, confs
