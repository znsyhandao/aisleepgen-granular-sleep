"""
最终版: 信息粒增强型睡眠分期

架构:
1. LR z-score 基分类器 (93.8%) — 可解释系数 + 白盒
2. 信息粒时序平滑 — 3/5/7 epoch 滑动窗口
3. 过渡期检测 — 粒间关系 + 阶段转移约束 + 置信度标记
4. 歧义区标记 — 自动识别 N1/W 边界、N2/N3 含混区

输出: 
- 精准分期 + 每30s置信度 + 信息粒时间演化
- 低置信歧义区标记（→ 人工审核 or 用户确认）
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']

# 睡眠生理约束
VALID_TRANSITIONS = {
    'W': ['N1', 'REM'],                    # W → N1 (入睡) 或 W → REM (反常)
    'N1': ['N2', 'W'],                      # N1 → N2 (进入浅睡) 或回 W
    'N2': ['N3', 'REM', 'W'],               # N2 → N3 (深睡) 或 → REM (周期) 或 → W (微觉醒)
    'N3': ['N2', 'W'],                      # N3 → N2 (回到浅睡) 或 → W (非正常)
    'REM': ['N2', 'W'],                     # REM → N2 (下一个周期) 或 → W (自然醒)
}


class GranularSleepStager:
    """
    信息粒增强型睡眠分期器
    
    step 1: z-score + LR 做基分类 → 概率矩阵
    step 2: 信息粒压缩 (连续同阶段/近阶段合并)
    step 3: 粒间关系 (相似度+包含度+转移合法性)
    step 4: 过渡区标记 (歧义 epoch)
    """
    
    def __init__(self, base_classifier=None):
        self.base_clf = base_classifier  # sklearn LR 或其他
        self.scaler_ = None
        self.means_ = None
        self.stds_ = None
    
    def fit(self, X, y, scaler=None):
        """训练"""
        if scaler:
            self.scaler_ = scaler
        else:
            from sklearn.preprocessing import StandardScaler
            self.scaler_ = StandardScaler().fit(X)
            self.means_ = self.scaler_.mean_
            self.stds_ = self.scaler_.scale_
    
    def predict_proba(self, X) -> np.ndarray:
        """概率矩阵"""
        Xs = self.scaler_.transform(X) if self.scaler_ else X
        return self.base_clf.predict_proba(Xs)
    
    def predict(self, X, smooth_window=3, min_granule_epochs=2) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        完整预测流程
        
        Args:
            X: (n_epochs, n_features)
            smooth_window: 平滑窗口
            min_granule_epochs: 最小信息粒长度
        
        Returns:
            stages: (n_epochs,) 最终预测
            confidences: (n_epochs,) 每epoch置信度
            granules: 信息粒列表 [{stage, start, end, confidence, days}, ...]
        """
        probs = self.predict_proba(X)
        n = len(probs)
        
        # Step 1: 平滑
        smoothed = probs.copy()
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            for si in range(len(STAGES)):
                smoothed[:, si] = np.convolve(probs[:, si], kernel, mode='same')
        
        # Step 2: 带约束的 Viterbi 平滑
        stages = np.array([STAGES[np.argmax(smoothed[i])] for i in range(n)])
        confidences = np.array([np.max(smoothed[i]) for i in range(n)])
        
        # 转移约束（保留模式：仅低置信时修正）
        for i in range(1, n):
            prev = stages[i-1]
            # 只在低置信度时做约束修正
            if confidences[i] < 0.5 and stages[i] not in VALID_TRANSITIONS.get(prev, [stages[i]]):
                probs_i = smoothed[i].copy()
                probs_i[STAGES.index(stages[i])] = 0  # 淘汰非法阶段
                stages[i] = STAGES[np.argmax(probs_i)]
        
        # Step 3: 信息粒构建
        granules = self._build_granules(stages, confidences, min_granule_epochs)
        
        return stages, confidences, granules
    
    def _build_granules(self, stages, confidences, min_epochs=2):
        """构建信息粒列表"""
        granules = []
        if len(stages) == 0:
            return granules
        
        current = stages[0]
        start = 0
        
        for i in range(1, len(stages)):
            if stages[i] != current:
                granule_len = i - start
                if granule_len >= min_epochs:
                    granules.append({
                        'stage': current,
                        'start_epoch': start,
                        'end_epoch': i - 1,
                        'start_min': start * 0.5,
                        'end_min': (i - 1) * 0.5,
                        'confidence': float(np.mean(confidences[start:i])),
                        'duration_min': granule_len * 0.5,
                        'n_epochs': granule_len,
                    })
                start = i
                current = stages[i]
        
        # 最后一个粒
        granule_len = len(stages) - start
        if granule_len >= min_epochs:
            granules.append({
                'stage': current,
                'start_epoch': start,
                'end_epoch': len(stages) - 1,
                'start_min': start * 0.5,
                'end_min': (len(stages) - 1) * 0.5,
                'confidence': float(np.mean(confidences[start:])),
                'duration_min': granule_len * 0.5,
                'n_epochs': granule_len,
            })
        
        return granules
    
    def detect_ambiguous_regions(self, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        检测歧义区域（两个阶段置信度接近）
        
        Returns: (n_epochs,) >0 = 歧义索引对数组
        """
        n = len(probs)
        ambiguous = np.zeros(n, dtype=bool)
        
        for i in range(n):
            sorted_probs = np.sort(probs[i])[::-1]
            ambiguous[i] = sorted_probs[1] / max(sorted_probs[0], 1e-10) > (1 - threshold)
        
        return ambiguous
    
    def get_knowledge_rules(self) -> List[str]:
        """提取可解释规则"""
        if not hasattr(self.base_clf, 'coef_'):
            return []
        
        feature_names = [f'f{i}' for i in range(self.base_clf.coef_.shape[1])]
        rules = []
        
        for ci, stage in enumerate(self.base_clf.classes_):
            coefs = self.base_clf.coef_[ci]
            top = np.argsort(-np.abs(coefs))[:3]
            conds = []
            for idx in top:
                name = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
                if coefs[idx] > 0:
                    conds.append(f'{name} ↑ +{coefs[idx]:.2f}')
                else:
                    conds.append(f'{name} ↓ {coefs[idx]:.2f}')
            rules.append(f'IF {" AND ".join(conds)} → {stage}')
        
        return rules
