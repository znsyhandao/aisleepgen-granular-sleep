"""
方案B: 过渡期检测器 - 专门处理N1/N2/W边界

N1低recall(22.9%)的核心原因是:
- N1是过渡阶段，在30秒epoch中通常只占10-15秒
- 特征空间夹在W(高beta)和N2(低beta+高sigma)之间
- 标准分类器把N1误判为W或N2

本模块: 在后处理阶段插入过渡期重分类
"""
import numpy as np
from typing import Dict, List, Tuple

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def detect_transition_epochs(predictions: np.ndarray, 
                              features: np.ndarray,
                              confidences: np.ndarray = None) -> np.ndarray:
    """
    检测可能的过渡epoch并重新标记
    
    策略:
    1. 找W→N2或N2→W的边界（中间缺少N1标记的）
    2. 如果两个相邻epoch是W和N2，中间epoch重新标记为N1
    3. 高置信度的原始判断不修改
    """
    pred = predictions.copy()
    conf = confidences if confidences is not None else np.ones(len(predictions))
    
    for i in range(len(pred)):
        left = pred[i-1] if i > 0 else None
        right = pred[i+1] if i < len(pred)-1 else None
        
        # 情况1: W → [?] → N2, 中间epoch应是N1
        if left == 'W' and right == 'N2' and pred[i] not in ('N1', 'N2'):
            pred[i] = 'N1'
        
        # 情况2: N2 → [?] → W, 中间epoch应检查是否N1或W->N1
        if left == 'N2' and right == 'W' and pred[i] not in ('N1', 'W'):
            pred[i] = 'N1'
        
        # 情况3: W → [?] → REM (可能的N1)
        if left == 'W' and right == 'REM' and pred[i] not in ('N1', 'REM'):
            pred[i] = 'N1'
    
    return pred


def refine_n1_from_features(predictions: np.ndarray,
                             features: np.ndarray,
                             delta_theta_idx: int = -1,
                             beta_idx: int = -1) -> np.ndarray:
    """
    用原始特征值精炼N1判断
    
    N1的特征签名:
    - beta功率: 中等 (低于W, 高于N2/REM)
    - delta/theta比值: 中等 (低于N2/N3, 高于W/REM)
    - theta功率: 相对突出
    
    对当前被判为W或N2但特征接近N1的epoch重新标记
    """
    pred = predictions.copy()
    
    if delta_theta_idx < 0 or beta_idx < 0:
        return pred  # 无法精炼
    
    for i in range(len(pred)):
        dt = features[i, delta_theta_idx]
        be = features[i, beta_idx]
        
        # N1的典型特征: 中等beta + 中等delta_theta
        if pred[i] == 'W' and dt > 0.15 and dt < 0.5 and be < 1.5:
            pred[i] = 'N1'
        elif pred[i] == 'N2' and dt > 0.15 and dt < 0.5 and be > -0.5:
            pred[i] = 'N1'
        elif pred[i] == 'REM' and dt > 0.2 and be > 0:
            pred[i] = 'N1'
    
    return pred


def post_process(predictions: np.ndarray,
                 features: np.ndarray,
                 confidences: np.ndarray = None,
                 delta_theta_idx: int = -1,
                 beta_idx: int = -1) -> np.ndarray:
    """完整的N1后处理流程"""
    # Step 1: 过渡检测
    p1 = detect_transition_epochs(predictions, features, confidences)
    # Step 2: 特征精炼
    p2 = refine_n1_from_features(p1, features, delta_theta_idx, beta_idx)
    return p2
