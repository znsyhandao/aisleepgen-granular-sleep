"""
方案B: 过渡期检测器（仅过渡检测+过预测抑制）
v4: 只做W↔N2边界N1恢复，不做特征精炼
"""
import numpy as np

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def post_process(predictions, features=None, confidences=None,
                 delta_theta_idx=-1, beta_idx=-1):
    """仅过渡检测 + 连续N1抑制"""
    pred = predictions.copy()
    n = len(pred)
    
    # Step 1: 过渡期N1恢复
    for i in range(n):
        left = pred[i-1] if i > 0 else None
        right = pred[i+1] if i < n-1 else None
        if left == 'W' and right in ('N2', 'REM') and pred[i] not in ('N1', 'N2'):
            pred[i] = 'N1'
        elif left == 'N2' and right == 'W' and pred[i] not in ('N1', 'W'):
            pred[i] = 'N1'
    
    # Step 2: 连续N1>3个的块恢复原始判断
    i = 0
    while i < n:
        if pred[i] == 'N1':
            j = i
            while j < n and pred[j] == 'N1':
                j += 1
            if j - i > 3:
                for k in range(i, j):
                    if predictions[k] != 'N1':
                        pred[k] = predictions[k]
            i = j
        else:
            i += 1
    
    return pred
