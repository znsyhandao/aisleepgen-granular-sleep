"""
子epoch N1检测 v2 — 用微状态方差代替频谱

v1失败原因：5秒子窗口(100Hz*5=500样本)计算频谱不稳定
v2策略：子窗口内用beta_power的波动性和theta短爆检测N1
"""
import numpy as np


def sub_epoch_variance_features(data, sfreq, epoch_samples):
    """
    用子窗口统计量检测N1
    
    每个5秒子窗口提取:
    - beta功率的波动性（std/mean）
    - theta/beta比值
    - 零交叉率变化
    """
    # 滤波参数（简单差分=高通>~5Hz）
    win = int(5 * sfreq)
    n_wins = min(6, data.shape[0] // win)
    
    feats = np.zeros((n_wins, 3))
    
    for i in range(n_wins):
        seg = data[i*win : (i+1)*win]
        if len(seg) < win // 2:
            continue
        
        # 高频成分（用差分近似beta 16-30Hz）
        diff_seg = np.diff(seg)
        # 包络（hilbert近似用rectify+lowpass）
        rms = np.sqrt(np.mean(seg**2))
        env_var = np.std(np.abs(seg)) / (rms + 1e-15)
        
        # 过零率（频率代理）
        zcr = np.sum(np.abs(np.diff(np.sign(seg)))) / (2 * len(seg))
        
        # 慢波成分（低通近似：滑动平均的波动性）
        slow = np.convolve(seg, np.ones(5)/5, mode='same')
        slow_var = np.std(seg - slow) / (rms + 1e-15)
        
        feats[i, 0] = env_var    # 包络波动性（N1时局部不稳定）
        feats[i, 1] = zcr        # 过零率（N1时介于W和N2之间）
        feats[i, 2] = slow_var   # 慢波残留（N1时无慢波，所以残留大）
    
    return feats


def detect_n1_subepoch_v2(sub_feats):
    """
    基于统计特征的N1检测
    
    N1模式（5秒窗口级）:
    - env_var: 中等（比W低但比N2高）
    - zcr: 中等（W高、N2低）
    - slow_var: 中等偏高（无稳定节律）
    """
    if len(sub_feats) < 4:
        return 0.0
    
    # z-score
    means = np.mean(sub_feats, axis=0)
    stds = np.std(sub_feats, axis=0) + 1e-15
    z = (sub_feats - means) / stds
    
    # N1：env_var不极端, zcr不极端, slow_var不极端
    n1_count = 0
    for i in range(len(sub_feats)):
        if (-0.8 < z[i,0] < 0.8 and 
            -0.8 < z[i,1] < 0.8 and
            -0.5 < z[i,2] < 1.0):
            n1_count += 1
    
    return n1_count / len(sub_feats)
