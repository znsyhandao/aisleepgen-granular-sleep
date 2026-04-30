"""
可变粒度 v2: 频谱轨迹突变检测 (Change-Point Detection)

核心洞见: 不需要手写N1规则
只需检测"频谱能量在2-5秒内发生突变"的位置
然后让LR分类器在这些突变区做精细分类

直觉: 睡眠阶段的转换必然伴随频谱能量的突变
    W→N1: beta下降, theta上升(缓慢)
    N1→N2: sigma上升(纺锤波)
    N2→N3: delta上升, beta持续下降
    N3→REM: 全频带转换
    REM→W: beta急剧上升

不识别N1 — 而是识别"变化的时刻" → 在这些时刻做高密度分类
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def _band_energy_trajectory(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    计算全频带能量轨迹, 每1秒一个点
    
    用短时傅里叶(250ms窗口, 50ms步长)提取4个关键频带的能量
    
    Returns:
        (n_timepoints, 4) 矩阵: [delta, theta, sigma, beta] 轨迹
    """
    from scipy import signal
    
    # 250ms窗口, 50ms步长 → 20帧/秒
    nperseg = int(0.25 * sfreq)
    noverlap = int(0.2 * sfreq)
    
    f, t, Sxx = signal.spectrogram(data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    
    # 4个频带: delta(0.5-4), theta(4-8), sigma(11-16), beta(16-30)
    bands = [(0.5, 4), (4, 8), (11, 16), (16, 30)]
    traj = np.zeros((len(t), len(bands)))
    
    for bi, (lo, hi) in enumerate(bands):
        mask = (f >= lo) & (f < hi)
        if np.any(mask):
            traj[:, bi] = np.log10(np.mean(Sxx[mask, :], axis=0) + 1e-15)
        else:
            traj[:, bi] = -10
    
    # 下采样到每1秒一个点
    target_rate = 1  # 每秒
    ratio = int(sfreq / (nperseg - noverlap) / target_rate)
    if ratio > 0:
        n_sec = traj.shape[0] // ratio
        traj_sec = np.mean(traj[:n_sec*ratio].reshape(n_sec, ratio, -1), axis=1)
    else:
        traj_sec = traj
    
    return traj_sec  # (n_seconds, 4)


def detect_transition_regions(data: np.ndarray, sfreq: float, 
                                smooth_sigma: float = 2.0,
                                threshold: float = 0.3) -> np.ndarray:
    """
    检测频谱突变区域
    
    Args:
        data: 原始EEG
        sfreq: 采样率
        smooth_sigma: 高斯平滑sigma (秒)
        threshold: 突变概率阈值
        
    Returns:
        transition_prob: (n_seconds,) 每秒的突变概率
    """
    traj = _band_energy_trajectory(data, sfreq)  # (n_seconds, 4)
    n = len(traj)
    if n < 10:
        return np.zeros(n)
    
    # 平滑
    smooth = np.zeros_like(traj)
    for bi in range(traj.shape[1]):
        smooth[:, bi] = gaussian_filter1d(traj[:, bi], sigma=smooth_sigma)
    
    # 计算每一秒的"变化度": 当前帧与前帧的绝对差之和
    change = np.zeros(n)
    for i in range(1, n):
        change[i] = np.sum(np.abs(smooth[i] - smooth[i-1]))
    
    # 归一化到 [0, 1]
    if np.max(change) > 0:
        change = change / np.max(change)
    
    return change


def adaptive_n1_detection(data: np.ndarray, sfreq: float,
                           epoch_preds: np.ndarray,
                           change_prob: np.ndarray = None) -> np.ndarray:
    """
    基于突变检测的N1重分类
    
    只在突变区做精细分类, 平稳区保持原判
    
    Args:
        data: 原始EEG
        sfreq: 采样率
        epoch_preds: 传统30秒预测
        change_prob: 每秒突变概率 (None则自动计算)
        
    Returns:
        refined_preds: 精炼后的30秒预测
    """
    if change_prob is None:
        change_prob = detect_transition_regions(data, sfreq)
    
    refined = epoch_preds.copy()
    n_epochs = len(epoch_preds)
    epoch_len = int(30 * sfreq)
    
    # 每秒 → 每30秒epoch的突变率
    for ei in range(n_epochs):
        start_s = ei * 30
        end_s = min(start_s + 30, len(change_prob))
        if start_s >= len(change_prob):
            break
        
        # 这个epoch的平均突变率
        epoch_change = np.mean(change_prob[start_s:end_s])
        
        # 突变率高 + 原判W/N2 → 可能N1
        if epoch_change > 0.3 and refined[ei] in ('W', 'N2'):
            # N1特判: beta波动检查
            refined[ei] = '_REVIEW_N1'
    
    return refined
