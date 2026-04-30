"""
颠覆式 v3: 睡眠连续谱追踪 (Sleep Continuum Tracking)

放弃: 
  - 30秒固定epoch
  - 5个离散阶段
  - 先分类后平滑

采用:
  - 每秒一个"状态向量" (delta, theta, sigma, beta 相对能量)
  - 状态向量在连续空间中移动
  - 睡眠阶段 = 当前状态向量在连续空间中的位置
  - N1 = "从W到N2的路径中段"

数学: 每分钟的4维状态向量在4D空间中画一条轨迹
     W  = (0.2, 0.2, 0.2, 0.4)  高beta
     N1 = (0.3, 0.3, 0.2, 0.2)  过渡
     N2 = (0.4, 0.3, 0.2, 0.1)  sigma波
     N3 = (0.6, 0.2, 0.1, 0.1)  高delta
     REM = (0.2, 0.3, 0.1, 0.4) 低delta, 高beta

不再说"这个epoch是N1"
而是说"此刻状态向量在W→N2路径上, 进度45%"
"""
import numpy as np
from scipy import integrate
from scipy import signal as sg, integrate
from scipy.ndimage import gaussian_filter1d

# 5个参考状态的4维向量 (delta_pct, theta_pct, sigma_pct, beta_pct)
REFERENCE_STATES = {
    'W':   np.array([0.15, 0.20, 0.15, 0.50]),
    'N1':  np.array([0.25, 0.30, 0.20, 0.25]),
    'N2':  np.array([0.40, 0.25, 0.20, 0.15]),
    'N3':  np.array([0.60, 0.20, 0.10, 0.10]),
    'REM': np.array([0.15, 0.30, 0.10, 0.45]),
}


def compute_continuum_trajectory(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    计算睡眠连续谱轨迹
    
    Args:
        data: 原始EEG
        sfreq: 采样率
    
    Returns:
        trajectory: (n_seconds, 4) 每秒的4维状态向量
    """
    # 1秒窗口, 每1秒滑动, 计算4频带相对能量
    win = int(sfreq)
    step = int(sfreq)
    
    n_sec = (len(data) - win) // step + 1
    if n_sec < 1:
        return np.zeros((1, 4))
    
    traj = np.zeros((n_sec, 4))
    
    for i in range(n_sec):
        seg = data[i*step : i*step + win]
        if len(seg) < win // 2:
            continue
        
        # FFT
        fft = np.fft.rfft(seg * np.hanning(len(seg)))
        psd = np.abs(fft)**2 / len(seg)
        freqs = np.fft.rfftfreq(len(seg), 1/sfreq)
        
        # 4频带
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'sigma': (11, 16), 'beta': (16, 30)}
        total = 1e-15
        for bi, (name, (lo, hi)) in enumerate(bands.items()):
            mask = (freqs >= lo) & (freqs < hi)
            p = integrate.trapezoid(psd[mask], freqs[mask]) if np.any(mask) else 1e-15
            traj[i, bi] = p
            total += p
        
        # 归一化为相对能量
        if total > 0:
            traj[i] /= total
    
    return traj


def state_to_stage(state_vector: np.ndarray) -> str:
    """
    状态向量 → 最近的参考阶段
    
    用余弦相似度找最近邻
    """
    best_dist = float('inf')
    best_stage = 'W'
    
    for stage, ref in REFERENCE_STATES.items():
        # 欧氏距离
        dist = np.sqrt(np.sum((state_vector - ref)**2))
        if dist < best_dist:
            best_dist = dist
            best_stage = stage
    
    return best_stage


def trajectory_distance_to_ref(traj: np.ndarray, ref_stage: str) -> np.ndarray:
    """计算轨迹到参考状态的距离"""
    ref = REFERENCE_STATES[ref_stage]
    return np.sqrt(np.sum((traj - ref)**2, axis=1))


def estimate_n1_progress(traj: np.ndarray) -> np.ndarray:
    """
    估计轨迹在W→N2路径上的进度
    
    0.0 = 完全W, 1.0 = 完全N2
    0.3~0.7 = N1区
    
    路径向量: N2 - W
    当前投影: (traj_i - W) 在路径上的投影长度 / 路径总长度
    """
    w = REFERENCE_STATES['W']
    n2 = REFERENCE_STATES['N2']
    path = n2 - w
    path_len = np.linalg.norm(path)
    if path_len < 1e-10:
        return np.zeros(len(traj))
    
    progress = np.zeros(len(traj))
    for i in range(len(traj)):
        proj = np.dot(traj[i] - w, path) / path_len
        progress[i] = np.clip(proj / path_len, 0, 1)
    
    return progress


def continuum_staging(data: np.ndarray, sfreq: float) -> tuple:
    """
    连续谱睡眠追踪主入口
    
    Returns:
        stages: 每秒的睡眠阶段
        trajectory: 每秒的4维状态向量
        n1_progress: 每秒的N1进度
        distance_to_ref: 每秒到各参考状态的距离 (n_sec, 5)
    """
    traj = compute_continuum_trajectory(data, sfreq)
    n = len(traj)
    
    # 平滑轨迹
    smooth = np.zeros_like(traj)
    for bi in range(4):
        smooth[:, bi] = gaussian_filter1d(traj[:, bi], sigma=5)
    
    # 每秒的阶段和进度
    stages = []
    for i in range(n):
        stages.append(state_to_stage(smooth[i]))
    
    n1_progress = estimate_n1_progress(smooth)
    
    # 到各参考状态的距离
    dists = np.zeros((n, 5))
    for si, stage in enumerate(['W', 'N1', 'N2', 'N3', 'REM']):
        dists[:, si] = trajectory_distance_to_ref(smooth, stage)
    
    return np.array(stages), smooth, n1_progress, dists


def epoch_stages_from_continuum(stages_per_sec: np.ndarray, 
                                  n1_progress: np.ndarray) -> np.ndarray:
    """
    把每秒的连续谱阶段聚合成30秒epoch
    
    这不是简单的投票
    而是看30秒内状态向量的轨迹经过哪些区域
    """
    n_epochs = len(stages_per_sec) // 30
    epoch_stages = []
    
    for ei in range(n_epochs):
        start = ei * 30
        end = min(start + 30, len(stages_per_sec))
        sec_stages = stages_per_sec[start:end]
        sec_progress = n1_progress[start:end]
        
        # N1检测: 进度在0.2-0.8之间的秒数超过33%
        n1_fraction = np.mean((sec_progress > 0.2) & (sec_progress < 0.8))
        
        if n1_fraction > 0.3:
            epoch_stages.append('N1')
        else:
            # 多数投票
            from collections import Counter
            counter = Counter(sec_stages)
            # 排除'?' 
            if '?' in counter: del counter['?']
            if counter:
                epoch_stages.append(counter.most_common(1)[0][0])
            else:
                epoch_stages.append('W')
    
    return np.array(epoch_stages)
