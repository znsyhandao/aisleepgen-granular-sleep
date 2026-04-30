"""
频谱特征提取 — Spectral Feature Extraction

提取每个 epoch 的频谱特征用于粒计算。
直接复用现有 sleep_spindle_analysis 的分析能力。
"""

import numpy as np
import mne
from scipy import signal
from typing import Dict, List, Optional, Tuple


# 频带定义 (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'sigma': (11.0, 16.0),  # 纺锤波频带
    'beta': (16.0, 30.0),
    'gamma': (30.0, 45.0),
}


def compute_epoch_spectrum(data: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算单通道epoch的功率谱密度
    
    参数:
        data: epoch 数据 (n_samples,)
        sfreq: 采样率 (Hz)
        
    返回:
        freqs: 频率数组
        psd: 功率谱密度
    """
    nperseg = min(256, len(data) // 2)
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    """计算指定频带的平均功率"""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def extract_epoch_features(data: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    从单通道EEG epoch提取所有频带特征
    
    参数:
        data: epoch EEG 数据
        sfreq: 采样率
        
    返回:
        {特征名: 值} 字典
    """
    freqs, psd = compute_epoch_spectrum(data, sfreq)
    
    features = {}
    total_power = 0.0
    
    for name, band in FREQ_BANDS.items():
        bp = band_power(freqs, psd, band)
        features[f'{name}_power'] = bp
        total_power += bp
    
    # 相对功率 (归一化)
    if total_power > 0:
        for name in FREQ_BANDS:
            features[f'{name}_ratio'] = features[f'{name}_power'] / total_power
    
    # 频谱边缘频率 (SEF)
    features['sef_50'] = _spectral_edge_frequency(freqs, psd, 0.50)
    features['sef_95'] = _spectral_edge_frequency(freqs, psd, 0.95)
    
    # δ/θ 比值 (睡眠深度指标)
    features['delta_theta_ratio'] = features.get('delta_power', 0) / (features.get('theta_power', 0) + 1e-10)
    features['alpha_delta_ratio'] = features.get('alpha_power', 0) / (features.get('delta_power', 0) + 1e-10)
    
    return features


def extract_multi_channel_features(raw: mne.io.Raw, 
                                    eeg_chs: List[str],
                                    epoch_duration: float = 30.0) -> np.ndarray:
    """
    从多通道EEG数据提取所有epoch特征
    
    参数:
        raw: MNE Raw 对象
        eeg_chs: EEG 通道名列表（取第一个用于分析）
        epoch_duration: epoch 时长 (秒)
        
    返回:
        feature_matrix: (n_epochs, n_features) 特征矩阵
        feature_names: 特征名列表
    """
    if not eeg_chs:
        eeg_chs = [ch for ch in raw.ch_names if 'EEG' in ch or 'eeg' in ch]
    if not eeg_chs:
        eeg_chs = [raw.ch_names[0]]
    
    ch = eeg_chs[0]  # 用第一个EEG通道
    sfreq = raw.info['sfreq']
    
    # 提取数据
    data, _ = raw[ch]
    data = data.flatten()
    
    # 分段
    epoch_len = int(epoch_duration * sfreq)
    n_epochs = len(data) // epoch_len
    
    # 提取第一个epoch的特征（用于确定特征数量）
    first_epoch = data[:epoch_len]
    first_features = extract_epoch_features(first_epoch, sfreq)
    feature_names = list(first_features.keys())
    n_features = len(feature_names)
    
    # 构建特征矩阵
    feature_matrix = np.zeros((n_epochs, n_features))
    feature_matrix[0] = list(first_features.values())
    
    for i in range(1, n_epochs):
        epoch_data = data[i * epoch_len : (i + 1) * epoch_len]
        if len(epoch_data) < epoch_len // 2:
            continue  # 跳过最后一个不完整的epoch
        feat = extract_epoch_features(epoch_data, sfreq)
        feature_matrix[i] = [feat.get(name, 0.0) for name in feature_names]
    
    return feature_matrix, feature_names


def _spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray, 
                              percentile: float) -> float:
    """频谱边缘频率: 低于该频率的功率占总功率的百分比"""
    cum_power = np.cumsum(psd)
    total_power = cum_power[-1]
    if total_power <= 0:
        return 0.0
    target = total_power * percentile
    idx = np.searchsorted(cum_power, target)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])
