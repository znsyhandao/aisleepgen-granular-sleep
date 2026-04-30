"""轻量纺锤波检测 — 从眠小兔系统移植"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import butter, sosfiltfilt, hilbert, welch


def detect_spindles(eeg_data, sfreq, amplitude_threshold=2.0):
    """
    检测 EEG 信号中的睡眠纺锤波 (11-16 Hz)

    Parameters
    ----------
    eeg_data : ndarray, shape (n_samples,)
        一维 EEG 信号
    sfreq : float
        采样率 (Hz)
    amplitude_threshold : float
        幅度阈值倍数 (相对于 RMS), 默认 2.0

    Returns
    -------
    n_spindles : int
        检测到的纺锤波数量
    density_per_second : float
        每秒纺锤波密度
    spindle_features_list : list of dict
        每个纺锤波的详细信息:
            start_time, end_time, duration, frequency, amplitude, symmetry, type
    """
    min_dur, max_dur = 0.5, 3.0  # 纺锤波时长范围 (秒)

    # 11-16Hz 带通滤波 (butter 4阶, sosfiltfilt)
    low, high = 11.0, 16.0
    sos = butter(4, [low, high], btype='band', fs=sfreq, output='sos')
    filtered = sosfiltfilt(sos, eeg_data)

    # Hilbert 包络
    envelope = np.abs(hilbert(filtered))

    # 平滑包络 (100ms 滑动窗口)
    window_len = int(sfreq * 0.1)
    if window_len < 1:
        window_len = 1
    envelope_smooth = np.convolve(
        envelope,
        np.ones(window_len) / window_len,
        mode='same'
    )

    # RMS * amplitude_threshold
    rms = np.sqrt(np.mean(envelope_smooth ** 2))
    threshold = rms * amplitude_threshold

    # 检测候选纺锤波
    mask = envelope_smooth > threshold
    diff_mask = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(diff_mask == 1)[0]
    ends = np.where(diff_mask == -1)[0]

    spindle_features_list = []
    for start, end in zip(starts, ends):
        duration = (end - start) / sfreq
        if min_dur <= duration <= max_dur:
            segment = filtered[start:end]
            segment_env = envelope_smooth[start:end]

            # 频率估计 (Welch PSD 峰值)
            freq = _estimate_frequency(segment, sfreq)
            amplitude_val = float(np.max(segment_env))
            symmetry = _calculate_symmetry(segment_env)

            spindle_features_list.append({
                'start_time': start / sfreq,
                'end_time': end / sfreq,
                'duration': float(duration),
                'frequency': float(freq),
                'amplitude': amplitude_val,
                'symmetry': symmetry,
                'type': 'fast' if freq >= 13 else 'slow'
            })

    n_spindles = len(spindle_features_list)
    total_time = len(eeg_data) / sfreq
    density_per_second = n_spindles / total_time if total_time > 0 else 0.0

    return n_spindles, density_per_second, spindle_features_list


def epoch_spindle_density(eeg_epoch, sfreq, amplitude_threshold=2.0):
    """
    计算单个 epoch 的纺锤波密度

    Parameters
    ----------
    eeg_epoch : ndarray
        单个 epoch 的 EEG 数据
    sfreq : float
        采样率
    amplitude_threshold : float
        幅度阈值

    Returns
    -------
    density : float
        该 epoch 的纺锤波密度 (个/秒)
    count : int
        该 epoch 内的纺锤波数量
    """
    n, density, _ = detect_spindles(eeg_epoch, sfreq, amplitude_threshold)
    return density, n


def _estimate_frequency(segment, sfreq):
    """估计纺锤波片段的峰值频率"""
    if len(segment) < 10:
        return 13.0
    nperseg = min(256, len(segment))
    freqs, psd = welch(segment, sfreq, nperseg=nperseg)
    mask = (freqs >= 11.0) & (freqs <= 16.0)
    if np.any(mask):
        peak_idx = np.argmax(psd[mask])
        return float(freqs[mask][peak_idx])
    return 13.0


def _calculate_symmetry(envelope):
    """计算纺锤波包络的对称性"""
    if len(envelope) < 3:
        return 0.5
    peak_idx = np.argmax(envelope)
    if peak_idx == 0 or peak_idx == len(envelope) - 1:
        return 0.5
    rise_area = np.trapezoid(envelope[:peak_idx + 1])
    fall_area = np.trapezoid(envelope[peak_idx:])
    total = rise_area + fall_area
    if total == 0:
        return 0.5
    return float(rise_area / total)
