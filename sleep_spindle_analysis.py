#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眠小兔专业睡眠分析系统
整合呼吸事件、睡眠分期、纺锤波分析的完整解决方案
基于AASM标准，支持多通道分析，提供临床级报告
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import mne
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import welch, butter, filtfilt, hilbert
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import traceback

warnings.filterwarnings('ignore')

# ==================== 配置类 ====================
@dataclass
class AnalysisConfig:
    """分析配置参数"""
    # 基本参数
    epoch_duration: int = 30  # 秒
    min_event_duration: int = 10  # 秒
    
    # 频带定义 (Hz)
    freq_bands: Dict = None
    
    # 呼吸事件阈值
    apnea_threshold: float = 0.1  # 气流下降90%
    hypopnea_threshold: float = 0.7  # 气流下降30%
    
    # 纺锤波参数
    spindle_freq_range: Tuple = (11, 16)  # Hz
    spindle_duration_range: Tuple = (0.5, 3.0)  # 秒
    spindle_amplitude_threshold: float = 2.0  # 标准差倍数
    
    # 血氧参数
    spo2_drop_threshold: float = 3.0  # 百分比
    spo2_drop_duration: int = 10  # 秒
    
    # 输出设置
    output_dir: str = "sleep_analysis_results"
    figure_dpi: int = 150
    save_intermediate: bool = False
    
    def __post_init__(self):
        if self.freq_bands is None:
            self.freq_bands = {
                'delta': (0.5, 4.0),
                'theta': (4.0, 8.0),
                'alpha': (8.0, 13.0),
                'sigma': (11.0, 16.0),
                'beta': (13.0, 30.0),
                'gamma': (30.0, 45.0)
            }

@dataclass
class SignalQuality:
    """信号质量评估"""
    channel: str
    snr: float
    artifact_percent: float
    flat_percent: float
    quality_score: float
    is_usable: bool

# ==================== 日志设置 ====================
def setup_logging(log_dir: str = "logs"):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"sleep_analysis_{datetime.now():%Y%m%d_%H%M%S}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==================== 信号质量评估 ====================
class SignalQualityAssessor:
    """信号质量评估器"""
    
    def __init__(self, sfreq: float):
        self.sfreq = sfreq
        self.logger = logging.getLogger(__name__)
    
    def assess(self, data: np.ndarray, channel: str) -> SignalQuality:
        """评估单个通道信号质量"""
        try:
            # 1. 计算信噪比
            snr = self._calculate_snr(data)
            
            # 2. 检测伪迹
            artifact_percent = self._detect_artifacts(data)
            
            # 3. 检测平坦线
            flat_percent = self._detect_flat_lines(data)
            
            # 4. 计算综合评分
            quality_score = self._calculate_quality_score(
                snr, artifact_percent, flat_percent
            )
            
            return SignalQuality(
                channel=channel,
                snr=snr,
                artifact_percent=artifact_percent,
                flat_percent=flat_percent,
                quality_score=quality_score,
                is_usable=quality_score > 0.7
            )
            
        except Exception as e:
            self.logger.error(f"信号质量评估失败 {channel}: {e}")
            return SignalQuality(channel, 0, 100, 100, 0, False)
    
    def _calculate_snr(self, data: np.ndarray) -> float:
        """计算信噪比"""
        try:
            # 信号功率
            signal_power = np.mean(data ** 2)
            
            # 噪声功率（使用高频成分）
            freqs, psd = welch(data, self.sfreq, nperseg=min(256, len(data)))
            noise_mask = freqs > 30
            if np.any(noise_mask):
                noise_power = np.mean(psd[noise_mask])
            else:
                noise_power = np.percentile(psd, 10)
            
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            return max(0, snr)
            
        except Exception:
            return 0
    
    def _detect_artifacts(self, data: np.ndarray) -> float:
        """检测伪迹百分比"""
        try:
            # 基于振幅的伪迹检测
            mean_amp = np.mean(np.abs(data))
            std_amp = np.std(data)
            
            # 超过均值±5倍标准差的点视为伪迹
            artifact_mask = np.abs(data - mean_amp) > 5 * std_amp
            
            return np.sum(artifact_mask) / len(data) * 100
            
        except Exception:
            return 100
    
    def _detect_flat_lines(self, data: np.ndarray) -> float:
        """检测平坦线百分比"""
        try:
            # 连续相同值检测
            diff_data = np.diff(data)
            flat_mask = np.abs(diff_data) < 1e-6
            
            # 使用卷积识别连续平坦段
            window = int(self.sfreq)  # 1秒窗口
            conv = np.convolve(flat_mask.astype(int), np.ones(window), mode='same')
            flat_regions = conv > window * 0.9
            
            return np.sum(flat_regions) / len(data) * 100
            
        except Exception:
            return 100
    
    def _calculate_quality_score(self, snr: float, artifact: float, flat: float) -> float:
        """计算综合质量评分"""
        # 归一化各指标
        snr_score = min(snr / 20, 1.0)  # SNR 20dB为满分
        artifact_score = max(1 - artifact / 50, 0)  # 伪迹<50%为及格
        flat_score = max(1 - flat / 20, 0)  # 平坦线<20%为及格
        
        # 加权平均
        quality = 0.4 * snr_score + 0.3 * artifact_score + 0.3 * flat_score
        return min(max(quality, 0), 1)

# ==================== 特征提取器 ====================
class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, sfreq: float, config: AnalysisConfig):
        self.sfreq = sfreq
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_eeg_features(self, data: np.ndarray) -> Dict:
        """提取EEG特征"""
        features = {}
        
        try:
            # 时域特征
            features['mean'] = float(np.mean(data))
            features['std'] = float(np.std(data))
            features['variance'] = float(np.var(data))
            features['peak_to_peak'] = float(np.ptp(data))
            features['rms'] = float(np.sqrt(np.mean(data ** 2)))
            features['zero_crossing_rate'] = self._zero_crossing_rate(data)
            
            # 频域特征
            freqs, psd = welch(data, self.sfreq, nperseg=min(256, len(data)))
            total_power = np.trapz(psd, freqs)
            
            for band_name, (low, high) in self.config.freq_bands.items():
                mask = (freqs >= low) & (freqs <= high)
                if np.any(mask):
                    band_power = np.trapz(psd[mask], freqs[mask])
                    features[f'{band_name}_power'] = float(band_power)
                    features[f'{band_name}_ratio'] = float(
                        band_power / total_power if total_power > 0 else 0
                    )
                    
                    # 峰值频率
                    peak_idx = np.argmax(psd[mask])
                    features[f'{band_name}_peak_freq'] = float(freqs[mask][peak_idx])
                else:
                    features[f'{band_name}_power'] = 0.0
                    features[f'{band_name}_ratio'] = 0.0
                    features[f'{band_name}_peak_freq'] = 0.0
            
            # 谱熵
            features['spectral_entropy'] = self._spectral_entropy(psd)
            
            # Hjorth参数
            hjorth = self._hjorth_parameters(data)
            features.update(hjorth)
            
        except Exception as e:
            self.logger.error(f"EEG特征提取失败: {e}")
            
        return features
    
    def extract_eog_features(self, data: np.ndarray) -> Dict:
        """提取EOG特征（眼动）"""
        features = {}
        
        try:
            # 快速眼动检测
            features['rem_density'] = self._detect_rem(data)
            
            # 眨眼检测
            features['blink_rate'] = self._detect_blinks(data)
            
        except Exception:
            pass
            
        return features
    
    def extract_emg_features(self, data: np.ndarray) -> Dict:
        """提取EMG特征（肌电）"""
        features = {}
        
        try:
            # 肌电活动强度
            features['emg_activity'] = float(np.mean(np.abs(data)))
            
            # 肌电变异
            features['emg_variability'] = float(np.std(data))
            
        except Exception:
            pass
            
        return features
    
    def _zero_crossing_rate(self, data: np.ndarray) -> float:
        """过零率"""
        return float(np.sum(np.diff(np.signbit(data))) / len(data))
    
    def _spectral_entropy(self, psd: np.ndarray) -> float:
        """谱熵"""
        psd_norm = psd / (np.sum(psd) + 1e-10)
        return float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
    
    def _hjorth_parameters(self, data: np.ndarray) -> Dict:
        """Hjorth参数"""
        diff1 = np.diff(data)
        diff2 = np.diff(diff1)
        
        var0 = np.var(data)
        var1 = np.var(diff1)
        var2 = np.var(diff2)
        
        mobility = np.sqrt(var1 / (var0 + 1e-10))
        complexity = np.sqrt(var2 / (var1 + 1e-10)) / (mobility + 1e-10)
        
        return {
            'hjorth_mobility': float(mobility),
            'hjorth_complexity': float(complexity)
        }
    
    def _detect_rem(self, data: np.ndarray) -> float:
        """检测REM（简化版）"""
        # 实际应用中需要更复杂的算法
        return float(np.std(data) / (np.mean(np.abs(data)) + 1e-10))
    
    def _detect_blinks(self, data: np.ndarray) -> float:
        """检测眨眼（简化版）"""
        # 实际应用中需要更复杂的算法
        return float(np.sum(np.abs(data) > 3 * np.std(data)) / len(data))

# ==================== 睡眠分期器 ====================
class SleepStager:
    """睡眠分期器（基于规则+机器学习）"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.stage_labels = {
            0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R', 
            5: 'Movement', 6: 'Unscored'
        }
        
    def stage_sleep(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """执行睡眠分期"""
        try:
            # 尝试加载预训练模型
            self._load_model()
            
            if self.model is not None:
                # 使用模型分期
                stages = self._stage_with_model(features_df)
            else:
                # 使用规则分期
                stages = self._stage_with_rules(features_df)
            
            # 添加阶段标签
            stages_df = pd.DataFrame({
                'epoch': range(len(stages)),
                'stage': stages,
                'stage_label': [self.stage_labels.get(s, 'Unknown') for s in stages]
            })
            
            return stages_df
            
        except Exception as e:
            self.logger.error(f"睡眠分期失败: {e}")
            return pd.DataFrame()
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            # 尝试导入scikit-learn
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            
            # 检查是否有预训练模型
            model_path = Path(__file__).parent / "models" / "sleep_staging_model.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                self.logger.info("已加载预训练睡眠分期模型")
                
        except ImportError:
            self.logger.warning("未找到scikit-learn，使用规则分期")
        except Exception as e:
            self.logger.warning(f"模型加载失败，使用规则分期: {e}")
    
    def _stage_with_model(self, features_df: pd.DataFrame) -> List[int]:
        """使用模型分期"""
        # 准备特征矩阵
        feature_cols = [col for col in features_df.columns 
                       if col not in ['epoch', 'start_time', 'end_time']]
        X = features_df[feature_cols].fillna(0).values
        
        # 预测
        stages = self.model.predict(X)
        return stages.tolist()
    
    def _stage_with_rules(self, features_df: pd.DataFrame) -> List[int]:
        """使用规则分期（改进版）"""
        stages = []
        
        for _, row in features_df.iterrows():
            # 获取关键特征
            delta_ratio = row.get('delta_ratio', 0)
            theta_ratio = row.get('theta_ratio', 0)
            alpha_ratio = row.get('alpha_ratio', 0)
            sigma_ratio = row.get('sigma_ratio', 0)
            eeg_std = row.get('std', 0)
            
            # 多规则决策
            stage = self._apply_staging_rules(
                delta_ratio, theta_ratio, alpha_ratio, 
                sigma_ratio, eeg_std
            )
            
            stages.append(stage)
        
        return stages
    
    def _apply_staging_rules(self, delta: float, theta: float, 
                            alpha: float, sigma: float, eeg_std: float) -> int:
        """应用分期规则"""
        
        # 规则1: 清醒 (高alpha，高变异性)
        if alpha > 0.3 and eeg_std > 20:
            return 0
        
        # 规则2: 深睡N3 (高delta)
        if delta > 0.5:
            return 3
        
        # 规则3: N2 (有纺锤波)
        if sigma > 0.15:
            return 2
        
        # 规则4: REM (低振幅，混合频率)
        if eeg_std < 15 and 0.1 < theta < 0.4 and alpha < 0.2:
            return 4
        
        # 规则5: N1 (过渡期)
        if 0.1 < alpha < 0.3 and eeg_std < 25:
            return 1
        
        # 默认: N1
        return 1

# ==================== 呼吸事件检测器 ====================
class RespiratoryEventDetector:
    """呼吸事件检测器（基于AASM标准）"""
    
    def __init__(self, sfreq: float, config: AnalysisConfig):
        self.sfreq = sfreq
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def detect_events(self, resp_data: np.ndarray, 
                     spo2_data: Optional[np.ndarray] = None) -> pd.DataFrame:
        """检测呼吸事件"""
        events = []
        
        try:
            # 预处理
            resp_filtered = self._preprocess_respiratory(resp_data)
            
            # 计算包络
            resp_envelope = np.abs(hilbert(resp_filtered))
            baseline = np.median(resp_envelope)
            
            # 检测呼吸暂停
            apnea_mask = resp_envelope < self.config.apnea_threshold * baseline
            apnea_events = self._find_events(
                apnea_mask, 
                self.config.min_event_duration,
                'apnea'
            )
            events.extend(apnea_events)
            
            # 检测低通气
            hypopnea_mask = resp_envelope < self.config.hypopnea_threshold * baseline
            hypopnea_events = self._find_events(
                hypopnea_mask,
                self.config.min_event_duration,
                'hypopnea'
            )
            events.extend(hypopnea_events)
            
            # 如果有血氧数据，验证事件
            if spo2_data is not None:
                events = self._validate_with_spo2(events, spo2_data)
            
            # 转换为DataFrame
            events_df = pd.DataFrame(events) if events else pd.DataFrame()
            
            return events_df
            
        except Exception as e:
            self.logger.error(f"呼吸事件检测失败: {e}")
            return pd.DataFrame()
    
    def _preprocess_respiratory(self, data: np.ndarray) -> np.ndarray:
        """预处理呼吸信号"""
        # 带通滤波 (0.1-5 Hz)
        nyquist = self.sfreq / 2
        b, a = butter(4, [0.1/nyquist, 5/nyquist], btype='band')
        filtered = filtfilt(b, a, data)
        
        # 去除基线漂移
        filtered = signal.detrend(filtered)
        
        return filtered
    
    def _find_events(self, mask: np.ndarray, min_duration: float, 
                    event_type: str) -> List[Dict]:
        """查找事件"""
        events = []
        min_samples = int(min_duration * self.sfreq)
        
        # 寻找连续区域
        diff_mask = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        starts = np.where(diff_mask == 1)[0]
        ends = np.where(diff_mask == -1)[0]
        
        for start, end in zip(starts, ends):
            duration = (end - start) / self.sfreq
            if duration >= min_duration:
                events.append({
                    'type': event_type,
                    'start_sample': int(start),
                    'end_sample': int(end),
                    'start_time': start / self.sfreq,
                    'end_time': end / self.sfreq,
                    'duration': float(duration)
                })
        
        return events
    
    def _validate_with_spo2(self, events: List[Dict], 
                           spo2_data: np.ndarray) -> List[Dict]:
        """用血氧验证事件"""
        validated_events = []
        
        for event in events:
            start = int(event['start_sample'])
            end = int(event['end_sample'])
            
            # 检查事件期间的血氧下降
            if end < len(spo2_data):
                baseline_spo2 = np.median(spo2_data[max(0, start-100):start])
                min_spo2 = np.min(spo2_data[start:end])
                spo2_drop = baseline_spo2 - min_spo2
                
                event['spo2_baseline'] = float(baseline_spo2)
                event['spo2_nadir'] = float(min_spo2)
                event['spo2_drop'] = float(spo2_drop)
                event['validated'] = spo2_drop >= self.config.spo2_drop_threshold
                
                validated_events.append(event)
        
        return validated_events

# ==================== 纺锤波检测器 ====================
class SpindleDetector:
    """纺锤波检测器"""
    
    def __init__(self, sfreq: float, config: AnalysisConfig):
        self.sfreq = sfreq
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def detect_spindles(self, eeg_data: np.ndarray) -> pd.DataFrame:
        """检测纺锤波"""
        spindles = []
        
        try:
            # 带通滤波
            low, high = self.config.spindle_freq_range
            sos = butter(4, [low, high], btype='band', fs=self.sfreq, output='sos')
            filtered = signal.sosfiltfilt(sos, eeg_data)
            
            # 计算包络
            envelope = np.abs(hilbert(filtered))
            
            # 平滑包络
            window = int(self.sfreq * 0.1)  # 100ms
            envelope_smooth = np.convolve(
                envelope, 
                np.ones(window)/window, 
                mode='same'
            )
            
            # 计算阈值
            rms = np.sqrt(np.mean(envelope_smooth ** 2))
            threshold = rms * self.config.spindle_amplitude_threshold
            
            # 检测候选纺锤波
            mask = envelope_smooth > threshold
            
            # 合并相邻区域
            diff_mask = np.diff(np.concatenate(([0], mask.astype(int), [0])))
            starts = np.where(diff_mask == 1)[0]
            ends = np.where(diff_mask == -1)[0]
            
            for start, end in zip(starts, ends):
                duration = (end - start) / self.sfreq
                min_dur, max_dur = self.config.spindle_duration_range
                
                if min_dur <= duration <= max_dur:
                    # 提取纺锤波片段
                    segment = filtered[start:end]
                    segment_env = envelope_smooth[start:end]
                    
                    # 计算特征
                    freq = self._estimate_frequency(segment)
                    amplitude = float(np.max(segment_env))
                    symmetry = self._calculate_symmetry(segment_env)
                    
                    spindles.append({
                        'start_idx': int(start),
                        'end_idx': int(end),
                        'start_time': start / self.sfreq,
                        'end_time': end / self.sfreq,
                        'duration': float(duration),
                        'frequency': float(freq),
                        'amplitude': amplitude,
                        'symmetry': symmetry,
                        'type': 'fast' if freq >= 13 else 'slow'
                    })
            
            self.logger.info(f"检测到 {len(spindles)} 个纺锤波")
            
        except Exception as e:
            self.logger.error(f"纺锤波检测失败: {e}")
        
        return pd.DataFrame(spindles)
    
    def _estimate_frequency(self, segment: np.ndarray) -> float:
        """估计频率"""
        if len(segment) < 10:
            return 13.0
        
        freqs, psd = welch(segment, self.sfreq, nperseg=min(256, len(segment)))
        low, high = self.config.spindle_freq_range
        mask = (freqs >= low) & (freqs <= high)
        
        if np.any(mask):
            peak_idx = np.argmax(psd[mask])
            return float(freqs[mask][peak_idx])
        
        return 13.0
    
    def _calculate_symmetry(self, envelope: np.ndarray) -> float:
        """计算对称性"""
        if len(envelope) < 3:
            return 0.5
        
        peak_idx = np.argmax(envelope)
        
        if peak_idx == 0 or peak_idx == len(envelope) - 1:
            return 0.5
        
        rise_area = np.trapz(envelope[:peak_idx + 1])
        fall_area = np.trapz(envelope[peak_idx:])
        total = rise_area + fall_area
        
        return float(rise_area / total) if total > 0 else 0.5

# ==================== 睡眠分析器主类 ====================
class SleepAnalyzer:
    """睡眠分析器主类"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = setup_logging()
        self.raw = None
        self.sfreq = None
        self.channels = {}
        self.quality_report = {}
        self.results = {}
        
    def load_edf(self, edf_path: str) -> bool:
        """加载EDF文件"""
        try:
            self.logger.info(f"加载EDF文件: {edf_path}")
            
            if not os.path.exists(edf_path):
                self.logger.error(f"文件不存在: {edf_path}")
                return False
            
            self.raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            
            self.logger.info(f"采样率: {self.sfreq} Hz")
            self.logger.info(f"通道数: {len(self.raw.ch_names)}")
            self.logger.info(f"时长: {self.raw.n_times / self.sfreq / 3600:.2f} 小时")
            
            self._identify_channels()
            self._assess_signal_quality()
            
            return True
            
        except Exception as e:
            self.logger.error(f"EDF加载失败: {e}")
            return False
    
    def _identify_channels(self):
        """识别通道类型"""
        self.channels = {
            'eeg': [], 'eog': [], 'emg': [], 
            'ecg': [], 'resp': [], 'spo2': [], 'other': []
        }
        
        for ch in self.raw.ch_names:
            ch_lower = ch.lower()
            ch_type = self.raw.get_channel_types(picks=ch)[0]
            
            if ch_type == 'eeg':
                self.channels['eeg'].append(ch)
            elif ch_type == 'eog':
                self.channels['eog'].append(ch)
            elif ch_type == 'emg':
                self.channels['emg'].append(ch)
            elif 'ecg' in ch_lower:
                self.channels['ecg'].append(ch)
            elif any(k in ch_lower for k in ['resp', 'flow', 'air']):
                self.channels['resp'].append(ch)
            elif any(k in ch_lower for k in ['spo2', 'sao2', 'ox']):
                self.channels['spo2'].append(ch)
            else:
                self.channels['other'].append(ch)
        
        self.logger.info(f"通道识别结果: { {k: len(v) for k, v in self.channels.items()} }")
    
    def _assess_signal_quality(self):
        """评估信号质量"""
        assessor = SignalQualityAssessor(self.sfreq)
        
        for ch_type, channels in self.channels.items():
            for ch in channels:
                data = self.raw.get_data(picks=ch)[0]
                quality = assessor.assess(data, ch)
                self.quality_report[ch] = asdict(quality)
        
        usable_channels = sum(1 for q in self.quality_report.values() if q['is_usable'])
        self.logger.info(f"可用通道: {usable_channels}/{len(self.quality_report)}")
    
    def analyze(self) -> Dict:
        """执行完整分析"""
        self.logger.info("开始睡眠分析...")
        
        # 1. 睡眠分期分析
        if self.channels['eeg']:
            self.logger.info("执行睡眠分期分析...")
            self.results['sleep_staging'] = self._analyze_sleep_staging()
        
        # 2. 呼吸事件分析
        if self.channels['resp']:
            self.logger.info("执行呼吸事件分析...")
            self.results['respiratory'] = self._analyze_respiratory()
        
        # 3. 纺锤波分析
        if self.channels['eeg']:
            self.logger.info("执行纺锤波分析...")
            self.results['spindles'] = self._analyze_spindles()
        
        # 4. 生成综合报告
        self.results['summary'] = self._generate_summary()
        
        # 5. 保存结果
        self._save_results()
        
        return self.results
    
    def _analyze_sleep_staging(self) -> Dict:
        """睡眠分期分析"""
        if not self.channels['eeg']:
            return {}
        
        # 选择最佳EEG通道
        eeg_channel = self._select_best_channel('eeg')
        
        # 提取数据
        eeg_data, times = self.raw[eeg_channel, :]
        eeg_data = eeg_data.flatten()
        
        # 创建epochs
        samples_per_epoch = int(self.config.epoch_duration * self.sfreq)
        n_epochs = len(eeg_data) // samples_per_epoch
        
        # 提取特征
        feature_extractor = FeatureExtractor(self.sfreq, self.config)
        features_list = []
        
        for epoch_idx in range(n_epochs):
            start = epoch_idx * samples_per_epoch
            end = start + samples_per_epoch
            epoch_data = eeg_data[start:end]
            
            features = feature_extractor.extract_eeg_features(epoch_data)
            features['epoch'] = epoch_idx
            features['start_time'] = times[start]
            features['end_time'] = times[min(end, len(times)-1)]
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # 睡眠分期
        stager = SleepStager(self.config)
        stages_df = stager.stage_sleep(features_df)
        
        # 计算睡眠指标
        metrics = self._calculate_sleep_metrics(stages_df)
        
        return {
            'channel': eeg_channel,
            'features': features_df.to_dict('records') if self.config.save_intermediate else [],
            'stages': stages_df.to_dict('records'),
            'metrics': metrics
        }
    
    def _analyze_respiratory(self) -> Dict:
        """呼吸事件分析"""
        if not self.channels['resp']:
            return {}
        
        # 选择呼吸通道
        resp_channel = self._select_best_channel('resp')
        resp_data = self.raw.get_data(picks=resp_channel)[0]
        
        # 获取血氧数据（如果有）
        spo2_data = None
        if self.channels['spo2']:
            spo2_channel = self._select_best_channel('spo2')
            spo2_data = self.raw.get_data(picks=spo2_channel)[0]
        
        # 检测事件
        detector = RespiratoryEventDetector(self.sfreq, self.config)
        events_df = detector.detect_events(resp_data, spo2_data)
        
        # 计算指标
        if not events_df.empty:
            duration_hours = len(resp_data) / self.sfreq / 3600
            apnea_count = len(events_df[events_df['type'] == 'apnea'])
            hypopnea_count = len(events_df[events_df['type'] == 'hypopnea'])
            total_events = len(events_df)
            
            ahi = total_events / duration_hours
            
            # 血氧指标
            spo2_metrics = {}
            if spo2_data is not None and 'validated' in events_df.columns:
                validated = events_df[events_df['validated'] == True]
                odi = len(validated) / duration_hours
                spo2_metrics = {
                    'odi': odi,
                    'min_spo2': float(np.min(spo2_data)),
                    'mean_spo2': float(np.mean(spo2_data))
                }
            
            return {
                'channel': resp_channel,
                'total_events': total_events,
                'apnea_count': int(apnea_count),
                'hypopnea_count': int(hypopnea_count),
                'ahi': float(ahi),
                'events': events_df.to_dict('records') if self.config.save_intermediate else [],
                'spo2_metrics': spo2_metrics,
                'severity': self._classify_ahi(ahi)
            }
        
        return {}
    
    def _analyze_spindles(self) -> Dict:
        """纺锤波分析"""
        if not self.channels['eeg']:
            return {}
        
        # 选择中央区EEG
        eeg_channels = self.channels['eeg']
        central_channels = [ch for ch in eeg_channels if 'C' in ch]
        spindle_channel = central_channels[0] if central_channels else eeg_channels[0]
        
        # 提取数据
        eeg_data = self.raw.get_data(picks=spindle_channel)[0]
        
        # 检测纺锤波
        detector = SpindleDetector(self.sfreq, self.config)
        spindles_df = detector.detect_spindles(eeg_data)
        
        if not spindles_df.empty:
            duration_hours = len(eeg_data) / self.sfreq / 3600
            
            # 统计
            stats = {
                'count': len(spindles_df),
                'density': len(spindles_df) / (duration_hours * 60),  # 个/分钟
                'mean_duration': float(spindles_df['duration'].mean()),
                'mean_frequency': float(spindles_df['frequency'].mean()),
                'mean_amplitude': float(spindles_df['amplitude'].mean()),
                'slow_count': int(sum(spindles_df['type'] == 'slow')),
                'fast_count': int(sum(spindles_df['type'] == 'fast'))
            }
            
            # 类型比例
            if stats['count'] > 0:
                stats['slow_percent'] = stats['slow_count'] / stats['count'] * 100
                stats['fast_percent'] = stats['fast_count'] / stats['count'] * 100
            
            return {
                'channel': spindle_channel,
                'statistics': stats,
                'spindles': spindles_df.to_dict('records') if self.config.save_intermediate else []
            }
        
        return {}
    
    def _select_best_channel(self, channel_type: str) -> str:
        """选择最佳通道"""
        channels = self.channels.get(channel_type, [])
        if not channels:
            return None
        
        # 根据信号质量选择
        best_channel = None
        best_quality = -1
        
        for ch in channels:
            quality = self.quality_report.get(ch, {})
            score = quality.get('quality_score', 0)
            if score > best_quality:
                best_quality = score
                best_channel = ch
        
        return best_channel or channels[0]
    
    def _calculate_sleep_metrics(self, stages_df: pd.DataFrame) -> Dict:
        """计算睡眠指标"""
        if stages_df.empty:
            return {}
        
        total_epochs = len(stages_df)
        total_time_min = total_epochs * self.config.epoch_duration / 60
        
        # 各阶段统计
        stage_counts = stages_df['stage'].value_counts()
        
        # 睡眠时间（排除清醒）
        sleep_epochs = sum(stages_df['stage'].isin([1, 2, 3, 4]))
        sleep_time_min = sleep_epochs * self.config.epoch_duration / 60
        
        # 睡眠效率
        sleep_efficiency = (sleep_time_min / total_time_min * 100) if total_time_min > 0 else 0
        
        # 睡眠潜伏期（第一个睡眠epoch）
        sleep_indices = stages_df[stages_df['stage'].isin([1, 2, 3, 4])].index
        sleep_latency_min = (sleep_indices[0] * self.config.epoch_duration / 60) if len(sleep_indices) > 0 else total_time_min
        
        # REM潜伏期
        rem_indices = stages_df[stages_df['stage'] == 4].index
        rem_latency_min = None
        if len(rem_indices) > 0 and len(sleep_indices) > 0:
            rem_latency_min = (rem_indices[0] - sleep_indices[0]) * self.config.epoch_duration / 60
        
        # 各阶段百分比
        stage_percentages = {}
        for stage in range(5):  # 0-4
            count = stage_counts.get(stage, 0)
            stage_percentages[self.stager.stage_labels.get(stage, f'Stage{stage}')] = \
                (count / total_epochs * 100) if total_epochs > 0 else 0
        
        return {
            'total_time_hours': total_time_min / 60,
            'sleep_time_hours': sleep_time_min / 60,
            'sleep_efficiency': sleep_efficiency,
            'sleep_latency_min': sleep_latency_min,
            'rem_latency_min': rem_latency_min,
            'stage_percentages': stage_percentages
        }
    
    def _classify_ahi(self, ahi: float) -> str:
        """AHI分级"""
        if ahi < 5:
            return '正常'
        elif ahi < 15:
            return '轻度'
        elif ahi < 30:
            return '中度'
        else:
            return '重度'
    
    def _generate_summary(self) -> Dict:
        """生成综合摘要"""
        summary = {
            'analysis_time': datetime.now().isoformat(),
            'file_info': {
                'sampling_rate': self.sfreq,
                'channels': {k: len(v) for k, v in self.channels.items()},
                'signal_quality': {
                    'usable_channels': sum(1 for q in self.quality_report.values() if q['is_usable']),
                    'average_quality': np.mean([q['quality_score'] for q in self.quality_report.values()]) if self.quality_report else 0
                }
            }
        }
        
        # 整合关键指标
        if 'sleep_staging' in self.results:
            staging = self.results['sleep_staging']
            summary['sleep_metrics'] = staging.get('metrics', {})
        
        if 'respiratory' in self.results:
            resp = self.results['respiratory']
            summary['respiratory_metrics'] = {
                'ahi': resp.get('ahi', 0),
                'severity': resp.get('severity', '未知'),
                'apnea_count': resp.get('apnea_count', 0),
                'hypopnea_count': resp.get('hypopnea_count', 0)
            }
            if 'spo2_metrics' in resp:
                summary['respiratory_metrics'].update(resp['spo2_metrics'])
        
        if 'spindles' in self.results:
            spindles = self.results['spindles']
            summary['spindle_metrics'] = spindles.get('statistics', {})
        
        return summary
    
    def _save_results(self):
        """保存结果"""
        try:
            # 创建输出目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(self.config.output_dir) / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存JSON报告
            report_path = output_dir / 'sleep_analysis_report.json'
            
            # 转换numpy类型
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            report = convert_numpy(self.results)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 生成可视化
            self._generate_visualizations(output_dir)
            
            # 生成文本报告
            self._generate_text_report(output_dir)
            
            self.logger.info(f"结果已保存到: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
    
    def _generate_visualizations(self, output_dir: Path):
        """生成可视化图表"""
        try:
            # 睡眠结构图
            if 'sleep_staging' in self.results:
                self._plot_hypnogram(output_dir)
            
            # 呼吸事件图
            if 'respiratory' in self.results and self.results['respiratory'].get('events'):
                self._plot_respiratory_events(output_dir)
            
            # 纺锤波分布图
            if 'spindles' in self.results and self.results['spindles'].get('spindles'):
                self._plot_spindle_distribution(output_dir)
            
            # 综合仪表板
            self._plot_dashboard(output_dir)
            
        except Exception as e:
            self.logger.error(f"可视化生成失败: {e}")
    
    def _plot_hypnogram(self, output_dir: Path):
        """绘制睡眠结构图"""
        try:
            stages_data = self.results['sleep_staging']['stages']
            if not stages_data:
                return
            
            df = pd.DataFrame(stages_data)
            
            fig, ax = plt.subplots(figsize=(15, 5))
            
            # 时间轴（小时）
            times = [d['epoch'] * self.config.epoch_duration / 3600 for d in stages_data]
            
            # 颜色映射
            colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red', 4: 'purple'}
            
            # 绘制
            for i in range(len(times)-1):
                stage = df.iloc[i]['stage']
                ax.fill_between([times[i], times[i+1]], [stage, stage], [stage+0.8, stage+0.8],
                               color=colors.get(stage, 'gray'), alpha=0.7)
            
            ax.set_xlabel('时间 (小时)')
            ax.set_ylabel('睡眠阶段')
            ax.set_yticks([0, 1, 2, 3, 4])
            ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'])
            ax.set_title('睡眠结构图 (Hypnogram)')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'hypnogram.png', dpi=self.config.figure_dpi)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"睡眠结构图绘制失败: {e}")
    
    def _plot_respiratory_events(self, output_dir: Path):
        """绘制呼吸事件图"""
        try:
            events = self.results['respiratory']['events']
            if not events:
                return
            
            df = pd.DataFrame(events)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
            
            # 事件时间分布
            times = [e['start_time'] / 3600 for e in events]  # 小时
            durations = [e['duration'] for e in events]
            types = [e['type'] for e in events]
            
            colors = {'apnea': 'red', 'hypopnea': 'orange'}
            
            for i, (t, d, typ) in enumerate(zip(times, durations, types)):
                ax1.bar(t, d, width=0.02, color=colors.get(typ, 'gray'), 
                       alpha=0.7, label=typ if i == 0 else "")
            
            ax1.set_xlabel('时间 (小时)')
            ax1.set_ylabel('事件时长 (秒)')
            ax1.set_title('呼吸事件分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 事件类型统计
            type_counts = df['type'].value_counts()
            ax2.bar(type_counts.index, type_counts.values, 
                   color=[colors.get(t, 'gray') for t in type_counts.index])
            ax2.set_xlabel('事件类型')
            ax2.set_ylabel('数量')
            ax2.set_title('呼吸事件统计')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'respiratory_events.png', dpi=self.config.figure_dpi)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"呼吸事件图绘制失败: {e}")
    
    def _plot_spindle_distribution(self, output_dir: Path):
        """绘制纺锤波分布图"""
        try:
            spindles_data = self.results['spindles']['spindles']
            if not spindles_data:
                return
            
            df = pd.DataFrame(spindles_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 频率分布
            ax1 = axes[0, 0]
            ax1.hist(df['frequency'], bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax1.axvline(df['frequency'].mean(), color='red', linestyle='--', 
                       label=f"均值: {df['frequency'].mean():.2f} Hz")
            ax1.set_xlabel('频率 (Hz)')
            ax1.set_ylabel('数量')
            ax1.set_title('纺锤波频率分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 时长分布
            ax2 = axes[0, 1]
            ax2.hist(df['duration'], bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax2.axvline(df['duration'].mean(), color='red', linestyle='--',
                       label=f"均值: {df['duration'].mean():.2f} 秒")
            ax2.set_xlabel('时长 (秒)')
            ax2.set_ylabel('数量')
            ax2.set_title('纺锤波时长分布')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 振幅分布
            ax3 = axes[1, 0]
            ax3.hist(df['amplitude'], bins=30, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(df['amplitude'].mean(), color='red', linestyle='--',
                       label=f"均值: {df['amplitude'].mean():.3f}")
            ax3.set_xlabel('振幅')
            ax3.set_ylabel('数量')
            ax3.set_title('纺锤波振幅分布')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 类型比例
            ax4 = axes[1, 1]
            type_counts = df['type'].value_counts()
            ax4.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                   colors=['lightblue', 'lightcoral'])
            ax4.set_title('纺锤波类型分布')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'spindle_distribution.png', dpi=self.config.figure_dpi)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"纺锤波分布图绘制失败: {e}")
    
    def _plot_dashboard(self, output_dir: Path):
        """绘制综合仪表板"""
        try:
            fig = plt.figure(figsize=(20, 12))
            
            # 创建网格布局
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 标题
            fig.suptitle(f'眠小兔睡眠分析报告 - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                        fontsize=16, fontweight='bold')
            
            # 睡眠指标卡片
            ax_metrics = fig.add_subplot(gs[0, :])
            ax_metrics.axis('off')
            
            metrics_text = ""
            if 'sleep_metrics' in self.results.get('summary', {}):
                sm = self.results['summary']['sleep_metrics']
                metrics_text += f"总睡眠时间: {sm.get('sleep_time_hours', 0):.1f}小时 | "
                metrics_text += f"睡眠效率: {sm.get('sleep_efficiency', 0):.1f}% | "
                metrics_text += f"睡眠潜伏期: {sm.get('sleep_latency_min', 0):.1f}分钟"
            
            if 'respiratory_metrics' in self.results.get('summary', {}):
                rm = self.results['summary']['respiratory_metrics']
                metrics_text += f"\nAHI: {rm.get('ahi', 0):.1f} ({rm.get('severity', '未知')}) | "
                metrics_text += f"ODI: {rm.get('odi', 0):.1f} | "
                metrics_text += f"最低血氧: {rm.get('min_spo2', 0):.1f}%"
            
            if 'spindle_metrics' in self.results.get('summary', {}):
                spm = self.results['summary']['spindle_metrics']
                metrics_text += f"\n纺锤波密度: {spm.get('density', 0):.2f}/分钟 | "
                metrics_text += f"平均频率: {spm.get('mean_frequency', 0):.1f}Hz"
            
            ax_metrics.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            # 睡眠结构图（小图）
            ax_hypno = fig.add_subplot(gs[1, 0])
            ax_hypno.set_title('睡眠结构')
            if 'sleep_staging' in self.results:
                stages_data = self.results['sleep_staging'].get('stages', [])
                if stages_data:
                    times = [d['epoch'] * self.config.epoch_duration / 3600 for d in stages_data[:500]]  # 只显示前500个
                    stages = [d['stage'] for d in stages_data[:500]]
                    ax_hypno.plot(times, stages, 'b-', linewidth=0.5)
                    ax_hypno.set_ylim(-0.5, 4.5)
                    ax_hypno.set_xlabel('小时')
                    ax_hypno.set_ylabel('阶段')
            
            # 呼吸事件分布（小图）
            ax_resp = fig.add_subplot(gs[1, 1])
            ax_resp.set_title('呼吸事件')
            if 'respiratory' in self.results:
                events = self.results['respiratory'].get('events', [])
                if events:
                    df = pd.DataFrame(events)
                    if 'type' in df.columns:
                        type_counts = df['type'].value_counts()
                        ax_resp.bar(type_counts.index, type_counts.values, 
                                   color=['red', 'orange'])
                        ax_resp.set_ylabel('数量')
            
            # 纺锤波分布（小图）
            ax_spindle = fig.add_subplot(gs[1, 2])
            ax_spindle.set_title('纺锤波频率')
            if 'spindles' in self.results:
                spindles = self.results['spindles'].get('spindles', [])
                if spindles:
                    df = pd.DataFrame(spindles)
                    ax_spindle.hist(df['frequency'], bins=20, alpha=0.7, color='purple')
                    ax_spindle.set_xlabel('频率 (Hz)')
                    ax_spindle.set_ylabel('数量')
            
            # 阶段比例饼图
            ax_pie = fig.add_subplot(gs[2, 0])
            if 'sleep_staging' in self.results:
                metrics = self.results['sleep_staging'].get('metrics', {})
                percentages = metrics.get('stage_percentages', {})
                if percentages:
                    labels = list(percentages.keys())
                    sizes = list(percentages.values())
                    ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%',
                             colors=['blue', 'green', 'orange', 'red', 'purple'])
                    ax_pie.set_title('睡眠阶段比例')
            
            # 临床建议
            ax_advice = fig.add_subplot(gs[2, 1:])
            ax_advice.axis('off')
            
            advice = "【临床建议】\n"
            
            # 基于AHI的建议
            if 'respiratory_metrics' in self.results.get('summary', {}):
                severity = self.results['summary']['respiratory_metrics'].get('severity', '')
                if severity == '重度':
                    advice += "• 重度睡眠呼吸暂停，建议立即就医\n"
                elif severity == '中度':
                    advice += "• 中度睡眠呼吸暂停，建议专科评估\n"
                elif severity == '轻度':
                    advice += "• 轻度睡眠呼吸暂停，建议生活方式干预\n"
                else:
                    advice += "• 呼吸事件正常，无需特殊处理\n"
            
            # 基于睡眠效率的建议
            if 'sleep_metrics' in self.results.get('summary', {}):
                efficiency = self.results['summary']['sleep_metrics'].get('sleep_efficiency', 0)
                if efficiency < 85:
                    advice += "• 睡眠效率偏低，建议改善睡眠卫生\n"
                
                latency = self.results['summary']['sleep_metrics'].get('sleep_latency_min', 0)
                if latency > 30:
                    advice += "• 入睡困难，建议放松训练\n"
            
            # 基于纺锤波的建议
            if 'spindle_metrics' in self.results.get('summary', {}):
                density = self.results['summary']['spindle_metrics'].get('density', 0)
                if density < 1.5:
                    advice += "• 纺锤波密度偏低，可能N2睡眠质量一般\n"
            
            ax_advice.text(0.1, 0.5, advice, fontsize=12, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
            
            plt.savefig(output_dir / 'dashboard.png', dpi=self.config.figure_dpi, 
                       bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"仪表板绘制失败: {e}")
    
    def _generate_text_report(self, output_dir: Path):
        """生成文本报告"""
        try:
            report_path = output_dir / 'analysis_report.txt'
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("眠小兔专业睡眠分析报告\n")
                f.write("=" * 70 + "\n\n")
                
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"记录时长: {self.results['summary']['file_info']['sampling_rate']:.1f} Hz\n\n")
                
                f.write("【关键指标】\n")
                f.write("-" * 40 + "\n")
                
                if 'sleep_metrics' in self.results['summary']:
                    sm = self.results['summary']['sleep_metrics']
                    f.write(f"总睡眠时间: {sm.get('sleep_time_hours', 0):.1f} 小时\n")
                    f.write(f"睡眠效率: {sm.get('sleep_efficiency', 0):.1f}%\n")
                    f.write(f"睡眠潜伏期: {sm.get('sleep_latency_min', 0):.1f} 分钟\n")
                    if sm.get('rem_latency_min'):
                        f.write(f"REM潜伏期: {sm['rem_latency_min']:.1f} 分钟\n")
                
                if 'respiratory_metrics' in self.results['summary']:
                    rm = self.results['summary']['respiratory_metrics']
                    f.write(f"\nAHI指数: {rm.get('ahi', 0):.1f} ({rm.get('severity', '未知')})\n")
                    f.write(f"呼吸暂停: {rm.get('apnea_count', 0)} 次\n")
                    f.write(f"低通气: {rm.get('hypopnea_count', 0)} 次\n")
                    if 'odi' in rm:
                        f.write(f"ODI指数: {rm['odi']:.1f}\n")
                        f.write(f"最低血氧: {rm.get('min_spo2', 0):.1f}%\n")
                
                if 'spindle_metrics' in self.results['summary']:
                    spm = self.results['summary']['spindle_metrics']
                    f.write(f"\n纺锤波数量: {spm.get('count', 0)} 个\n")
                    f.write(f"纺锤波密度: {spm.get('density', 0):.2f} 个/分钟\n")
                    f.write(f"平均频率: {spm.get('mean_frequency', 0):.1f} Hz\n")
                    f.write(f"慢纺锤波: {spm.get('slow_percent', 0):.1f}%\n")
                    f.write(f"快纺锤波: {spm.get('fast_percent', 0):.1f}%\n")
                
                f.write("\n【临床参考范围】\n")
                f.write("-" * 40 + "\n")
                f.write("AHI: <5 正常, 5-15 轻度, 15-30 中度, >30 重度\n")
                f.write("睡眠效率: >85% 正常\n")
                f.write("睡眠潜伏期: <30 分钟 正常\n")
                f.write("纺锤波密度: 1.5-3.5 个/分钟 正常\n")
                
                f.write("\n【注意事项】\n")
                f.write("-" * 40 + "\n")
                f.write("1. 本报告基于自动分析，仅供参考\n")
                f.write("2. 最终诊断需由专业医生结合临床判断\n")
                f.write("3. 如有睡眠问题，建议咨询睡眠专科医生\n")
                
                f.write("\n" + "=" * 70 + "\n")
            
            self.logger.info(f"文本报告已生成: {report_path}")
            
        except Exception as e:
            self.logger.error(f"文本报告生成失败: {e}")

# ==================== 主函数 ====================
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='眠小兔专业睡眠分析系统')
    parser.add_argument('edf_file', nargs='?', help='EDF文件路径')
    parser.add_argument('-o', '--output', default='sleep_analysis_results', help='输出目录')
    parser.add_argument('-c', '--config', help='配置文件路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    # 设置编码
    if sys.stdout.encoding != 'UTF-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # 确定EDF文件路径
    if args.edf_file:
        edf_path = args.edf_file
    else:
        # 默认测试文件
        edf_path = r"D:\openclaw\AISleepGen\data\edf\SC4001E0-PSG.edf"
        print(f"使用默认测试文件: {edf_path}")
    
    # 加载配置
    config = AnalysisConfig(output_dir=args.output)
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            print(f"已加载配置文件: {args.config}")
        except Exception as e:
            print(f"配置文件加载失败: {e}")
    
    print("\n" + "=" * 70)
    print("眠小兔专业睡眠分析系统 v2.0")
    print("=" * 70 + "\n")
    
    # 创建分析器
    analyzer = SleepAnalyzer(config)
    
    # 加载EDF
    if not analyzer.load_edf(edf_path):
        print("❌ EDF文件加载失败")
        sys.exit(1)
    
    # 执行分析
    print("\n🔍 开始分析...\n")
    results = analyzer.analyze()
    
    # 显示结果摘要
    print("\n" + "=" * 70)
    print("📊 分析结果摘要")
    print("=" * 70)
    
    if 'sleep_metrics' in results.get('summary', {}):
        sm = results['summary']['sleep_metrics']
        print(f"\n💤 睡眠指标:")
        print(f"  总睡眠时间: {sm.get('sleep_time_hours', 0):.1f} 小时")
        print(f"  睡眠效率: {sm.get('sleep_efficiency', 0):.1f}%")
    
    if 'respiratory_metrics' in results.get('summary', {}):
        rm = results['summary']['respiratory_metrics']
        print(f"\n🌬️ 呼吸指标:")
        print(f"  AHI: {rm.get('ahi', 0):.1f} ({rm.get('severity', '未知')})")
        if 'min_spo2' in rm:
            print(f"  最低血氧: {rm['min_spo2']:.1f}%")
    
    if 'spindle_metrics' in results.get('summary', {}):
        spm = results['summary']['spindle_metrics']
        print(f"\n🌀 纺锤波指标:")
        print(f"  密度: {spm.get('density', 0):.2f} 个/分钟")
        print(f"  平均频率: {spm.get('mean_frequency', 0):.1f} Hz")
    
    print(f"\n📁 结果保存在: {Path(config.output_dir).absolute()}")
    print("\n" + "=" * 70)
    print("✅ 分析完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
