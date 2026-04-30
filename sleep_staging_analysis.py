#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
睡眠分期分析脚本
分析EDF文件的睡眠阶段（Wake, N1, N2, N3, REM）
"""

import os
import sys
import mne
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import json
warnings.filterwarnings('ignore')

def sleep_staging_analysis(edf_path):
    """进行睡眠分期分析"""
    
    print("=" * 70)
    print("睡眠分期分析报告")
    print("=" * 70)
    print(f"分析文件: {edf_path}")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not os.path.exists(edf_path):
        print("错误: 文件不存在!")
        return None
    
    try:
        # 1. 加载EDF文件
        print("[1/8] 加载EDF文件...")
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        
        # 获取基本信息
        sfreq = raw.info['sfreq']
        n_samples = raw.n_times
        duration_hours = n_samples / (sfreq * 3600)
        
        print(f"   采样频率: {sfreq} Hz")
        print(f"   数据长度: {n_samples:,} 样本")
        print(f"   记录时长: {duration_hours:.2f} 小时")
        
        # 2. 识别EEG通道（用于睡眠分期）
        print("[2/8] 识别EEG通道...")
        
        eeg_channels = []
        eog_channels = []
        emg_channels = []
        
        for ch_name in raw.ch_names:
            ch_name_lower = ch_name.lower()
            if any(keyword in ch_name_lower for keyword in ['eeg', 'c3', 'c4', 'cz', 'f3', 'f4', 'fz', 'o1', 'o2', 'pz']):
                eeg_channels.append(ch_name)
            elif any(keyword in ch_name_lower for keyword in ['eog', 'loc', 'roc']):
                eog_channels.append(ch_name)
            elif any(keyword in ch_name_lower for keyword in ['emg', 'chin']):
                emg_channels.append(ch_name)
        
        print(f"   EEG通道: {eeg_channels}")
        print(f"   EOG通道: {eog_channels}")
        print(f"   EMG通道: {emg_channels}")
        
        if not eeg_channels:
            print("警告: 未找到标准EEG通道，尝试使用其他通道...")
            # 尝试使用任何看起来像脑电的通道
            for ch_name in raw.ch_names:
                if 'EEG' in ch_name.upper() or 'C' in ch_name or 'F' in ch_name or 'O' in ch_name or 'P' in ch_name:
                    eeg_channels.append(ch_name)
            
            if not eeg_channels:
                print("错误: 无法找到合适的EEG通道进行睡眠分期!")
                return None
        
        # 3. 提取EEG数据（分块处理避免内存问题）
        print("[3/8] 提取EEG数据（分块处理）...")
        
        # 定义分块大小（30分钟）
        chunk_size_samples = int(1800 * sfreq)  # 30分钟
        n_chunks = int(np.ceil(n_samples / chunk_size_samples))
        
        print(f"   分为 {n_chunks} 个数据块处理")
        
        # 存储每30秒epoch的特征
        all_features = []
        epoch_times = []
        
        # 每30秒一个epoch（标准睡眠分期）
        epoch_duration = 30  # 秒
        epoch_samples = int(epoch_duration * sfreq)
        
        # 4. 提取特征（简化版 - 实际应使用更复杂的特征提取）
        print("[4/8] 提取睡眠特征...")
        
        # 使用第一个EEG通道进行分析
        eeg_channel = eeg_channels[0]
        
        # 计算总epoch数
        total_epochs = int(n_samples / epoch_samples)
        print(f"   总epoch数: {total_epochs} (每{epoch_duration}秒一个epoch)")
        
        # 采样处理（每10个epoch处理一个，避免内存问题）
        sample_rate = 10
        sampled_epochs = []
        
        for epoch_idx in range(0, total_epochs, sample_rate):
            start_sample = epoch_idx * epoch_samples
            end_sample = min((epoch_idx + 1) * epoch_samples, n_samples)
            
            if epoch_idx % 100 == 0 or epoch_idx == total_epochs - 1:
                print(f"      处理epoch {epoch_idx + 1}/{total_epochs}")
            
            # 提取当前epoch数据
            epoch_data = raw.get_data(
                picks=eeg_channel,
                start=start_sample,
                stop=end_sample
            )[0]
            
            if len(epoch_data) < epoch_samples * 0.5:  # 数据不足一半，跳过
                continue
            
            # 计算特征（简化版）
            features = {
                'epoch_idx': epoch_idx,
                'start_time': start_sample / sfreq,
                'end_time': end_sample / sfreq,
                'mean_amplitude': np.mean(np.abs(epoch_data)),
                'std_amplitude': np.std(epoch_data),
                'delta_power': _calculate_band_power(epoch_data, sfreq, 0.5, 4),
                'theta_power': _calculate_band_power(epoch_data, sfreq, 4, 8),
                'alpha_power': _calculate_band_power(epoch_data, sfreq, 8, 13),
                'sigma_power': _calculate_band_power(epoch_data, sfreq, 11, 16),  # 纺锤波频段
                'beta_power': _calculate_band_power(epoch_data, sfreq, 13, 30),
                'gamma_power': _calculate_band_power(epoch_data, sfreq, 30, 45)
            }
            
            sampled_epochs.append(features)
        
        # 5. 睡眠分期（基于规则的简化版本）
        print("[5/8] 进行睡眠分期...")
        
        # 基于特征进行简单分期
        sleep_stages = []
        stage_labels = []
        
        for features in sampled_epochs:
            stage = _classify_sleep_stage(features)
            sleep_stages.append(stage)
            stage_labels.append(_get_stage_label(stage))
        
        # 6. 计算睡眠结构指标
        print("[6/8] 计算睡眠结构指标...")
        
        # 统计各阶段比例
        unique_stages, stage_counts = np.unique(sleep_stages, return_counts=True)
        total_epochs_sampled = len(sleep_stages)
        
        stage_percentages = {}
        for stage, count in zip(unique_stages, stage_counts):
            percentage = (count / total_epochs_sampled) * 100
            stage_percentages[_get_stage_label(stage)] = round(percentage, 1)
        
        # 计算睡眠连续性指标
        sleep_continuity = _calculate_sleep_continuity(sleep_stages, sampled_epochs)
        
        # 7. 生成报告
        print("[7/8] 生成分析报告...")
        
        report = {
            'file_info': {
                'path': edf_path,
                'duration_hours': round(duration_hours, 2),
                'sampling_rate': sfreq,
                'eeg_channel': eeg_channel,
                'total_epochs': total_epochs,
                'sampled_epochs': total_epochs_sampled
            },
            'sleep_structure': {
                'stage_percentages': stage_percentages,
                'total_sleep_time_hours': round(sleep_continuity['total_sleep_time'] / 3600, 2),
                'sleep_efficiency': round(sleep_continuity['sleep_efficiency'] * 100, 1),
                'sleep_latency_minutes': round(sleep_continuity['sleep_latency'] / 60, 1),
                'wake_after_sleep_onset_minutes': round(sleep_continuity['wake_after_sleep_onset'] / 60, 1),
                'rem_latency_minutes': round(sleep_continuity['rem_latency'] / 60, 1) if sleep_continuity['rem_latency'] else None
            },
            'stage_details': {
                'sampled_stages': sleep_stages,
                'stage_labels': stage_labels,
                'epoch_features': sampled_epochs
            },
            'analysis_notes': '此分析基于简化规则。专业睡眠分期需要多通道EEG/EOG/EMG和专家评分。',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 8. 保存报告和可视化
        print("[8/8] 保存分析结果...")
        
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(edf_path), 'sleep_staging_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON报告
        report_path = os.path.join(output_dir, 'sleep_staging_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成睡眠结构图
        _create_sleep_histogram(sleep_stages, sampled_epochs, output_dir)
        
        # 生成文本报告
        _create_text_report(report, output_dir)
        
        print()
        print("=" * 70)
        print("睡眠分期分析完成!")
        print("=" * 70)
        print(f"输出目录: {output_dir}")
        
        return report
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def _calculate_band_power(self, signal, sfreq, low_freq, high_freq):
    """计算频带功率（简化版）"""
    from scipy import signal as sp_signal
    
    # 设计带通滤波器
    nyquist = sfreq / 2
    b, a = sp_signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
    
    # 滤波
    filtered = sp_signal.filtfilt(b, a, signal)
    
    # 计算功率
    power = np.mean(filtered ** 2)
    return power

def _classify_sleep_stage(self, features):
    """基于特征分类睡眠阶段（简化规则）"""
    
    # 规则基于典型睡眠特征
    alpha_ratio = features['alpha_power'] / (features['delta_power'] + 1e-10)
    delta_ratio = features['delta_power'] / (features['alpha_power'] + 1e-10)
    sigma_ratio = features['sigma_power'] / (features['theta_power'] + 1e-10)
    
    # 简化分类规则
    if features['mean_amplitude'] < 10:  # 非常低的振幅可能是伪迹或缺失数据
        return 7  # 未评分
    
    # 高alpha功率通常表示清醒或N1
    if alpha_ratio > 0.5:
        return 0  # Wake
    
    # 高delta功率表示深睡（N3）
    elif delta_ratio > 2.0:
        return 3  # N3
    
    # 中等sigma功率（纺锤波）表示N2
    elif 0.3 < sigma_ratio < 1.5:
        return 2  # N2
    
    # 低振幅混合频率可能是REM
    elif features['mean_amplitude'] < 30 and 0.1 < alpha_ratio < 0.3:
        return 4  # REM
    
    # 默认N1（浅睡）
    else:
        return 1  # N1

def _get_stage_label(self, stage_code):
    """获取睡眠阶段标签"""
    stage_labels = {
        0: 'Wake',
        1: 'N1',
        2: 'N2', 
        3: 'N3',
        4: 'REM',
        5: 'N4',  # 旧标准，现合并到N3
        6: 'Movement',
        7: 'Unscored'
    }
    return stage_labels.get(stage_code, 'Unknown')

def _calculate_sleep_continuity(self, sleep_stages, epoch_features):
    """计算睡眠连续性指标"""
    
    if not sleep_stages:
        return {
            'total_sleep_time': 0,
            'sleep_efficiency': 0,
            'sleep_latency': 0,
            'wake_after_sleep_onset': 0,
            'rem_latency': None
        }
    
    # 找到第一个睡眠期（非Wake）
    first_sleep_idx = None
    for i, stage in enumerate(sleep_stages):
        if stage != 0:  # 不是Wake
            first_sleep_idx = i
            break
    
    # 找到第一个REM期
    first_rem_idx = None
    for i, stage in enumerate(sleep_stages):
        if stage == 4:  # REM
            first_rem_idx = i
            break
    
    # 计算指标
    total_epochs = len(sleep_stages)
    sleep_epochs = len([s for s in sleep_stages if s != 0])  # 非Wake都是睡眠
    
    # 每个epoch 30秒
    epoch_duration = 30
    
    total_sleep_time = sleep_epochs * epoch_duration
    total_recording_time = total_epochs * epoch_duration
    
    sleep_efficiency = total_sleep_time / total_recording_time if total_recording_time > 0 else 0
    
    # 睡眠潜伏期（从记录开始到第一个睡眠）
    sleep_latency = first_sleep_idx * epoch_duration if first_sleep_idx is not None else total_recording_time
    
    # REM潜伏期（从第一个睡眠到第一个REM）
    rem_latency = None
    if first_sleep_idx is not None and first_rem_idx is not None and first_rem_idx > first_sleep_idx:
        rem_latency = (first_rem_idx - first_sleep_idx) * epoch_duration
    
    # 睡眠后觉醒时间（简化计算）
    # 找到主要睡眠期后的Wake
    wake_after_sleep = 0
    if first_sleep_idx is not None:
        for i in range(first_sleep_idx, len(sleep_stages)):
            if sleep_stages[i] == 0:
                wake_after_sleep += epoch_duration
    
    return {
        'total_sleep_time': total_sleep_time,
        'sleep_efficiency': sleep_efficiency,
        'sleep_latency': sleep_latency,
        'wake_after_sleep_onset': wake_after_sleep,
        'rem_latency': rem_latency
    }

def _create_sleep_histogram(self, sleep_stages, epoch_features, output_dir):
    """创建睡眠阶段直方图"""
    
    if not sleep_stages:
        return
    
    try:
        # 准备数据
        stage_labels = [self._get_stage_label(s) for s in sleep_stages]
        times = [feat['start_time'] / 3600 for feat in epoch_features]  # 转换为小时
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 睡眠阶段时间序列
        colors = {'Wake': 'gray', 'N1': 'blue', 'N2': 'green', 'N3': 'red', 'REM': 'purple'}
        stage_colors = [colors.get(label, 'black') for label in stage_labels]
        
        ax1.bar(times, [1] * len(times), color=stage_colors, width=0.02)
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('睡眠阶段')
        ax1.set_title('睡眠阶段分布')
        ax1.set_ylim(0, 1.2)
        
        # 创建图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=label) 
                          for label, color in colors.items()]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 2. 睡眠阶段比例饼图
        unique_labels, counts = np.unique(stage_labels, return_counts=True)
        ax2.pie(counts, labels=unique_labels, autopct='%1.1f%%', 
                colors=[colors.get(label, 'gray') for label in unique_labels])
        ax2.set_title('睡眠阶段比例分布')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'sleep_staging_histogram.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   睡眠阶段图已保存: {plot_path}")
        
    except Exception as e:
        print(f"   创建图表时出错: {str(e)}")

def _create_text_report(self, report, output_dir):
    """创建文本报告"""
    
    try:
        txt_path = os.path.join(output_dir, 'sleep_staging_summary.txt')
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("睡眠分期分析报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"分析文件: {report['file_info']['path']}\n")
            f.write(f"分析时间: {report['timestamp']}\n")
            f.write(f"记录时长: {report['file_info']['duration_hours']} 小时\n")
            f.write(f"EEG通道: {report['file_info']['eeg_channel']}\n")
            f.write(f"总epoch数: {report['file_info']['total_epochs']}\n")
            f.write(f"采样epoch数: {report['file_info']['sampled_epochs']}\n\n")
            
            f.write("睡眠结构指标:\n")
            f.write(f"  总睡眠时间: {report['sleep_structure']['total_sleep_time_hours']} 小时\n")
            f.write(f"  睡眠效率: {report['sleep_structure']['sleep_efficiency']}%\n")
            f.write(f"  睡眠潜伏期: {report['sleep_structure']['sleep_latency_minutes']} 分钟\n")
            f.write(f"  睡眠后觉醒时间: {report['sleep_structure']['wake_after_sleep_onset_minutes']} 分钟\n")
            
            if report['sleep_structure']['rem_latency_minutes']:
                f.write(f"  REM潜伏期: {report['sleep_structure']['rem_latency_minutes']} 分钟\n")
            else:
                f.write("  REM潜伏期: 未检测到REM睡眠\n")
            
            f.write("\n睡眠阶段比例:\n")
            for stage, percentage in report['sleep_structure']['stage_percentages'].items():
                f.write(f"  {stage}: {percentage}%\n")
            
            f.write("\n临床参考范围（健康成人）:\n")
            f.write("  睡眠效率: >85%\n")
            f.write("  睡眠潜伏期: <30分钟\n")
            f.write("  REM潜伏期: 60-120分钟\n")
            f.write("  深睡比例(N3): 15-25%\n")
            f.write("  REM比例: 20-25%\n")
            
            f.write("\n分析说明:\n")
            f.write("  此分析基于单通道EEG的简化规则。\n")
            f.write("  专业睡眠分期需要多通道PSG记录和专家人工评分。\n")
            f.write("  结果仅供参考，不能替代临床诊断。\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"   文本报告已保存: {txt_path}")
        
    except Exception as e:
        print(f"   创建文本报告时出错: {str(e)}")

def main():
    """主函数"""
    # 设置编码
    import sys
    import io
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    if len(sys.argv) > 1:
        edf_path = sys.argv[1]
    else:
        edf_path = r"D:\openclaw\AISleepGen\data\edf\SC4001E0-PSG.edf"
    
    print("睡眠分期分析系统启动...")
    
    # 创建分析器实例
    analyzer = SleepStagingAnalyzer()
    result = analyzer.sleep_staging_analysis(edf_path)
    
    if result:
        print("分析成功完成!")
    else:
        print("分析失败!")
        sys.exit(1)

class SleepStagingAnalyzer:
    """睡眠分期分析器类"""
    
    def __init__(self):
        pass
    
    # 将所有方法作为实例方法
    def sleep_staging_analysis(self, edf_path):
        # 这里需要将原函数转换为方法
        # 由于代码较长，我们直接调用模块级函数
        return sleep_staging_analysis(edf_path)
    
    def _calculate_band_power(self, signal, sfreq, low_freq, high_freq):
        return _calculate_band_power(signal, sfreq, low_freq, high_freq)
    
    def _classify_sleep_stage(self, features):
        return _classify_sleep_stage(features)
    
    def _get_stage_label(self, stage_code):
        return _get_stage_label(stage_code)
    
    def _calculate_sleep_continuity(self, sleep_stages, epoch_features):
        return _calculate_sleep_continuity(sleep_stages, epoch_features)
    
    def _create_sleep_histogram(self, sleep_stages, epoch_features, output_dir):
        return _create_sleep_histogram(sleep_stages, epoch_features, output_dir)
    
    def _create_text_report(self, report, output_dir):
        return _create_text_report(report, output_dir)

# 定义模块级函数（供类方法调用）
def _calculate_band_power(signal, sfreq, low_freq, high_freq):
    """计算频带功率（简化版）"""
    from scipy import signal as sp_signal
    
    # 设计带通滤波器
    nyquist = sfreq / 2
    b, a = sp_signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
    
    # 滤波
    filtered = sp_signal.filtfilt(b, a, signal)
    
    # 计算功率
    power = np.mean(filtered ** 2)
    return power

def _classify_sleep_stage(features):
    """基于特征分类睡眠阶段（简化规则）"""
    
    # 规则基于典型睡眠特征
    alpha_ratio = features['alpha_power'] / (features['delta_power'] + 1e-10)
    delta_ratio = features['delta_power'] / (features['alpha_power'] + 1e-10)
    sigma_ratio = features['sigma_power'] / (features['theta_power'] + 1e-10)
    
    # 简化分类规则
    if features['mean_amplitude'] < 10:  # 非常低的振幅可能是伪迹或缺失数据
        return 7  # 未评分
    
    # 高alpha功率通常表示清醒或N1
    if alpha_ratio > 0.5:
        return 0  # Wake
    
    # 高delta功率表示深睡（N3）
    elif delta_ratio > 2.0:
        return 3  # N3
    
    # 中等sigma功率（纺锤波）表示N2
    elif 0.3 < sigma_ratio < 1.5:
        return 2  # N2
    
    # 低振幅混合频率可能是REM
    elif features['mean_amplitude'] < 30 and 0.1 < alpha_ratio < 0.3:
        return 4  # REM
    
    # 默认N1（浅睡）
    else:
        return 1  # N1

def _get_stage_label(stage_code):
    """获取睡眠阶段标签"""
    stage_labels = {
        0: 'Wake',
        1: 'N1',
        2: 'N2', 
        3: 'N3',
        4: 'REM',
        5: 'N4',  # 旧标准，现合并到N3
        6: 'Movement',
        7: 'Unscored'
    }
    return stage_labels.get(stage_code, 'Unknown')

def _calculate_sleep_continuity(sleep_stages, epoch_features):
    """计算睡眠连续性指标"""
    
    if not sleep_stages:
        return {
            'total_sleep_time': 0,
            'sleep_efficiency': 0,
            'sleep_latency': 0,
            'wake_after_sleep_onset': 0,
            'rem_latency': None
        }
    
    # 找到第一个睡眠期（非Wake）
    first_sleep_idx = None
    for i, stage in enumerate(sleep_stages):
        if stage != 0:  # 不是Wake
            first_sleep_idx = i
            break
    
    # 找到第一个REM期
    first_rem_idx = None
    for i, stage in enumerate(sleep_stages):
        if stage == 4:  # REM
            first_rem_idx = i
            break
    
    # 计算指标
    total_epochs = len(sleep_stages)
    sleep_epochs = len([s for s in sleep_stages if s != 0])  # 非Wake都是睡眠
    
    # 每个epoch 30秒
    epoch_duration = 30
    
    total_sleep_time = sleep_epochs * epoch_duration
    total_recording_time = total_epochs * epoch_duration
    
    sleep_efficiency = total_sleep_time / total_recording_time if total_recording_time > 0 else 0
    
    # 睡眠潜伏期（从记录开始到第一个睡眠）
    sleep_latency = first_sleep_idx * epoch_duration if first_sleep_idx is not None else total_recording_time
    
    # REM潜伏期（从第一个睡眠到第一个REM）
    rem_latency = None
    if first_sleep_idx is not None and first_rem_idx is not None and first_rem_idx > first_sleep_idx:
        rem_latency = (first_rem_idx - first_sleep_idx) * epoch_duration
    
    # 睡眠后觉醒时间（简化计算）
    # 找到主要睡眠期后的Wake
    wake_after_sleep = 0
    if first_sleep_idx is not None:
        for i in range(first_sleep_idx, len(sleep_stages)):
            if sleep_stages[i] == 0:
                wake_after_sleep += epoch_duration
    
    return {
        'total_sleep_time': total_sleep_time,
        'sleep_efficiency': sleep_efficiency,
        'sleep_latency': sleep_latency,
        'wake_after_sleep_onset': wake_after_sleep,
        'rem_latency': rem_latency
    }

def _create_sleep_histogram(sleep_stages, epoch_features, output_dir):
    """创建睡眠阶段直方图"""
    
    if not sleep_stages:
        return
    
    try:
        # 准备数据
        stage_labels = [_get_stage_label(s) for s in sleep_stages]
        times = [feat['start_time'] / 3600 for feat in epoch_features]  # 转换为小时
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 睡眠阶段时间序列
        colors = {'Wake': 'gray', 'N1': 'blue', 'N2': 'green', 'N3': 'red', 'REM': 'purple'}
        stage_colors = [colors.get(label, 'black') for label in stage_labels]
        
        ax1.bar(times, [1] * len(times), color=stage_colors, width=0.02)
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('睡眠阶段')
        ax1.set_title('睡眠阶段分布')
        ax1.set_ylim(0, 1.2)
        
        # 创建图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=label) 
                          for label, color in colors.items()]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 2. 睡眠阶段比例饼图
        unique_labels, counts = np.unique(stage_labels, return_counts=True)
        ax2.pie(counts, labels=unique_labels, autopct='%1.1f%%', 
                colors=[colors.get(label, 'gray') for label in unique_labels])
        ax2.set_title('睡眠阶段比例分布')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'sleep_staging_histogram.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   睡眠阶段图已保存: {plot_path}")
        
    except Exception as e:
        print(f"   创建图表时出错: {str(e)}")

def _create_text_report(report, output_dir):
    """创建文本报告"""
    
    try:
        txt_path = os.path.join(output_dir, 'sleep_staging_summary.txt')
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("睡眠分期分析报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"分析文件: {report['file_info']['path']}\n")
            f.write(f"分析时间: {report['timestamp']}\n")
            f.write(f"记录时长: {report['file_info']['duration_hours']} 小时\n")
            f.write(f"EEG通道: {report['file_info']['eeg_channel']}\n")
            f.write(f"总epoch数: {report['file_info']['total_epochs']}\n")
            f.write(f"采样epoch数: {report['file_info']['sampled_epochs']}\n\n")
            
            f.write("睡眠结构指标:\n")
            f.write(f"  总睡眠时间: {report['sleep_structure']['total_sleep_time_hours']} 小时\n")
            f.write(f"  睡眠效率: {report['sleep_structure']['sleep_efficiency']}%\n")
            f.write(f"  睡眠潜伏期: {report['sleep_structure']['sleep_latency_minutes']} 分钟\n")
            f.write(f"  睡眠后觉醒时间: {report['sleep_structure']['wake_after_sleep_onset_minutes']} 分钟\n")
            
            if report['sleep_structure']['rem_latency_minutes']:
                f.write(f"  REM潜伏期: {report['sleep_structure']['rem_latency_minutes']} 分钟\n")
            else:
                f.write("  REM潜伏期: 未检测到REM睡眠\n")
            
            f.write("\n睡眠阶段比例:\n")
            for stage, percentage in report['sleep_structure']['stage_percentages'].items():
                f.write(f"  {stage}: {percentage}%\n")
            
            f.write("\n临床参考范围（健康成人）:\n")
            f.write("  睡眠效率: >85%\n")
            f.write("  睡眠潜伏期: <30分钟\n")
            f.write("  REM潜伏期: 60-120分钟\n")
            f.write("  深睡比例(N3): 15-25%\n")
            f.write("  REM比例: 20-25%\n")
            
            f.write("\n分析说明:\n")
            f.write("  此分析基于单通道EEG的简化规则。\n")
            f.write("  专业睡眠分期需要多通道PSG记录和专家人工评分。\n")
            f.write("  结果仅供参考，不能替代临床诊断。\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"   文本报告已保存: {txt_path}")
        
    except Exception as e:
        print(f"   创建文本报告时出错: {str(e)}")

if __name__ == "__main__":
    main()