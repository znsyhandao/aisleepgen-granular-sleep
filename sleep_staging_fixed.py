#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版睡眠分期分析
调整分类规则，解决Unscored问题
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
from scipy import signal
import warnings
import json
warnings.filterwarnings('ignore')

def analyze_sleep_stages(edf_path):
    """修复版睡眠分期分析"""
    
    print("=" * 70)
    print("修复版睡眠分期分析")
    print("=" * 70)
    print(f"文件: {edf_path}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. 加载文件
        print("[1/6] 加载EDF文件...")
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        
        sfreq = raw.info['sfreq']
        n_samples = raw.n_times
        duration_hours = n_samples / (sfreq * 3600)
        
        print(f"   时长: {duration_hours:.1f}小时")
        print(f"   采样率: {sfreq}Hz")
        print(f"   通道: {len(raw.ch_names)}个")
        
        # 2. 识别通道
        print("[2/6] 识别睡眠相关通道...")
        
        # 显示所有通道
        print("   所有通道:")
        for i, ch in enumerate(raw.ch_names[:10]):  # 只显示前10个
            print(f"     {i+1}. {ch}")
        if len(raw.ch_names) > 10:
            print(f"    ... 还有{len(raw.ch_names)-10}个通道")
        
        # 自动识别通道类型
        eeg_chans = []
        eog_chans = []
        emg_chans = []
        resp_chans = []
        
        for ch in raw.ch_names:
            ch_lower = ch.lower()
            if 'eeg' in ch_lower:
                eeg_chans.append(ch)
            elif 'eog' in ch_lower:
                eog_chans.append(ch)
            elif 'emg' in ch_lower:
                emg_chans.append(ch)
            elif 'resp' in ch_lower:
                resp_chans.append(ch)
        
        print(f"   EEG通道: {eeg_chans}")
        print(f"   EOG通道: {eog_chans}")
        print(f"   EMG通道: {emg_chans}")
        print(f"   呼吸通道: {resp_chans}")
        
        # 3. 选择分析通道
        if eeg_chans:
            analysis_channel = eeg_chans[0]
            print(f"   使用EEG通道进行分析: {analysis_channel}")
        elif resp_chans:
            analysis_channel = resp_chans[0]
            print(f"   使用呼吸通道进行分析: {analysis_channel}")
        else:
            print("   错误: 没有找到合适的分析通道!")
            return None
        
        # 4. 提取和分析数据
        print("[3/6] 提取和分析数据...")
        
        # 每30秒一个epoch（标准睡眠分期）
        epoch_sec = 30
        epoch_samples = int(epoch_sec * sfreq)
        total_epochs = int(n_samples / epoch_samples)
        
        print(f"   总epoch数: {total_epochs} (每{epoch_sec}秒)")
        
        # 采样分析（每10个epoch分析一个，加快速度）
        sample_rate = 20
        sampled_epochs = []
        
        for epoch_idx in range(0, total_epochs, sample_rate):
            start = epoch_idx * epoch_samples
            end = min((epoch_idx + 1) * epoch_samples, n_samples)
            
            if epoch_idx % 200 == 0:
                print(f"      处理epoch {epoch_idx}/{total_epochs}")
            
            # 提取数据
            data = raw.get_data(picks=analysis_channel, start=start, stop=end)[0]
            
            if len(data) < epoch_samples * 0.8:  # 数据不足80%，跳过
                continue
            
            # 计算特征
            features = calculate_epoch_features(data, sfreq, epoch_idx, start/sfreq)
            sampled_epochs.append(features)
        
        print(f"   分析完成: {len(sampled_epochs)}个采样epoch")
        
        # 5. 睡眠分期
        print("[4/6] 进行睡眠分期...")
        
        sleep_stages = []
        for features in sampled_epochs:
            stage = classify_sleep_stage_fixed(features)
            sleep_stages.append(stage)
        
        # 统计
        stage_counts = {}
        for stage in sleep_stages:
            label = get_stage_label(stage)
            stage_counts[label] = stage_counts.get(label, 0) + 1
        
        total = len(sleep_stages)
        print("   睡眠阶段统计:")
        for label, count in stage_counts.items():
            percentage = (count / total) * 100
            print(f"     {label}: {count}个 ({percentage:.1f}%)")
        
        # 6. 生成报告
        print("[5/6] 生成报告...")
        
        # 计算睡眠指标
        sleep_metrics = calculate_sleep_metrics(sleep_stages, sampled_epochs)
        
        report = {
            'file_info': {
                'path': edf_path,
                'duration_hours': round(duration_hours, 2),
                'sampling_rate': sfreq,
                'analysis_channel': analysis_channel,
                'total_epochs': total_epochs,
                'sampled_epochs': len(sampled_epochs)
            },
            'sleep_stages': {
                'counts': stage_counts,
                'percentages': {k: round((v/total)*100, 1) for k, v in stage_counts.items()},
                'raw_stages': sleep_stages
            },
            'sleep_metrics': sleep_metrics,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 7. 保存结果
        print("[6/6] 保存结果...")
        
        output_dir = os.path.join(os.path.dirname(edf_path), 'sleep_staging_fixed')
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON报告
        report_path = os.path.join(output_dir, 'sleep_staging_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 文本报告
        txt_path = os.path.join(output_dir, 'sleep_staging_summary.txt')
        create_text_report(report, txt_path)
        
        # 可视化
        create_visualizations(sleep_stages, sampled_epochs, output_dir)
        
        print()
        print("=" * 70)
        print("分析完成!")
        print(f"输出目录: {output_dir}")
        print("=" * 70)
        
        return report
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_epoch_features(data, sfreq, epoch_idx, start_time):
    """计算epoch特征"""
    
    # 基本统计
    mean_amp = np.mean(np.abs(data))
    std_amp = np.std(data)
    
    # 频带功率（使用更稳定的计算方法）
    def band_power(data, sfreq, low, high):
        from scipy.signal import welch
        freqs, psd = welch(data, sfreq, nperseg=min(256, len(data)))
        mask = (freqs >= low) & (freqs <= high)
        return np.mean(psd[mask]) if np.any(mask) else 0
    
    delta_power = band_power(data, sfreq, 0.5, 4)
    theta_power = band_power(data, sfreq, 4, 8)
    alpha_power = band_power(data, sfreq, 8, 13)
    sigma_power = band_power(data, sfreq, 11, 16)
    beta_power = band_power(data, sfreq, 13, 30)
    
    # 总功率用于归一化
    total_power = delta_power + theta_power + alpha_power + sigma_power + beta_power + 1e-10
    
    features = {
        'epoch_idx': epoch_idx,
        'start_time': start_time,
        'mean_amplitude': mean_amp,
        'std_amplitude': std_amp,
        'delta_ratio': delta_power / total_power,
        'theta_ratio': theta_power / total_power,
        'alpha_ratio': alpha_power / total_power,
        'sigma_ratio': sigma_power / total_power,
        'beta_ratio': beta_power / total_power,
        'total_power': total_power
    }
    
    return features

def classify_sleep_stage_fixed(features):
    """修复版睡眠阶段分类"""
    
    mean_amp = features['mean_amplitude']
    alpha_ratio = features['alpha_ratio']
    delta_ratio = features['delta_ratio']
    sigma_ratio = features['sigma_ratio']
    
    # 调试信息
    debug = False
    
    if debug:
        print(f"  特征: amp={mean_amp:.3f}, alpha={alpha_ratio:.3f}, delta={delta_ratio:.3f}, sigma={sigma_ratio:.3f}")
    
    # 规则1: 检查数据质量
    if mean_amp < 1.0:  # 降低幅度阈值
        if debug: print("   -> Unscored (低幅度)")
        return 7  # Unscored
    
    # 规则2: 高Alpha表示清醒
    if alpha_ratio > 0.3:  # 降低Alpha阈值
        if debug: print("   -> Wake (高Alpha)")
        return 0  # Wake
    
    # 规则3: 高Delta表示深睡
    if delta_ratio > 0.4:  # 调整Delta阈值
        if debug: print("   -> N3 (高Delta)")
        return 3  # N3
    
    # 规则4: 中等Sigma表示N2
    if 0.1 < sigma_ratio < 0.3:  # 调整Sigma范围
        if debug: print("   -> N2 (中等Sigma)")
        return 2  # N2
    
    # 规则5: 低幅度+中等Alpha可能是REM
    if mean_amp < 50 and 0.05 < alpha_ratio < 0.2:
        if debug: print("   -> REM (低幅度混合)")
        return 4  # REM
    
    # 默认: N1（浅睡）
    if debug: print("   -> N1 (默认)")
    return 1  # N1

def get_stage_label(stage_code):
    """获取阶段标签"""
    labels = {
        0: 'Wake',
        1: 'N1',
        2: 'N2',
        3: 'N3',
        4: 'REM',
        5: 'N4',
        6: 'Movement',
        7: 'Unscored'
    }
    return labels.get(stage_code, 'Unknown')

def calculate_sleep_metrics(stages, epochs):
    """计算睡眠指标"""
    
    if not stages:
        return {
            'total_sleep_time_hours': 0,
            'sleep_efficiency_percent': 0,
            'sleep_latency_minutes': 0,
            'rem_latency_minutes': None,
            'wake_percent': 0,
            'n1_percent': 0,
            'n2_percent': 0,
            'n3_percent': 0,
            'rem_percent': 0
        }
    
    # 统计各阶段
    stage_counts = {}
    for stage in stages:
        label = get_stage_label(stage)
        stage_counts[label] = stage_counts.get(label, 0) + 1
    
    total = len(stages)
    
    # 计算百分比
    percentages = {}
    for label, count in stage_counts.items():
        percentages[label] = round((count / total) * 100, 1)
    
    # 睡眠时间（每个epoch 30秒）
    sleep_epochs = total - stage_counts.get('Wake', 0) - stage_counts.get('Unscored', 0)
    total_sleep_time = sleep_epochs * 30  # 秒
    
    # 睡眠效率
    total_time = total * 30  # 秒
    sleep_efficiency = (total_sleep_time / total_time) * 100 if total_time > 0 else 0
    
    # 睡眠潜伏期（找到第一个非Wake/Unscored）
    sleep_latency = 0
    for i, stage in enumerate(stages):
        if stage not in [0, 7]:  # 不是Wake或Unscored
            sleep_latency = i * 30 / 60  # 转换为分钟
            break
    
    # REM潜伏期
    rem_latency = None
    for i, stage in enumerate(stages):
        if stage == 4:  # REM
            # 找到第一个睡眠期
            first_sleep = None
            for j in range(i):
                if stages[j] not in [0, 7]:
                    first_sleep = j
                    break
            
            if first_sleep is not None:
                rem_latency = (i - first_sleep) * 30 / 60  # 分钟
            break
    
    return {
        'total_sleep_time_hours': round(total_sleep_time / 3600, 2),
        'sleep_efficiency_percent': round(sleep_efficiency, 1),
        'sleep_latency_minutes': round(sleep_latency, 1),
        'rem_latency_minutes': round(rem_latency, 1) if rem_latency else None,
        'wake_percent': percentages.get('Wake', 0),
        'n1_percent': percentages.get('N1', 0),
        'n2_percent': percentages.get('N2', 0),
        'n3_percent': percentages.get('N3', 0),
        'rem_percent': percentages.get('REM', 0),
        'unscored_percent': percentages.get('Unscored', 0)
    }

def create_text_report(report, output_path):
    """创建文本报告"""
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("睡眠分期分析报告（修复版）\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"文件: {report['file_info']['path']}\n")
            f.write(f"分析时间: {report['analysis_time']}\n")
            f.write(f"记录时长: {report['file_info']['duration_hours']}小时\n")
            f.write(f"分析通道: {report['file_info']['analysis_channel']}\n")
            f.write(f"采样epoch数: {report['file_info']['sampled_epochs']}\n\n")
            
            f.write("睡眠指标:\n")
            metrics = report['sleep_metrics']
            f.write(f"  总睡眠时间: {metrics['total_sleep_time_hours']}小时\n")
            f.write(f"  睡眠效率: {metrics['sleep_efficiency_percent']}%\n")
            f.write(f"  睡眠潜伏期: {metrics['sleep_latency_minutes']}分钟\n")
            
            if metrics['rem_latency_minutes']:
                f.write(f"  REM潜伏期: {metrics['rem_latency_minutes']}分钟\n")
            else:
                f.write("  REM潜伏期: 未检测到REM睡眠\n")
            
            f.write("\n睡眠阶段比例:\n")
            for stage, percent in report['sleep_stages']['percentages'].items():
                f.write(f"  {stage}: {percent}%\n")
            
            f.write("\n临床参考（健康成人）:\n")
            f.write("  睡眠效率: >85%\n")
            f.write("  睡眠潜伏期: <30分钟\n")
            f.write("  REM潜伏期: 60-120分钟\n")
            f.write("  深睡(N3): 15-25%\n")
            f.write("  REM: 20-25%\n")
            f.write("  浅睡(N1): 5-10%\n")
            f.write("  N2: 45-55%\n")
            
            f.write("\n注意: 此分析基于简化算法，仅供参考。\n")
            f.write("专业睡眠分期需要多通道PSG和专家评分。\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"   文本报告已保存: {output_path}")
        
    except Exception as e:
        print(f"   创建文本报告错误: {str(e)}")

def create_visualizations(stages, epochs, output_dir):
    """创建可视化图表"""
    
    try:
        if not stages:
            return
        
        # 1. 睡眠阶段时间序列
        times = [e['start_time'] / 3600 for e in epochs]  # 转换为小时
        stage_labels = [get_stage_label(s) for s in stages]
        
        # 颜色映射
        colors = {
            'Wake': '#FF6B6B',
            'N1': '#4ECDC4',
            'N2': '#45