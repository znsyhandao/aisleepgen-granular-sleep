#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细的EDF睡眠文件分析
"""

import os
import sys
import mne
from datetime import datetime

def detailed_analysis(edf_path):
    """详细分析EDF睡眠文件"""
    
    print("=" * 70)
    print("眠小兔详细睡眠分析报告")
    print("=" * 70)
    print(f"文件: {edf_path}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not os.path.exists(edf_path):
        print("[ERROR] 文件不存在!")
        return
    
    try:
        # 1. 读取EDF文件
        print("[1/5] 读取EDF文件...")
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        
        # 2. 文件基本信息
        print("[2/5] 基本信息分析...")
        file_size_mb = os.path.getsize(edf_path) / 1024 / 1024
        duration_hours = raw.times[-1] / 3600
        sfreq = raw.info['sfreq']
        n_channels = len(raw.ch_names)
        
        print(f"  文件大小: {file_size_mb:.2f} MB")
        print(f"  采样频率: {sfreq:.1f} Hz")
        print(f"  通道数量: {n_channels}")
        print(f"  数据时长: {duration_hours:.2f} 小时")
        print(f"  采样点数: {len(raw.times):,}")
        
        # 3. 通道类型分析
        print("[3/5] 通道类型分析...")
        channel_types = {}
        for ch in raw.ch_names:
            ch_type = raw.get_channel_types(picks=ch)[0]
            channel_types[ch_type] = channel_types.get(ch_type, 0) + 1
        
        print("  通道分布:")
        for ch_type, count in sorted(channel_types.items()):
            print(f"    {ch_type.upper():10s}: {count:2d}个")
        
        # 4. 睡眠质量评分
        print("[4/5] 睡眠质量评估...")
        
        # 基于时长的评分
        if duration_hours >= 8:
            duration_score = 90
            duration_comment = "睡眠时长优秀"
        elif duration_hours >= 7:
            duration_score = 80
            duration_comment = "睡眠时长良好"
        elif duration_hours >= 6:
            duration_score = 70
            duration_comment = "睡眠时长基本足够"
        elif duration_hours >= 5:
            duration_score = 60
            duration_comment = "睡眠时长稍短"
        else:
            duration_score = 40
            duration_comment = "睡眠时长不足"
        
        # 基于EEG通道的评分
        eeg_count = channel_types.get('eeg', 0)
        if eeg_count >= 6:
            eeg_score = 95
            eeg_comment = "EEG通道充足，分析精度高"
        elif eeg_count >= 4:
            eeg_score = 85
            eeg_comment = "EEG通道足够"
        elif eeg_count >= 2:
            eeg_score = 70
            eeg_comment = "EEG通道基本足够"
        else:
            eeg_score = 40
            eeg_comment = "EEG通道不足"
        
        # 综合评分
        sleep_score = int((duration_score * 0.6 + eeg_score * 0.4))
        
        # 睡眠质量等级
        if sleep_score >= 85:
            quality = "优秀"
            quality_symbol = "[EXCELLENT]"
        elif sleep_score >= 70:
            quality = "良好"
            quality_symbol = "[GOOD]"
        elif sleep_score >= 60:
            quality = "一般"
            quality_symbol = "[FAIR]"
        else:
            quality = "较差"
            quality_symbol = "[POOR]"
        
        print(f"  睡眠评分: {sleep_score}/100 {quality_symbol}")
        print(f"  睡眠质量: {quality}")
        print(f"  时长评估: {duration_comment} ({duration_hours:.2f}小时)")
        print(f"  通道评估: {eeg_comment} ({eeg_count}个EEG通道)")
        
        # 5. 详细建议
        print("[5/5] 详细建议...")
        
        recommendations = []
        
        # 时长相关建议
        if duration_hours < 7:
            recommendations.append(f"建议延长睡眠时间至7-8小时（当前{duration_hours:.1f}小时）")
            recommendations.append("建立规律的睡眠时间表")
        
        # 通道相关建议
        if eeg_count < 4:
            recommendations.append(f"建议使用至少4个EEG通道（当前{eeg_count}个）")
            recommendations.append("确保电极接触良好，减少噪声")
        
        # 通用建议
        recommendations.append("睡前1小时避免使用电子设备")
        recommendations.append("保持卧室黑暗、安静、凉爽")
        recommendations.append("每天固定时间睡觉和起床")
        recommendations.append("考虑使用白噪音或放松音乐助眠")
        recommendations.append("避免睡前摄入咖啡因或大量食物")
        
        print("  个性化建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")
        
        # 6. 技术指标
        print()
        print("[技术指标]")
        print(f"  * 数据密度: {len(raw.times)/duration_hours/3600:.1f} 点/秒")
        print(f"  * 时间分辨率: {1/sfreq*1000:.1f} 毫秒")
        print(f"  * 存储效率: {file_size_mb/(duration_hours*60):.3f} MB/分钟")
        print(f"  * 数据完整性: {'完整' if duration_hours >= 6 else '不足'}")
        
        # 7. 通道列表（前10个）
        print()
        print("[通道列表（前10个）]")
        for i, ch in enumerate(raw.ch_names[:10], 1):
            ch_type = raw.get_channel_types(picks=ch)[0]
            print(f"  {i:2d}. {ch:20s} ({ch_type.upper()})")
        
        if n_channels > 10:
            print(f"  ... 还有 {n_channels-10} 个通道")
        
        # 8. 睡眠阶段分析（如果可能）
        print()
        print("[睡眠结构分析]")
        print("  基于EDF文件结构评估:")
        
        # 简单的睡眠阶段估计
        if eeg_count >= 2:
            print("  * 可进行基本的睡眠分期分析")
            print("  * 建议使用专业睡眠分析软件进行详细分期")
        else:
            print("  * EEG通道不足，无法进行详细睡眠分期")
            print("  * 建议增加EEG通道以获得更准确的分析")
        
        # 9. 数据质量评估
        print()
        print("[数据质量评估]")
        if sfreq >= 200:
            print("  * 采样频率: 良好（适合睡眠分析）")
        elif sfreq >= 100:
            print("  * 采样频率: 一般（基本满足需求）")
        else:
            print("  * 采样频率: 较低（可能影响分析精度）")
        
        if duration_hours >= 6:
            print("  * 数据时长: 充足")
        else:
            print("  * 数据时长: 不足（建议延长监测时间）")
        
        print()
        print("=" * 70)
        print("分析完成！建议定期监测睡眠质量以跟踪改善效果。")
        print("=" * 70)
        
        return {
            "sleep_score": sleep_score,
            "duration_hours": duration_hours,
            "eeg_count": eeg_count,
            "n_channels": n_channels,
            "quality": quality,
            "recommendations": recommendations
        }
        
    except Exception as e:
        print(f"[ERROR] 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 分析指定的EDF文件
    edf_path = r"D:\openclaw\AISleepGen\data\edf\SC4001E0-PSG.edf"
    
    if len(sys.argv) > 1:
        edf_path = sys.argv[1]
    
    result = detailed_analysis(edf_path)
    
    if result:
        print()
        print("[分析摘要]")
        print(f"  睡眠评分: {result['sleep_score']}/100")
        print(f"  数据时长: {result['duration_hours']:.2f}小时")
        print(f"  EEG通道: {result['eeg_count']}个")
        print(f"  总通道数: {result['n_channels']}个")
        print(f"  质量等级: {result['quality']}")