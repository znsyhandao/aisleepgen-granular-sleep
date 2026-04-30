"""
信息粒时序压缩 + 睡眠质量评分

从原始 epoch 级预测 → 信息粒 → 睡眠质量评分
"""
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = {'W': '#ff6b6b', 'N1': '#ffd93d', 'N2': '#6bcb77', 'N3': '#4d96ff', 'REM': '#9b59b6'}

# AASM 标准的睡眠周期模式
# 正常夜间: W → N1 → N2 → N3 → N2 → REM → (循环)
NORMAL_CYCLE = ['W', 'N1', 'N2', 'N3', 'N2', 'REM']


def info_epoch_to_granules(pred_stages, confidences, min_epochs=2):
    """
    epoch 级预测 → 信息粒列表
    
    一个信息粒 = 连续同阶段的 epoch 块
    
    Returns: [{'stage','start_epoch','end_epoch','duration_min','confidence'}, ...]
    """
    granules = []
    if len(pred_stages) == 0: return granules
    
    current, start = pred_stages[0], 0
    for i in range(1, len(pred_stages)):
        if pred_stages[i] != current:
            dur = (i - start) * 0.5
            if dur >= min_epochs * 0.5:
                granules.append({
                    'stage': current, 'start_epoch': start,
                    'end_epoch': i-1, 'duration_min': dur,
                    'n_epochs': i - start,
                    'confidence': float(np.mean(confidences[start:i])),
                })
            start, current = i, pred_stages[i]
    
    dur = (len(pred_stages) - start) * 0.5
    if dur >= min_epochs * 0.5:
        granules.append({
            'stage': current, 'start_epoch': start,
            'end_epoch': len(pred_stages)-1, 'duration_min': dur,
            'n_epochs': len(pred_stages) - start,
            'confidence': float(np.mean(confidences[start:])),
        })
    
    return granules


def compress_similar_granules(granules, max_merge_gap_min=3):
    """
    压缩信息粒：合并相邻的短过渡粒
    
    例如: N2(2min) + N1(1min) + N2(5min) → N2(8min) 
    如果中间的 N1 太短（<3min），把它视作噪声
    """
    if len(granules) < 3:
        return granules
    
    merged = [granules[0]]
    i = 1
    while i < len(granules):
        curr = granules[i]
        prev = merged[-1]
        
        # 检查能否合并：当前粒很短且在两个相同阶段之间
        if curr['duration_min'] < max_merge_gap_min and i + 1 < len(granules):
            next_g = granules[i + 1]
            if prev['stage'] == next_g['stage']:
                # 合并 prev + curr + next
                merged[-1] = {
                    'stage': prev['stage'],
                    'start_epoch': prev['start_epoch'],
                    'end_epoch': next_g['end_epoch'],
                    'duration_min': prev['duration_min'] + curr['duration_min'] + next_g['duration_min'],
                    'n_epochs': prev['n_epochs'] + curr['n_epochs'] + next_g['n_epochs'],
                    'confidence': min(prev['confidence'], next_g['confidence']),
                    'merged': True,
                    'merged_min': curr['duration_min'],
                    'merged_stage': curr['stage'],
                }
                i += 2
                continue
        
        merged.append(curr)
        i += 1
    
    return merged


def build_sleep_hypnogram(pred_stages, confidences):
    """
    生成完整的睡眠脑电图（hypnogram 图数据）
    
    Returns: {
        'epochs': [...],  # 简化后的粒序列
        'summary': {...}, # 统计
        'cycles': [...],  # 睡眠周期
    }
    """
    granules = info_epoch_to_granules(pred_stages, confidences)
    compressed = compress_similar_granules(granules)
    
    # 统计
    total_min = len(pred_stages) * 0.5
    stage_min = {s: sum(g['duration_min'] for g in compressed if g['stage'] == s) for s in STAGES}
    total_sleep = sum(v for s, v in stage_min.items() if s != 'W')
    
    # 睡眠效率
    sleep_efficiency = total_sleep / total_min * 100 if total_min > 0 else 0
    
    # 睡眠潜伏期：入睡前 W 的时间
    sleep_latency = 0
    for g in compressed:
        if g['stage'] == 'W':
            sleep_latency += g['duration_min']
        else:
            break
    
    # WASO (Wake After Sleep Onset)
    waso = 0
    sleep_started = False
    for g in compressed:
        if g['stage'] != 'W':
            sleep_started = True
        if sleep_started and g['stage'] == 'W':
            waso += g['duration_min']
    
    # 睡眠周期检测
    cycles = []
    current_cycle = []
    for g in compressed:
        if g['stage'] == 'W' and len(current_cycle) > 2:
            if current_cycle:
                cycles.append(current_cycle)
            current_cycle = []
        else:
            current_cycle.append(g)
    if current_cycle:
        cycles.append(current_cycle)
    
    # 过渡期占比（不稳定睡眠）
    transition_min = sum(g['duration_min'] for g in compressed 
                         if g['stage'] == 'N1' or (g.get('merged') and g['merged_stage'] == 'N1'))
    
    return {
        'granules_raw': granules,
        'granules': compressed,
        'n_granules': len(compressed),
        'summary': {
            'total_min': total_min,
            'total_hours': total_min / 60,
            'sleep_efficiency_pct': sleep_efficiency,
            'sleep_latency_min': sleep_latency,
            'waso_min': waso,
            'transition_min': transition_min,
            'transition_pct': transition_min / total_sleep * 100 if total_sleep > 0 else 0,
            'stage_min': stage_min,
            'stage_pct': {s: stage_min[s]/total_min*100 for s in STAGES},
            'n_cycles': len(cycles),
        },
        'cycles': cycles,
    }


def sleep_quality_score(hypnogram: dict) -> dict:
    """
    睡眠质量综合评分 (0-100)
    
    评分维度:
    1. 睡眠效率 (30分): 实际睡眠/总卧床时间
    2. 深睡比例 (20分): N3 占睡眠时间的比例
    3. REM 比例 (20分): REM 占睡眠时间的比例
    4. 睡眠连续性 (15分): WASO 时间
    5. 入睡速度 (15分): 睡眠潜伏期
    
    Returns: {total_score, dimensions: {...}, grade, ...}
    """
    s = hypnogram['summary']
    total_sleep = s['total_min'] - s['stage_min'].get('W', 0)
    
    # 1. 睡眠效率 (30分)
    eff = s['sleep_efficiency_pct']
    score_eff = min(30, max(0, (eff - 50) / 50 * 30))
    
    # 2. 深睡N3比例 (20分) 目标 15-25%
    n3_pct = s['stage_pct'].get('N3', 0)
    score_n3 = max(0, 20 - abs(n3_pct - 20) * 2) if total_sleep > 0 else 0
    
    # 3. REM比例 (20分) 目标 20-25%
    rem_pct = s['stage_pct'].get('REM', 0)
    score_rem = max(0, 20 - abs(rem_pct - 22) * 1.5) if total_sleep > 0 else 0
    
    # 4. 连续性 (15分) WASO 越少越好
    waso = s.get('waso_min', 0)
    score_waso = max(0, 15 - waso * 0.5)
    
    # 5. 入睡速度 (15分) 潜伏期 <30min
    latency = s.get('sleep_latency_min', 0)
    score_lat = max(0, min(15, 15 - latency * 0.5))
    
    total = score_eff + score_n3 + score_rem + score_waso + score_lat
    
    if total >= 85: grade = '优秀'
    elif total >= 70: grade = '良好'
    elif total >= 50: grade = '一般'
    else: grade = '需要关注'
    
    return {
        'total_score': round(total, 1),
        'grade': grade,
        'dimensions': {
            'sleep_efficiency': round(score_eff, 1),
            'deep_sleep_N3': round(score_n3, 1),
            'REM': round(score_rem, 1),
            'continuity': round(score_waso, 1),
            'sleep_latency': round(score_lat, 1),
        }
    }
