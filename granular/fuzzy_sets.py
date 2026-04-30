"""
模糊集定义 — Fuzzy Sets for Sleep Stage Granulation

将生理信号特征模糊化为"语言变量"：
- Delta功率: {低, 中, 高}
- Alpha功率: {低, 中, 高}
- Sigma功率(纺锤波): {弱, 中, 强}
- 眼动EOG能量: {无, 弱, 强}
- EMG肌电: {低, 中, 高}

每个模糊集用梯形/三角形隶属度函数定义。
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable


class FuzzySet:
    """模糊集: 语言标签 + 隶属度函数"""
    
    def __init__(self, name: str, mf: Callable[[float], float]):
        self.name = name
        self.mf = mf  # 隶属度函数: x ∈ R → [0, 1]
    
    def membership(self, x: float) -> float:
        return self.mf(x)


def _trapezoid(a: float, b: float, c: float, d: float) -> Callable:
    """梯形隶属度函数: 在 [b, c] 为1, [a,b]和[c,d]为斜坡"""
    def mf(x):
        if x <= a or x >= d:
            return 0.0
        if a < x < b:
            return (x - a) / (b - a)
        if b <= x <= c:
            return 1.0
        if c < x < d:
            return (d - x) / (d - c)
        return 0.0
    return mf


def _triangle(a: float, b: float, c: float) -> Callable:
    """三角形隶属度函数"""
    def mf(x):
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return (x - a) / (b - a)
        if b < x < c:
            return (c - x) / (c - b)
        return 0.0
    return mf


# =========================================================================
# 频带功率模糊集
# 阈值基于 Sleep Cassette 数据集的典型分布 (归一化功率 0-1)
# =========================================================================

def delta_power_sets() -> List[FuzzySet]:
    """Delta 功率 (0.5-4 Hz): 慢波活动的指标"""
    return [
        FuzzySet("低", _trapezoid(0.0, 0.0, 0.15, 0.30)),
        FuzzySet("中", _triangle(0.15, 0.35, 0.55)),
        FuzzySet("高", _trapezoid(0.40, 0.60, 1.0, 1.0)),
    ]


def theta_power_sets() -> List[FuzzySet]:
    """Theta 功率 (4-8 Hz): 困倦/浅睡指标"""
    return [
        FuzzySet("低", _trapezoid(0.0, 0.0, 0.10, 0.25)),
        FuzzySet("中", _triangle(0.15, 0.30, 0.50)),
        FuzzySet("高", _trapezoid(0.35, 0.55, 1.0, 1.0)),
    ]


def alpha_power_sets() -> List[FuzzySet]:
    """Alpha 功率 (8-13 Hz): 放松/清醒指标"""
    return [
        FuzzySet("低", _trapezoid(0.0, 0.0, 0.08, 0.20)),
        FuzzySet("中", _triangle(0.12, 0.25, 0.45)),
        FuzzySet("高", _trapezoid(0.30, 0.50, 1.0, 1.0)),
    ]


def sigma_power_sets() -> List[FuzzySet]:
    """Sigma 功率 (11-16 Hz): 纺锤波指标 (N2标志)"""
    return [
        FuzzySet("弱", _trapezoid(0.0, 0.0, 0.05, 0.15)),
        FuzzySet("中", _triangle(0.08, 0.18, 0.35)),
        FuzzySet("强", _trapezoid(0.25, 0.40, 1.0, 1.0)),
    ]


def beta_power_sets() -> List[FuzzySet]:
    """Beta 功率 (16-30 Hz): 活跃/清醒指标"""
    return [
        FuzzySet("低", _trapezoid(0.0, 0.0, 0.05, 0.12)),
        FuzzySet("中", _triangle(0.08, 0.16, 0.30)),
        FuzzySet("高", _trapezoid(0.22, 0.35, 1.0, 1.0)),
    ]


# =========================================================================
# 生理特征模糊集
# =========================================================================

def eog_energy_sets() -> List[FuzzySet]:
    """EOG 能量: 眼动指标 (REM标志)"""
    return [
        FuzzySet("无", _trapezoid(0.0, 0.0, 0.05, 0.12)),
        FuzzySet("弱", _triangle(0.08, 0.18, 0.35)),
        FuzzySet("强", _trapezoid(0.25, 0.40, 1.0, 1.0)),
    ]


def emg_tone_sets() -> List[FuzzySet]:
    """EMG 肌电张力: 肌张力指标 (REM肌张力缺失)"""
    return [
        FuzzySet("低(无张力)", _trapezoid(0.0, 0.0, 0.08, 0.18)),
        FuzzySet("中", _triangle(0.10, 0.25, 0.45)),
        FuzzySet("高(有张力)", _trapezoid(0.35, 0.55, 1.0, 1.0)),
    ]


def spindle_density_sets() -> List[FuzzySet]:
    """纺锤波密度: epoch内纺锤波数量"""
    return [
        FuzzySet("无", _trapezoid(0, 0, 0.5, 1.5)),
        FuzzySet("少量", _triangle(0.5, 2.0, 4.0)),
        FuzzySet("中量", _triangle(2.5, 5.0, 8.0)),
        FuzzySet("大量", _trapezoid(6.0, 8.0, 20, 20)),
    ]


def sawtooth_wave_sets() -> List[FuzzySet]:
    """锯齿波密度: REM标志"""
    return [
        FuzzySet("无", _trapezoid(0, 0, 0.5, 1.5)),
        FuzzySet("有", _triangle(0.5, 2.0, 5.0)),
        FuzzySet("多", _trapezoid(3.0, 5.0, 10, 10)),
    ]


# =========================================================================
# 特征轴 → 模糊集 映射
# =========================================================================

FEATURE_FUZZY_MAP = {
    'delta_power': delta_power_sets,
    'theta_power': theta_power_sets,
    'alpha_power': alpha_power_sets,
    'sigma_power': sigma_power_sets,
    'beta_power': beta_power_sets,
    'eog_energy': eog_energy_sets,
    'emg_tone': emg_tone_sets,
    'spindle_density': spindle_density_sets,
    'sawtooth_density': sawtooth_wave_sets,
}


def fuzzify(feature_name: str, value: float) -> List[Tuple[str, float]]:
    """
    将一个特征值模糊化为 (语言标签, 隶属度) 列表
    
    参数:
        feature_name: 特征名 (如 'delta_power')
        value: 特征值
        
    返回:
        [(语言标签1, 隶属度1), (语言标签2, 隶属度2), ...]
    """
    if feature_name not in FEATURE_FUZZY_MAP:
        raise ValueError(f"Unknown feature: {feature_name}")
    
    sets = FEATURE_FUZZY_MAP[feature_name]()
    result = [(s.name, s.membership(value)) for s in sets]
    # 过滤掉隶属度为0的（可选项）
    result = [(name, mu) for name, mu in result if mu > 0]
    return result
