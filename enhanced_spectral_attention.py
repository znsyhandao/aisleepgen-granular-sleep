"""
增强版SpectralAttention - 实现时频融合和多头注意力机制
基于现有架构的渐进式升级，保持向后兼容性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class TimeDomainEncoder(nn.Module):
    """时域特征编码器"""
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        features = self.conv_layers(x)  # (batch_size, hidden_dim, 1)
        features = features.squeeze(-1)  # (batch_size, hidden_dim)
        return features


class FrequencyDomainEncoder(nn.Module):
    """频域特征编码器 - 基于STFT的频域分析"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_fft: int = 64):
        super().__init__()
        self.n_fft = n_fft
        self.hidden_dim = hidden_dim
        
        # 频带特征提取
        self.freq_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(64, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        # 对每个通道进行STFT
        stft_features = []
        for i in range(input_dim):
            channel_data = x[:, :, i]  # (batch_size, seq_len)
            
            # 计算STFT
            stft = torch.stft(
                channel_data, 
                n_fft=self.n_fft, 
                hop_length=self.n_fft//2, 
                return_complex=True
            )
            
            # 计算幅度谱
            magnitude = torch.abs(stft)  # (batch_size, freq_bins, time_frames)
            stft_features.append(magnitude.unsqueeze(1))  # 添加通道维度
        
        # 合并所有通道
        stft_combined = torch.cat(stft_features, dim=1)  # (batch_size, input_dim, freq_bins, time_frames)
        
        # 通过卷积网络提取特征
        features = self.freq_conv(stft_combined.unsqueeze(1))  # 添加批次维度
        features = features.view(batch_size, -1)
        features = self.fc(features)
        
        return features


class TimeFrequencyFusion(nn.Module):
    """时频融合模块 - 核心创新点"""
    def __init__(self, time_dim: int, freq_dim: int, hidden_dim: int):
        super().__init__()
        
        # 门控融合机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(time_dim + freq_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 特征转换
        self.time_transform = nn.Linear(time_dim, hidden_dim)
        self.freq_transform = nn.Linear(freq_dim, hidden_dim)
        
        # 残差连接
        self.residual = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, time_features: torch.Tensor, freq_features: torch.Tensor) -> torch.Tensor:
        # 特征转换
        time_transformed = self.time_transform(time_features)
        freq_transformed = self.freq_transform(freq_features)
        
        # 计算融合权重
        concatenated = torch.cat([time_features, freq_features], dim=-1)
        gate_weights = self.fusion_gate(concatenated)
        
        # 加权融合
        fused_features = gate_weights * time_transformed + (1 - gate_weights) * freq_transformed
        
        # 残差连接
        residual = self.residual(fused_features)
        fused_features = fused_features + residual
        
        return fused_features


class EnhancedSpectralAttention(nn.Module):
    """
    增强版SpectralAttention - 实现时频融合和多头注意力
    
    核心创新：
    1. 时频域深度融合（非简单拼接）
    2. 多头注意力机制
    3. 残差连接和层归一化
    4. 向后兼容现有接口
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 时域编码器
        self.time_encoder = TimeDomainEncoder(input_dim, hidden_dim)
        
        # 频域编码器
        self.freq_encoder = FrequencyDomainEncoder(input_dim, hidden_dim)
        
        # 时频融合模块
        self.fusion_module = TimeFrequencyFusion(hidden_dim, hidden_dim, hidden_dim)
        
        # 多头注意力机制
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 输出投影（保持与原始维度兼容）
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            attention_mask: 注意力掩码
            
        Returns:
            torch.Tensor: 输出张量 (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 时域特征提取
        time_features = self.time_encoder(x)  # (batch_size, hidden_dim)
        
        # 频域特征提取
        freq_features = self.freq_encoder(x)  # (batch_size, hidden_dim)
        
        # 时频融合
        fused_features = self.fusion_module(time_features, freq_features)  # (batch_size, hidden_dim)
        
        # 扩展为序列长度（为了与多头注意力兼容）
        fused_features_expanded = fused_features.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)
        
        # 多头注意力
        attended_features, attention_weights = self.multihead_attention(
            query=fused_features_expanded,
            key=fused_features_expanded,
            value=fused_features_expanded,
            attn_mask=attention_mask,
            need_weights=True
        )
        
        # 残差连接和层归一化
        attended_features = self.layer_norm1(attended_features + fused_features_expanded)
        
        # 前馈网络
        ff_output = self.feed_forward(attended_features)
        
        # 残差连接和层归一化
        output = self.layer_norm2(ff_output + attended_features)
        
        # 输出投影（保持维度兼容）
        output = self.output_projection(output)
        
        return output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取注意力权重（用于可视化）"""
        if hasattr(self.multihead_attention, 'attention_weights'):
            return self.multihead_attention.attention_weights
        return None


# 兼容性包装器 - 确保现有代码无需修改
class SpectralAttention(EnhancedSpectralAttention):
    """
    向后兼容的SpectralAttention类
    保持与现有代码的接口一致性
    """
    def __init__(self, input_dim: int, **kwargs):
        # 设置合理的默认参数
        default_kwargs = {
            'hidden_dim': 256,
            'num_heads': 8,
            'dropout': 0.1
        }
        default_kwargs.update(kwargs)
        
        super().__init__(input_dim=input_dim, **default_kwargs)


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size, seq_len, input_dim = 32, 100, 64
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    # 测试增强版SpectralAttention
    model = EnhancedSpectralAttention(input_dim=input_dim)
    output = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试兼容性
    compatible_model = SpectralAttention(input_dim=input_dim)
    compatible_output = compatible_model(test_input)
    print(f"兼容版本输出形状: {compatible_output.shape}")
    
    print("[OK] 增强版SpectralAttention实现成功！")