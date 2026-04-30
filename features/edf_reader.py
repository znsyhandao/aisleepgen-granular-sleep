"""
轻量级 EDF 读取器（零依赖）
不需要 mne/pyedflib，纯 Python struct 解析

EDF 格式: https://www.edfplus.info/specs/edf.html
"""

import struct
import numpy as np
from typing import Dict, List, Tuple, Optional


class EDFSignal:
    """单通道 EDF 信号"""
    def __init__(self, label: str, data: np.ndarray, sfreq: float,
                 physical_min: float, physical_max: float,
                 digital_min: int, digital_max: int):
        self.label = label
        self.data = data
        self.sfreq = sfreq
        self.physical_min = physical_min
        self.physical_max = physical_max
        self.digital_min = digital_min
        self.digital_max = digital_max


class EDFReader:
    """轻量 EDF 读取器"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.signals: List[EDFSignal] = []
        self._parse()
    
    def _parse(self):
        with open(self.filepath, 'rb') as f:
            header = f.read(256)
            
            # Parse header (ASCII 固定长度字段)
            self.patient_id = header[:80].decode('ascii', errors='replace').strip()
            self.record_info = header[80:160].decode('ascii', errors='replace').strip()
            self.start_date = header[168:176].decode('ascii', errors='replace')
            self.start_time = header[176:184].decode('ascii', errors='replace')
            
            self.header_length = struct.unpack('<H', header[184:186])[0]
            reserved = header[186:192]
            self.n_records = struct.unpack('<H', header[236:238])[0]
            self.record_duration = struct.unpack('<H', header[244:246])[0] / 1e7  # in seconds
            
            self.n_signals = struct.unpack('<H', header[252:254])[0]
            
            # Read signal headers
            signal_headers_raw = f.read(self.n_signals * 256)
            
            self.labels = []
            self.physical_dims = []
            self.physical_mins = []
            self.physical_maxs = []
            self.digital_mins = []
            self.digital_maxs = []
            self.n_samples_per_record = []
            
            for i in range(self.n_signals):
                offset = i * 256
                sh = signal_headers_raw[offset:offset+256]
                
                label = sh[:16].decode('ascii', errors='replace').strip()
                phys_dim = sh[80:88].decode('ascii', errors='replace').strip()
                
                # EDF 字段是固定 8 字符，但有些文件有额外 meta 字段偏移
                # 安全解析：尝试读取固定位置，失败则用默认值
                try:
                    phys_min = float(sh[88:96].decode('ascii', errors='replace').strip()[:8])
                except:
                    phys_min = -1000.0
                try:
                    phys_max = float(sh[96:104].decode('ascii', errors='replace').strip()[:8])
                except:
                    phys_max = 1000.0
                try:
                    dig_min = int(sh[104:112].decode('ascii', errors='replace').strip()[:8])
                except:
                    dig_min = -32768
                try:
                    dig_max = int(sh[112:120].decode('ascii', errors='replace').strip()[:8])
                except:
                    dig_max = 32767
                
                try:
                    n_samples = int(sh[184:192].decode('ascii', errors='replace').strip()[:8])
                except:
                    n_samples = 1
                
                self.labels.append(label)
                self.physical_dims.append(phys_dim)
                self.physical_mins.append(phys_min)
                self.physical_maxs.append(phys_max)
                self.digital_mins.append(dig_min)
                self.digital_maxs.append(dig_max)
                self.n_samples_per_record.append(n_samples)
            
            # Read signal data
            self._read_signal_data(f)
    
    def _read_signal_data(self, f):
        """读取所有记录（数据块）"""
        total_samples_per_signal = [0] * self.n_signals
        all_data = [[] for _ in range(self.n_signals)]
        
        for record_idx in range(self.n_records if self.n_records > 0 else 1):
            for sig_idx in range(self.n_signals):
                n = self.n_samples_per_record[sig_idx]
                raw_bytes = f.read(n * 2)  # 每个样本2字节 (16-bit integer)
                if len(raw_bytes) < n * 2:
                    break
                
                # 解析数字信号
                dig_data = np.frombuffer(raw_bytes, dtype=np.int16)
                
                # 转换为物理值
                phys_min = self.physical_mins[sig_idx]
                phys_max = self.physical_maxs[sig_idx]
                dig_min = self.digital_mins[sig_idx]
                dig_max = self.digital_maxs[sig_idx]
                
                if dig_max != dig_min:
                    phys_data = phys_min + (dig_data.astype(float) - dig_min) * (phys_max - phys_min) / (dig_max - dig_min)
                else:
                    phys_data = dig_data.astype(float)
                
                all_data[sig_idx].append(phys_data)
        
        # 构建信号对象
        for sig_idx in range(self.n_signals):
            data = np.concatenate(all_data[sig_idx])
            sfreq = self.n_samples_per_record[sig_idx] / self.record_duration if self.record_duration > 0 else 1.0
            
            sig = EDFSignal(
                label=self.labels[sig_idx],
                data=data,
                sfreq=sfreq,
                physical_min=self.physical_mins[sig_idx],
                physical_max=self.physical_maxs[sig_idx],
                digital_min=self.digital_mins[sig_idx],
                digital_max=self.digital_maxs[sig_idx],
            )
            self.signals.append(sig)
    
    def get_signal(self, name_substring: str) -> Optional[EDFSignal]:
        """通过名称子串获取信号"""
        for sig in self.signals:
            if name_substring.lower() in sig.label.lower():
                return sig
        return None
    
    def get_eeg_signals(self) -> List[EDFSignal]:
        """获取所有 EEG 信号"""
        eeg_keywords = ['eeg', 'fpz', 'pz', 'cz', 'oz', 'c3', 'c4', 'f3', 'f4']
        result = []
        for sig in self.signals:
            label_lower = sig.label.lower()
            if any(kw in label_lower for kw in eeg_keywords):
                result.append(sig)
        return result
    
    def summary(self) -> Dict:
        return {
            'file': self.filepath,
            'patient': self.patient_id,
            'n_signals': self.n_signals,
            'n_records': self.n_records,
            'record_duration_s': self.record_duration,
            'signals': [(s.label, s.sfreq, len(s.data)) for s in self.signals],
        }
