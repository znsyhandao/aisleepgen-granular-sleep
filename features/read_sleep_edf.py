"""超轻量 EDF 读取器 — 专为 Sleep Cassette 数据集优化
跳过复杂 header，直接读 raw int16，自己算缩放因子
"""

import struct
import numpy as np
from typing import List, Optional


def read_sleep_edf(psg_path, hypno_path=None):
    """
    读取 Sleep Cassette EDF 文件
    返回: (eeg_data, sfreq, stages) 
      - eeg_data: 完整 EEG 信号 (n_samples,)
      - sfreq: 采样率 (Hz)
      - stages: 睡眠阶段列表 (n_epochs,)
    """
    # === 读取 PSG (波形数据) ===
    with open(psg_path, 'rb') as f:
        header = f.read(256)
        
        # 关键字段
        n_signals = int(header[252:254].decode('ascii'))
        header_len = int(header[184:186].decode('ascii'))
        record_duration = float(header[244:252].decode('ascii').strip())  # 通常是 30s
        
        # 读取信号头
        sig_headers = f.read(n_signals * 256)
        
        labels = []
        samples_per_record = []
        dig_min_list = []
        dig_max_list = []
        phys_min_list = []
        phys_max_list = []
        
        for i in range(n_signals):
            sh = sig_headers[i*256:(i+1)*256]
            labels.append(sh[:16].decode('ascii', errors='replace').strip())
            
            label_lower = labels[-1].lower()
            # 跳过注释信号和非波形信号
            if 'annotation' in label_lower:
                continue
            
            try:
                sps_str = sh[184:192].decode('ascii', errors='replace').strip()
                if not sps_str or not sps_str.replace('-','').replace('.','').isdigit():
                    continue
                sps = int(sps_str)
                if sps <= 0:
                    continue
            except:
                continue
            
            try: pmin = float(sh[88:96].decode('ascii', errors='replace').strip())
            except: pmin = -1000.0
            try: pmax = float(sh[96:104].decode('ascii', errors='replace').strip())
            except: pmax = 1000.0
            try: dmin = int(sh[104:112].decode('ascii', errors='replace').strip())
            except: dmin = -32768
            try: dmax = int(sh[112:120].decode('ascii', errors='replace').strip())
            except: dmax = 32767
            
            phys_min_list.append(pmin)
            phys_max_list.append(pmax)
            dig_min_list.append(dmin)
            dig_max_list.append(dmax)
            samples_per_record.append(sps)
        
        # 确定数据块大小
        record_bytes = sum(s * 2 for s in samples_per_record)
        file_size = __import__('os').path.getsize(psg_path)
        data_bytes = file_size - header_len
        n_records = data_bytes // record_bytes if record_bytes > 0 else 0
        
        # 读取全部信号数据
        all_signals = [[] for _ in range(n_signals)]
        for rec in range(min(n_records, 5000)):  # 上限防止内存爆
            for sig_idx in range(n_signals):
                n = samples_per_record[sig_idx]
                raw = f.read(n * 2)
                if len(raw) < n * 2:
                    break
                
                label = labels[sig_idx].lower()
                if 'annotation' in label or 'edf annotations' in label:
                    all_signals[sig_idx].append(np.array([0.0]))
                    continue
                
                dig = np.frombuffer(raw, dtype=np.int16)
                pmax = phys_max_list[sig_idx]
                pmin = phys_min_list[sig_idx]
                dmax = dig_max_list[sig_idx]
                dmin = dig_min_list[sig_idx]
                
                if dmax != dmin:
                    phys = pmin + (dig.astype(float) - dmin) * (pmax - pmin) / (dmax - dmin)
                else:
                    phys = dig.astype(float)
                all_signals[sig_idx].append(phys)
        
        # 合并信号
        signals = []
        for sig_idx in range(n_signals):
            if all_signals[sig_idx]:
                try:
                    data = np.concatenate(all_signals[sig_idx])
                except:
                    data = np.array(all_signals[sig_idx][0])
                sfreq = samples_per_record[sig_idx] / record_duration
                signals.append({
                    'label': labels[sig_idx],
                    'data': data,
                    'sfreq': sfreq,
                })
    
    # === 找到第一个 EEG 信号 ===
    eeg_keywords = ['eeg', 'fpz', 'pz', 'cz', 'oz', 'c3', 'c4']
    eeg_sig = None
    for s in signals:
        label_lower = s['label'].lower()
        if any(kw in label_lower for kw in eeg_keywords):
            eeg_sig = s
            break
    if eeg_sig is None and signals:
        eeg_sig = signals[0]  # fallback
    
    sfreq = eeg_sig['sfreq']
    eeg_data = eeg_sig['data']
    
    # === 读取 Hypnogram (睡眠阶段标注) ===
    stages = None
    if hypno_path:
        try:
            stages = _parse_hypnogram(hypno_path, sfreq, len(eeg_data))
        except Exception as e:
            print(f'Hypnogram parse warning: {e}')
    
    return eeg_data, sfreq, stages


def _parse_hypnogram(hypno_path, sfreq, n_samples):
    """解析 Hypnogram EDF 文件为阶段标签列表"""
    stages = []
    with open(hypno_path, 'rb') as f:
        header = f.read(256)
        n_signals = int(header[252:254].decode('ascii'))
        hdr_len = int(header[184:186].decode('ascii'))
        
        # Skip to annotations
        f.seek(hdr_len + n_signals * 256)
        
        # Parse TAL (Timestamped Annotation List)
        raw = f.read(50000)  # Hypnogram 很小
        
        text = raw.decode('ascii', errors='replace')
        
        # Sleep Cassette 格式: +N<epoch_dur> Stage
        # 例如: "+30 Sleep stage 2"
        stage_map = {'W': 'W', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'N3', 'R': 'REM', '?': '?'}
        
        annotations = []
        i = 0
        while i < len(text):
            if text[i] == '+':
                # 找结束符
                j = text.find('\x00', i)
                if j < 0: j = text.find('\n', i)
                if j < 0: j = min(i + 100, len(text))
                record = text[i:j].strip('\x00').strip()
                
                # 解析: +{onset} {description}
                parts = record.split(None, 1)
                if len(parts) >= 1:
                    try:
                        onset = float(parts[0]) if parts[0].replace('.','').isdigit() else 0
                    except:
                        onset = 0
                    desc = parts[-1] if len(parts) > 1 else ''
                    
                    if desc:
                        stage_code = desc[-1].upper()  # 最后一个字母是阶段码
                        stage = stage_map.get(stage_code, '?')
                        if stage != '?':
                            epoch_idx = int(round(onset / 30))
                            stages.append((epoch_idx, stage))
                
                i = j + 1
            else:
                i += 1
    
    # 转换为 epoch 标签列表
    if not stages:
        return None
    
    max_epoch = max(s[0] for s in stages) + 1
    stage_labels = ['?'] * max_epoch
    for epoch_idx, stage in stages:
        if epoch_idx < max_epoch:
            stage_labels[epoch_idx] = stage
    
    return stage_labels
