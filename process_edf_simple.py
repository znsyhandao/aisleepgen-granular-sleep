"""
简单EDF处理测试
"""

import os
import sys
from datetime import datetime

print("=" * 70)
print("EDF Processing Test")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# 检查EDF文件
edf_dir = "sleep-edf-database/sleep-cassette"

if not os.path.exists(edf_dir):
    print(f"Error: EDF directory not found: {edf_dir}")
    sys.exit(1)

# 列出文件
import glob
edf_files = glob.glob(os.path.join(edf_dir, "*.edf"))
hypno_files = [f for f in edf_files if "Hypnogram" in f]
psg_files = [f for f in edf_files if "PSG" in f]

print(f"Found {len(edf_files)} EDF files")
print(f"  PSG files: {len(psg_files)}")
print(f"  Hypnogram files: {len(hypno_files)}")

if len(psg_files) == 0:
    print("No PSG files found")
    sys.exit(1)

# 测试第一个文件
test_file = psg_files[0]
print(f"\nTesting file: {os.path.basename(test_file)}")
print(f"File size: {os.path.getsize(test_file) / (1024*1024):.1f} MB")

try:
    print("\nTrying to import MNE...")
    import mne
    print(f"MNE version: {mne.__version__}")
    
    # 尝试读取EDF文件
    print("\nReading EDF file...")
    raw = mne.io.read_raw_edf(test_file, preload=False, verbose=False)
    
    print(f"Success! File info:")
    print(f"  Channels: {raw.info['nchan']}")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]:.1f} seconds")
    
    # 显示通道
    print(f"\nChannel names (first 10):")
    for i, ch in enumerate(raw.ch_names[:10]):
        print(f"  {i+1}. {ch}")
    
    if len(raw.ch_names) > 10:
        print(f"  ... and {len(raw.ch_names)-10} more")
    
    # 尝试加载对应的hypnogram
    base_name = os.path.basename(test_file).replace("PSG", "Hypnogram")
    hypno_file = None
    
    for h_file in hypno_files:
        if base_name in os.path.basename(h_file):
            hypno_file = h_file
            break
    
    if hypno_file:
        print(f"\nFound hypnogram: {os.path.basename(hypno_file)}")
        
        try:
            hypno_raw = mne.io.read_raw_edf(hypno_file, preload=False, verbose=False)
            print(f"Hypnogram channels: {hypno_raw.info['nchan']}")
            print(f"Hypnogram sampling rate: {hypno_raw.info['sfreq']} Hz")
            
            # 加载数据
            hypno_data = hypno_raw.get_data()
            print(f"Hypnogram data shape: {hypno_data.shape}")
            
            if hypno_data.size > 0:
                labels = hypno_data[0]  # 假设第一个通道是标签
                unique_labels = np.unique(labels)
                print(f"Unique labels: {unique_labels}")
                print(f"Number of labels: {len(labels)}")
                
                # 睡眠阶段映射
                stage_map = {
                    0: "Wake",
                    1: "N1",
                    2: "N2",
                    3: "N3",
                    4: "REM",
                    5: "Movement",
                    6: "Unscored"
                }
                
                print("\nLabel distribution:")
                for label in unique_labels:
                    count = np.sum(labels == label)
                    percentage = count / len(labels) * 100
                    stage_name = stage_map.get(int(label), f"Unknown({label})")
                    print(f"  {stage_name}: {count} ({percentage:.1f}%)")
        
        except Exception as e:
            print(f"Error reading hypnogram: {e}")
    
    print("\n" + "=" * 70)
    print("SUCCESS: EDF files can be read!")
    print("Next steps:")
    print("1. Extract features from EDF data")
    print("2. Train model on real sleep data")
    print("3. Validate with cross-subject testing")
    print("=" * 70)
    
except ImportError:
    print("MNE not installed in current environment")
    print("Try: pip install mne")
    
except Exception as e:
    print(f"Error processing EDF: {e}")
    import traceback
    traceback.print_exc()

# 需要numpy
import numpy as np