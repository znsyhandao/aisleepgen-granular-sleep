"""双人音频测试: 先跑10分钟验证"""
import sys, os, json
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import scipy.io.wavfile as wav

# Load 10min test
sr, data = wav.read(r'D:\AISleepGen_Optimized\sleep_record\test_10min.wav')
print(f'Loaded: {len(data)/sr/60:.0f}min @ {sr}Hz')

# Audio stats
print(f'Range: [{data.min():.4f}, {data.max():.4f}]')
print(f'RMS: {np.sqrt(np.mean(data**2)):.4f}')

# Blind source separation - try full 10min first
from sklearn.decomposition import FastICA

# Sampling - use 1/10 to fit within memory
step = 10
data_sampled = data[::step]
print(f'Sampled: {len(data_sampled)} ({len(data)/sr/60:.0f}min reduced to {len(data_sampled)/sr*step:.0f}s)')

# Time-delay embedding for single-channel ICA
delay = int(0.5 * sr / step)  # 0.5s delay
n_channels = 2
n_samples = len(data_sampled) - delay * n_channels
X = np.column_stack([data_sampled[i*delay:i*delay+n_samples] for i in range(n_channels)])

print(f'ICA input: {X.shape}')

ica = FastICA(n_components=2, random_state=42, max_iter=1000)
S = ica.fit_transform(X)

# Reconstruct to full length
# For each source, interleave back
source1 = np.zeros(len(data))
source2 = np.zeros(len(data))
s1_sampled = S[:, 0]
s2_sampled = S[:, 1]

# Resample back
import scipy.signal as ss
source1_recon = ss.resample(s1_sampled, len(data))
source2_recon = ss.resample(s2_sampled, len(data))

print(f'Separated source1 RMS: {np.sqrt(np.mean(source1_recon**2)):.4f}')
print(f'Separated source2 RMS: {np.sqrt(np.mean(source2_recon**2)):.4f}')

# Try simple alternative: frequency-based separation
# High-pass and low-pass to simulate each person's breathing
from scipy import signal as scipy_signal

# Person 1: low-frequency dominant (slower/deeper breathing)
sos1 = scipy_signal.butter(4, 200, btype='low', fs=sr, output='sos')
person1 = scipy_signal.sosfiltfilt(sos1, data)

# Person 2: high-frequency dominant (faster/shallower breathing)  
sos2 = scipy_signal.butter(4, 200, btype='high', fs=sr, output='sos')
person2 = scipy_signal.sosfiltfilt(sos2, data)

# Now run granular rules on both
print('\n--- 快速频谱分析 ---')
import sys; sys.path.insert(0, r'D:\AISleepGen_GranularSleep')
from features.spectral import compute_epoch_spectrum, band_power
EDF_BANDS = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'sigma': (11,16), 'beta': (16,30)}

for label, sig in [('Person1(Low)', person1), ('Person2(High)', person2)]:
    epoch_len = int(30 * sr)
    features_all = []
    
    for i in range(0, min(60, len(sig)//epoch_len)):
        epoch = sig[i*epoch_len:(i+1)*epoch_len]
        freqs, psd = compute_epoch_spectrum(epoch, sr)
        vec = []
        for name, band in EDF_BANDS.items():
            vec.append(np.log10(max(band_power(freqs, psd, band), 1e-15)))
        features_all.append(vec)
    
    if features_all:
        fmean = np.mean(features_all, axis=0)
        print(f'{label}: δ={fmean[0]:+.1f} θ={fmean[1]:+.1f} α={fmean[2]:+.1f} σ={fmean[3]:+.1f} β={fmean[4]:+.1f}')
    else:
        print(f'{label}: (no epochs)')

print('\nDone!')
# Cleanup
os.remove(r'D:\AISleepGen_Optimized\sleep_record\test_10min.wav')
