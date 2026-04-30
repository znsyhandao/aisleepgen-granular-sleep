import sys, numpy as np
sys.stdout = open(1,'w',encoding='utf-8',closefd=False)
from features.spindle import detect_spindles, epoch_spindle_density
sfreq = 100
t = np.arange(0, 30, 1/sfreq)
eeg = 0.5 * np.random.randn(len(t))
spindle = 2.0 * np.sin(2*np.pi*13*t)
idx = (t >= 10) & (t < 11)
eeg[idx] += spindle[idx]
n, density, features = detect_spindles(eeg, sfreq)
print(f'Spindles detected: {n}, density per sec: {density:.4f}')
for f in features:
    dur = f['duration']; freq = f['frequency']; tp = f['type']
    print(f'  dur={dur:.3f}s freq={freq:.1f}Hz type={tp}')
d, c = epoch_spindle_density(eeg, sfreq)
print(f'Epoch density: {d:.4f}/s, count={c}')
print('TEST PASSED')
