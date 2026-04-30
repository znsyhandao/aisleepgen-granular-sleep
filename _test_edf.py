import sys, traceback
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)
sys.path.insert(0, r'D:\AISleepGen_GranularSleep')

from features.edf_reader import EDFReader

try:
    print("Reading PSG (SC4001E0-PSG.edf)...")
    reader = EDFReader(r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette\SC4001E0-PSG.edf')
    s = reader.summary()
    print('Signals:', len(s['signals']))
    for label, sfreq, n in s['signals'][:10]:
        print(f'  [{label:20s}] {sfreq:6.1f}Hz  {n} samples')
    
    eeg = reader.get_eeg_signals()
    print(f'\nEEG signals: {len(eeg)}')
    for sig in eeg[:3]:
        print(f'  {sig.label:20s} {sig.sfreq:.1f}Hz {sig.data.shape} mean={sig.data.mean():.2f}')
    
    # Test a quick FFT on first 30s of first EEG
    if eeg:
        sig = eeg[0]
        epoch = sig.data[:int(30 * sig.sfreq)]
        from scipy import signal
        freqs, psd = signal.welch(epoch, fs=sig.sfreq, nperseg=256)
        print(f'\nFFT test: {len(epoch)} samples -> {len(freqs)} freq bins')
        print(f'  Delta power (0.5-4Hz): {float(__import__("numpy").trapz(psd[(freqs>=0.5)&(freqs<=4)], freqs[(freqs>=0.5)&(freqs<=4)])):.4f}')
        
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc(file=sys.stdout)
