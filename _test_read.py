import sys, time
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)
sys.path.insert(0, r'D:\AISleepGen_GranularSleep')

t0 = time.time()
from features.read_sleep_edf import read_sleep_edf

edf_dir = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'

print("Reading PSG + Hypnogram...")
eeg, sfreq, stages = read_sleep_edf(
    f'{edf_dir}/SC4001E0-PSG.edf',
    f'{edf_dir}/SC4001EC-Hypnogram.edf'
)

print(f'\nEEG: {len(eeg)} samples @ {sfreq:.0f}Hz = {len(eeg)/sfreq/3600:.1f} hours')
print(f'Stages: {len(stages)} epochs = {len(stages)*30/60:.0f} min')

if stages:
    unique, counts = __import__('numpy').unique(stages, return_counts=True)
    for s, c in zip(unique, counts):
        print(f'  Stage {s}: {c} epochs ({c*30/60:.1f} min)')

print(f'\nTotal: {time.time()-t0:.2f}s')
