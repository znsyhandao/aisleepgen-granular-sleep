import sys, traceback
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)
sys.path.insert(0, r'D:\AISleepGen_GranularSleep')

try:
    from features.edf_reader import EDFReader
    print("Import OK")
    
    # 先读小的 Hypnogram
    print("Reading hypnogram...")
    hypno = EDFReader(r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette\SC4001EC-Hypnogram.edf')
    for sig in hypno.signals:
        print(f'  Signal: {sig.label} len={len(sig.data)}')
        print(f'  Data[:30]: {sig.data[:30]}')
    print("Hypnogram OK")
    
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc(file=sys.stdout)
