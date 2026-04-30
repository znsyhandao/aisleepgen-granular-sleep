import sys
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

with open(r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette\SC4001E0-PSG.edf', 'rb') as f:
    header = f.read(256)
    n_signals = int(header[252:254].decode('ascii'))
    print(f'n_signals: {n_signals}')
    
    sh = f.read(min(n_signals * 256, 512))
    
    for i in range(min(3, n_signals)):
        offset = i * 256
        print(f'\n--- Signal {i} ---')
        for j in range(0, 256, 8):
            chunk = sh[offset+j:offset+j+8]
            text = chunk.decode('ascii', errors='replace').strip()
            if text:
                print(f'  [{j:3d}] {text}')
