import sys, struct, os
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)

edf_dir = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'

with open(f'{edf_dir}/SC4001E0-PSG.edf', 'rb') as f:
    h = f.read(256)
    
    # n_records is ASCII right-justified 8 chars
    n_rec_str = h[236:244].decode('ascii')
    print(f'n_records raw: [{repr(n_rec_str)}]')
    try:
        n_records = int(n_rec_str.strip())
    except:
        n_records = -1
    print(f'n_records parsed: {n_records}')
    
    dur_str = h[244:252].decode('ascii')
    dur = float(dur_str.strip())
    print(f'duration: {dur}')
    
    n_sig = int(h[252:254])
    hdr_len = int(h[184:186])
    print(f'n_signals: {n_sig}, header_len: {hdr_len}')
    
    sig_hdrs = f.read(n_sig * 256)
    
    for i in range(n_sig):
        sh = sig_hdrs[i*256:(i+1)*256]
        label = sh[:16].decode('ascii', errors='replace').strip()
        n_samp_str = sh[184:192].decode('ascii')
        n_samp = int(n_samp_str.strip())
        phys_min = float(sh[88:96].decode('ascii').strip())
        phys_max = float(sh[96:104].decode('ascii').strip())
        dig_min = int(sh[104:112].decode('ascii').strip())
        dig_max = int(sh[112:120].decode('ascii').strip())
        print(f'  Sig {i}: {label:20s} {n_samp:5d} samp  phys=[{phys_min},{phys_max}]  dig=[{dig_min},{dig_max}]')
    
    # 计算总记录数
    bytes_per_rec = sum(int(sh[184:192].decode('ascii').strip()) * 2 for i in range(n_sig) 
                       for sh in [sig_hdrs[i*256:(i+1)*256]])
    data_bytes = os.path.getsize(f'{edf_dir}/SC4001E0-PSG.edf') - hdr_len
    actual_records = data_bytes // bytes_per_rec if bytes_per_rec > 0 else 0
    print(f'\nbytes/record: {bytes_per_rec}')
    print(f'data bytes: {data_bytes}')
    print(f'actual records: {actual_records}')
    samp_per_chan = int(sig_hdrs[0*256+184:0*256+192].decode("ascii").strip())
    print(f'total samples/channel: {actual_records * samp_per_chan}')
