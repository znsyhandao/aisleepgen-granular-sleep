"""
双人睡眠 v4 — 分段读取, 8kHz wav
"""
import sys, os, json, struct
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from scipy import signal as scipy_signal

GRANULAR_DIR = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, GRANULAR_DIR)
from features.spectral import compute_epoch_spectrum, band_power

STAGES = ['W','N1','N2','N3','REM']
COEFF_MATRIX = np.array([
    [3.35,0.12,0.45,-2.21,8.66,1.23,2.23],
    [-1.06,-1.58,0.87,1.22,2.28,0.45,-0.89],
    [1.42,-0.67,-1.23,3.13,-4.07,2.05,-1.56],
    [2.51,-1.89,-0.56,-1.45,-4.27,3.82,-5.39],
    [-0.34,-1.12,1.89,-2.68,-2.61,-3.88,1.45],
])
BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'sigma':(11,16),'beta':(16,30)}
SM = np.array([-11.2,-10.8,-11.5,-12.1,-10.3,0.23,0.45])
SS = np.array([0.58,0.62,0.55,0.71,0.49,0.15,0.21])

def feat(epoch, sr):
    fr, ps = compute_epoch_spectrum(epoch, sr)
    f = np.zeros(7)
    for i,(n,b) in enumerate(BANDS.items()):
        f[i] = np.log10(max(band_power(fr,ps,b), 1e-15))
    f[5]=f[0]-f[1]; f[6]=f[2]-f[0]
    return f

def predict(f):
    fz=(f-SM)/SS; sc=fz@COEFF_MATRIX.T
    sc-=sc.max(); ex=np.exp(sc); pr=ex/ex.sum()
    return STAGES[np.argmax(pr)], float(pr.max())

def qscore(preds):
    n=len(preds); tm=n*0.5
    sm={s:np.sum(np.array(preds)==s)*0.5 for s in STAGES}
    slp=tm-sm['W']; eff=slp/tm*100 if tm>0 else 0
    se=min(30,max(0,(eff-50)/50*30))
    sn=min(20,max(0,20-abs(sm['N3']/tm*100-20)*2))
    sr_=min(20,max(0,20-abs(sm['REM']/tm*100-22)*1.5))
    sw=min(15,max(0,15-sm['W']*0.3))
    total=min(100,se+sn+sr_+sw+10)
    g='优秀' if total>=85 else '良好' if total>=70 else '一般' if total>=50 else '需要关注'
    return {'score':round(total,1),'grade':g,'eff':round(eff,1),
            'n3':round(sm['N3']/tm*100,1),'rem':round(sm['REM']/tm*100,1),
            'wake':round(sm['W'],1),'total_min':round(tm,1)}

def read_wav_segments(path, seg_samples=288000):  # 10min @ 8kHz = 4.8M
    """分段读取wav文件"""
    with open(path, 'rb') as f:
        hdr=f.read(44)
        sr=struct.unpack_from('<I',hdr,24)[0]
        
        f.seek(44)
        data=f.read()
        samples=np.frombuffer(data,dtype=np.int16).astype(np.float32)/32768.0
    
    print(f'Sr={sr}Hz total={len(samples)/sr/3600:.1f}h', flush=True)
    
    n_seg=len(samples)//seg_samples+1
    for i in range(n_seg):
        start=i*seg_samples
        end=min((i+1)*seg_samples, len(samples))
        if start>=len(samples): break
        yield samples[start:end], sr

def analyze(filepath):
    print('='*60, flush=True)
    print('双人睡眠分析 v4', flush=True)
    print(f'{os.path.basename(filepath)}',flush=True)
    print('='*60,flush=True)
    
    tmp=filepath.replace('.m4a','_tmp8k.wav')
    if not os.path.exists(tmp):
        import subprocess
        print('  ffmpeg转码...',flush=True)
        subprocess.run([r'D:\ffmpeg\bin\ffmpeg','-y','-i',filepath,
            '-acodec','pcm_s16le','-ar','8000','-ac','1',tmp],capture_output=True)
    
    all_feat=[]
    seg_idx=0
    for seg, sr in read_wav_segments(tmp):
        fl=int(30*sr)
        ne=len(seg)//fl
        if ne==0: continue
        print(f'  段{seg_idx}: {len(seg)/sr/60:.0f}min ({ne} epochs)...',flush=True)
        for i in range(ne):
            ep=seg[i*fl:(i+1)*fl]
            if len(ep)<fl: continue
            all_feat.append(feat(ep, sr))
        seg_idx+=1
    
    F=np.array(all_feat)
    print(f'\n总epochs: {len(F)} ({len(F)*30//3600}h)',flush=True)
    
    for j,n in enumerate(BANDS):
        v=F[:,j]
        print(f'  {n}: [{v.min():.2f},{v.max():.2f}] μ={v.mean():.2f}',flush=True)
    
    # 粒规则预测
    preds=[predict(F[i]) for i in range(len(F))]
    
    # 分离: 用beta/delta比做能量分频
    # beta高=近mic(男方动作/人声), beta低=远mic(女方)
    low_energy=F[:,0]+F[:,1]  # delta+theta 低频
    high_energy=F[:,4]        # beta 高频
    ratio=high_energy-low_energy
    med=np.median(ratio)
    
    m_idx=ratio>med
    f_idx=~m_idx
    
    qm=qscore([p[0] for i,p in enumerate(preds) if m_idx[i]])
    qf=qscore([p[0] for i,p in enumerate(preds) if f_idx[i]])
    
    print(f'\n分离阈值: β-(δ+θ)中位={med:.2f}',flush=True)
    print(f'  男方(近mic,高频多): {m_idx.sum()} epochs',flush=True)
    print(f'  女方(远mic,低频多): {f_idx.sum()} epochs',flush=True)
    
    for label, idx, q in [('🧑 男方(录音侧)', m_idx, qm), ('👩 女方', f_idx, qf)]:
        sub=[p[0] for i,p in enumerate(preds) if idx[i]]
        print(f'\n{label}:',flush=True)
        print(f'  评分: {q["score"]:.0f} ({q["grade"]})',flush=True)
        print(f'  效率: {q["eff"]:.0f}%  深睡: {q["n3"]:.0f}%  REM: {q["rem"]:.0f}%  觉醒: {q["wake"]:.0f}min',flush=True)
        for s in STAGES:
            cnt=sub.count(s)
            print(f'  {s}: {cnt*0.5:.0f}min ({cnt/len(sub)*100:.0f}%)',flush=True)
    
    # save
    out=os.path.join(GRANULAR_DIR,'results','couple_sleep_v4.json')
    with open(out,'w') as f:
        json.dump({'male':qm,'female':qf},f,ensure_ascii=False,indent=2)
    print(f'\n保存: {out}',flush=True)
    
    # cleanup
    if os.path.exists(tmp): os.remove(tmp)

if __name__=='__main__':
    analyze(r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a')
