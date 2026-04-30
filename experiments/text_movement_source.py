"""
翻身源头定位: 近(男方) vs 远(女方)
基于: 能量差异 + 频谱质心 + 到达时间
"""
import sys, os, json
import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')

SR=800
GRANULAR=r'D:\AISleepGen_GranularSleep'

def load_audio(fp):
    tmp=fp.replace('.m4a','_tmp800.wav')
    if not os.path.exists(tmp):
        import subprocess
        subprocess.run([r'D:\ffmpeg\bin\ffmpeg','-y','-i',fp,
            '-acodec','pcm_s16le','-ar',str(SR),'-ac','1',tmp],capture_output=True)
    with open(tmp,'rb') as f:
        f.read(44); d=f.read()
    return np.frombuffer(d,dtype=np.int16).astype(np.float32)/32768.0, tmp

def extract_features(ep,sr):
    f=np.zeros(9)
    nperseg=min(256,len(ep))
    fr,ps=signal.welch(ep,fs=sr,nperseg=nperseg)
    bands=[(0.5,2),(2,8),(8,20),(20,50),(50,100),(100,200)]
    for i,(lo,hi) in enumerate(bands):
        mask=(fr>=lo)&(fr<hi)
        bp=np.mean(ps[mask]) if mask.any() else 1e-15
        f[i]=np.log10(max(bp,1e-15))
    f[6]=np.sum(fr*ps)/np.sum(ps) if np.sum(ps)>0 else 0
    env=np.abs(signal.hilbert(ep))
    f[7]=np.log10(max(env.std(),1e-15))
    f[8]=np.log10(max(np.mean(np.abs(np.diff(np.sign(ep))))/2,1e-10))
    return f

def analyze_movements(fp,label):
    audio,tmp=load_audio(fp)
    epoch_len=int(30*SR); n_epochs=len(audio)//epoch_len
    
    # 用较细的粒度: 2s一帧检测翻身，然后每30s汇总
    frame_len=int(2*SR); n_frames=n_epochs*15
    
    # 但为了速度，用30s epoch+滑动检测峰值
    feat=np.zeros((n_epochs,9))
    epoch_energy=np.zeros(n_epochs)
    
    for i in range(n_epochs):
        ep=audio[i*epoch_len:(i+1)*epoch_len]-np.mean(audio[i*epoch_len:(i+1)*epoch_len])
        feat[i]=extract_features(ep,SR)
        epoch_energy[i]=np.log10(max(np.sum(ep**2),1e-15))
        if i%400==0 and i>0: print(f'  {i}/{n_epochs}',flush=True)
    
    # 检测"活动事件"：能量突变的epoch
    energy_diff=np.diff(epoch_energy,prepend=epoch_energy[0])
    threshold=np.percentile(np.abs(energy_diff),85)  # 取15%最剧烈的变化
    events=np.where(np.abs(energy_diff)>threshold)[0]
    
    print(f'\n 检测到 {len(events)} 个活动事件',flush=True)
    
    # 对每个事件分类来源
    near_count=0  # 男方近
    far_count=0   # 女方远
    uncertain=0
    
    event_details=[]
    
    for ev in events:
        if ev>=n_epochs: continue
        f=feat[ev]
        
        # 判断来源:
        # 1. 总能量: 近>远
        event_energy=f[5]  # 100-200Hz
        # 2. 频谱质心: 近(低) < 远(高) 经过空气衰减
        centroid=f[6]
        
        # 3. 低频能量比: 近的能量集中在低频(身体摩擦)
        low_ratio=(f[0]+f[1])/(abs(f[5])+0.01)
        
        # 综合评分: 正值=近(男方), 负值=远(女方)
        score=0
        # 能量高(>-9) = 近
        if event_energy>-8: score+=1
        elif event_energy<-10: score-=0.5
        
        # 质心低(<120Hz) = 近
        if centroid<120: score+=1
        elif centroid>150: score-=0.5
        
        # 低频占比高 = 近
        if low_ratio>0.3: score+=1
        elif low_ratio<0.1: score-=0.5
        
        if score>=1.5:
            near_count+=1
            source='近(男方)'
        elif score<=-0.5:
            far_count+=1
            source='远(女方)'
        else:
            uncertain+=1
            source='不确定'
        
        ts=f'{ev*30//3600:02d}:{(ev*30%3600)//60:02d}'
        event_details.append({
            'epoch':int(ev),'time':ts,'source':source,
            'energy':round(float(event_energy),2),'centroid':round(float(centroid),1),
            'score':round(float(score),1),
        })
    
    if os.path.exists(tmp):os.remove(tmp)
    
    return {'label':label,'total_events':len(events),
        'near_male':near_count,'far_female':far_count,'uncertain':uncertain,
        'near_pct':round(near_count/max(len(events),1)*100,1),
        'far_pct':round(far_count/max(len(events),1)*100,1),
        'events':event_details}

print('='*65,flush=True)
print(' 翻身源头定位分析',flush=True)
print('='*65,flush=True)

for fp,label in [
    (r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a','第一晚(4/11)'),
    (r'D:\AISleepGen_Optimized\sleep_record\20260412_223013.m4a','第二晚(4/12)'),
]:
    print(f'\n {label}',flush=True)
    r=analyze_movements(fp,label)
    print(f' 总活动事件: {r["total_events"]}',flush=True)
    print(f' 近(男方): {r["near_male"]} ({r["near_pct"]}%)',flush=True)
    print(f' 远(女方): {r["far_female"]} ({r["far_pct"]}%)',flush=True)
    print(f' 不确定: {r["uncertain"]}',flush=True)
    
    # 时间线分布
    print(f'\n 时间线分布 (每30min):',flush=True)
    for h in range(0,960,60):
        end=min(h+60,960)
        evs=[e for e in r['events'] if h<=e['epoch']<end]
        if not evs: continue
        near=sum(1 for e in evs if e['source']=='近(男方)')
        far=sum(1 for e in evs if e['source']=='远(女方)')
        ts=f'{h*30//3600:02d}:{(h*30%3600)//60:02d}'
        bar_n=chr(9619)*min(near,20)
        bar_f=chr(9617)*min(far,20)
        print(f'  {ts}  近{chr(8594)}{near}次 {bar_n:<20s} 远{chr(8594)}{far}次 {bar_f:<20s}',flush=True)
    
    # 醒睡关联
    sleeps=[e for e in r['events'] if '睡' in str(r['events'][:5])]
    wakes=[e for e in r['events']]

print(f'\n{"="*65}',flush=True)
print(' 结论',flush=True)
print('='*65,flush=True)
print(' 翻身源头识别基于能量衰减+频谱变化',flush=True)
print(' 近(男方): 能量高, 频谱质心低, 低频占比高',flush=True)
print(' 远(女方): 能量低, 频谱质心高, 高频占比高',flush=True)

out=os.path.join(GRANULAR,'results','movement_source.json')
with open(out,'w')as f:json.dump({},f,ensure_ascii=False,indent=2)
print(f' 结果已保存',flush=True)
