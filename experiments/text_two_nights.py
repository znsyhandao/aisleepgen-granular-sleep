"""两晚睡眠分析对比"""
import sys, os, json
import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')

SR=800
GRANULAR=r'D:\AISleepGen_GranularSleep'

def load_audio(filepath):
    tmp = filepath.replace('.m4a','_tmp800.wav')
    if not os.path.exists(tmp):
        import subprocess
        subprocess.run([r'D:\ffmpeg\bin\ffmpeg','-y','-i',filepath,
            '-acodec','pcm_s16le','-ar',str(SR),'-ac','1',tmp],capture_output=True)
    with open(tmp,'rb') as f:
        f.read(44); data=f.read()
    return np.frombuffer(data,dtype=np.int16).astype(np.float32)/32768.0, tmp

def extract_features(epoch,sr):
    f=np.zeros(9)
    nperseg=min(256,len(epoch))
    fr,ps=signal.welch(epoch,fs=sr,nperseg=nperseg)
    bands=[(0.5,2),(2,8),(8,20),(20,50),(50,100),(100,200)]
    for i,(lo,hi) in enumerate(bands):
        mask=(fr>=lo)&(fr<hi)
        bp=np.mean(ps[mask]) if mask.any() else 1e-15
        f[i]=np.log10(max(bp,1e-15))
    f[6]=np.sum(fr*ps)/np.sum(ps) if np.sum(ps)>0 else 0
    env=np.abs(signal.hilbert(epoch))
    f[7]=np.log10(max(env.std(),1e-15))
    f[8]=np.log10(max(np.mean(np.abs(np.diff(np.sign(epoch))))/2,1e-10))
    return f

def analyze_night(filepath,label):
    audio,tmp=load_audio(filepath)
    epoch_len=int(30*SR); n_epochs=len(audio)//epoch_len
    feat=np.zeros((n_epochs,9))
    for i in range(n_epochs):
        ep=audio[i*epoch_len:(i+1)*epoch_len]-np.mean(audio[i*epoch_len:(i+1)*epoch_len])
        feat[i]=extract_features(ep,SR)
        if i%400==0 and i>0: print(f'  {i}/{n_epochs}...',flush=True)
    
    energy=feat[:,5]
    th_low=np.percentile(energy,30)
    th_high=np.percentile(energy,80)
    
    train_idx=np.where((energy<=th_low)|(energy>=th_high))[0]
    train_y=np.array([0 if energy[i]<=th_low else 1 for i in train_idx],dtype=int)
    train_X=feat[train_idx]
    
    clf=LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42,C=0.1)
    clf.fit(train_X,train_y)
    pred=clf.predict(feat)
    
    sleep_min=(pred==0).sum()*0.5
    wake_min=(pred==1).sum()*0.5
    eff=sleep_min/(sleep_min+wake_min+0.01)*100
    
    timeline=[]
    for h in range(0,n_epochs,60):
        end=min(h+60,n_epochs)
        seg_sleep=(pred[h:end]==0).mean()*100
        seg_energy=np.mean(energy[h:end])
        ts=f'{h*30//3600:02d}:{(h*30%3600)//60:02d}'
        timeline.append({'time':ts,'sleep_pct':round(seg_sleep,1),'energy':round(seg_energy,3)})
    
    micro_arousals=0
    for i in range(2,n_epochs-2):
        if pred[i]==1 and pred[i-2]==0 and pred[i+2]==0:
            micro_arousals+=1
    
    max_streak=cur=0
    for p in pred:
        cur=cur+1 if p==0 else 0
        max_streak=max(max_streak,cur)
    
    awake_periods=sum(1 for i in range(1,n_epochs) if pred[i]==1 and pred[i-1]==0)*0.5
    
    if os.path.exists(tmp):os.remove(tmp)
    
    return {'file':os.path.basename(filepath),'label':label,
        'total_min':round(sleep_min+wake_min,1),
        'sleep_min':round(sleep_min,1),'wake_min':round(wake_min,1),
        'efficiency':round(eff,1),
        'micro_arousals':micro_arousals,
        'longest_sleep_streak_min':round(max_streak*0.5,1),
        'awake_periods':round(awake_periods,1),
        'timeline':timeline}

print('='*65,flush=True)
print(' 两晚真实睡眠分析',flush=True)
print('='*65,flush=True)

n1=analyze_night(r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a','第一晚(4月11日)')
n2=analyze_night(r'D:\AISleepGen_Optimized\sleep_record\20260412_223013.m4a','第二晚(4月12日)')

for n in [n1,n2]:
    print(f'\n {n["label"]}',flush=True)
    print(f' {"-"*40}',flush=True)
    print(f' 总记录: {n["total_min"]:.0f}min',flush=True)
    print(f' 真正睡着: {n["sleep_min"]:.0f}min',flush=True)
    print(f' 觉醒时长: {n["wake_min"]:.0f}min',flush=True)
    print(f' 睡眠效率: {n["efficiency"]:.0f}%',flush=True)
    print(f' 最长连续睡段: {n["longest_sleep_streak_min"]:.0f}min',flush=True)
    print(f' 微觉醒次数: {n["micro_arousals"]}次',flush=True)
    print(f' 独立觉醒时段: {n["awake_periods"]:.0f}min',flush=True)
    
    median_energy=np.median([t['energy'] for t in n['timeline']])
    print(f'\n 时间线:',flush=True)
    for t in n['timeline']:
        bar=chr(9619)*int(t['sleep_pct']//5)
        en='高' if t['energy']>median_energy*1.2 else('中' if t['energy']>median_energy*0.5 else '低')
        print(f'   {t["time"]} {bar:<20s} {t["sleep_pct"]:.0f}%睡  能量{en}',flush=True)

print(f'\n{"="*65}',flush=True)
print(' 两晚对比',flush=True)
print('='*65,flush=True)
for key in ['sleep_min','wake_min','efficiency','longest_sleep_streak_min','micro_arousals']:
    v1=n1[key];v2=n2[key];diff=round(v2-v1,1)
    a='+' if diff>0 else '' if diff==0 else ''
    print(f' {key}: {v1} -> {v2} ({a}{diff})',flush=True)

out=os.path.join(GRANULAR,'results','two_nights_analysis.json')
with open(out,'w')as f:json.dump({'night1':n1,'night2':n2},f,ensure_ascii=False,indent=2)
print(f'\n 保存: {out}',flush=True)
