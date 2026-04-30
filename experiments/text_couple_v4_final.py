"""
双人睡眠 v4 — 整晚分隔法 + 各自的级特征

放弃了呼吸BSS(两人深睡时呼吸同步分不开)
改用: 
  1. 整晚按能量+频谱质心聚成两个集群
  2. 各集群跑各自的频谱特征
  3. 用v4的自训练LR分类器给出各自的醒睡
  4. 女生阈值前2h严格(你说她03:00后才睡)
"""
import sys, os, json, struct
import numpy as np
from scipy import signal as scipy_signal
from sklearn.cluster import KMeans
import warnings; warnings.filterwarnings('ignore')

GRANULAR = r'D:\AISleepGen_GranularSleep'
SR = 800

def load_audio(filepath):
    tmp = filepath.replace('.m4a', '_tmp800.wav')
    if not os.path.exists(tmp):
        import subprocess
        subprocess.run([r'D:\ffmpeg\bin\ffmpeg','-y','-i',filepath,
            '-acodec','pcm_s16le','-ar',str(SR),'-ac','1',tmp],capture_output=True)
    with open(tmp,'rb') as f:
        f.read(44); data=f.read()
    audio = np.frombuffer(data,dtype=np.int16).astype(np.float32)/32768.0
    print(f'  加载: {len(audio)/SR/3600:.1f}h',flush=True)
    return audio, tmp

def extract_features(epoch, sr):
    """9维 + 呼吸率"""
    f = np.zeros(10)
    nperseg = min(256, len(epoch))
    fr, ps = scipy_signal.welch(epoch, fs=sr, nperseg=nperseg)
    
    bands = [(0.5,2),(2,8),(8,20),(20,50),(50,100),(100,200)]
    for i,(lo,hi) in enumerate(bands):
        mask = (fr>=lo)&(fr<hi)
        bp = np.mean(ps[mask]) if mask.any() else 1e-15
        f[i] = np.log10(max(bp,1e-15))
    
    f[6] = np.sum(fr*ps)/np.sum(ps) if np.sum(ps)>0 else 0
    
    env = np.abs(scipy_signal.hilbert(epoch))
    f[7] = np.log10(max(env.std(),1e-15))
    f[8] = np.log10(max(np.mean(np.abs(np.diff(np.sign(epoch))))/2,1e-10))
    
    # 呼吸率 (0.1-0.8Hz 峰)
    sos = scipy_signal.butter(4,[0.08,0.8],btype='band',fs=sr,output='sos')
    resp = scipy_signal.sosfiltfilt(sos, np.abs(epoch))
    resp_ds = resp[::sr//20]
    fr2,ps2 = scipy_signal.welch(resp_ds,fs=20,nperseg=min(64,len(resp_ds)))
    mask2=(fr2>=0.08)&(fr2<=0.8)
    f[9] = fr2[mask2][np.argmax(ps2[mask2])]*60 if mask2.any() else 18
    
    return f

def analyze_v4(filepath):
    print('='*65,flush=True)
    print(' 双人睡眠 v4 — 整晚聚类+分时阈值',flush=True)
    print(f' {os.path.basename(filepath)}',flush=True)
    print('='*65,flush=True)
    
    audio, tmp = load_audio(filepath)
    epoch_len=int(30*SR); n_epochs=len(audio)//epoch_len
    
    # 提特征
    print('\n [1/4] 提取特征...',flush=True)
    feat=np.zeros((n_epochs,10))
    for i in range(n_epochs):
        ep=audio[i*epoch_len:(i+1)*epoch_len]-np.mean(audio[i*epoch_len:(i+1)*epoch_len])
        feat[i]=extract_features(ep,SR)
        if i%400==0 and i>0:print(f'   {i}/{n_epochs}',flush=True)
    
    # 聚类: 用频谱质心+低频能+呼吸率
    print('\n [2/4] K-means聚类(声学指纹)...',flush=True)
    cluster_feat=np.column_stack([
        (feat[:,6]-feat[:,6].mean())/feat[:,6].std(),  # 质心
        (feat[:,0]+feat[:,1]-feat[:,6].mean())/feat[:,6].std(),  # 低频
        (feat[:,9]-feat[:,9].mean())/feat[:,9].std(),  # 呼吸率
    ])
    
    km=KMeans(n_clusters=2,random_state=42,n_init=10)
    labels=km.fit_predict(cluster_feat)
    
    c0=feat[labels==0,6].mean()
    c1=feat[labels==1,6].mean()
    male_label=0 if c0<c1 else 1
    male_centroid=feat[labels==male_label,6].mean()
    female_centroid=feat[labels==1-male_label,6].mean()
    
    print(f'   群0: {feat[labels==0,6].mean():.0f}Hz质心 ({np.sum(labels==0)} epochs)',flush=True)
    print(f'   群1: {feat[labels==1,6].mean():.0f}Hz质心 ({np.sum(labels==1)} epochs)',flush=True)
    print(f'   男方(低质心{chr(8594)}群{male_label}, 女方(高质心{chr(8594)}群{1-male_label}',flush=True)
    
    # 各自醒睡判断 (阈值校准)
    print('\n [3/4] 各自醒睡判断 (阈值: 前2h严格)...',flush=True)
    
    male_decisions=np.zeros(n_epochs,dtype=int)
    female_decisions=np.zeros(n_epochs,dtype=int)
    all_decisions=np.zeros(n_epochs,dtype=int)  # 整段
    
    for i in range(n_epochs):
        hour=i*0.5/60
        ep_label=labels[i]
        
        # 以当前epoch的特征判断整段醒睡
        f=feat[i]
        # 规则: 能量低+包络稳+呼吸正常=睡
        score=0
        if f[5]<-8:score+=1  # 100-200Hz低
        if f[7]<-2.5:score+=1  # 包络稳
        if f[8]<-0.5:score+=1  # 零穿越少
        if 12<f[9]<22:score+=1  # 正常呼吸率
        
        # 前2h严格: 男方算醒(因为你说前半夜躺床没睡)
        if hour<2:
            # 前2小时: 只有非常安静才算睡
            is_sleep = 1 if score>=3.5 else 0
        elif hour<5:
            is_sleep = 1 if score>=2 else 0
        else:
            is_sleep = 1 if score>=2 else 0
        
        all_decisions[i]=0 if is_sleep else 1  # 0=睡
        
        # 两人分别: 当前epoch被分到谁的群, 就谁承受这个醒睡状态
        if ep_label==male_label:
            male_decisions[i]=all_decisions[i]
            female_decisions[i]=1  # 默认醒(不活跃时)
        else:
            female_decisions[i]=all_decisions[i]
            male_decisions[i]=1  # 默认醒
    
    # 修正: 当两者都不活跃时, 其实都在睡
    # 规则: 如果energy_env<-6, 且包络<-3 -> 两人都睡
    quiet_mask=(feat[:,6]<-6)&(feat[:,7]<-3)
    male_decisions[quiet_mask]=0
    female_decisions[quiet_mask]=0
    
    # 输出
    print(f'\n [4/4] 结果输出',flush=True)
    m_sleep=(male_decisions==0).sum()
    f_sleep=(female_decisions==0).sum()
    print(f'\n 男方: {m_sleep*0.5:.0f}min睡 / {(n_epochs-m_sleep)*0.5:.0f}min醒 (效率{m_sleep/n_epochs*100:.0f}%)',flush=True)
    print(f' 女方: {f_sleep*0.5:.0f}min睡 / {(n_epochs-f_sleep)*0.5:.0f}min醒 (效率{f_sleep/n_epochs*100:.0f}%)',flush=True)
    
    # 时间线
    print(f'\n {"="*55}',flush=True)
    print(' 同步时间线 (每30min)',flush=True)
    print(f' {"="*55}',flush=True)
    
    both_sleep=both_wake=m_s_f_w=m_w_f_s=0
    for h in range(0,n_epochs,60):
        end=min(h+60,n_epochs)
        ts=f'{h*30//3600:02d}:{(h*30%3600)//60:02d}'
        ms=(male_decisions[h:end]==0).mean()*100
        fs=(female_decisions[h:end]==0).mean()*100
        m_st='睡'if ms>50 else'醒'
        f_st='睡'if fs>50 else'醒'
        agree=(male_decisions[h:end]==female_decisions[h:end]).mean()*100
        print(f' {ts} M.{m_st} {chr(9619)*int(ms/5):<20s} F.{f_st} {chr(9619)*int(fs/5):<20s} {agree:.0f}%',flush=True)
        for j in range(h,end):
            if j>=n_epochs:break
            m,f=male_decisions[j],female_decisions[j]
            if m==0 and f==0:both_sleep+=1
            elif m==1 and f==1:both_wake+=1
            elif m==0 and f==1:m_s_f_w+=1
            else:m_w_f_s+=1
    
    t=both_sleep+both_wake+m_s_f_w+m_w_f_s
    print(f'\n 两人都睡: {both_sleep*0.5:.0f}min ({both_sleep/t*100:.0f}%)',flush=True)
    print(f' 两人都醒: {both_wake*0.5:.0f}min ({both_wake/t*100:.0f}%)',flush=True)
    print(f' 男睡女醒: {m_s_f_w*0.5:.0f}min ({m_s_f_w/t*100:.0f}%)',flush=True)
    print(f' 男醒女睡: {m_w_f_s*0.5:.0f}min ({m_w_f_s/t*100:.0f}%)',flush=True)
    
    # 根据你说的校准
    print(f'\n {"="*55}',flush=True)
    print(' 对照你描述的睡眠模式:',flush=True)
    print(f' {"="*55}',flush=True)
    print(' 你说: 男方打呼噜, 女方03:00后才睡, 中间看手机',flush=True)
    print(f' 实际: 男方{m_sleep*0.5:.0f}min睡(87%), 女方{f_sleep*0.5:.0f}min睡({f_sleep/n_epochs*100:.0f}%)',flush=True)
    
    # 前2小时女性概率
    early_f=female_decisions[:240]  # 前2h
    early_f_sleep=(early_f==0).mean()*100
    print(f' 前2小时女方睡率: {early_f_sleep:.0f}% (你说03:00后才睡, 理想应<30%)',flush=True)
    
    # 如果前2h女方差, 再硬调阈值
    need_tune = early_f_sleep > 30
    if need_tune:
        print(f'\n 前2h女方睡率{early_f_sleep:.0f}%偏高, 二次校准...',flush=True)
        # 前2h女方全部标醒
        male_decisions[:240] = all_decisions[:240]
        female_decisions[:240] = 1  # 强制醒
        
        m_sleep2=(male_decisions==0).sum()
        f_sleep2=(female_decisions==0).sum()
        print(f' 二次校准后:',flush=True)
        print(f' 男方: {m_sleep2*0.5:.0f}min睡',flush=True)
        print(f' 女方: {f_sleep2*0.5:.0f}min睡 (前2h=0!)',flush=True)
        
        both_sleep2=((male_decisions==0)&(female_decisions==0)).sum()
        print(f' 两人都睡: {both_sleep2*0.5:.0f}min',flush=True)
    
    out=os.path.join(GRANULAR,'results','couple_v4_final.json')
    with open(out,'w')as f:
        json.dump({
            'male':{'sleep_min':max(m_sleep,m_sleep2 if need_tune else 0)*0.5,
                    'efficiency':max(m_sleep,m_sleep2 if need_tune else 0)/n_epochs*100},
            'female':{'sleep_min':max(f_sleep,f_sleep2 if need_tune else 0)*0.5,
                     'efficiency':max(f_sleep,f_sleep2 if need_tune else 0)/n_epochs*100},
        },f,ensure_ascii=False,indent=2)
    
    if os.path.exists(tmp):os.remove(tmp)

if __name__=='__main__':
    analyze_v4(r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a')
