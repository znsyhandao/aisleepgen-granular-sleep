"""
双人睡眠分析 v5 — 声学指纹分离 + 各自醒睡曲线

核心改进:
  1. 用频谱质心(spectral centroid)区分男女/不同声音源
  2. 各自epoch独立预测醒睡
  3. 输出各自时间线
"""
import sys, os, json
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from scipy import signal as scipy_signal

GRANULAR = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, GRANULAR)

SR = 800

def load(filepath):
    tmp = filepath.replace('.m4a', '_tmp800.wav')
    if not os.path.exists(tmp):
        import subprocess
        subprocess.run([r'D:\ffmpeg\bin\ffmpeg','-y','-i',filepath,
            '-acodec','pcm_s16le','-ar',str(SR),'-ac','1',tmp],capture_output=True)
    with open(tmp,'rb') as f:
        f.read(44); data=f.read()
    audio = np.frombuffer(data,dtype=np.int16).astype(np.float32)/32768.0
    hours = len(audio)/SR/3600
    print(f'  加载: {hours:.1f}h @ {SR}Hz',flush=True)
    return audio, tmp


def extract_rich_features(epoch, sr):
    """9维声学特征: 频谱质心+频带能量+包络"""
    f = np.zeros(9)
    # Welch
    nperseg = min(256, len(epoch))
    fr, ps = scipy_signal.welch(epoch, fs=sr, nperseg=nperseg)
    
    # 频带
    bands = [(0.5,2),(2,8),(8,20),(20,50),(50,100),(100,200)]
    for i,(lo,hi) in enumerate(bands):
        mask = (fr>=lo)&(fr<hi)
        bp = np.mean(ps[mask]) if mask.any() else 1e-15
        f[i] = np.log10(max(bp,1e-15))
    
    # 频谱质心
    f[6] = np.sum(fr*ps)/np.sum(ps) if np.sum(ps)>0 else 0
    
    # 包络标准差
    env_hilbert = np.abs(scipy_signal.hilbert(epoch))
    f[7] = np.log10(max(env_hilbert.std(),1e-15))
    
    # 零穿越率
    f[8] = np.log10(max(np.mean(np.abs(np.diff(np.sign(epoch))))/2,1e-10))
    
    return f


def predict_wake_sleep(features):
    """从特征预测醒/睡 — 用v4训练的LR模型系数"""
    # 系数从 audio_sleep_classifier.json (LR coef_)
    # 简单版本: 用能量包络 (低能量+低频谱质心=睡)
    
    # 从9维提取关键维
    # f[5]=100-200Hz能, f[7]=包络std, f[6]=质心
    energy_hi = features[5]  # 100-200Hz
    env_var = features[7]    # 包络变异
    centroid = features[6]   # 频谱质心Hz
    
    # 多维组合: 安静睡眠 = 低能量 + 低变异 + 稳定呼吸频率
    # 夜间呼吸声集中在50-200Hz
    sleep_likelihood = 0
    
    # 条件1: 100-200Hz能量低 (安静)
    if energy_hi < -8: sleep_likelihood += 1
    # 条件2: 包络稳定 (呼吸规律)
    if env_var < -2.5: sleep_likelihood += 1
    # 条件3: 频谱质心低 (没有高频噪音)
    if centroid < 120: sleep_likelihood += 1
    # 条件4: 低频能量有能量 (有呼吸)
    if features[1] > -8: sleep_likelihood += 1  # 2-8Hz有能量=呼吸存在
    
    return 0 if sleep_likelihood >= 3 else 1  # 0=睡, 1=醒


def analyze_v5(filepath):
    print('='*60,flush=True)
    print('双人睡眠 v5 — 声学指纹分离',flush=True)
    print(f'{os.path.basename(filepath)}',flush=True)
    print('='*60,flush=True)
    
    audio, tmp = load(filepath)
    
    epoch_len = int(30*SR)
    n_epochs = len(audio)//epoch_len
    print(f'\n[1/3] 提取特征 ({n_epochs} epochs)...',flush=True)
    
    all_feat = np.zeros((n_epochs, 9))
    for i in range(n_epochs):
        ep = audio[i*epoch_len:(i+1)*epoch_len]-np.mean(audio[i*epoch_len:(i+1)*epoch_len])
        all_feat[i] = extract_rich_features(ep, SR)
        if i%400==0 and i>0: print(f'  {i}/{n_epochs}...',flush=True)
    
    # ==========================================
    # 分离策略: 频谱质心 + 包络变异
    # - 男方呼噜: 低频(100-300Hz)+包络规则=周期性
    # - 女方看手机: 高频(300Hz+)+包络不规则
    # - 两人都安静: 需分离呼吸节奏
    # ==========================================
    
    centroid = all_feat[:, 6]  # 频谱质心
    env_energy = all_feat[:, 5]  # 100-200Hz
    low_energy = all_feat[:, 0] + all_feat[:, 1]  # 0.5-8Hz低频
    
    # 用质心分: 男声/呼噜 = 低质心, 女声/手机 = 高质心
    # 但频带能量组合更鲁棒
    male_signature = low_energy - env_energy  # 低频-高频
    # 归一化
    ms_norm = (male_signature - male_signature.mean()) / male_signature.std()
    
    # K-means 聚类分2群
    from sklearn.cluster import KMeans
    cluster_feat = np.column_stack([
        ms_norm,
        (centroid - centroid.mean())/centroid.std(),
        (env_energy - env_energy.mean())/env_energy.std(),
    ])
    
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(cluster_feat)
    
    # 确定哪个是男方(质心低的群)
    c0 = centroid[labels==0].mean()
    c1 = centroid[labels==1].mean()
    male_label = 0 if c0 < c1 else 1
    
    print(f'\n[2/3] K-means分离:',flush=True)
    print(f'  群0(质心={c0:.1f}Hz): {(labels==0).sum()} epochs',flush=True)
    print(f'  群1(质心={c1:.1f}Hz): {(labels==1).sum()} epochs',flush=True)
    print(f'  男方(低频多) = 群{male_label}',flush=True)
    
    # ==========================================
    # 各自醒睡预测
    # ==========================================
    print(f'\n[3/3] 各自醒睡时间线...',flush=True)
    
    results = {}
    for label_str, lbl in [('M. 男方(呼噜/低频)', male_label), ('F. 女方(手机/高频)', 1-male_label)]:
        idx = labels == lbl
        sub_feat = all_feat[idx]
        
        if len(sub_feat) == 0:
            print(f'{label_str}: 无数据',flush=True)
            continue
        
        preds = np.array([predict_wake_sleep(sub_feat[i]) for i in range(len(sub_feat))])
        sleep_pct = (preds==0).mean()*100
        wake_pct = (preds==1).mean()*100
        
        results[label_str] = {
            'total_epochs': len(sub_feat),
            'total_min': round(len(sub_feat)*0.5, 1),
            'sleep_min': round((preds==0).sum()*0.5, 1),
            'wake_min': round((preds==1).sum()*0.5, 1),
            'sleep_efficiency': round(sleep_pct, 1),
            'timeline': [],
        }
        
        # 时间线 (每30min)
        for h in range(0, len(sub_feat), 60):
            end = min(h+60, len(sub_feat))
            seg_sleep = (preds[h:end]==0).mean()*100
            ts = f'{h*30//3600:02d}:{(h*30%3600)//60:02d}'
            results[label_str]['timeline'].append(f'{ts} {chr(9619)*int(seg_sleep/5)}{chr(9617)*int((100-seg_sleep)/5)} {seg_sleep:.0f}%')
        
        print(f'{label_str}:',flush=True)
        print(f'  总时长: {results[label_str]["total_min"]:.0f}min',flush=True)
        print(f'  睡着: {results[label_str]["sleep_min"]:.0f}min (效率{results[label_str]["sleep_efficiency"]:.0f}%)',flush=True)
        print(f'  觉醒: {results[label_str]["wake_min"]:.0f}min',flush=True)
        print(f'  时间线:',flush=True)
        for line in results[label_str]['timeline'][::2]:  # 每小时显示
            print(f'    {line}',flush=True)
    
    # 同时刻对比
    print(f'\n{"="*60}',flush=True)
    print(f'同时间对比 (每30min)',flush=True)
    print(f'{"="*60}',flush=True)
    male_idx = labels == male_label
    female_idx = labels == 1-male_label
    
    for h in range(0, n_epochs, 60):
        end = min(h+60, n_epochs)
        ts = f'{h*30//3600:02d}:{(h*30%3600)//60:02d}'
        
        m_epochs = male_idx[h:end]
        f_epochs = female_idx[h:end]
        
        if m_epochs.sum() > 0 and f_epochs.sum() > 0:
            # 各自在这个时间段的醒睡比例
            m_pred = predict_wake_sleep(all_feat[h:h+60].mean(axis=0))
            f_pred = predict_wake_sleep(all_feat[h:h+60].mean(axis=0))
            
            m_bar = '睡' if m_pred==0 else '醒'
            f_bar = '睡' if f_pred==0 else '醒'
            print(f'  {ts}  M.{m_bar}  F.{f_bar}',flush=True)
    
    # 保存 (去掉emoji避免GBK编码问题)
    out = os.path.join(GRANULAR, 'results', 'couple_sleep_v5.json')
    with open(out,'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n保存: {out}',flush=True)
    
    # cleanup
    if os.path.exists(tmp): os.remove(tmp)
    
    return results


if __name__ == '__main__':
    analyze_v5(r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a')
