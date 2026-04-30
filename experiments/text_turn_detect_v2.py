"""
翻身检测 v2: 只抓短时冲击峰(真正身体翻动), 过滤呼吸/呼噜/环境
"""
import sys, os, json
import numpy as np
from scipy import signal
import warnings; warnings.filterwarnings('ignore')

SR=800; GRANULAR=r'D:\AISleepGen_GranularSleep'

def load_audio(fp):
    tmp=fp.replace('.m4a','_tmp800.wav')
    if not os.path.exists(tmp):
        import subprocess
        subprocess.run([r'D:\ffmpeg\bin\ffmpeg','-y','-i',fp,
            '-acodec','pcm_s16le','-ar',str(SR),'-ac','1',tmp],capture_output=True)
    with open(tmp,'rb') as f:
        f.read(44); d=f.read()
    return np.frombuffer(d,dtype=np.int16).astype(np.float32)/32768.0, tmp

def detect_turns(audio, sr=SR):
    """
    翻身检测: 短时能量冲击峰 (0.5-2s) 
    
    翻身 vs 呼吸/呼噜:
    呼吸: 每分钟12-20次, 每次~2s, 能量缓慢变化
    翻身: 每次~1s, 能量突变>5倍背景
    呼噜: 持续100-300Hz, 有周期性
    说话: 0.3-3s, 频谱宽
    """
    # 1. 提取短时能量包络 (50ms帧 = 40 samples @ 800Hz)
    frame_len = int(0.05 * sr)  # 50ms
    hop = frame_len // 2
    n_frames = (len(audio) - frame_len) // hop
    
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        f = audio[i*hop:i*hop+frame_len]
        energy[i] = np.sum(f**2)
    
    # 2. 能量变化率 (帧间差分)
    energy_log = np.log10(energy + 1e-15)
    diff = np.abs(np.diff(energy_log, prepend=energy_log[0]))
    
    # 3. 翻身特征: 能量快速上升 > 持续0.5s > 快速下降
    # 呼吸: 能量缓慢上升0.5s, 缓慢下降0.5s
    # 翻身: 能量急剧上升0.1s, 持续0.5-1s, 下降0.1s
    
    # 短时冲击检测: 后向差分(上升速度) * 前向差分(下降速度) 都大
    rise = np.zeros(n_frames)  # 上升速度
    fall = np.zeros(n_frames)  # 下降速度
    
    for i in range(10, n_frames-10):
        # 前0.5s平均 vs 后0.5s平均 (10帧=0.5s)
        pre = np.mean(energy_log[max(0,i-10):i])
        post = np.mean(energy_log[i:min(n_frames,i+10)])
        rise[i] = energy_log[i] - pre
        fall[i] = energy_log[i] - post
    
    # 翻身: 快速上升3帧内(>背景的3倍) + 下降 > 上升的50%
    bg_level = np.median(energy_log)
    impact_threshold = bg_level + 0.5  # 约3倍能量
    
    turns = []
    in_turn = False
    turn_start = 0
    turn_peak = 0
    
    for i in range(n_frames):
        ts = i * hop / sr  # 时间戳秒
        
        is_impact = (energy_log[i] > impact_threshold) and (rise[i] > 0.3)
        
        if is_impact and not in_turn:
            in_turn = True
            turn_start = ts
            turn_peak = energy_log[i]
        elif in_turn:
            if energy_log[i] > turn_peak:
                turn_peak = energy_log[i]
            # 能量回落开始或超过2s未结束
            if (fall[i] > 0.15 and energy_log[i] < turn_peak - 0.3) or (ts - turn_start > 3):
                turn_duration = ts - turn_start
                if 0.3 < turn_duration < 2.5:  # 翻身0.3-2.5s
                    turns.append({
                        'start_sec': round(turn_start, 1),
                        'duration': round(turn_duration, 2),
                        'peak_energy': round(float(turn_peak), 3),
                        'amplitude': round(float(turn_peak - bg_level), 3),
                    })
                in_turn = False
    
    return turns, energy_log, bg_level


def classify_turn_source(turn, audio, sr=SR):
    """判断翻身来源: 近=男方 vs 远=女方"""
    s = int(turn['start_sec'] * sr)
    d = int(turn['duration'] * sr)
    ep = audio[s:min(s+d+int(0.5*sr), len(audio))]
    if len(ep) < sr//4: return '不确定', 0
    
    ep = ep - np.mean(ep)
    nperseg = min(128, len(ep))
    fr, ps = signal.welch(ep, fs=sr, nperseg=nperseg)
    
    # 特征
    centroid = np.sum(fr*ps)/np.sum(ps) if np.sum(ps)>0 else 100
    low = np.log10(max(np.mean(ps[(fr>=0.5)&(fr<100)])+1e-15, 1e-15))
    mid = np.log10(max(np.mean(ps[(fr>=100)&(fr<300)])+1e-15, 1e-15))
    high = np.log10(max(np.mean(ps[(fr>=300)&(fr<500)])+1e-15, 1e-15))
    
    # 近: 能量大+质心低+低频多(身体摩擦)
    total_energy = np.sum(ep**2)
    
    score = 0
    if total_energy > 0.01: score += 2      # 能量很大=近
    elif total_energy > 0.003: score += 1   # 中等
    if centroid < 120: score += 1            # 低频=近
    elif centroid > 200: score -= 0.5       # 高频=远
    if low > mid: score += 1                 # 摩擦声低频多=近
    if high < -5: score += 0.5              # 缺乏高频=近(空气衰减)
    
    if score >= 2.5: return '近(男方)', round(score,1)
    elif score <= 0.5: return '远(女方)', round(score,1)
    else: return '不确定', round(score,1)


def analyze(fp, label):
    audio, tmp = load_audio(fp)
    print(f'  检测翻身中...', flush=True)
    turns, energy, bg = detect_turns(audio)
    
    print(f'  找到 {len(turns)} 个翻身事件', flush=True)
    
    near=far=uncertain=0
    near_times=[]
    far_times=[]
    
    for t in turns:
        source, sc = classify_turn_source(t, audio)
        t['source']=source
        t['score']=sc
        
        if source=='近(男方)': near+=1; near_times.append(t['start_sec']/60)
        elif source=='远(女方)': far+=1; far_times.append(t['start_sec']/60)
        else: uncertain+=1
    
    total = len(turns)
    print(f'  近(男方): {near}次 ({near/total*100:.0f}%)', flush=True)
    print(f'  远(女方): {far}次 ({far/total*100:.0f}%)', flush=True)
    print(f'  不确定: {uncertain}次 ({uncertain/total*100:.0f}%)', flush=True)
    
    # 每小时分布
    print(f'\n  每小时分布:', flush=True)
    for h in range(8):
        start_m=h*60; end_m=(h+1)*60
        hn=sum(1 for t in near_times if start_m<=t<end_m)
        hf=sum(1 for t in far_times if start_m<=t<end_m)
        if hn+hf>0:
            bar_n=chr(9619)*min(hn,15)
            bar_f=chr(9617)*min(hf,15)
            print(f'   {h}:00-{h+1}:00  近{hn}次 {bar_n:<15s} 远{hf}次 {bar_f:<15s}', flush=True)
    
    if os.path.exists(tmp): os.remove(tmp)
    return {'label':label,'total_turns':total,'near_male':near,'far_female':far,'uncertain':uncertain}

print('='*65,flush=True)
print(' 翻身检测 v2 - 短时冲击峰',flush=True)
print('='*65,flush=True)

for fp,label in [
    (r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a','第一晚'),
    (r'D:\AISleepGen_Optimized\sleep_record\20260412_223013.m4a','第二晚'),
]:
    print(f'\n {label}',flush=True)
    r=analyze(fp,label)

print(f'\n{"="*65}',flush=True)
print(' 正常人整晚翻身: 20-40次',flush=True)
print(' 低于20次可能睡太死, 高于60次可能睡眠质量差',flush=True)
print(f'{"="*65}',flush=True)
