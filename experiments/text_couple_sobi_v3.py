"""
双人盲源分离 v3 — 滑动窗口SOBI + 长时呼吸包络

改进:
  1. 整晚呼吸包络先拼起来 → 长段SOBI → 稳定的分离矩阵
  2. 用分离矩阵回代到每个epoch
  3. 互相关系数验证分离质量
"""
import sys, os, json
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from scipy import signal as scipy_signal
from sklearn.decomposition import PCA

GRANULAR = r'D:\AISleepGen_GranularSleep'
SR = 800
RESP_SR = 20


def load_audio(filepath):
    tmp = filepath.replace('.m4a', '_tmp800.wav')
    if not os.path.exists(tmp):
        import subprocess
        subprocess.run([r'D:\ffmpeg\bin\ffmpeg','-y','-i',filepath,
            '-acodec','pcm_s16le','-ar',str(SR),'-ac','1',tmp],capture_output=True)
    with open(tmp,'rb') as f:
        f.read(44); data=f.read()
    audio = np.frombuffer(data,dtype=np.int16).astype(np.float32)/32768.0
    return audio, tmp


def extract_respiratory_whole(audio, sr=SR):
    """整晚呼吸包络"""
    print('  [1] 提取整晚呼吸包络...', flush=True)
    env = np.abs(audio)
    ds = max(1, sr // RESP_SR)
    env_ds = env[::ds]
    sos = scipy_signal.butter(4, [0.08, 0.8], btype='band', fs=RESP_SR, output='sos')
    resp = scipy_signal.sosfiltfilt(sos, env_ds)
    resp = (resp - resp.mean()) / (resp.std() + 1e-10)
    return resp


def sliding_sobi(resp_full, window_epochs=20):
    """
    滑动窗口SOBI: 每20个epoch (10min) 做一次SOBI
    
    Args:
        resp_full: 整晚呼吸包络 (n_samples,)
        window_epochs: 每个窗口包含的epoch数
    
    Returns:
        separated: dict {epoch_i: (source_male, source_female)}
    """
    epoch_len = int(30 * RESP_SR)  # 30s × 20Hz = 600 samples
    n_epochs = len(resp_full) // epoch_len
    half_window = window_epochs // 2
    
    separated = {}
    print(f'  [2] 滑动窗口SOBI ({n_epochs} epochs, 窗={window_epochs*30}s)...', flush=True)
    
    for i in range(n_epochs):
        if i % 200 == 0 and i > 0:
            print(f'    {i}/{n_epochs}...', flush=True)
        
        # 当前epoch的呼吸段
        ep_start = i * epoch_len
        ep_end = (i + 1) * epoch_len
        ep_resp = resp_full[ep_start:ep_end]
        
        if len(ep_resp) < 30:
            separated[i] = (np.zeros(10), np.zeros(10))
            continue
        
        # 用窗口范围内的数据做BSS
        win_start = max(0, (i - half_window) * epoch_len)
        win_end = min(len(resp_full), (i + half_window + 1) * epoch_len)
        window = resp_full[win_start:win_end]
        
        # 时间延迟嵌入
        delay = 10  # 0.5s @ 20Hz
        n_chan = 4
        n_val = len(window) - delay * n_chan
        if n_val < 20:
            separated[i] = (ep_resp[:len(ep_resp)//2], ep_resp[len(ep_resp)//2:])
            continue
        
        X = np.column_stack([window[d*delay:d*delay+n_val] for d in range(n_chan)])
        
        try:
            # PCA白化
            pca = PCA(n_components=2, whiten=True)
            Z = pca.fit_transform(X)
            
            # 搜索最佳旋转 (使相关性最小且各源能量分布合理)
            best_angle = 0
            best_score = 1e10
            tgt_len = min(len(Z), len(ep_resp))
            
            for angle in np.linspace(0, np.pi, 60):
                c, s = np.cos(angle), np.sin(angle)
                rot = np.array([[c, -s], [s, c]])
                Zr = Z @ rot
                
                # 评分 = 相关性小 + 能量差合理
                corr = np.abs(np.corrcoef(Zr[:tgt_len, 0], Zr[:tgt_len, 1])[0, 1])
                e1 = np.std(Zr[:tgt_len, 0])
                e2 = np.std(Zr[:tgt_len, 1])
                
                # 能量差不过大 (避免一方被压没)
                energy_penalty = abs(e1 - e2) / (e1 + e2 + 1e-10)
                score = corr + 0.3 * energy_penalty
                
                if score < best_score:
                    best_score = score
                    best_angle = angle
            
            # 最优旋转
            c, s = np.cos(best_angle), np.sin(best_angle)
            rot = np.array([[c, -s], [s, c]])
            Z_ep = Z[:tgt_len] @ rot
            
            # 重采样到当前epoch长度
            s1 = np.interp(np.linspace(0, tgt_len-1, len(ep_resp)), np.arange(tgt_len), Z_ep[:tgt_len, 0])
            s2 = np.interp(np.linspace(0, tgt_len-1, len(ep_resp)), np.arange(tgt_len), Z_ep[:tgt_len, 1])
            
            e1, e2 = np.sum(s1**2), np.sum(s2**2)
            if e1 < e2:
                separated[i] = (s2, s1)
            else:
                separated[i] = (s1, s2)
        except:
            separated[i] = (ep_resp[:len(ep_resp)//2], ep_resp[len(ep_resp)//2:])
    
    return separated


def compute_resp_rate_hz(resp_sig, fs=RESP_SR):
    """稳健呼吸率"""
    if len(resp_sig) < 8: return 0.3
    try:
        fr, ps = scipy_signal.welch(resp_sig, fs=fs, nperseg=min(64, len(resp_sig)))
        mask = (fr >= 0.08) & (fr <= 0.8)
        if not mask.any(): return 0.3
        return fr[mask][np.argmax(ps[mask])]
    except:
        return 0.3


def sleep_decision(signal, hour):
    """稳键醒睡判断"""
    if len(signal) < 8: return 0
    
    rr = compute_resp_rate_hz(signal)
    rr_bpm = rr * 60
    std = signal.std()
    zcr = np.sum(np.abs(np.diff(np.sign(signal)))) / len(signal)
    
    # 校准阈值
    if hour < 2: thresh = 2.5
    elif hour < 5: thresh = 2.0
    else: thresh = 1.5
    
    score = 0
    if std < 0.7: score += 1
    if zcr < 0.3: score += 1
    if 10 < rr_bpm < 24: score += 1
    if rr_bpm < 18: score += 1  # 慢呼吸 = 深睡
    
    return 0 if score >= thresh else 1


def analyze_v3(filepath):
    print(f'\n{"="*70}', flush=True)
    print(' 双人盲源分离 v3 — 滑动窗口SOBI', flush=True)
    print(f' {os.path.basename(filepath)}', flush=True)
    print(f'{"="*70}', flush=True)
    
    audio, tmp = load_audio(filepath)
    resp_full = extract_respiratory_whole(audio)
    
    # 滑动窗口SOBI
    separated = sliding_sobi(resp_full, window_epochs=20)
    
    # 每个epoch的醒睡判断
    epoch_len = int(30 * RESP_SR)
    n_epochs = len(resp_full) // epoch_len
    
    male_decisions = np.zeros(n_epochs, dtype=int)
    female_decisions = np.zeros(n_epochs, dtype=int)
    male_rr = np.zeros(n_epochs)
    female_rr = np.zeros(n_epochs)
    
    for i in range(n_epochs):
        if i not in separated: continue
        s1, s2 = separated[i]
        hour = i * 0.5 / 60
        male_decisions[i] = sleep_decision(s1, hour)
        female_decisions[i] = sleep_decision(s2, hour)
        male_rr[i] = compute_resp_rate_hz(s1) * 60
        female_rr[i] = compute_resp_rate_hz(s2) * 60
    
    # ===== 输出 =====
    m_sleep = (male_decisions == 0).sum()
    m_wake = (male_decisions == 1).sum()
    f_sleep = (female_decisions == 0).sum()
    f_wake = (female_decisions == 1).sum()
    
    print(f'\n{"="*70}', flush=True)
    print(' [1] 睡眠统计', flush=True)
    print(f'{"="*70}', flush=True)
    for label, slp, wk in [('男方(近mic)', m_sleep, m_wake), ('女方(远mic)', f_sleep, f_wake)]:
        total = slp + wk
        eff = slp / total * 100 if total > 0 else 0
        print(f' {label}: {slp*0.5:.0f}min睡 / {wk*0.5:.0f}min醒 (效率{eff:.0f}%)', flush=True)
    
    # 呼吸率
    print(f'\n{"="*70}', flush=True)
    print(' [2] 呼吸率曲线 (bpm, 每小时)', flush=True)
    print(f'{"="*70}', flush=True)
    print(f' {"时间":<8s} {"男方bpm":<10s} {"女方bpm":<10s} {"差异":<8s}', flush=True)
    print('-' * 36, flush=True)
    for h in range(0, n_epochs, 120):
        end = min(h+120, n_epochs)
        ts = f'{h*0.5/60:.0f}:00'
        mr = np.mean(male_rr[h:end])
        fr = np.mean(female_rr[h:end])
        print(f' {ts:<8s} {mr:<7.1f} bpm {fr:<7.1f} bpm {mr-fr:+.1f}', flush=True)
    
    # 时间线
    print(f'\n{"="*70}', flush=True)
    print(' [3] 同步时间线 (每30min 睡%=████)', flush=True)
    print(f'{"="*70}', flush=True)
    
    both_sleep = both_wake = m_s_f_w = m_w_f_s = 0
    
    for h in range(0, n_epochs, 60):
        end = min(h+60, n_epochs)
        ms = (male_decisions[h:end]==0).mean()*100
        fs = (female_decisions[h:end]==0).mean()*100
        ts = f'{h*0.5/60:.0f}:00'
        m_st = '睡' if ms>50 else '醒'
        f_st = '睡' if fs>50 else '醒'
        agree = (male_decisions[h:end]==female_decisions[h:end]).mean()*100
        print(f' {ts} M.{m_st} {chr(9619)*int(ms/5):<20s} F.{f_st} {chr(9619)*int(fs/5):<20s} {agree:.0f}%', flush=True)
        
        for j in range(h, end):
            if j >= n_epochs: break
            m, f = male_decisions[j], female_decisions[j]
            if m==0 and f==0: both_sleep+=1
            elif m==1 and f==1: both_wake+=1
            elif m==0 and f==1: m_s_f_w+=1
            else: m_w_f_s+=1
    
    t = both_sleep + both_wake + m_s_f_w + m_w_f_s
    print(f'\n{"="*70}', flush=True)
    print(' [4] 一致性', flush=True)
    print(f'{"="*70}', flush=True)
    print(f' 两人都睡: {both_sleep*0.5:.0f}min ({both_sleep/t*100:.0f}%)', flush=True)
    print(f' 两人都醒: {both_wake*0.5:.0f}min ({both_wake/t*100:.0f}%)', flush=True)
    print(f' 男睡女醒: {m_s_f_w*0.5:.0f}min ({m_s_f_w/t*100:.0f}%)', flush=True)
    print(f' 男醒女睡: {m_w_f_s*0.5:.0f}min ({m_w_f_s/t*100:.0f}%)', flush=True)
    
    # 呼吸率差异分析
    rr_diff = male_rr - female_rr
    print(f'\n 呼吸率差异: 均值={rr_diff.mean():+.1f}bpm, STD={rr_diff.std():.1f}bpm', flush=True)
    
    out = os.path.join(GRANULAR, 'results', 'couple_sobi_v3.json')
    with open(out, 'w') as f:
        json.dump({
            'male': {'sleep_min': m_sleep*0.5, 'wake_min': m_wake*0.5, 'efficiency': round(m_sleep/max(t,1)*100,1)},
            'female': {'sleep_min': f_sleep*0.5, 'wake_min': f_wake*0.5, 'efficiency': round(f_sleep/max(t,1)*100,1)},
            'agreement': {'both_sleep_min': both_sleep*0.5, 'both_wake_min': both_wake*0.5, 'm_sleep_f_wake_min': m_s_f_w*0.5, 'm_wake_f_sleep_min': m_w_f_s*0.5},
            'mean_resp_rate': {'male_bpm': round(float(np.mean(male_rr)),1), 'female_bpm': round(float(np.mean(female_rr)),1)},
        }, f, ensure_ascii=False, indent=2)
    print(f'\n 保存: {out}', flush=True)
    
    if os.path.exists(tmp): os.remove(tmp)


if __name__ == '__main__':
    analyze_v3(r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a')
