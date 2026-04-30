"""
双人盲源分离 — 逐30s epoch FastICA 分离呼吸 + 各自醒睡判断

核心:
  1. 对每个30s epoch提取呼吸包络 (0.1-0.8Hz带通)
  2. FastICA分解呼吸包络 → 2路独立呼吸信号
  3. 各路的呼吸率变异性 → 醒睡打分
  4. 汇总所有epoch的双人结果
"""
import sys, os, json
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from scipy import signal as scipy_signal
from sklearn.decomposition import FastICA

GRANULAR = r'D:\AISleepGen_GranularSleep'
SR = 800  # 音频采样率
RESP_SR = 20  # 呼吸包络采样率


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


def extract_respiratory(envelope, sr=800):
    """
    从音频提取呼吸包络 (0.1-0.8Hz带通)
    
    返回: 呼吸包络信号, @ RESP_SR Hz
    """
    # 整流
    env = np.abs(envelope)
    
    # 降采样到RESP_SR
    ds = max(1, sr // RESP_SR)
    env_ds = env[::ds]
    
    # 带通
    sos = scipy_signal.butter(4, [0.08, 0.8], btype='band', fs=RESP_SR, output='sos')
    resp = scipy_signal.sosfiltfilt(sos, env_ds)
    
    # 归一化
    resp = (resp - resp.mean()) / (resp.std() + 1e-10)
    
    return resp


def ica_separate_two_sources(mixed_signal_1d, n_samples_ica=300):
    """
    单通道 → 时间延迟嵌入 → FastICA → 2路源信号
    
    Args:
        mixed_signal_1d: 呼吸包络信号 (n_samples,)
        n_samples_ica: 用于ICA的样本数
    
    Returns: (source1, source2) 各 n_samples 长
    """
    n = len(mixed_signal_1d)
    if n < 50:
        return mixed_signal_1d[:n//2], mixed_signal_1d[n//2:]
    
    # 时间延迟嵌入 (0.5s = 10 samples @ 20Hz)
    delay = 10
    n_channels = 3  # 用3通道延迟
    n_valid = n - delay * n_channels
    if n_valid < 10:
        return mixed_signal_1d[:n//2], mixed_signal_1d[n//2:]
    
    X = np.column_stack([mixed_signal_1d[i*delay:i*delay+n_valid] for i in range(n_channels)])
    
    try:
        ica = FastICA(n_components=2, random_state=42, max_iter=300)
        S = ica.fit_transform(X)
        
        # 重建到原始长度
        s1 = np.interp(np.linspace(0, len(mixed_signal_1d)-1, len(S)), 
                      np.arange(len(S)), S[:, 0])
        s2 = np.interp(np.linspace(0, len(mixed_signal_1d)-1, len(S)),
                      np.arange(len(S)), S[:, 1])
        
        # 按能量排序 (能量大的 = 近mic = 男方)
        e1 = np.sum(s1**2)
        e2 = np.sum(s2**2)
        if e1 < e2:
            return s2, s1  # s1=男方(能量大), s2=女方
        return s1, s2
    except:
        return mixed_signal_1d[:n//2], mixed_signal_1d[n//2:]


def sleep_score_from_resp(resp_signal):
    """
    从呼吸包络判断醒睡
    
    醒: 呼吸不规则 (高变异, 多0穿越)
    睡: 呼吸稳定规则 (低变异, 呼吸率0.2-0.35Hz)
    """
    if len(resp_signal) < 10:
        return 0  # 默认睡
    
    # 呼吸率 (主频)
    freqs, psd = scipy_signal.welch(resp_signal, fs=RESP_SR, nperseg=min(64, len(resp_signal)))
    resp_mask = (freqs >= 0.1) & (freqs <= 0.8)
    if resp_mask.any():
        resp_psd = psd[resp_mask]
        resp_freqs = freqs[resp_mask]
        peak_freq = resp_freqs[np.argmax(resp_psd)]
    else:
        peak_freq = 0.3
    
    # 包络变异
    env_std = np.std(resp_signal)
    
    # 零穿越率
    zcr = np.sum(np.abs(np.diff(np.sign(resp_signal)))) / len(resp_signal)
    
    # 呼吸稳定性: 0=睡, 1=醒
    # 稳定呼吸: 包络std<0.8, zcr<0.3, 呼吸率0.15-0.4Hz
    sleep_signals = 0
    if env_std < 0.8: sleep_signals += 1
    if zcr < 0.3: sleep_signals += 1
    if 0.15 < peak_freq < 0.4: sleep_signals += 1
    
    return 0 if sleep_signals >= 2 else 1  # 0=睡, 1=醒


def analyze_bss_full(filepath):
    print('='*65, flush=True)
    print(' 双人盲源分离 — 逐epoch呼吸ICA + 各自醒睡', flush=True)
    print(f' {os.path.basename(filepath)}', flush=True)
    print('='*65, flush=True)
    
    # 1. 加载
    audio, tmp = load_audio(filepath)
    epoch_len = int(30 * SR)
    n_epochs = len(audio) // epoch_len
    print(f'\n 总epochs: {n_epochs}', flush=True)
    
    # 2. 逐epoch盲源分离
    results_male = {'sleep_epochs': 0, 'wake_epochs': 0}
    results_female = {'sleep_epochs': 0, 'wake_epochs': 0}
    
    timeline = []
    
    for i in range(n_epochs):
        if i % 100 == 0 and i > 0:
            print(f'  进度: {i}/{n_epochs} ({i/n_epochs*100:.0f}%)...', flush=True)
        
        ep = audio[i*epoch_len:(i+1)*epoch_len]
        ep = ep - ep.mean()
        
        # 呼吸包络提取
        resp_env = extract_respiratory(ep)
        
        if len(resp_env) < 20:  # 至少1秒数据
            # 无法分离, 用整段做单一判断
            resp_avg = sleep_score_from_resp(resp_env)
            results_male['sleep_epochs' if resp_avg==0 else 'wake_epochs'] += 1
            results_female['sleep_epochs' if resp_avg==0 else 'wake_epochs'] += 1
            timeline.append({'epoch': i, 'time': f'{i*30//3600:02d}:{(i*30%3600)//60:02d}',
                           'male': '睡' if resp_avg==0 else '醒', 
                           'female': '睡' if resp_avg==0 else '醒'})
            continue
        
        # ICA分离
        s_male, s_female = ica_separate_two_sources(resp_env)
        
        # 判断各自醒睡
        m_state = sleep_score_from_resp(s_male)
        f_state = sleep_score_from_resp(s_female)
        
        if m_state == 0: results_male['sleep_epochs'] += 1
        else: results_male['wake_epochs'] += 1
        
        if f_state == 0: results_female['sleep_epochs'] += 1
        else: results_female['wake_epochs'] += 1
        
        timeline.append({
            'epoch': i,
            'time': f'{i*30//3600:02d}:{(i*30%3600)//60:02d}',
            'male': '睡' if m_state == 0 else '醒',
            'female': '睡' if f_state == 0 else '醒',
        })
    
    # 3. 输出
    print(f'\n{"="*65}', flush=True)
    for label, res, role in [('男方(近mic)', results_male, '🧑'), ('女方(远mic)', results_female, '👩')]:
        total = res['sleep_epochs'] + res['wake_epochs']
        eff = res['sleep_epochs'] / total * 100 if total > 0 else 0
        print(f' {role} {label}:', flush=True)
        print(f'   总: {total*0.5:.0f}min ({total} epochs)', flush=True)
        print(f'   睡着: {res["sleep_epochs"]*0.5:.0f}min (效率 {eff:.0f}%)', flush=True)
        print(f'   觉醒: {res["wake_epochs"]*0.5:.0f}min', flush=True)
    
    # 4. 时间线 (每30min对比)
    print(f'\n{"="*65}', flush=True)
    print(' 双人同步时间线 (每30min)', flush=True)
    print(f'{"="*65}', flush=True)
    
    for h in range(0, n_epochs, 60):
        end = min(h+60, n_epochs)
        seg = timeline[h:end]
        if not seg: continue
        
        m_sleep = sum(1 for t in seg if t['male'] == '睡')
        f_sleep = sum(1 for t in seg if t['female'] == '睡')
        n_seg = len(seg)
        
        ts = seg[0]['time']
        
        # 状态判定
        m_state = '睡' if m_sleep > n_seg/2 else '醒'
        f_state = '睡' if f_sleep > n_seg/2 else '醒'
        
        # 共识度
        agree = sum(1 for t in seg if t['male'] == t['female'])
        agree_pct = agree / n_seg * 100
        
        bar_m = chr(9619) * int(m_sleep/n_seg*20) + chr(9617) * int((1-m_sleep/n_seg)*20)
        bar_f = chr(9619) * int(f_sleep/n_seg*20) + chr(9617) * int((1-f_sleep/n_seg)*20)
        
        print(f' {ts}  M.{m_state} {bar_m}  F.{f_state} {bar_f}  同步率:{agree_pct:.0f}%', flush=True)
    
    # 5. 共识分析
    both_sleep = sum(1 for t in timeline if t['male']=='睡' and t['female']=='睡')
    both_wake = sum(1 for t in timeline if t['male']=='醒' and t['female']=='醒')
    m_sleep_f_wake = sum(1 for t in timeline if t['male']=='睡' and t['female']=='醒')
    m_wake_f_sleep = sum(1 for t in timeline if t['male']=='醒' and t['female']=='睡')
    
    print(f'\n{"="*65}', flush=True)
    print(' 一致性分析', flush=True)
    print(f'{"="*65}', flush=True)
    print(f' 两人都睡: {both_sleep*0.5:.0f}min ({both_sleep/len(timeline)*100:.0f}%)', flush=True)
    print(f' 两人都醒: {both_wake*0.5:.0f}min ({both_wake/len(timeline)*100:.0f}%)', flush=True)
    print(f' 男睡女醒: {m_sleep_f_wake*0.5:.0f}min ({m_sleep_f_wake/len(timeline)*100:.0f}%)', flush=True)
    print(f' 男醒女睡: {m_wake_f_sleep*0.5:.0f}min ({m_wake_f_sleep/len(timeline)*100:.0f}%)', flush=True)
    
    # 保存
    out = os.path.join(GRANULAR, 'results', 'couple_bss.json')
    with open(out, 'w') as f:
        json.dump({
            'male': {'sleep_min': results_male['sleep_epochs']*0.5, 'wake_min': results_male['wake_epochs']*0.5, 'efficiency': round(results_male['sleep_epochs']/(results_male['sleep_epochs']+results_male['wake_epochs']+0.001)*100,1)},
            'female': {'sleep_min': results_female['sleep_epochs']*0.5, 'wake_min': results_female['wake_epochs']*0.5, 'efficiency': round(results_female['sleep_epochs']/(results_female['sleep_epochs']+results_female['wake_epochs']+0.001)*100,1)},
            'agreement': {'both_sleep_min': both_sleep*0.5, 'both_wake_min': both_wake*0.5, 'm_sleep_f_wake_min': m_sleep_f_wake*0.5, 'm_wake_f_sleep_min': m_wake_f_sleep*0.5},
            'timeline': timeline,
        }, f, ensure_ascii=False, indent=2)
    print(f'\n 保存: {out}', flush=True)
    
    # cleanup
    if os.path.exists(tmp):
        os.remove(tmp)
    
    return results_male, results_female


if __name__ == '__main__':
    analyze_bss_full(r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a')
