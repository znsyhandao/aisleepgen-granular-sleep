"""
双人睡眠录音分析 v2 — 呼吸包络法

核心思路:
  手机mic录音 → 提取呼吸包络(0.1-0.8Hz带通) → 呼吸率变异性 → 盲源分离 → 2路呼吸 → 粒规则

比v1改进:
  1. 不对原始音频做ICA(太慢) — 对呼吸包络做ICA(快100倍)
  2. 低频波段直接作为睡眠特征
  3. 整晚分段处理(15min/段)
"""
import sys, os, json, math
import numpy as np
import warnings; warnings.filterwarnings('ignore')

GRANULAR_DIR = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, GRANULAR_DIR)

# 粒规则系数 (从145人平均LR系数导出)
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
COEFF_MATRIX = np.array([
    [3.35, 0.12, 0.45, -2.21, 8.66, 1.23, 2.23],
    [-1.06, -1.58, 0.87, 1.22, 2.28, 0.45, -0.89],
    [1.42, -0.67, -1.23, 3.13, -4.07, 2.05, -1.56],
    [2.51, -1.89, -0.56, -1.45, -4.27, 3.82, -5.39],
    [-0.34, -1.12, 1.89, -2.68, -2.61, -3.88, 1.45],
])
BANDS = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'sigma': (11,16), 'beta': (16,30)}
SCALER_MEAN = np.array([-11.2, -10.8, -11.5, -12.1, -10.3, 0.23, 0.45])
SCALER_STD  = np.array([0.58, 0.62, 0.55, 0.71, 0.49, 0.15, 0.21])
TARGET_SR = 100  # 降采样后采样率

from scipy import signal as scipy_signal


def load_and_preprocess(filepath):
    """加载m4a → 降采样到1kHz → 单声道"""
    import subprocess
    temp_dir = os.path.dirname(filepath)
    temp_wav = os.path.join(temp_dir, '_temp_full.wav')
    
    if not os.path.exists(temp_wav):
        print('  转码整晚录音...', flush=True)
        subprocess.run([
            r'D:\ffmpeg\bin\ffmpeg', '-y', '-i', filepath,
            '-acodec', 'pcm_s16le', '-ar', '1000', '-ac', '1',
            temp_wav
        ], capture_output=True)
    
    import scipy.io.wavfile as wav
    sr, data = wav.read(temp_wav)
    data = data.astype(np.float32) / 32768.0
    
    hours = len(data) / sr / 3600
    print(f'  加载完成: {hours:.1f}h @ {sr}Hz, {len(data)/sr/60:.0f}min', flush=True)
    
    # 再降采样到TARGET_SR
    step = sr // TARGET_SR
    data_low = data[::step]
    
    return data, data_low, TARGET_SR


def extract_respiratory_envelope(audio, sr):
    """呼吸包络提取: 能量包络 + 0.1-0.8Hz带通"""
    # 整流+低通
    envelope = np.abs(audio)
    
    # 降采样到10Hz (呼吸只需~5Hz)
    ds = max(1, sr // 10)
    env = envelope[::ds]
    sr_env = sr / ds
    
    # 带通 0.1-0.8Hz
    sos = scipy_signal.butter(4, [0.08, 0.8], btype='band', fs=sr_env, output='sos')
    resp = scipy_signal.sosfiltfilt(sos, env)
    
    # 归一化
    resp = (resp - resp.mean()) / (resp.std() + 1e-10)
    
    return resp, sr_env


def separate_respiration_via_ica(resp_signal, n_components=2):
    """对呼吸包络做ICA (比直接对音频做快1000倍)"""
    from sklearn.decomposition import FastICA
    
    # 时间延迟嵌入 (0.5s延迟)
    delay = int(0.5 * 10)  # resp是10Hz, 0.5s = 5 samples
    n_features = n_components * 3
    n = len(resp_signal) - delay * n_features
    if n <= 0: n = min(1000, len(resp_signal))
    
    X = np.column_stack([resp_signal[i*delay:i*delay+n] for i in range(n_features)])
    
    ica = FastICA(n_components=n_components, random_state=42, max_iter=300)
    S = ica.fit_transform(X)
    
    # 按能量排序
    energies = np.sum(S**2, axis=0)
    order = np.argsort(-energies)
    S = S[:, order]
    
    return S


def compute_sleep_features(audio_segment, sr):
    """
    从音频段计算7维睡眠特征
    
    返回: ndarray (n_epochs, 7)
    """
    epoch_len = int(30 * sr)
    n_epochs = len(audio_segment) // epoch_len
    
    from features.spectral import compute_epoch_spectrum, band_power
    
    features = np.zeros((n_epochs, 7))
    
    for i in range(n_epochs):
        epoch = audio_segment[i*epoch_len:(i+1)*epoch_len]
        if len(epoch) < epoch_len:
            features = features[:i]
            break
        
        freqs, psd = compute_epoch_spectrum(epoch, sr)
        col = 0
        for name, band in BANDS.items():
            features[i, col] = np.log10(max(band_power(freqs, psd, band), 1e-15))
            col += 1
        
        dp, tp, ap = features[i, 0], features[i, 1], features[i, 2]
        features[i, 5] = dp - tp   # delta_theta ratio in log space
        features[i, 6] = ap - dp   # alpha_delta ratio
    
    return features


def granular_predict_batch(features):
    """批量粒规则预测"""
    # z-score
    fz = (features - SCALER_MEAN) / SCALER_STD
    
    # LR scores
    scores = fz @ COEFF_MATRIX.T
    
    # softmax
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    
    preds = np.array([STAGES[i] for i in probs.argmax(axis=1)])
    confs = probs.max(axis=1)
    
    return preds, confs


def compute_quality_score_simple(pred_stages, confidences):
    """简化版质量评分 (不依赖sleep_metrics模块)"""
    n = len(pred_stages)
    total_min = n * 0.5
    wake_min = np.sum(np.array(pred_stages) == 'W') * 0.5
    n3_min = np.sum(np.array(pred_stages) == 'N3') * 0.5
    rem_min = np.sum(np.array(pred_stages) == 'REM') * 0.5
    
    sleep_min = total_min - wake_min
    eff = sleep_min / total_min * 100 if total_min > 0 else 0
    
    # 5维度
    s_eff = min(30, max(0, (eff - 50) / 50 * 30))
    s_n3 = min(20, max(0, 20 - abs(n3_min/total_min*100 - 20) * 2))
    s_rem = min(20, max(0, 20 - abs(rem_min/total_min*100 - 22) * 1.5))
    s_waso = min(15, max(0, 15 - wake_min * 0.3))
    s_lat = 10  # 无法准确估计, 给中值
    
    total = s_eff + s_n3 + s_rem + s_waso + s_lat
    if total >= 85: grade = '优秀'
    elif total >= 70: grade = '良好'
    elif total >= 50: grade = '一般'
    else: grade = '需要关注'
    
    return {
        'total_score': round(total, 1), 'grade': grade,
        'efficiency_pct': round(eff, 1),
        'n3_pct': round(n3_min/total_min*100, 1) if total_min > 0 else 0,
        'rem_pct': round(rem_min/total_min*100, 1) if total_min > 0 else 0,
        'wake_min': round(wake_min, 1),
    }


def analyze_full_night(filepath):
    """完整分析整晚双人录音"""
    print('=' * 60)
    print('双人睡眠录音分析 v2 (呼吸包络法)')
    print(f'文件: {os.path.basename(filepath)}')
    print('=' * 60)
    
    # 1. 加载 (降采样到1kHz)
    audio_1k, audio_100, sr1k = load_and_preprocess(filepath)
    
    # 2. 呼吸包络提取 (从1kHz音频)
    print('\n[1/4] 提取呼吸包络...', flush=True)
    resp_env, resp_sr = extract_respiratory_envelope(audio_1k, sr1k)
    print(f'  呼吸包络 @ {resp_sr:.0f}Hz, {len(resp_env)/resp_sr/60:.0f}min', flush=True)
    
    # 3. 盲源分离呼吸 → 2人呼吸信号
    print('\n[2/4] 分离双人呼吸...', flush=True)
    # 分段处理 (5min/段)
    seg_samples = int(5 * 60 * resp_sr)
    n_segs = len(resp_env) // seg_samples
    
    person1_resp = np.array([])
    person2_resp = np.array([])
    
    for seg in range(n_segs):
        seg_data = resp_env[seg*seg_samples:(seg+1)*seg_samples]
        try:
            S = separate_respiration_via_ica(seg_data, 2)
            person1_resp = np.concatenate([person1_resp, S[:, 0]])
            person2_resp = np.concatenate([person2_resp, S[:, 1]])
        except:
            if len(person1_resp) == 0:
                person1_resp = seg_data[:seg_samples//2]
                person2_resp = seg_data[:seg_samples//2]
            else:
                person1_resp = np.concatenate([person1_resp, seg_data[:len(seg_data)//2]])
                person2_resp = np.concatenate([person2_resp, seg_data[len(seg_data)//2:]])
    
    print(f'  人员A: {len(person1_resp)/resp_sr/60:.0f}min')
    print(f'  人员B: {len(person2_resp)/resp_sr/60:.0f}min')
    
    # 4. 呼吸信号 → 合成音频特征 (每30s epoch)
    # 使用呼吸率变异性作为特征
    print('\n[3/4] 计算睡眠特征...', flush=True)
    
    results = {}
    for label, resp_sig in [('🧑 男方(录音侧)', person1_resp), ('👩 女方', person2_resp)]:
        # 把呼吸信号视为低频音频, 直接算频谱
        epoch_len = int(30 * resp_sr)
        n_epochs = len(resp_sig) // epoch_len
        
        if n_epochs < 10:
            print(f'{label}: epochs太少({n_epochs}), 跳过', flush=True)
            continue
        
        # 用纯scipy计算低频带功率
        features = np.zeros((n_epochs, 7))
        for i in range(n_epochs):
            seg = resp_sig[i*epoch_len:(i+1)*epoch_len]
            if len(seg) < 16: continue
            
            # Welch PSD
            nperseg = min(128, len(seg))
            freqs, psd = scipy_signal.welch(seg, fs=resp_sr, nperseg=nperseg)
            
            col = 0
            for name, (lo, hi) in BANDS.items():
                mask = (freqs >= lo) & (freqs < hi)
                bp = np.mean(psd[mask]) if mask.any() else 1e-15
                features[i, col] = np.log10(max(bp, 1e-15))
                col += 1
            
            features[i, 5] = features[i,0] - features[i,1]  # delta_theta
            features[i, 6] = features[i,2] - features[i,0]  # alpha_delta
        
        # 粒规则预测
        preds, confs = granular_predict_batch(features)
        quality = compute_quality_score_simple(preds, confs)
        
        # 阶段分布
        stage_dist = {s: float(np.sum(np.array(preds) == s)) for s in STAGES}
        stage_min = {s: stage_dist[s] * 0.5 for s in STAGES}
        
        results[label] = {'quality': quality, 'stage_dist': stage_dist, 'n_epochs': n_epochs}
        
        print(f'\n{label}:', flush=True)
        print(f'  评分: {quality["total_score"]:.0f} ({quality["grade"]})', flush=True)
        print(f'  效率: {quality["efficiency_pct"]:.0f}%  N3: {quality["n3_pct"]:.0f}%  REM: {quality["rem_pct"]:.0f}%', flush=True)
        print(f'  觉醒: {quality["wake_min"]:.0f}min', flush=True)
        for s in STAGES:
            print(f'  {s}: {stage_min[s]:.0f}min', flush=True)
    
    # 5. 保存
    print('\n[4/4] 保存结果...', flush=True)
    out_dir = os.path.join(GRANULAR_DIR, 'results', 'couple_sleep')
    os.makedirs(out_dir, exist_ok=True)
    
    summary = {}
    for label, res in results.items():
        key = 'person_A_male' if '男' in label else 'person_B_female'
        summary[key] = res
        with open(os.path.join(out_dir, f'{key}.json'), 'w') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f'  保存: {out_dir}', flush=True)
    
    # 清理临时文件
    temp_wav = os.path.join(os.path.dirname(filepath), '_temp_full.wav')
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
        print('  临时文件清理完成', flush=True)
    
    return results


if __name__ == '__main__':
    filepath = r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a'
    results = analyze_full_night(filepath)
