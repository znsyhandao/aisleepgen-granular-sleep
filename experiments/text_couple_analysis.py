"""
双人睡眠录音 盲源分离 + 粒规则双路评分

输入: 手机单mic录音 (整晚, 双人睡眠)
流程: 盲源分离 → 两路呼吸信号 → 各30s epoch频谱 → 各粒规则评分
"""
import sys, os, json
import numpy as np
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, r'D:\AISleepGen_GranularSleep')
from features.spectral import compute_epoch_spectrum, band_power

EDF_BANDS = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'sigma': (11,16), 'beta': (16,30)}
FEATURE_NAMES = [f'{n}_power' for n in EDF_BANDS] + ['delta_theta_ratio', 'alpha_delta_ratio']

# ===== 粒规则系数 (从145人平均LR系数导出) =====
STAGE_COEFFS = {
    'W':   [3.35, 0.12, 0.45, -2.21, 8.66, 1.23, 2.23],
    'N1':  [-1.06, -1.58, 0.87, 1.22, 2.28, 0.45, -0.89],
    'N2':  [1.42, -0.67, -1.23, 3.13, -4.07, 2.05, -1.56],
    'N3':  [2.51, -1.89, -0.56, -1.45, -4.27, 3.82, -5.39],
    'REM': [-0.34, -1.12, 1.89, -2.68, -2.61, -3.88, 1.45],
}
STAGE_NAMES = list(STAGE_COEFFS.keys())
COEFF_MATRIX = np.array([STAGE_COEFFS[s] for s in STAGE_NAMES])

# SC4001 scaler (从145人批量实验)
SCALER_MEAN = np.array([-11.2, -10.8, -11.5, -12.1, -10.3, 0.23, 0.45])
SCALER_STD  = np.array([0.58, 0.62, 0.55, 0.71, 0.49, 0.15, 0.21])


def load_audio(filepath):
    """加载m4a音频文件"""
    import subprocess
    temp_wav = filepath.replace('.m4a', '_temp.wav')
    
    # 用ffmpeg转码 (需要ffmpeg在PATH)
    if not os.path.exists(temp_wav):
        print(f'  转码: {os.path.basename(filepath)} ...', flush=True)
        subprocess.run([
            'ffmpeg', '-y', '-i', filepath,
            '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
            temp_wav
        ], capture_output=True)
    
    import scipy.io.wavfile as wav
    sr, data = wav.read(temp_wav)
    
    # 转换为float
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    
    print(f'  采样率: {sr}Hz, 时长: {len(data)/sr/3600:.1f}h', flush=True)
    return data, sr


def blind_source_separation(mixed_signal, n_sources=2):
    """
    盲源分离: FastICA
    
    输入: 单通道混合信号
    输出: n_sources 路分离信号
    """
    from sklearn.decomposition import FastICA
    
    # 构建延迟嵌入 (单通道 → 多通道, 使用时间延迟)
    delay = int(0.1 * 44100)  # 100ms 延迟
    n_frames = len(mixed_signal) - delay * n_sources
    if n_frames <= 0:
        return np.array([mixed_signal, mixed_signal * 0.5])
    
    # 创建多路
    channels = []
    for i in range(n_sources):
        channels.append(mixed_signal[i*delay:i*delay+n_frames])
    
    X = np.column_stack(channels)
    
    # FastICA
    ica = FastICA(n_components=n_sources, random_state=42, max_iter=500)
    S = ica.fit_transform(X)
    
    # 按能量排序 (确保一致性)
    energies = np.sum(S**2, axis=0)
    order = np.argsort(-energies)
    
    return S[:, order].T  # [n_sources, n_samples]


def extract_respiration(signal, sr=44100):
    """从音频提取呼吸包络 (0.1-0.8Hz 带通)"""
    from scipy import signal as scipy_signal
    
    # 降采样到100Hz (呼吸频率低)
    downsample = int(sr / 100)
    if downsample > 1:
        signal = scipy_signal.decimate(signal, downsample, ftype='fir')
    sr_new = 100
    
    # 绝对值 + 低通 (提取包络)
    envelope = np.abs(signal)
    
    # 带通滤波 0.1-0.8Hz (呼吸范围)
    sos = scipy_signal.butter(4, [0.08, 0.8], btype='band', fs=sr_new, output='sos')
    resp = scipy_signal.sosfiltfilt(sos, envelope)
    
    # 降采样到2Hz (30s epoch = 60 samples)
    downsample2 = int(sr_new / 2)
    if downsample2 > 1:
        resp = resp[::downsample2]
    
    return resp


def person_sleep_analysis(audio_segment, sr):
    """单路音频 → 睡眠评分"""
    epoch_len = int(30 * sr)
    total_epochs = len(audio_segment) // epoch_len
    
    stage_counts = {s: 0 for s in STAGE_NAMES}
    epoch_scores = []
    
    print(f'  分析 {total_epochs} epochs ({total_epochs*0.5:.0f}min)...', flush=True)
    
    for i in range(total_epochs):
        epoch = audio_segment[i*epoch_len:(i+1)*epoch_len]
        if len(epoch) < epoch_len:
            break
        
        # 频谱特征
        freqs, psd = compute_epoch_spectrum(epoch, sr)
        features = np.zeros(7)
        col = 0
        for name, band in EDF_BANDS.items():
            features[col] = np.log10(max(band_power(freqs, psd, band), 1e-15))
            col += 1
        dp, tp, ap = features[0], features[1], features[2]
        features[5] = dp - tp   # delta_theta ratio in log space
        features[6] = ap - dp   # alpha_delta ratio
        
        # z-score
        features_z = (features - SCALER_MEAN) / SCALER_STD
        
        # LR softmax
        scores = features_z @ COEFF_MATRIX.T
        scores -= scores.max()
        exp_s = np.exp(scores)
        probs = exp_s / exp_s.sum()
        
        pred = STAGE_NAMES[np.argmax(probs)]
        conf = probs.max()
        
        stage_counts[pred] = stage_counts.get(pred, 0) + 1
        epoch_scores.append({
            'epoch': i,
            'stage': pred,
            'confidence': float(conf),
            'time': f'{i*30//3600:02d}:{(i*30%3600)//60:02d}',
        })
    
    # 汇总
    from granular.sleep_metrics import build_sleep_hypnogram, sleep_quality_score
    pred_stages = np.array([e['stage'] for e in epoch_scores])
    confidences = np.array([e['confidence'] for e in epoch_scores])
    
    hypno = build_sleep_hypnogram(pred_stages, confidences)
    score = sleep_quality_score(hypno)
    
    # 阶段分布
    stage_dist = {}
    for s in STAGE_NAMES:
        n = stage_counts.get(s, 0)
        stage_dist[s] = {
            'epochs': n,
            'minutes': n * 0.5,
            'pct': n / total_epochs * 100 if total_epochs > 0 else 0,
        }
    
    return {
        'n_epochs': total_epochs,
        'score': score,
        'hypnogram': hypno,
        'stage_distribution': stage_dist,
        'epochs': epoch_scores,
    }


def analyze_couple_recording(filepath):
    """完整分析双人录音"""
    print(f'\n{"="*60}', flush=True)
    print(f'双人睡眠录音分析', flush=True)
    print(f'文件: {os.path.basename(filepath)}', flush=True)
    print(f'{"="*60}', flush=True)
    
    # 1. 加载
    mixed_audio, sr = load_audio(filepath)
    
    # 2. 盲源分离 (只取前30分钟做延迟嵌入, 整晚内存不够)
    # 分段处理: 每30分钟一轮
    segment_minutes = 30
    segment_samples = segment_minutes * 60 * sr
    n_segments = len(mixed_audio) // segment_samples
    
    print(f'\n分段分离 ({n_segments} × {segment_minutes}min)...', flush=True)
    
    results = {'person_A': [], 'person_B': [], 'raw': []}
    
    for seg in range(n_segments):
        seg_start = seg * segment_samples
        seg_end = (seg + 1) * segment_samples
        segment = mixed_audio[seg_start:seg_end]
        
        try:
            sources = blind_source_separation(segment, n_sources=2)
            if len(sources) >= 2:
                results['person_A'].append(sources[0])
                results['person_B'].append(sources[1])
            results['raw'].append(segment)
        except Exception as e:
            print(f'  段{seg}: 分离失败 {str(e)[:50]}', flush=True)
            if len(results['raw']) > 0:
                results['person_A'].append(results['raw'][-1] * 0.5)
                results['person_B'].append(results['raw'][-1] * 0.3)
            continue
        
        if (seg + 1) % 5 == 0:
            print(f'  已处理 {(seg+1)*segment_minutes}min...', flush=True)
    
    print(f'  分离完成: {len(results["person_A"])} 段', flush=True)
    
    # 3. 拼接
    person_A_audio = np.concatenate(results['person_A']) if results['person_A'] else mixed_audio[:segment_samples]
    person_B_audio = np.concatenate(results['person_B']) if results['person_B'] else mixed_audio[:segment_samples]
    
    # 4. 各自分析
    print(f'\n{"="*60}', flush=True)
    print(f'【男方】(首选, 录音位置侧) 睡眠分析', flush=True)
    print(f'{"="*60}', flush=True)
    result_a = person_sleep_analysis(person_A_audio, sr)
    
    print(f'\n{"="*60}', flush=True)
    print(f'【女方】睡眠分析', flush=True)
    print(f'{"="*60}', flush=True)
    result_b = person_sleep_analysis(person_B_audio, sr)
    
    # 5. 输出汇总
    print(f'\n{"="*60}', flush=True)
    print(f'双人睡眠质量对比', flush=True)
    print(f'{"="*60}', flush=True)
    
    for label, res in [('🧑 男方', result_a), ('👩 女方', result_b)]:
        score = res['score']
        dist = res['stage_distribution']
        print(f'\n{label}:')
        print(f'  综合评分: {score["total_score"]:.0f}/{score["grade"]}')
        print(f'  有效分析: {res["n_epochs"]*0.5:.0f}min ({res["n_epochs"]} epochs)')
        print(f'  深睡(N3):  {dist["N3"]["pct"]:.1f}% ({dist["N3"]["minutes"]:.0f}min)')
        print(f'  REM:        {dist["REM"]["pct"]:.1f}% ({dist["REM"]["minutes"]:.0f}min)')
        print(f'  浅睡(N2):  {dist["N2"]["pct"]:.1f}% ({dist["N2"]["minutes"]:.0f}min)')
        print(f'  过渡(N1):  {dist["N1"]["pct"]:.1f}% ({dist["N1"]["minutes"]:.0f}min)')
        print(f'  觉醒(W):   {dist["W"]["pct"]:.1f}% ({dist["W"]["minutes"]:.0f}min)')
        
        dims = score['dimensions']
        print(f'  效率: {dims["sleep_efficiency"]/30*100:.0f}%')
        print(f'  WASO: {res["hypnogram"]["summary"].get("waso_min",0):.0f}min')
    
    return result_a, result_b


if __name__ == '__main__':
    import sys
    filepath = r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a'
    
    if not os.path.exists(filepath):
        print(f'文件不存在: {filepath}')
        sys.exit(1)
    
    result_a, result_b = analyze_couple_recording(filepath)
    
    # 保存结果
    out_dir = r'D:\AISleepGen_GranularSleep\results\couple_sleep'
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'person_A_male.json'), 'w') as f:
        json.dump({'score': result_a['score'], 'stage_distribution': result_a['stage_distribution']}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, 'person_B_female.json'), 'w') as f:
        json.dump({'score': result_b['score'], 'stage_distribution': result_b['stage_distribution']}, f, indent=2, ensure_ascii=False)
    
    print(f'\n结果已保存: {out_dir}', flush=True)
