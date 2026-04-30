"""
双人睡眠录音分析 v3 — 能量分帧 + 频谱重映射

改进:
  1. 用原始1kHz音频直接做epoch频谱 (不是呼吸包络)
  2. 分帧时按能量高低分离两路 (近mic者声音更大)
  3. 频谱映射校准
"""
import sys, os, json
import numpy as np
import warnings; warnings.filterwarnings('ignore')

GRANULAR_DIR = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, GRANULAR_DIR)

from scipy import signal as scipy_signal
from features.spectral import compute_epoch_spectrum, band_power

# 粒规则
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


def load_audio_fast(filepath):
    """用ffmpeg转码到8kHz (保持低频分辨率的平衡)"""
    temp_wav = filepath.replace('.m4a', '_tmp8k.wav')
    if not os.path.exists(temp_wav):
        import subprocess
        print('  转码到8kHz...', flush=True)
        subprocess.run([
            r'D:\ffmpeg\bin\ffmpeg', '-y', '-i', filepath,
            '-acodec', 'pcm_s16le', '-ar', '8000', '-ac', '1',
            temp_wav
        ], capture_output=True)
    
    # 不加载整个文件, 用mmap分段读
    import mmap
    sr = 8000
    print(f'  文件大小: {os.path.getsize(temp_wav)/1e6:.0f}MB, 准备分段读取', flush=True)
    return None, sr, temp_wav  # 返回None, 实际在analyze里分段


def process_segment(seg_data, sr):
    """处理一段音频 → 特征矩阵"""
    frame_len = int(30 * sr)
    n_frames = len(seg_data) // frame_len
    features = np.zeros((n_frames, 7))
    
    for i in range(n_frames):
        epoch = seg_data[i*frame_len:(i+1)*frame_len]
        if len(epoch) < frame_len: continue
        
        try:
            features[i] = compute_epoch_features(epoch, sr)
        except:
            features[i] = np.array([-15, -15, -15, -15, -15, 0, 0])
    
    return features


def analyze_couple_v3(filepath):
    print('=' * 60)
    print('双人睡眠分析 v3 — 分段处理')
    print('=' * 60)
    
    _, sr, tmp_wav = load_audio_fast(filepath)
    import scipy.io.wavfile as wav
    
    # 分段: 每1小时一块
    seg_hours = 1.0
    seg_samples = int(seg_hours * 3600 * sr)
    frame_len = int(30 * sr)
    
    seg_idx = 0
    all_features = []
    
    with open(tmp_wav, 'rb') as f:
        # skip wav header (44 bytes)
        header = f.read(44)
        # get total samples from header
        data_size = os.path.getsize(tmp_wav) - 44
        n_total = data_size // 2  # int16
        
        for offset in range(44, os.path.getsize(tmp_wav), seg_samples * 2):
            n_read = min(seg_samples, (os.path.getsize(tmp_wav) - offset) // 2)
            if n_read <= 0: break
            
            # read segment
            f.seek(offset)
            raw = f.read(n_read * 2)
            seg = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            
            print(f'  段{seg_idx}: {len(seg)/sr/60:.0f}min...', flush=True)
            feat = process_segment(seg, sr)
            all_features.append(feat)
            seg_idx += 1
    
    total_features = np.concatenate(all_features, axis=0)
    print(f'\n总epochs: {len(total_features)} ({len(total_features)*30//3600}h)', flush=True)


def compute_epoch_features(epoch, sr):
    """单epoch 7维特征"""
    freqs, psd = compute_epoch_spectrum(epoch, sr)
    features = np.zeros(7)
    col = 0
    for name, band in BANDS.items():
        features[col] = np.log10(max(band_power(freqs, psd, band), 1e-15))
        col += 1
    features[5] = features[0] - features[1]  # delta_theta
    features[6] = features[2] - features[0]  # alpha_delta
    return features


def granular_predict(features):
    """粒规则预测"""
    fz = (features - SCALER_MEAN) / SCALER_STD
    scores = fz @ COEFF_MATRIX.T
    scores -= scores.max()
    probs = np.exp(scores) / np.exp(scores).sum()
    pred = STAGES[np.argmax(probs)]
    return pred, float(probs.max())


def quality_from_preds(preds):
    """从预测序列算质量评分"""
    n = len(preds)
    if n == 0: return {'total_score': 0, 'grade': '未知'}
    total_min = n * 0.5
    stage_min = {s: np.sum(np.array(preds) == s) * 0.5 for s in STAGES}
    sleep_min = total_min - stage_min['W']
    eff = sleep_min / total_min * 100 if total_min > 0 else 0
    
    s_eff = min(30, max(0, (eff - 50) / 50 * 30))
    s_n3 = min(20, max(0, 20 - abs(stage_min['N3']/total_min*100 - 20) * 2))
    s_rem = min(20, max(0, 20 - abs(stage_min['REM']/total_min*100 - 22) * 1.5))
    s_waso = min(15, max(0, 15 - stage_min['W'] * 0.3))
    s_lat = 10
    
    total = min(100, s_eff + s_n3 + s_rem + s_waso + s_lat)
    grade = '优秀' if total >= 85 else '良好' if total >= 70 else '一般' if total >= 50 else '需要关注'
    
    return {'total_score': round(total, 1), 'grade': grade, 'eff_pct': round(eff, 1),
            'n3_pct': round(stage_min['N3']/total_min*100, 1),
            'rem_pct': round(stage_min['REM']/total_min*100, 1),
            'wake_min': round(stage_min['W'], 1)}


def analyze_couple_v3(filepath):
    print('=' * 60)
    print('双人睡眠分析 v3 — 能量分帧 + 1kHz频谱')
    print(f'文件: {os.path.basename(filepath)}')
    print('=' * 60)
    
    # 1. 加载 (分段)
    _, sr, tmp_wav = load_audio_fast(filepath)
    import scipy.io.wavfile as wav
    
    # 3. 用能量分两层 — 高能帧(近mic=男方觉醒/动作), 低能帧(远mic=女方)
    # 但两人都睡时能量差异小 — 不用硬切分, 用低频/高频比做软分离
    total_features = np.zeros((len(frames), 7))
    print('\n[2/3] 计算频谱特征...', flush=True)
    
    for i in range(len(frames)):
        if i % 200 == 0 and i > 0:
            print(f'  {i}/{len(frames)} epochs ({i*30/3600:.1f}h)...', flush=True)
        try:
            total_features[i] = compute_epoch_features(frames[i], sr)
        except:
            total_features[i] = np.array([-12, -11, -12, -13, -11, -1, 0])
    
    # 5. 特征分布分析 (先看看原始功率值)
    print('\n  原始log功率范围:', flush=True)
    for j, name in enumerate(BANDS):
        log_p = total_features[:, j]
        print(f'    {name}: [{log_p.min():.2f}, {log_p.max():.2f}] mean={log_p.mean():.2f}', flush=True)
    
    # 5. 粒规则预测 (整晚整体)
    print('\n[3/3] 粒规则预测...', flush=True)
    all_preds = []
    for i in range(len(total_features)):
        pred, conf = granular_predict(total_features[i])
        all_preds.append(pred)
    
    # 按低频/高频比分离双人
    # 近mic者: 呼吸+体动声音大 → 低频高 = 男方
    # 远mic者: 混响+环境声 → 高频占比高 = 女方
    low_power = total_features[:, 0]  # delta
    high_power = total_features[:, 4]  # beta
    ratio = low_power - high_power  # log space
    
    # 用ratio中位切割
    med_ratio = np.median(ratio)
    idx_male = ratio > med_ratio    # 低频多 = 男方近
    idx_female = ratio <= med_ratio  # 高频多 = 女方远
    
    preds_male = np.array(all_preds)[idx_male]
    preds_female = np.array(all_preds)[idx_female]
    
    q_male = quality_from_preds(preds_male)
    q_female = quality_from_preds(preds_female)
    
    print(f'\n分离阈值: ratio中位={med_ratio:.2f}', flush=True)
    print(f'  男方(近mic,低频多): {idx_male.sum()} epochs', flush=True)
    print(f'  女方(远mic,高频多): {idx_female.sum()} epochs', flush=True)
    
    for label, preds, q in [('🧑 男方', preds_male, q_male), ('👩 女方', preds_female, q_female)]:
        print(f'\n{label}:', flush=True)
        print(f'  综合评分: {q["total_score"]:.0f} ({q["grade"]})', flush=True)
        print(f'  效率: {q["eff_pct"]:.0f}%', flush=True)
        print(f'  深睡N3: {q["n3_pct"]:.0f}%   REM: {q["rem_pct"]:.0f}%  觉醒: {q["wake_min"]:.0f}min', flush=True)
        
        stage_dist = {s: int(np.sum(np.array(preds) == s)) for s in STAGES}
        for s in STAGES:
            print(f'  {s}: {stage_dist[s]*0.5:.0f}min ({stage_dist[s]/len(preds)*100:.0f}%)', flush=True)
    
    return q_male, q_female


if __name__ == '__main__':
    filepath = r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a'
    analyze_couple_v3(filepath)
    
    # 清理
    tmp = filepath.replace('.m4a', '_tmp8k.wav')
    if os.path.exists(tmp): os.remove(tmp)
