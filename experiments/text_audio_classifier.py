"""
音频睡眠分类器 v1 — 从真实录音自训练

核心思路:
  录音里安静时段 = 睡着, 有声音时段 = 觉醒
  用这些自标注数据训练一个音频频谱 → 醒/睡 分类器
"""
import sys, os, json, struct
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from scipy import signal as scipy_signal

GRANULAR = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, GRANULAR)

# 音频专用频带 (手机mic能录到的范围)
AUDIO_BANDS = {
    'sub_low':   (0.5, 2),    # 深层呼吸
    'low':       (2, 8),      # 呼吸/翻身
    'mid':       (8, 20),     # 环境/细微动作  
    'high':      (20, 50),    # 人声/翻身声
    'ultra':     (50, 100),   # 高频噪音/远距离声音
    'breath_var': (0.1, 0.8), # 呼吸率变异性
    'energy_env': (0, 0),     # 总能量
}
BAND_NAMES = list(AUDIO_BANDS.keys())  # 7个特征
SR = 800  # 降到800Hz (0.5-100Hz够用)


def extract_audio_features(audio_seg, sr):
    """从音频段提取7维特征"""
    features = {}
    
    # Welch PSD
    nperseg = min(256, len(audio_seg))
    freqs, psd = scipy_signal.welch(audio_seg, fs=sr, nperseg=nperseg)
    
    # 各频带功率 (log)
    col = 0
    for name, (lo, hi) in AUDIO_BANDS.items():
        if name == 'energy_env':
            features[name] = np.log10(max(np.mean(audio_seg**2), 1e-15))
        else:
            mask = (freqs >= lo) & (freqs < hi)
            bp = np.mean(psd[mask]) if mask.any() else 1e-15
            features[name] = np.log10(max(bp, 1e-15))
    
    # 额外特征: 零穿越率 (人声vs安静)
    zcr = np.mean(np.abs(np.diff(np.sign(audio_seg)))) / 2
    features['zcr'] = np.log10(max(zcr, 1e-10))
    
    # 包络标准差 (翻身/动作)
    envelope = np.abs(audio_seg)
    env_low = scipy_signal.decimate(envelope, max(1, sr//20))
    features['env_std'] = np.log10(max(env_low.std(), 1e-15))
    
    return np.array([features[b] for b in BAND_NAMES] + [features['zcr'], features['env_std']])


def self_label_audio(filepath):
    """
    自标注: 按能量分安静/活跃时段
    
    安静(睡) = 低能量连续段
    活跃(醒) = 高能量段(翻身/说话)
    """
    print('='*60, flush=True)
    print('音频睡眠分类器 — 自训练', flush=True)
    print(f'{os.path.basename(filepath)}', flush=True)
    print('='*60, flush=True)
    
    # 1. 加载 (800Hz)
    tmp = filepath.replace('.m4a', '_tmp800.wav')
    if not os.path.exists(tmp):
        import subprocess
        print('  转码到800Hz...', flush=True)
        subprocess.run([
            r'D:\ffmpeg\bin\ffmpeg', '-y', '-i', filepath,
            '-acodec', 'pcm_s16le', '-ar', str(SR), '-ac', '1', tmp
        ], capture_output=True)
    
    with open(tmp, 'rb') as f:
        f.read(44)
        data = f.read()
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    
    print(f'  {len(audio)/SR/3600:.1f}h @ {SR}Hz', flush=True)
    
    # 2. 逐30s epoch提取特征
    epoch_len = int(30 * SR)
    n_epochs = len(audio) // epoch_len
    n_epochs = min(n_epochs, 960)  # 最多8h
    
    print(f'\n[1/2] 提取特征 ({n_epochs} epochs)...', flush=True)
    features = np.zeros((n_epochs, 9))
    energies = np.zeros(n_epochs)
    
    for i in range(n_epochs):
        if i % 200 == 0 and i > 0:
            print(f'  {i}/{n_epochs}...', flush=True)
        ep = audio[i*epoch_len:(i+1)*epoch_len]
        ep = ep - ep.mean()  # DC
        features[i] = extract_audio_features(ep, SR)
        energies[i] = np.sqrt(np.mean(ep**2))
    
    print(f'\n  能量统计: μ={energies.mean():.6f} σ={energies.std():.6f}', flush=True)
    print(f'  特征范围:', flush=True)
    for j, name in enumerate(BAND_NAMES + ['zcr','env_std']):
        print(f'    {name}: [{features[:,j].min():.2f}, {features[:,j].max():.2f}]', flush=True)
    
    # 3. 自标注: 能量最低的30% = 睡, 最高的20% = 醒
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(features)
    fn = scaler.transform(features)
    
    # 剩余50% = 未知, 不用于训练
    p30 = np.percentile(energies, 30)
    p80 = np.percentile(energies, 80)
    
    quiet_mask = energies < p30
    active_mask = energies > p80
    
    labels_init = np.full(n_epochs, -1, dtype=int)  # -1=未知
    labels_init[quiet_mask] = 0  # 睡
    labels_init[active_mask] = 1  # 醒
    
    train_mask = labels_init >= 0
    X_train = fn[train_mask]
    y_train = labels_init[train_mask]
    
    n_sleep = (y_train == 0).sum()
    n_wake = (y_train == 1).sum()
    print(f'\n[2/2] 自标注: 睡={n_sleep}epochs, 醒={n_wake}epochs, 未知={n_epochs-n_sleep-n_wake}epochs', flush=True)
    
    if n_sleep < 10 or n_wake < 10:
        print(f'  两类样本太少, 无法训练', flush=True)
        return None
    
    # 4. 训练音频睡眠分类器 (LR)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    clf = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced')
    X_train = fn
    y_train = labels_init
    
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f'\n  交叉验证准确率: {cv_scores.mean()*100:.1f}%±{cv_scores.std()*100:.1f}', flush=True)
    
    clf.fit(X_train, y_train)
    
    # 5. 预测整晚 + 双人
    y_pred = clf.predict(X_train)
    probs = clf.predict_proba(X_train)
    
    # 逐epoch醒/睡
    sleep_probs = probs[:, 0]  # 睡的概率
    
    # 双人分离: 低频段高的epoch = 近mic(男方)
    # 音频band: sub_low(呼吸), low, mid
    male_energy = features[:, 0] + features[:, 1]  # sub_low + low
    female_energy = features[:, 2] + features[:, 3]  # mid + high
    ratio = male_energy - female_energy
    med_r = np.median(ratio)
    
    male_epochs = ratio > med_r
    female_epochs = ~male_epochs
    
    # 每个人员的睡/醒分布
    male_sleep_pct = (y_pred[male_epochs] == 0).mean() * 100 if male_epochs.sum() > 0 else 0
    female_sleep_pct = (y_pred[female_epochs] == 0).mean() * 100 if female_epochs.sum() > 0 else 0
    
    print(f'\n===== 双人睡眠分析结果 =====', flush=True)
    print(f'分离阈值(低频/高频比): {med_r:.2f}', flush=True)
    print(f'  男方(近mic): {male_epochs.sum()} epochs ({male_epochs.sum()*0.5:.0f}min)', flush=True)
    print(f'  女方(远mic): {female_epochs.sum()} epochs ({female_epochs.sum()*0.5:.0f}min)', flush=True)
    
    # 眠时间计算
    male_sleep_min = (y_pred[male_epochs] == 0).sum() * 0.5
    male_total_min = male_epochs.sum() * 0.5
    female_sleep_min = (y_pred[female_epochs] == 0).sum() * 0.5
    female_total_min = female_epochs.sum() * 0.5
    
    print(f'\n🧑 男方(录音侧):', flush=True)
    print(f'  睡眠时间: {male_sleep_min:.0f}min / {male_total_min:.0f}min', flush=True)
    print(f'  睡眠效率: {male_sleep_min/male_total_min*100:.0f}%' if male_total_min > 0 else 'N/A', flush=True)
    print(f'  觉醒: {male_total_min - male_sleep_min:.0f}min', flush=True)
    
    print(f'\n👩 女方:', flush=True)
    print(f'  睡眠时间: {female_sleep_min:.0f}min / {female_total_min:.0f}min', flush=True)
    print(f'  睡眠效率: {female_sleep_min/female_total_min*100:.0f}%' if female_total_min > 0 else 'N/A', flush=True)
    print(f'  觉醒: {female_total_min - female_sleep_min:.0f}min', flush=True)
    
    # 整晚睡/醒时间线 (每30min)
    print(f'\n整晚时间线 (每30min):', flush=True)
    for h in range(0, n_epochs, 60):  # 每30min
        end = min(h+60, n_epochs)
        seg_sleep = (y_pred[h:end] == 0).mean()
        seg_male = male_epochs[h:end].mean() if male_epochs[h:end].sum() > 0 else 0.5
        bar_sleep = '█' * int(seg_sleep * 20) + '░' * int((1-seg_sleep) * 20)
        time_label = f'{h*30//3600:02d}:{(h*30%3600)//60:02d}'
        print(f'  {time_label} {bar_sleep} {seg_sleep*100:.0f}%睡 M:{seg_male*100:.0f}%', flush=True)
    
    # 保存模型和结果
    result = {
        'male': {
            'total_min': round(male_total_min, 1),
            'sleep_min': round(male_sleep_min, 1),
            'efficiency': round(male_sleep_min/male_total_min*100, 1) if male_total_min > 0 else 0,
        },
        'female': {
            'total_min': round(female_total_min, 1),
            'sleep_min': round(female_sleep_min, 1),
            'efficiency': round(female_sleep_min/female_total_min*100, 1) if female_total_min > 0 else 0,
        },
        'clf_coefficients': clf.coef_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
    }
    
    out = os.path.join(GRANULAR, 'results', 'audio_sleep_classifier.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f'\n结果保存: {out}', flush=True)
    
    # 清理
    if os.path.exists(tmp):
        os.remove(tmp)
        print('临时文件已清理', flush=True)
    
    return result


if __name__ == '__main__':
    result = self_label_audio(r'D:\AISleepGen_Optimized\sleep_record\20260411_225359.m4a')
