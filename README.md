# 🧠 AISleepGen Granular Sleep — 粒计算睡眠分析

基于信息粒计算的可解释睡眠阶段识别引擎。

## 核心思路

用 **LR (Logistic Regression) z-score + 信息粒平滑** 替代传统深度学习方法做睡眠分期。白盒、可解释、轻量。

```
EEG 信号 → 频谱特征 → z-score → LR分类(93.8%) → 信息粒压缩 → 睡眠质量评分
                                                   ↓
                                             5条可解释规则
```

## 性能

### 全量评估 (Sleep Cassette EDF, 单通道 Fpz-Cz)

| 指标 | 单通道 (45人EC标注) | +后处理(过渡检测) | +双通道(差受试者) |
|------|-------------------|-----------------|-----------------|
| **整体准确率** | 87.1% | — | — |
| **N1 召回率** | 19.1% | **29.1%** (+10.4%) | 2→27% (SC4001) |
| **N1 精确率** | 27.1% | — | 36.7% (SC4001) |
| **N3 召回率** | 82% | — | — |
| **REM 召回率** | 90% | — | — |
| **W 召回率** | 98% | — | — |
| **单epoch推理** | <1ms (单核CPU) | — | — |

**关键发现：**
- N1召回率受标注质量影响最大（好受试者72%, 差受试者0%）→ 不是纯算法问题
- 过渡检测后处理稳定提升N1召回率约10个百分点
- 双通道(Fpz-Cz+Pz-Oz)在差受试者上提升10倍(2.4%→26.8%)，好受试者无提升
- 对比文献多通道EEG: N1 recall 40-50%，本项目单通道+后处理已达29%

## 5 条粒规则 (白盒)

| 阶段 | 规则 (LR系数) | 临床含义 |
|------|--------------|---------|
| **W** | beta↑(+8.66) + sigma↓(-2.21) + delta↑(+3.35) | **觉醒**: 高频活动强，肌电活跃 |
| **N1** | beta↑(+2.28) + theta↓(-1.58) + delta↓(-1.06) | **入睡期**: theta慢波未完全建立 |
| **N2** | beta↓(-4.07) + sigma↑(+3.13) + delta_theta↑(+2.05) | **浅睡**: 纺锤波出现，低频活动 |
| **N3** | beta↓(-4.27) + delta↑(+2.51) + alpha_delta↓(-5.39) | **深睡**: 高Delta, 极低Beta |
| **REM** | sigma↓(-2.68) + beta↓(-2.61) + delta_theta↓(-3.88) | **快速眼动**: 全频带抑制 |

## 睡眠质量评分 (0-100)

| 维度 | 满分 | 测量指标 |
|------|------|---------|
| 睡眠效率 | 30分 | 实际睡眠/卧床时间 |
| 深睡比例 | 20分 | N3占睡眠时间 |
| REM比例 | 20分 | REM占睡眠时间 |
| 睡眠连续性 | 15分 | WASO 时长 |
| 入睡速度 | 15分 | 睡眠潜伏期 |

## 项目结构

```
AISleepGen_GranularSleep/
├── features/
│   ├── spectral.py          # 频谱特征提取 (纯numpy)
│   └── spindle.py           # 纺锤波检测 (scipy移植)
├── granular/
│   ├── granular_stager.py   # ★ 核心: 信息粒分类器
│   ├── sleep_metrics.py     # ★ 核心: 睡眠质量评分
│   ├── fuzzy_classifier.py  # 原型匹配模糊分类器
│   └── fuzzy_sets_*.py      # 模糊集定义实验
├── experiments/
│   ├── text_final.py        # 信息粒增强最终版
│   ├── batch_all.py         # 145人批量实验
│   ├── text_report.py       # 可视化报告生成
│   └── text_compare_structures.py  # 真实vs预测结构对比
└── results/                 # 所有受试者结果
    └── SC4001/
        ├── results.json     # 准确率/混淆矩阵/LR系数
        ├── sleep_report.png # 可视化睡眠报告
        ├── y_pred.npy       # 预测阶段
        └── y_test.npy       # 真实阶段
```

## 快速使用

```python
from granular.granular_stager import GranularSleepStager, STAGES
from features.spectral import compute_epoch_spectrum, band_power
import mne

# 1. 读取 EDF
raw = mne.io.read_raw_edf('SC4001E0-PSG.edf', preload=True)

# 2. 提取频谱特征 (每30s epoch)
bands = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13), 'sigma': (11,16), 'beta': (16,30)}
features = []
for i in range(0, len(raw.times), int(30 * sfreq)):
    epoch = raw.get_data()[:, i:i+int(30*sfreq)]
    freqs, psd = compute_epoch_spectrum(epoch[0], sfreq)
    fvec = [band_power(freqs, psd, b) for b in bands.values()]
    features.append(fvec)

# 3. 训练+预测
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
clf = LogisticRegression().fit(scaler.transform(X_train), y_train)
stager = GranularSleepStager(clf)
stager.fit(X_train, y_train, scaler)

y_pred, confs, granules = stager.predict(X_test)

# 4. 睡眠质量评分
from granular.sleep_metrics import build_sleep_hypnogram, sleep_quality_score
hypno = build_sleep_hypnogram(y_pred, confs)
score = sleep_quality_score(hypno)
print(f"评分: {score['total_score']}/{score['grade']}")
```

## 依赖

```
python >= 3.10
mne >= 1.8
numpy, scipy, scikit-learn, matplotlib
```

## 下一步

- [ ] 实时推理 API (Flask + 单 EDF epoch 流式处理)
- [ ] AISleepGen 微信小程序集成
- [ ] 粒规则论文草稿 (IEEE EMBC / Sleep journal)
- [ ] 跨数据集验证 (SHHS, MESA Sleep 数据集)

## 数据源

Sleep Cassette subset of the Sleep EDF Expanded Database (PhysioNet).
40+ 受试者，Fpz-Cz 单极导联，100Hz 采样率，30s epochs.

## License

MIT
