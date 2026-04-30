# AISleepGen GranularSleep — 粒计算睡眠阶段识别

## 目标
用粒计算(Granular Computing) + 模糊逻辑做睡眠阶段检测，替代传统 CNN。
比深度学习更可解释，可发表论文，可集成进 AISleepGen 产品。

## 方法论

### 核心思路
1. **多粒度信息粒构建**
   - EEG/EOG/EMG 信号 → 时域/频域/非线性特征
   - 按 30s epoch（传统）→ 按生理事件粒度（过渡期、纺锤波、K复合波）
   - 粗粒度：睡眠阶段（W/N1/N2/N3/REM）
   - 细粒度：微结构（纺锤波、K复合波、慢波、眼动）

2. **模糊粒化**
   - 每个特征轴用模糊集编码（低/中/高 delta 功率）
   - 睡眠阶段边界不是硬的 → 用隶属度函数表示"过渡概率"

3. **粒计算推理**
   - 用粒的包含、重叠、邻近关系做推理
   - 而不是"softmax 出 5 个概率"（黑箱）

### 数据
- Sleep Cassette (SC) 数据集：153 受试者
- 每人一整夜 PSG + 专家标注 Hypnogram
- 标准 30s epoch × 5 阶段

### 评估
- 与传统 CNN/Random Forest 对比
- 重点指标：过渡期准确率（N1↔N2 边界 vs 稳态期）
- 可解释性：粒规则可视化

## 目录结构

```
D:\AISleepGen_GranularSleep\
├── data/                  EDF 读取缓存（轻量预处理）
│   └── metadata.csv       被试元数据
├── granular/              粒计算核心
│   ├── info_granules.py   信息粒构建
│   ├── fuzzy_sets.py      模糊集定义
│   └── granular_reason.py 粒推理引擎
├── features/              特征提取
│   ├── spectral.py        频谱特征
│   ├── nonlinear.py       非线性特征(Hurst, 熵)
│   └── micro_events.py    微结构事件检测(纺锤波/K复合波)
├── evaluation/            评估
│   ├── baseline_cnn.py    CNN 对比基线
│   ├── metrics.py         过渡期指标
│   └── interpretability.py  可解释性分析
├── experiments/           实验管理
│   ├── run_pipeline.py    主流水线
│   └── config.py          配置
├── results/               结果输出
├── paper/                 LaTeX 论文模板
└── README.md
```

## 实施计划

### Phase 1: 数据管道（本日）
- [ ] EDF 读取 + epoch 提取
- [ ] 频谱特征计算（delta/theta/alpha/sigma/beta）
- [ ] metadata 构建

### Phase 2: 粒计算核心（明日）
- [ ] 信息粒构建
- [ ] 模糊集定义
- [ ] 粒推理引擎

### Phase 3: 基线对比
- [ ] CNN 分类器
- [ ] 随机森林
- [ ] 粒计算 vs 基线对比

### Phase 4: 论文
- [ ] 结果分析
- [ ] 论文撰写
- [ ] 提交

## 护城河价值
1. 粒计算睡眠阶段识别 → 比 CNN 可解释
2. 过渡期检测精度提升（N1→N2 边界是临床难点）
3. 可集成进 AISleepGen → 产品卖点
4. 论文可发表 → 学术背书
