"""方案A: N1过渡区Boost - 直接替换"""
with open('granular/explainable_classifier.py', 'r', encoding='utf-8') as f:
    c = f.read()

# 简单的标记替换
old_marker = '# z-score 高 = 贡献高'
new_marker = '# z-score 高 = 贡献高 (保持原逻辑)'

old_end = '        return score'
new_end = '        # [方案A] N1过渡区区域Boost\n        if stage == "N1":\n            try:\n                bi = self.feature_names_.index("beta_power")\n                di = self.feature_names_.index("delta_theta_ratio")\n                bz, dz = features_z[bi], features_z[di]\n                if -1.0 < bz < 1.0 and -1.0 < dz < 1.0:\n                    score *= 1.3\n            except ValueError:\n                pass\n\n        return score'

# 找到最后一个 old_marker 和最后一个 old_end 之间的区域
last_marker = c.rfind(old_marker)
last_end = c.rfind(old_end)
if last_marker > 0 and last_end > last_marker:
    # 替换最后一个 old_marker 之后的区域
    before = c[:last_end]
    after = c[last_end + len(old_end):]
    c = before + new_end + after
    with open('granular/explainable_classifier.py', 'w', encoding='utf-8') as f:
        f.write(c)
    print('方案A: done (marker replacement)')
else:
    print('未找到标记')
