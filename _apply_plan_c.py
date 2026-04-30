"""方案C: 混合策略 - 在classifier中加入transition钩子"""
import sys
with open('granular/explainable_classifier.py', 'r', encoding='utf-8') as f:
    c = f.read()

old_marker = '# 阶段转移约束（Viterbi-like）'
new_block = """# [方案C] 过渡期后处理（N1提升）
        try:
            from granular.transition_detector import post_process as _post_n1
            _ft = X if hasattr(self, 'feature_names_') else X
            _dt_idx = -1
            _be_idx = -1
            if hasattr(self, 'feature_names_'):
                try: _dt_idx = self.feature_names_.index('delta_theta_ratio')
                except ValueError: pass
                try: _be_idx = self.feature_names_.index('beta_power')
                except ValueError: pass
            stages = list(_post_n1(np.array(stages), X, None, _dt_idx, _be_idx))
        except ImportError:
            pass
        
        # 阶段转移约束（Viterbi-like）"""

assert old_marker in c
c = c.replace(old_marker, new_block)
with open('granular/explainable_classifier.py', 'w', encoding='utf-8') as f:
    f.write(c)
print('方案C: done')
