import sys
with open('granular/granular_stager.py', 'r', encoding='utf-8') as f:
    c = f.read()

# N1保护：替换LOW_CONF区域
old = '        # 只对低于阈值的区域做平滑\n        LOW_CONF = 0.5\n        for i in range(n):\n            if base_conf[i] < LOW_CONF:\n                stages[i] = lr_classes[np.argmax(smoothed[i])]\n                confidences[i] = np.max(smoothed[i])'

new = '        # [L3] N1保护：N1不参与平滑（过渡期孤立epoch容易被吞）\n        n1_mask = base_stages == "N1"\n        LOW_CONF = 0.5\n        for i in range(n):\n            if base_conf[i] < LOW_CONF and not n1_mask[i]:\n                stages[i] = lr_classes[np.argmax(smoothed[i])]\n                confidences[i] = np.max(smoothed[i])'

assert old in c
c = c.replace(old, new)
with open('granular/granular_stager.py', 'w', encoding='utf-8') as f:
    f.write(c)
print('N1保护注入成功')
