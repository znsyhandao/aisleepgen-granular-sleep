import sys, os, json
sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)
import numpy as np

results_dir = r'D:\AISleepGen_GranularSleep\results'
results = []
for d in sorted(os.listdir(results_dir)):
    rpath = os.path.join(results_dir, d, 'results.json')
    if os.path.exists(rpath):
        r = json.load(open(rpath))
        results.append(r)

print(f'共 {len(results)} 个受试者')
print()

header = f'{"Subject":>7s}  {"Ep":>4s}  {"Acc":>5s}  {"AccNS":>5s}  {"W":>4s}  {"N1":>4s}  {"N2":>4s}  {"N3":>4s}  {"REM":>4s}'
print(header)
print('-' * 58)

accs = []; accs_ns = []
for r in results:
    subj = r['subject']
    acc = r['accuracy']
    acc_ns = r.get('accuracy_no_spindle', acc)
    rec = r['recalls']
    ep = r['n_epochs']
    line = f'{subj:>7s}  {ep:4d}  {acc:.3f}  {acc_ns:.3f}'
    for s in ['W','N1','N2','N3','REM']:
        line += f'  {rec[s]:.3f}'
    print(line)
    accs.append(acc); accs_ns.append(acc_ns)

print('-' * 58)
avg_line = '    AVG       '
avg_line += f'{np.mean(accs):.3f}  {np.mean(accs_ns):.3f}'
for s in ['W','N1','N2','N3','REM']:
    v = np.mean([r['recalls'][s] for r in results])
    avg_line += f'  {v:.3f}'
print(avg_line)

print(f'\n有纺锤波 vs 无纺锤波平均准确率: {np.mean(accs):.4f} vs {np.mean(accs_ns):.4f}')
diffs = [a - an for a, an in zip(accs, accs_ns)]
print(f'纺锤波带来的提升: {np.mean(diffs):+.4f} (max={max(diffs):+.4f}, min={min(diffs):+.4f})')

print(f'\n平均LR系数 (8特征, 5阶段):')
feature_names = results[0]['feature_names']
if 'spindle_density' not in feature_names:
    feature_names = feature_names + ['spindle_density']

coefs = None
n_coef = 0
for r in results:
    c = np.array(r['lr_coefficients'])
    if coefs is None:
        coefs = np.zeros(c.shape)
    if c.shape == coefs.shape:
        coefs += c
        n_coef += 1
coefs /= max(n_coef, 1)

labels = ['N1','N2','N3','REM','W']

print(f'\n  特征: {feature_names}')
for ci, stage in enumerate(labels):
    top = np.argsort(-np.abs(coefs[ci]))[:5]
    parts = []
    for i in top:
        sign = '↑' if coefs[ci][i] > 0 else '↓'
        parts.append(f'{feature_names[i]}{sign}({coefs[ci][i]:+.2f})')
    print(f'  {stage}: {", ".join(parts)}')
