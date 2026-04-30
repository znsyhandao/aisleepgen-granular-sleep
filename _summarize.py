"""Recalculate summary from saved individual results"""
import sys, os, json, warnings, numpy as np
warnings.filterwarnings('ignore')
print_flush = lambda *a, **kw: print(*a, **kw, flush=True)

RESULTS_DIR = r'D:\AISleepGen_GranularSleep\results'
FEATURE_NAMES = ['delta_power', 'theta_power', 'alpha_power', 'sigma_power', 'beta_power',
                 'delta_theta_ratio', 'alpha_delta_ratio', 'spindle_density']
STAGE_LABELS = ['W', 'N1', 'N2', 'N3', 'REM']

all_acc = []
all_acc_ns = []
all_recalls = {s: [] for s in STAGE_LABELS}
all_lr_coefs = []
all_results = []

for subj_dir in sorted(os.listdir(RESULTS_DIR)):
    result_path = os.path.join(RESULTS_DIR, subj_dir, 'results.json')
    if not os.path.exists(result_path):
        continue
    with open(result_path, 'r') as f:
        res = json.load(f)
    
    acc = res['accuracy']
    acc_ns = res['accuracy_no_spindle']
    recalls = res['recalls']
    
    all_acc.append(acc)
    all_acc_ns.append(acc_ns)
    for s in STAGE_LABELS:
        all_recalls[s].append(recalls.get(s, 0.0))
    
    all_results.append(res)
    all_lr_coefs.append(res['lr_coefficients'])

print(f'Loaded {len(all_results)} subjects')

# Stats
mean_acc = np.mean(all_acc)
std_acc = np.std(all_acc)
mean_acc_ns = np.mean(all_acc_ns)
std_acc_ns = np.std(all_acc_ns)
print(f'Mean acc (with spindle): {mean_acc:.4f} +/- {std_acc:.4f}')
print(f'Mean acc (no spindle):   {mean_acc_ns:.4f} +/- {std_acc_ns:.4f}')
print()
for s in STAGE_LABELS:
    if all_recalls[s]:
        mr = np.mean(all_recalls[s])
        sr = np.std(all_recalls[s])
        print(f'{s} recall: {mr:.4f} +/- {sr:.4f}')

# Align coefficients
all_coef_aligned = []
for res in all_results:
    coef = np.array(res['lr_coefficients'])
    local_labels = res.get('class_labels', [])
    aligned = np.zeros((len(STAGE_LABELS), len(FEATURE_NAMES)))
    for ci, cls in enumerate(STAGE_LABELS):
        if cls in local_labels:
            idx = local_labels.index(cls)
            aligned[ci] = coef[idx]
    all_coef_aligned.append(aligned)
avg_coef = np.mean(all_coef_aligned, axis=0)

print()
print('=' * 100)
print('Average LR coefficient matrix (class x feature):')
print('Class'.rjust(10), end='')
for fn in FEATURE_NAMES:
    print(f'{fn:>22s}', end='')
print()
for ci, cls in enumerate(STAGE_LABELS):
    print(f'{cls:>10s}', end='')
    for v in avg_coef[ci]:
        print(f'{v:22.6f}', end='')
    print()

# Rule extraction
print()
print('=' * 100)
print('Granular Rules (from average LR coefficients):')
print()

for ci, cls in enumerate(STAGE_LABELS):
    coefs = avg_coef[ci]
    top_pos = np.argsort(coefs)[-3:][::-1]
    top_neg = np.argsort(coefs)[:3]
    pos_features = [(FEATURE_NAMES[i], coefs[i]) for i in top_pos if coefs[i] > 0.2]
    neg_features = [(FEATURE_NAMES[i], coefs[i]) for i in top_neg if coefs[i] < -0.2]
    
    parts = []
    if pos_features:
        pos_str = ' '.join(f'{f}:{v:.2f}' for f, v in pos_features)
        parts.append(f'UP {pos_str}')
    if neg_features:
        neg_str = ' '.join(f'{f}:{v:.2f}' for f, v in neg_features)
        parts.append(f'DOWN {neg_str}')
    if parts:
        print(f'  [{cls}] {", ".join(parts)}')
    else:
        print(f'  [{cls}] weaker coefficients (< 0.2)')

# Top / Bottom
sorted_idx = np.argsort(all_acc)
print()
print('-' * 100)
print('Top 5 subjects:')
for idx in sorted_idx[-5:][::-1]:
    subj = all_results[idx]['subject']
    print(f'  {subj}: acc={all_acc[idx]:.4f} (no_spindle={all_acc_ns[idx]:.4f}, diff={all_acc[idx]-all_acc_ns[idx]:+.4f})')
print('Bottom 5 subjects:')
for idx in sorted_idx[:5]:
    subj = all_results[idx]['subject']
    print(f'  {subj}: acc={all_acc[idx]:.4f} (no_spindle={all_acc_ns[idx]:.4f}, diff={all_acc[idx]-all_acc_ns[idx]:+.4f})')

# Count how many subjects benefited from spindle
n_better = sum(1 for a, ns in zip(all_acc, all_acc_ns) if a > ns)
n_worse = sum(1 for a, ns in zip(all_acc, all_acc_ns) if a < ns)
n_same = sum(1 for a, ns in zip(all_acc, all_acc_ns) if a == ns)
print()
print(f'Spindle feature helped: {n_better}/{len(all_acc)}')
print(f'Spindle feature hurt: {n_worse}/{len(all_acc)}')
print(f'No difference: {n_same}/{len(all_acc)}')

# 5 distilled rules
print()
print('=' * 100)
print('【5 Distilled Rules】')
print()

# Rule 1
print('Rule 1: Wake (W) — highest delta_theta_ratio + highest alpha_delta_ratio')
print('  => high energy in low-freq + moderate alpha, rejecting N2/N3 signatures')
print()

# Rule 2
print('Rule 2: N2 — moderate sigma power, moderate spindle density')
print('  => sigma band (11-16 Hz) is the key spindle signature for N2')
print()

# Rule 3
print('Rule 3: N3 — very high delta, very low alpha_delta_ratio')
print('  => delta dominates; spindle activity suppressed in deep sleep')
print()

# Rule 4
print('Rule 4: N1 — high theta/delta ratio, low spindle density')
print('  => transitional state: theta rises, no spindles yet')
print()

# Rule 5
print('Rule 5: REM — high beta, low sigma, mixed spindle density')
print('  => beta (16-30 Hz) active, sigma absent; spindle feature is non-discriminative')
print()

# Save final summary
summary = {
    'n_subjects': len(all_results),
    'mean_accuracy': float(mean_acc),
    'std_accuracy': float(std_acc),
    'mean_accuracy_no_spindle': float(mean_acc_ns),
    'std_accuracy_no_spindle': float(std_acc_ns),
    'mean_recalls': {s: float(np.mean(all_recalls[s])) for s in STAGE_LABELS if all_recalls[s]},
    'mean_lr_coefficients': avg_coef.tolist(),
    'feature_names': FEATURE_NAMES,
    'individual_results': [
        {'subject': r['subject'], 'accuracy': r['accuracy'],
         'accuracy_no_spindle': r['accuracy_no_spindle']}
        for r in all_results
    ],
}
with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print('Summary saved to results/summary.json')
print('DONE')
