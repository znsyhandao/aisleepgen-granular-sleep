import sys, os, json
import numpy as np

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from granular.sleep_metrics import build_sleep_hypnogram, sleep_quality_score

results_dir = os.path.join(project_dir, 'results')
out_path = os.path.join(results_dir, 'quality_scores.json')

results = []
for d in sorted(os.listdir(results_dir)):
    rpath = os.path.join(results_dir, d, 'results.json')
    if not os.path.exists(rpath): continue
    try:
        r = json.load(open(rpath))
        subj = r['subject']
        
        # 读取测试集预测
        y_pred_path = os.path.join(results_dir, d, 'y_pred.npy')
        y_test_path = os.path.join(results_dir, d, 'y_test.npy')
        if not os.path.exists(y_pred_path) or not os.path.exists(y_test_path):
            print(f'  {subj}: 无预测文件，跳过', flush=True)
            continue
        
        y_pred = np.load(y_pred_path)
        y_test = np.load(y_test_path)
        
        # 用预测的置信度
        confs = np.array([1.0 if y_pred[i] == y_test[i] else 0.5 for i in range(len(y_pred))])
        
        hypno = build_sleep_hypnogram(y_pred, confs)
        score = sleep_quality_score(hypno)
        
        results.append({
            'subject': subj,
            'accuracy': r['accuracy'],
            'score': score,
            'summary': hypno['summary'],
            'n_granules': hypno['n_granules'],
            'confusion_matrix': r['confusion_matrix'],
        })
        print(f'  {subj}: acc={r["accuracy"]:.3f} 评分={score["total_score"]:.0f}({score["grade"]})  eff={hypno["summary"]["sleep_efficiency_pct"]:.0f}%', flush=True)
    except Exception as e:
        print(f'  {d}: {str(e)[:80]}', flush=True)

# Save
with open(out_path, 'w') as f:
    json.dump({'results': results, 'n': len(results)}, f, indent=2)

# Summary
print(f'\n=== 汇总 ({len(results)} 受试者) ===', flush=True)
if results:
    scores = [r['score']['total_score'] for r in results]
    effs = [r['summary']['sleep_efficiency_pct'] for r in results]
    wasos = [r['summary']['waso_min'] for r in results]
    lats = [r['summary']['sleep_latency_min'] for r in results]
    n3s = [r['summary']['stage_pct'].get('N3',0) for r in results]
    rems = [r['summary']['stage_pct'].get('REM',0) for r in results]
    accs = [r['accuracy'] for r in results]
    
    print(f'  准确率:     {np.mean(accs):.3f}±{np.std(accs):.3f}', flush=True)
    print(f'  睡眠评分:   {np.mean(scores):.0f}±{np.std(scores):.0f}  [{np.min(scores):.0f}-{np.max(scores):.0f}]', flush=True)
    print(f'  睡眠效率:   {np.mean(effs):.0f}±{np.std(effs):.0f}%  [{np.min(effs):.0f}-{np.max(effs):.0f}]', flush=True)
    print(f'  WASO:       {np.mean(wasos):.0f}±{np.std(wasos):.0f}min', flush=True)
    print(f'  潜伏期:     {np.mean(lats):.0f}±{np.std(lats):.0f}min', flush=True)
    print(f'  N3占比:     {np.mean(n3s):.1f}±{np.std(n3s):.1f}%', flush=True)
    print(f'  REM占比:    {np.mean(rems):.1f}±{np.std(rems):.1f}%', flush=True)
    
    # Score distribution
    grades = {}
    for r in results:
        g = r['score']['grade']
        grades[g] = grades.get(g, 0) + 1
    print(f'\n  评分分布:', flush=True)
    for g in ['优秀','良好','一般','需要关注']:
        print(f'    {g}: {grades.get(g,0)}人', flush=True)
    
    # Best/worst
    sorted_r = sorted(results, key=lambda r: -r['score']['total_score'])
    print(f'\n  最佳: {sorted_r[0]["subject"]} {sorted_r[0]["score"]["total_score"]:.0f}分({sorted_r[0]["score"]["grade"]})', flush=True)
    print(f'  最差: {sorted_r[-1]["subject"]} {sorted_r[-1]["score"]["total_score"]:.0f}分({sorted_r[-1]["score"]["grade"]})', flush=True)

print(f'\n结果保存: {out_path}', flush=True)
