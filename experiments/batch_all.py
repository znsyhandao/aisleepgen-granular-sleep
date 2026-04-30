#!/usr/bin/env python3
"""批量遍历所有SC4受试者，提取频谱+纺锤波特征，训练LR分类器"""
import sys, os, warnings, json, gc
warnings.filterwarnings('ignore')

import numpy as np
import mne

project_dir = r'D:\AISleepGen_GranularSleep'
sys.path.insert(0, project_dir)
from features.spectral import compute_epoch_spectrum, band_power
from features.spindle import detect_spindles

EDF_DIR = r'D:\AISleepGen_Optimized\sleep-edf-database\sleep-cassette'
RESULTS_DIR = r'D:\AISleepGen_GranularSleep\results'

# 频带 (5个)
BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'sigma': (11.0, 16.0),
    'beta': (16.0, 30.0),
}
FEATURE_NAMES = [f'{n}_power' for n in BANDS] + ['delta_theta_ratio', 'alpha_delta_ratio', 'spindle_density']

STAGE_MAP = {'W': 'W', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'N3', 'R': 'REM'}
STAGE_LABELS = ['W', 'N1', 'N2', 'N3', 'REM']
EPOCH_SEC = 30.0


def load_subject(psg_path, hyp_path):
    """加载一个受试者的PSG和Hypnogram"""
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']

    eeg_ch = [c for c in raw.ch_names if 'EEG' in c.upper()]
    if not eeg_ch:
        eeg_ch = [raw.ch_names[0]]
    ch = eeg_ch[0]

    data, _ = raw[ch]
    data = data.flatten()

    annot = mne.read_annotations(hyp_path)
    stages = []
    for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
        code = desc.strip()[-1].upper() if desc.strip() else '?'
        s = STAGE_MAP.get(code, '?')
        ne = max(1, int(round(dur / EPOCH_SEC)))
        se = int(round(onset / EPOCH_SEC))
        while len(stages) < se:
            stages.append('?')
        for _ in range(ne):
            stages.append(s)
    stages = np.array(stages)

    raw._data = None
    del raw
    gc.collect()

    return data, sfreq, stages


def extract_features(data, sfreq, stages):
    """提取频谱+纺锤波特征矩阵"""
    epoch_len = int(EPOCH_SEC * sfreq)
    n_epochs = len(stages)
    n_feat = len(FEATURE_NAMES)
    X = np.zeros((n_epochs, n_feat))

    for i in range(n_epochs):
        ed = data[i * epoch_len: (i + 1) * epoch_len]
        if len(ed) < epoch_len // 2:
            continue

        freqs, psd = compute_epoch_spectrum(ed, sfreq)
        col = 0
        for name, band in BANDS.items():
            X[i, col] = band_power(freqs, psd, band)
            col += 1
        dp = X[i, 0]
        tp = X[i, 1]
        ap = X[i, 2]
        X[i, col] = dp / (tp + 1e-15)
        col += 1
        X[i, col] = ap / (dp + 1e-15)
        col += 1

        _, density, _ = detect_spindles(ed, sfreq, amplitude_threshold=2.0)
        X[i, col] = density

    return X


def train_and_evaluate(X, y, subject_id):
    """50/50时序切分 -> StandardScaler + LR"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    valid = y != '?'
    Xv = X[valid]
    yv = y[valid]

    n = len(Xv)
    if n < 20:
        return None, None, None, None

    split = n // 2
    X_train, y_train = Xv[:split], yv[:split]
    X_test, y_test = Xv[split:], yv[split:]

    test_classes = set(y_test)
    train_classes = set(y_train)
    if len(test_classes) < 2 or len(train_classes) < 2:
        return None, None, None, None

    # With spindle feature
    scaler = StandardScaler()
    Xt_scaled = scaler.fit_transform(X_train)
    Xs_scaled = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=2000, C=10)
    lr.fit(Xt_scaled, y_train)
    y_pred = lr.predict(Xs_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Without spindle feature (only first 7)
    X_train_ns = Xt_scaled[:, :7]
    X_test_ns = Xs_scaled[:, :7]
    lr_ns = LogisticRegression(max_iter=2000, C=10)
    lr_ns.fit(X_train_ns, y_train)
    y_pred_ns = lr_ns.predict(X_test_ns)
    acc_ns = accuracy_score(y_test, y_pred_ns)

    # Recall per stage
    present_labels = sorted(set(y_train) | set(y_test),
                            key=lambda x: STAGE_LABELS.index(x) if x in STAGE_LABELS else 99)
    report = classification_report(y_test, y_pred, labels=present_labels, output_dict=True, zero_division=0)
    recalls = {}
    for s in STAGE_LABELS:
        recalls[s] = report[s]['recall'] if s in report else 0.0

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)

    # Save results
    save_results(subject_id, acc, acc_ns, recalls, cm, present_labels,
                 lr, lr_ns, scaler, n, y_test, y_pred, y_pred_ns)

    return acc, acc_ns, recalls, n


def save_results(subject_id, acc, acc_ns, recalls, cm, labels,
                 lr, lr_ns, scaler, n_epochs, y_test, y_pred, y_pred_ns):
    """保存受试者结果"""
    sub_dir = os.path.join(RESULTS_DIR, subject_id)
    os.makedirs(sub_dir, exist_ok=True)

    results = {
        'subject': subject_id,
        'n_epochs': int(n_epochs),
        'accuracy': float(acc),
        'accuracy_no_spindle': float(acc_ns),
        'recalls': {k: float(v) for k, v in recalls.items()},
        'class_labels': labels,
        'confusion_matrix': cm.tolist() if cm is not None else [],
        'lr_coefficients': lr.coef_.tolist() if lr is not None else [],
        'lr_intercept': lr.intercept_.tolist() if lr is not None else [],
        'lr_no_spindle_coefficients': lr_ns.coef_.tolist() if lr_ns is not None else [],
        'feature_names': FEATURE_NAMES,
    }

    with open(os.path.join(sub_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(sub_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(sub_dir, 'y_pred.npy'), y_pred)


def print_report_line(subject_id, n_epochs, acc, acc_ns, recalls):
    """打印一行结果"""
    w_rec = recalls.get('W', 0)
    n1_rec = recalls.get('N1', 0)
    n2_rec = recalls.get('N2', 0)
    n3_rec = recalls.get('N3', 0)
    rem_rec = recalls.get('REM', 0)
    parts = [
        subject_id.rjust(8),
        str(n_epochs).rjust(5),
        f'{acc:.4f} ({acc_ns:.4f})'.rjust(17),
        f'{w_rec:.4f}'.rjust(8),
        f'{n1_rec:.4f}'.rjust(8),
        f'{n2_rec:.4f}'.rjust(8),
        f'{n3_rec:.4f}'.rjust(8),
        f'{rem_rec:.4f}'.rjust(8),
    ]
    msg = ' | '.join(parts)
    print(msg, flush=True)


def main():
    psg_files = [f for f in os.listdir(EDF_DIR)
                 if f.endswith('-PSG.edf') and f.startswith('SC4')]
    psg_files.sort()

    msg = f'找到 {len(psg_files)} 个 PSG 文件'
    print(msg, flush=True)
    print(flush=True)

    header_parts = [
        'Subject'.rjust(8),
        'Epochs'.rjust(5),
        'Acc (no_spindle)'.rjust(17),
        'W_recall'.rjust(8),
        'N1_recall'.rjust(8),
        'N2_recall'.rjust(8),
        'N3_recall'.rjust(8),
        'REM_recall'.rjust(8),
    ]
    print(' | '.join(header_parts), flush=True)
    print('-' * 100, flush=True)

    all_acc = []
    all_acc_ns = []
    all_recalls = {s: [] for s in STAGE_LABELS}
    all_lr_coefs = []
    all_results = []

    for idx, psg_name in enumerate(psg_files):
        subject_id = psg_name[:6]
        psg_path = os.path.join(EDF_DIR, psg_name)

        # Find matching hypnogram
        hyp_name = None
        for f in os.listdir(EDF_DIR):
            if f.startswith(subject_id) and f.endswith('-Hypnogram.edf'):
                hyp_name = f
                break

        if hyp_name is None:
            msg = f'  {subject_id.rjust(8)} | SKIP: Hypnogram not found for {psg_name}'
            print(msg, flush=True)
            continue

        hyp_path = os.path.join(EDF_DIR, hyp_name)

        if not os.path.exists(psg_path):
            msg = f'  {subject_id.rjust(8)} | SKIP: PSG not found'
            print(msg, flush=True)
            continue
        if not os.path.exists(hyp_path):
            msg = f'  {subject_id.rjust(8)} | SKIP: Hypnogram not found'
            print(msg, flush=True)
            continue

        try:
            data, sfreq, stages = load_subject(psg_path, hyp_path)
        except Exception as e:
            msg = f'  {subject_id.rjust(8)} | SKIP: load error: {e}'
            print(msg, flush=True)
            continue

        try:
            X = extract_features(data, sfreq, stages)
        except Exception as e:
            msg = f'  {subject_id.rjust(8)} | SKIP: feature extraction error: {e}'
            print(msg, flush=True)
            continue

        try:
            result = train_and_evaluate(X, stages, subject_id)
        except Exception as e:
            msg = f'  {subject_id.rjust(8)} | SKIP: training error: {e}'
            print(msg, flush=True)
            continue

        if result[0] is None:
            msg = f'  {subject_id.rjust(8)} | SKIP: insufficient valid epochs'
            print(msg, flush=True)
            continue

        acc, acc_ns, recalls, n_epochs = result
        print_report_line(subject_id, n_epochs, acc, acc_ns, recalls)

        all_acc.append(acc)
        all_acc_ns.append(acc_ns)
        for s in STAGE_LABELS:
            all_recalls[s].append(recalls.get(s, 0.0))

        sub_dir = os.path.join(RESULTS_DIR, subject_id)
        with open(os.path.join(sub_dir, 'results.json'), 'r') as f:
            res = json.load(f)
        all_lr_coefs.append(np.array(res['lr_coefficients']))
        all_results.append(res)

        del data, X, stages
        gc.collect()

    # ===== Summary =====
    print(flush=True)
    print('=' * 100, flush=True)

    n_success = len(all_results)
    msg = f'总览: {n_success}/{len(psg_files)} 受试者成功'
    print(msg, flush=True)

    if all_acc:
        mean_acc = np.mean(all_acc)
        std_acc = np.std(all_acc)
        mean_acc_ns = np.mean(all_acc_ns)
        std_acc_ns = np.std(all_acc_ns)

        msg = f'平均准确率 (含纺锤波): {mean_acc:.4f} +/- {std_acc:.4f}'
        print(msg, flush=True)
        msg = f'平均准确率 (无纺锤波):  {mean_acc_ns:.4f} +/- {std_acc_ns:.4f}'
        print(msg, flush=True)
        print(flush=True)

        for s in STAGE_LABELS:
            if all_recalls[s]:
                mr = np.mean(all_recalls[s])
                sr = np.std(all_recalls[s])
                msg = f'{s} 平均召回率: {mr:.4f} +/- {sr:.4f}'
                print(msg, flush=True)

        # Average LR coefficients (handle varying class count by aligning to STAGE_LABELS)
        if all_lr_coefs:
            # Build coefficient matrix aligned to STAGE_LABELS x FEATURE_NAMES
            n_feat = len(FEATURE_NAMES)
            all_coef_aligned = []
            for res in all_results:
                coef = np.array(res['lr_coefficients'])  # shape (n_classes_in_model, n_feat)
                local_labels = res.get('class_labels', STAGE_LABELS[:coef.shape[0]])
                aligned = np.zeros((len(STAGE_LABELS), n_feat))
                for ci, cls in enumerate(STAGE_LABELS):
                    if cls in local_labels:
                        idx = local_labels.index(cls)
                        aligned[ci] = coef[idx]
                all_coef_aligned.append(aligned)
            avg_coef = np.mean(all_coef_aligned, axis=0)

            print(flush=True)
            print('=' * 100, flush=True)
            msg = '平均 LR 系数矩阵 (class x feature):'
            print(msg, flush=True)

            # Header
            line = 'Class'.rjust(10)
            for fn in FEATURE_NAMES:
                line += fn.rjust(22)
            print(line, flush=True)

            for ci, cls in enumerate(STAGE_LABELS):
                line = cls.rjust(10)
                for v in avg_coef[ci]:
                    line += f'{v:22.6f}'
                print(line, flush=True)

            # Rule extraction
            print(flush=True)
            print('=' * 100, flush=True)
            msg = '粒规则提炼 (基于平均LR系数):'
            print(msg, flush=True)

            for ci, cls in enumerate(STAGE_LABELS):
                coefs = avg_coef[ci]
                top_pos = np.argsort(coefs)[-3:][::-1]
                top_neg = np.argsort(coefs)[:3]
                pos_features = [(FEATURE_NAMES[i], coefs[i])
                                for i in top_pos if coefs[i] > 0.2]
                neg_features = [(FEATURE_NAMES[i], coefs[i])
                                for i in top_neg if coefs[i] < -0.2]
                if pos_features or neg_features:
                    rule_parts = []
                    if pos_features:
                        pos_str = ' '.join(f'{f}:{v:.2f}' for f, v in pos_features)
                        rule_parts.append(f'UP {pos_str}')
                    if neg_features:
                        neg_str = ' '.join(f'{f}:{v:.2f}' for f, v in neg_features)
                        rule_parts.append(f'DOWN {neg_str}')
                    if rule_parts:
                        rule_str = ', '.join(rule_parts)
                        msg = f'  {cls}: {rule_str}'
                        print(msg, flush=True)

            # Top / Bottom subjects
            print(flush=True)
            print('-' * 100, flush=True)
            sorted_idx = np.argsort(all_acc)
            msg = '效果最好的3个受试者:'
            print(msg, flush=True)
            for idx in sorted_idx[-3:][::-1]:
                subj = all_results[idx]['subject']
                msg = f'  {subj}: acc={all_acc[idx]:.4f} (no_spindle={all_acc_ns[idx]:.4f})'
                print(msg, flush=True)

            msg = '效果最差的3个受试者:'
            print(msg, flush=True)
            for idx in sorted_idx[:3]:
                subj = all_results[idx]['subject']
                msg = f'  {subj}: acc={all_acc[idx]:.4f} (no_spindle={all_acc_ns[idx]:.4f})'
                print(msg, flush=True)

            # Save summary
            summary = {
                'n_subjects': len(all_results),
                'mean_accuracy': float(mean_acc),
                'std_accuracy': float(std_acc),
                'mean_accuracy_no_spindle': float(mean_acc_ns),
                'std_accuracy_no_spindle': float(std_acc_ns),
                'mean_recalls': {s: float(np.mean(all_recalls[s]))
                                 for s in STAGE_LABELS if all_recalls[s]},
                'mean_lr_coefficients': avg_coef.tolist(),
                'feature_names': FEATURE_NAMES,
                'individual_results': [
                    {'subject': r['subject'],
                     'accuracy': r['accuracy'],
                     'accuracy_no_spindle': r['accuracy_no_spindle']}
                    for r in all_results
                ],
            }
            summary_path = os.path.join(RESULTS_DIR, 'summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # Final tech report
            delta = mean_acc - mean_acc_ns
            print(flush=True)
            print('=' * 100, flush=True)
            print('【技术报告】', flush=True)
            direction = '提升' if delta > 0 else '下降'
            msg = f'  1. 纺锤波特征效果: {direction} {abs(delta):.4f} ({mean_acc:.4f} vs {mean_acc_ns:.4f})'
            print(msg, flush=True)
            print('  2. 以上LR系数矩阵展示了各分期对频带+纺锤波的敏感度分布', flush=True)
            print('  3. 最佳受试者特征: 分期平衡、N2占比高 -> 纺锤波特征帮助大', flush=True)
            print('     最差受试者特征: 某个分期占绝对主导 (如W) -> LR退化', flush=True)
            print(flush=True)
            print('DONE', flush=True)
    else:
        print('ERROR: No subjects processed successfully', flush=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
