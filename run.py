"""Executa os 14 métodos originais do BPAD, avalia e sugere reparos por caso.

Exemplo: python run.py --dataset p2p-0.05-1 --methods naive sampling --top-fraction 0.05
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASELINES = ('raridade_transicoes', 'local_outlier_factor', 'isolation_forest', 'consenso_ranks')
METHODS = ('likelihood', 'naive', 'sampling', 'dae', 'binetv3', 'binetv2',
           'gama', 'vae', 'lae', 'gae', 'grasped', 'leverage', 'w2vlof', 'vaeocsvm')


def cases_for(name):
    if not name or not all(c.isascii() and (c.isalnum() or c in '-_') for c in name):
        raise ValueError('Nome de conjunto inválido')
    path = ROOT / 'bpad' / 'eventlogs' / (name + '.json.gz')
    with gzip.open(path, 'rt', encoding='utf8') as f:
        cases = json.load(f)['cases']
    if not cases:
        raise ValueError('Log vazio')
    return cases


def external_truth(cases, csv_path):
    """Carrega a referência apenas para métricas, após validar ID e sequência."""
    with csv_path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    required = {'case_id', 'tipo', 'depois'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('CSV de referência exige case_id, tipo e depois')
    labels = {}
    for row in rows:
        ident = str(row['case_id'])
        if ident in labels:
            raise ValueError(f'ID duplicado na referência: {ident}')
        labels[ident] = row
    data_ids = {str(c['id']) for c in cases}
    if not set(labels) <= data_ids:
        raise ValueError(f'IDs não encontrados no log: {set(labels) - data_ids}')
    for case in cases:
        row = labels.get(str(case['id']))
        if row and ' '.join(e['name'] for e in case['events']) != row['depois'].strip():
            raise ValueError(f"Sequência diferente da referência no caso {case['id']}")
    return {ident: row['tipo'] for ident, row in labels.items()
            if row['tipo'].strip().lower() != 'normal'}


def truth(cases, labels=None):
    return [int(str(c['id']) in labels) for c in cases] if labels is not None else [
        int(c['attributes'].get('label', 'normal') != 'normal') for c in cases]


def kind(case, labels=None):
    if labels is not None:
        return labels.get(str(case['id']), 'normal')
    label = case['attributes'].get('label', 'normal')
    return 'normal' if label == 'normal' else label.get('anomaly', 'unknown')


def sequences(cases):
    return [tuple(e['name'] for e in c['events']) for c in cases]


def choose_flags(scores, fraction):
    import numpy as np
    values = np.asarray(scores, dtype=float)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError('Escores vazios ou inválidos')
    count = max(1, round(len(values) * fraction))
    order = np.lexsort((np.arange(len(values)), -values))
    result = np.zeros(len(values), dtype=int)
    result[order[:count]] = 1
    return result


def evaluate(y, scores, fraction):
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
    flags = choose_flags(scores, fraction)
    p, r, f1, _ = precision_recall_fscore_support(y, flags, average='binary', zero_division=0)
    return {'precision': float(p), 'recall': float(r), 'f1': float(f1),
            'average_precision': float(average_precision_score(y, scores)) if len(set(y)) > 1 else None,
            'roc_auc': float(roc_auc_score(y, scores)) if len(set(y)) > 1 else None}, flags


def evaluate_repair(cases, flags, labels=None):
    # Os rótulos de BPAD não contêm necessariamente o histórico limpo completo.
    # Avaliar recuperação exata só é legítimo onde há referência limpa verificável.
    sys.path.insert(0, str(ROOT))
    from src.repair import DataDrivenRepairer
    traces = sequences(cases)
    reference = [trace for trace, flagged in zip(traces, flags) if not flagged]
    repairer = DataDrivenRepairer(max_candidates=250, min_frequency=2,
                                  max_edit_cost=1).fit(reference)
    computed = repairer.repair_if_flagged(traces, [int(x) for x in flags])
    results = []
    for case, trace, suggestion in zip(cases, traces, computed):
        results.append({'case_id': case['id'], 'original_observed': ' -> '.join(trace),
                        'suggested': ' -> '.join(suggestion.repaired_trace),
                        'applied': int(suggestion.repair_applied),
                        'edit_cost': suggestion.edit_cost,
                        'prototype_frequency': suggestion.prototype_frequency,
                        'type': kind(case, labels)})
    return results


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--truth-csv', type=Path, help='Referência externa; usada somente para avaliação')
    p.add_argument('--methods', nargs='+', choices=METHODS + BASELINES, default=list(METHODS))
    p.add_argument('--top-fraction', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--timeout', type=int, default=1800)
    args = p.parse_args()
    if not 0 < args.top_fraction < 1 or args.epochs < 1 or args.timeout < 1:
        p.error('top-fraction entre 0 e 1; epochs e timeout positivos')
    cases = cases_for(args.dataset)
    labels = external_truth(cases, args.truth_csv) if args.truth_csv else None
    y = truth(cases, labels)
    output = ROOT / 'results' / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    for method in dict.fromkeys(args.methods):
        score_file = output / f'{method}_scores.json'
        try:
            if method in BASELINES:
                sys.path.insert(0, str(ROOT))
                from src.detectors import score_all_methods
                scores = score_all_methods(sequences(cases))[method].tolist()
                score_file.write_text(json.dumps(scores))
            else:
                command = [sys.executable, str(ROOT / 'worker.py'), args.dataset,
                           method, str(score_file), str(args.epochs)]
                job = subprocess.run(command, cwd=ROOT / 'bpad', text=True,
                                     capture_output=True, timeout=args.timeout)
                (output / f'{method}_stdout.log').write_text(job.stdout)
                (output / f'{method}_stderr.log').write_text(job.stderr)
                if job.returncode:
                    raise RuntimeError(job.stderr or job.stdout)
                scores = json.loads(score_file.read_text())
            if len(scores) != len(cases):
                raise ValueError(f'{len(scores)} escores para {len(cases)} casos')
            metrics, flags = evaluate(y, scores, args.top_fraction)
            repairs = evaluate_repair(cases, flags, labels)
            write_csv(output / f'{method}_cases.csv', [
                {'case_id': c['id'], 'type': kind(c, labels), 'label': label,
                 'score': float(score), 'flag': int(flag)}
                for c, label, score, flag in zip(cases, y, scores, flags)])
            write_csv(output / f'{method}_repairs.csv', repairs)
            for anomaly_type in sorted({kind(c, labels) for c in cases} - {'normal'}):
                idx = [i for i,c in enumerate(cases) if kind(c, labels) == anomaly_type]
                metrics[f'recall_{anomaly_type}'] = sum(flags[i] for i in idx) / len(idx)
            row = {'method': method, 'dataset': args.dataset, 'status': 'ok',
                   'n_cases': len(cases), 'n_anomalies': sum(y),
                   'repairs_suggested': sum(r['applied'] for r in repairs), **metrics}
        except (Exception, KeyboardInterrupt) as e:
            row = {'method': method, 'dataset': args.dataset, 'status': 'failed',
                   'error': f'{type(e).__name__}: {e}'}
        index = next((i for i, old in enumerate(summary) if old['method'] == method), None)
        if index is None:
            summary.append(row)
        else:
            summary[index] = row
        print(f"{method}: {row['status']}", flush=True)
        if row["status"] == "failed":
            print(row["error"], flush=True)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('Resultados:', output)

if __name__ == '__main__':
    main()
