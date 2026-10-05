"""Cria um experimento XES controlado sem alterar o log original.

Uso: python preparar_xes.py --input data/real/BPI_Challenge_2012.xes.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

NS = ''
TAG = lambda name: f'{{{NS}}}{name}' if NS else name
ROOT = Path(__file__).resolve().parent


def field(element, key):
    return next((child for child in element if child.attrib.get('key') == key), None)


def name(element):
    value = field(element, 'concept:name')
    if value is None:
        raise ValueError('Falta concept:name em um trace ou evento')
    return value.attrib['value']


def sequence(trace):
    return [name(event) for event in trace.findall(TAG('event'))]


def change(trace, mutation, rng):
    events = trace.findall(TAG('event'))
    if mutation == 'omissao':
        trace.remove(events[rng.randrange(1, len(events) - 1)])
    elif mutation == 'repeticao':
        index = rng.randrange(1, len(events) - 1)
        trace.insert(list(trace).index(events[index]) + 1, deepcopy(events[index]))
    elif mutation == 'troca':
        a, b = field(events[0], 'concept:name'), field(events[1], 'concept:name')
        a.attrib['value'], b.attrib['value'] = b.attrib['value'], a.attrib['value']
    elif mutation == 'atividade_desconhecida':
        index = rng.randrange(1, len(events) - 1)
        inserted = deepcopy(events[index])
        field(inserted, 'concept:name').attrib['value'] = '__UNKNOWN__'
        trace.insert(list(trace).index(events[index]) + 1, inserted)
    else:
        raise ValueError(mutation)


def save_xes(tree, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, 'wb') as output:
        tree.write(output, encoding='utf-8', xml_declaration=True)


def prepare(source, dataset='bpic12-anomalias', count=1000, rate=.20, seed=42):
    global NS
    if not source.is_file():
        raise FileNotFoundError(source)
    if count < 20 or not (0 < rate < 1):
        raise ValueError('Casos >=20 e taxa entre 0 e 1 são obrigatórios')
    opener = gzip.open if str(source).endswith('.gz') else open
    with opener(source, 'rb') as stream:
        tree = ET.parse(stream)
    root = tree.getroot()
    NS = root.tag.split('}', 1)[0][1:] if root.tag.startswith('{') else ''
    traces = root.findall(TAG('trace'))
    eligible = {name(t): t for t in traces if len(t.findall(TAG('event'))) >= 3}
    if len(eligible) != len([t for t in traces if len(t.findall(TAG('event'))) >= 3]):
        raise ValueError('IDs duplicados no log original')
    rng = random.Random(seed)
    chosen = sorted(rng.sample(sorted(eligible), min(count, len(eligible))))
    normal_count = round(len(chosen) * rate)
    selected = rng.sample(chosen, normal_count)
    types = ('omissao', 'repeticao', 'troca', 'atividade_desconhecida')
    injected = dict(zip(selected, (types[i % 4] for i in range(len(selected)))))
    clean_root = ET.Element(root.tag, root.attrib)
    noisy_root = ET.Element(root.tag, root.attrib)
    # Conserva extensões e atributos globais do XES.
    for element in root:
        if element.tag != TAG('trace'):
            clean_root.append(deepcopy(element))
            noisy_root.append(deepcopy(element))
    rows = []
    for case_id in chosen:
        clean = deepcopy(eligible[case_id])
        observed = deepcopy(clean)
        before = sequence(clean)
        mutation = injected.get(case_id, 'normal')
        if mutation != 'normal':
            change(observed, mutation, rng)
        after = sequence(observed)
        if mutation != 'normal' and before == after:
            raise ValueError(f'A alteração não modificou o caso {case_id}')
        clean_root.append(clean)
        noisy_root.append(observed)
        rows.append({'case_id': case_id, 'tipo': mutation,
                     'is_anomaly': int(mutation != 'normal'),
                     'antes': ' '.join(before), 'depois': ' '.join(after)})
    folder = ROOT / 'data' / 'experimento' / dataset
    folder.mkdir(parents=True, exist_ok=True)
    save_xes(ET.ElementTree(clean_root), folder / 'amostra_limpa.xes.gz')
    save_xes(ET.ElementTree(noisy_root), folder / 'amostra_anomala.xes.gz')
    with (folder / 'rotulos.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'protocolo.json').write_text(json.dumps({
        'input': str(source), 'n_cases': len(chosen), 'n_modified': len(selected),
        'seed': seed, 'rate': rate, 'types': dict(Counter(injected.values()))
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(chosen)} casos XES; {len(selected)} alterados. Dados: {folder}')
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/real/BPI_Challenge_2012.xes.gz')
    parser.add_argument('--dataset', default='bpic12-anomalias')
    parser.add_argument('--cases', type=int, default=1000)
    parser.add_argument('--rate', type=float, default=.20)
    parser.add_argument('--seed', type=int, default=42)
    a = parser.parse_args()
    prepare(a.input, a.dataset, a.cases, a.rate, a.seed)
