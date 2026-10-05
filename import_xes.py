"""Converte XES/XES.GZ para os 14 métodos sem incorporar os rótulos."""
import argparse
import gzip
import json
from pathlib import Path
from src.log_io import read_xes_events, group_events

ROOT = Path(__file__).resolve().parent


def convert(source, dataset):
    if not dataset or not all(c.isascii() and (c.isalnum() or c in '-_') for c in dataset):
        raise ValueError('Nome do conjunto inválido')
    grouped = group_events(read_xes_events(source))
    cases = [{'id': case_id, 'attributes': {'label': 'normal'},
              'events': [{'name': event.activity, 'attributes': {},
                          'timestamp': None, 'timestamp_end': None}
                         for event in events]}
             for case_id, events in sorted(grouped.items())]
    destination = ROOT / 'bpad' / 'eventlogs' / f'{dataset}.json.gz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, 'wt', encoding='utf-8') as stream:
        json.dump({'attributes': {'source': str(source), 'labels': 'external only'},
                   'cases': cases}, stream, ensure_ascii=False)
    print(f'{len(cases)} casos importados sem rótulos: {destination}')
    return destination


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xes', type=Path, required=True)
    p.add_argument('--dataset', default='bpic12-anomalias')
    a = p.parse_args()
    convert(a.xes, a.dataset)
