"""Converte um log MXML do BIMP para o formato de entrada do BPAD.

Usa somente as atividades concluídas; mantém IDs e ordem dos casos. Os rótulos
de injeção nunca são incorporados ao arquivo lido pelos detectores.
"""
import argparse
import gzip
import json
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent


def convert(source, dataset):
    if not dataset or not all(c.isascii() and (c.isalnum() or c in '-_') for c in dataset):
        raise ValueError('Nome do conjunto inválido')
    if not source.is_file():
        raise FileNotFoundError(source)
    opener = gzip.open if str(source).endswith('.gz') else open
    cases = []
    with opener(source, 'rb') as stream:
        for _, case in ET.iterparse(stream, events=('end',)):
            if case.tag != 'ProcessInstance':
                continue
            case_id = case.attrib['id']
            events = []
            for entry in case.findall('AuditTrailEntry'):
                if entry.findtext('EventType') != 'complete':
                    continue
                name = entry.findtext('WorkflowModelElement')
                if not name or name.startswith('EVENT '):
                    continue
                events.append({'name': name, 'attributes': {},
                               'timestamp': None, 'timestamp_end': None})
            if not events:
                raise ValueError(f'Caso vazio: {case_id}')
            cases.append({'id': case_id, 'attributes': {'label': 'normal'},
                          'events': events})
            case.clear()
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('IDs repetidos no log')
    target = ROOT / 'bpad' / 'eventlogs' / f'{dataset}.json.gz'
    if target.exists():
        raise FileExistsError(f'Conjunto já existe: {target}. Escolha outro nome.')
    with gzip.open(target, 'wt', encoding='utf-8') as out:
        json.dump({'attributes': {'source': 'BIMP MXML; eventos complete; rótulos externos'},
                   'cases': cases}, out, ensure_ascii=False)
    print(f'{len(cases)} casos importados: {target}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mxml', required=True, type=Path)
    p.add_argument('--dataset', default='professora-piloto-100')
    a = p.parse_args()
    convert(a.mxml, a.dataset)
