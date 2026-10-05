"""Extrai 500 casos do MXML BIMP da professora, com 25 anomalias existentes."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data' / 'sintetico' / 'fonte_5000'
DEST = ROOT / 'data' / 'sintetico' / 'amostra_500'


def extract(path, selected, output):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        context = ET.iterparse(stream, events=('start','end'))
        event, log = next(context)
        if event != 'start' or log.tag != 'WorkflowLog':
            raise ValueError('Arquivo MXML sem raiz WorkflowLog')
        new_log = ET.Element(log.tag, log.attrib)
        process = None
        new_process = None
        found = set()
        for event, element in context:
            if event == 'start' and element.tag == 'Process':
                process = element
                new_process = ET.Element(element.tag, element.attrib)
            if event != 'end':
                continue
            if element.tag == 'ProcessInstance':
                case_id = element.attrib['id']
                if case_id in selected:
                    if case_id in found:
                        raise ValueError(f'ID duplicado: {case_id}')
                    new_process.append(deepcopy(element))
                    found.add(case_id)
                process.remove(element)
                element.clear()
            elif element.tag == 'Process':
                new_log.append(new_process)
                process = None
                element.clear()
            elif element in log and element.tag != 'Process':
                new_log.append(deepcopy(element))
                log.remove(element)
                element.clear()
    if found != selected:
        raise ValueError(f'IDs ausentes no log {path.name}: {len(selected - found)}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, 'wb') as stream:
        ET.ElementTree(new_log).write(stream, encoding='utf-8', xml_declaration=True)


def prepare(seed=2026):
    with (SOURCE / 'rotulos_5000_5pct.csv').open(encoding='utf-8-sig',newline='') as stream:
        labels=list(csv.DictReader(stream))
    unusual={row['case_id']:row for row in labels}
    original=SOURCE/'log_original_5000.mxml.gz'
    with gzip.open(original,'rb') as stream:
        case_ids={e.attrib['id'] for event,e in ET.iterparse(stream,events=('start',))
                  if e.tag=='ProcessInstance'}
    if len(case_ids)!=5000 or not set(unusual)<=case_ids:
        raise ValueError('Fonte de 5.000 casos ou rótulos inconsistentes')
    rng=random.Random(seed)
    positives=set(rng.sample(sorted(unusual),25))
    negatives=set(rng.sample(sorted(case_ids-set(unusual)),475))
    selected=positives|negatives
    extract(original,selected,DEST/'amostra_limpa_500.mxml.gz')
    extract(SOURCE/'log_anomalo_5000_5pct.mxml.gz',selected,
            DEST/'amostra_anomala_500.mxml.gz')
    with (DEST/'rotulos_500_5pct.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=labels[0])
        writer.writeheader()
        writer.writerows(unusual[case_id] for case_id in sorted(positives))
    (DEST/'protocolo.json').write_text(json.dumps({
        'source': 'modelo_professora.bpmn -> BIMP -> 5000 MXML -> amostra de 500',
        'n_cases':500,'n_modified':25,'rate':.05,'seed':seed,
        'ids_modified':sorted(positives),'ids_normal':sorted(negatives),
        'source_original_sha256':hashlib.sha256(original.read_bytes()).hexdigest()
    },ensure_ascii=False,indent=2),encoding='utf8')
    print('500 casos sintéticos extraídos: 25 alterados e 475 normais')
    return DEST


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=2026)
    prepare(p.parse_args().seed)
