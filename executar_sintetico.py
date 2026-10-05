"""Executa o log BIMP de 500 casos e atualiza o relatório IEEEtran."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from gerar_relatorio import generate
from import_mxml import convert
from preparar_sintetico import prepare
from run import METHODS, BASELINES

ROOT = Path(__file__).resolve().parent
DATASET = 'professora-500-5pct'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods', nargs='+', choices=METHODS+BASELINES,
                   default=['raridade_transicoes'])
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--regenerate', action='store_true', help='Refaz a amostra dos 5.000 casos')
    a=p.parse_args()
    target=ROOT/'data/sintetico/amostra_500/amostra_anomala_500.mxml.gz'
    if a.regenerate or not target.exists():
        prepare()
    converted=ROOT/'bpad/eventlogs'/f'{DATASET}.json.gz'
    if a.regenerate or not converted.exists():
        if a.regenerate:
            converted.unlink(missing_ok=True)
        convert(target, DATASET)
    status=subprocess.run([
        sys.executable,str(ROOT/'run.py'), '--dataset',DATASET,
        '--truth-csv',str(ROOT/'data/sintetico/amostra_500/rotulos_500_5pct.csv'),
        '--top-fraction','.05','--methods',*a.methods,
        '--epochs',str(a.epochs),'--timeout',str(a.timeout)
    ],cwd=ROOT)
    if status.returncode:
        raise SystemExit(status.returncode)
    generate(DATASET)


if __name__=='__main__':
    main()
