"""Executa: XES original -> anomalias XES -> 14 métodos -> relatório IEEEtran.

Exemplo: python executar.py --methods naive sampling likelihood
Padrão: tenta os 14 métodos e registra individualmente sucessos e falhas.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from preparar_xes import prepare
from import_xes import convert
from gerar_relatorio import generate
from run import METHODS

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'data/real/BPI_Challenge_2012.xes.gz')
    p.add_argument('--dataset', default='bpic12-anomalias')
    p.add_argument('--cases', type=int, default=1000)
    p.add_argument('--rate', type=float, default=.20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    args = p.parse_args()
    if not args.dataset or not all(c.isascii() and (c.isalnum() or c in '-_') for c in args.dataset):
        p.error('Dataset deve conter apenas letras ASCII, dígitos, - e _')
    if args.epochs < 1 or args.timeout < 1:
        p.error('Épocas e timeout devem ser positivos')
    folder = prepare(args.input, args.dataset, args.cases, args.rate, args.seed)
    convert(folder / 'amostra_anomala.xes.gz', args.dataset)
    result = subprocess.run([
        sys.executable, str(ROOT / 'run.py'), '--dataset', args.dataset,
        '--truth-csv', str(folder / 'rotulos.csv'), '--methods', *args.methods,
        '--top-fraction', str(args.rate), '--epochs', str(args.epochs),
        '--timeout', str(args.timeout),
    ], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)
    generate(args.dataset)


if __name__ == '__main__':
    main()
