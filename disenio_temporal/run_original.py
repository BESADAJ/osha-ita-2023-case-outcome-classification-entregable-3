"""Fase B del modelo original: validación anidada del ganador de la Fase A (balance='none').

Usa el mismo bucle anidado que la corrida base (osha3/nested.py: búsqueda en la muestra del
25 %, StratifiedGroupKFold(3) por establecimiento, reentrenamiento al 100 %, una medición en
la validación externa), pero escribe en results/original/runs/ para no tocar results/runs/.
Reanudable: las celdas con JSON ya escrito se saltan.

python run_original.py --model pt_bagging --workers 5 --n-evals 20
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed

from osha3.models import ORIGINAL
from osha3.nested import run_combination
from osha3.optimizers import OPTIMIZERS

ROOT = Path(__file__).resolve().parent / 'results' / 'original'
RUN_DIR = ROOT / 'runs'
LOG = ROOT / 'run_log.csv'
FOLD_SIZE = {1: 0.4, 2: 0.7, 3: 1.0}


def _task(model, opt, fold, n_evals):
    start = time.perf_counter()
    status = run_combination(model, 'none', opt, fold, n_evals=n_evals, n_jobs=1, run_dir=RUN_DIR)
    return model, opt, fold, status, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(ORIGINAL))
    ap.add_argument('--optimizers', nargs='+', default=list(OPTIMIZERS))
    ap.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--n-evals', type=int, default=20)
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()
    # Las más largas primero (pliegue 3), como en run_nested.py.
    tasks = sorted([(o, f) for o in args.optimizers for f in args.folds], key=lambda t: -FOLD_SIZE[t[1]])
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f'{len(tasks)} celdas | {args.model} | workers={args.workers} | n_evals={args.n_evals}', flush=True)
    new = not LOG.exists()
    with LOG.open('a', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(['finished_at', 'model', 'balance', 'optimizer', 'fold', 'status', 'seconds'])
        jobs = (delayed(_task)(args.model, o, f, args.n_evals) for o, f in tasks)
        for i, (m, o, f, status, sec) in enumerate(Parallel(n_jobs=args.workers, return_as='generator_unordered')(jobs), 1):
            writer.writerow([datetime.now().isoformat(timespec='seconds'), m, 'none', o, f, status, round(sec, 1)])
            fh.flush()
            print(f'[{i}/{len(tasks)}] {datetime.now():%H:%M} {m} none {o} fold{f}: {status} ({sec:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
