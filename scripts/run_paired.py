"""Corre (o reanuda) el experimento pareado (hasta 5×5; las celdas recortadas se marcan con un archivo vacío) del modelo original y los 7 benchmarks.

Uso: python scripts/run_paired.py [n_procesos] [repeticiones, p. ej. 0,1,2]
Cada celda se guarda en results/pareado/<modelo>/r<rep>_f<pliegue>.npz.
"""
import sys
import time

import _bootstrap  # noqa: F401
from joblib import Parallel, delayed

from osha_e3 import paired as P

if __name__ == '__main__':
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    reps = [int(r) for r in sys.argv[2].split(',')] if len(sys.argv) > 2 else range(P.REPEATS)
    todo = [j for j in P.jobs(reps) if not P.cell_path(*j).exists()]
    print(f'{len(todo)} celdas pendientes, {n_jobs} procesos', flush=True)
    t0 = time.time()
    Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(delayed(P.run)(*j) for j in todo)
    print(f'listo en {(time.time() - t0) / 60:.1f} min', flush=True)
