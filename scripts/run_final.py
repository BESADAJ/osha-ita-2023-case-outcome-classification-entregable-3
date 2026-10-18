"""Reajusta cada combinación con sus hiperparámetros finales y la evalúa en la prueba.

Uso: python scripts/run_final.py [clf|reg|all] [n_jobs]
"""
import sys
import warnings

import _bootstrap  # noqa: F401
from osha_e3 import config as C, final as F

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    C.set_global_seed()
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for task in (['reg', 'clf'] if which == 'all' else [which]):
        F.run_all(task, n_jobs=n_jobs)
        F.test_table(task).to_csv(C.RESULTS / f'test_{task}.csv', index=False)
        print('Tabla de prueba escrita:', C.RESULTS / f'test_{task}.csv', flush=True)
