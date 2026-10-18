"""Corre (o reanuda) la validación anidada de las 140 combinaciones y escribe la tabla maestra.

Uso: python scripts/run_nested.py [clf|reg|all] [n_jobs]
"""
import sys
import warnings

import _bootstrap  # noqa: F401
from osha_e3 import config as C, nested as N

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    C.set_global_seed()
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for task in (['reg', 'clf'] if which == 'all' else [which]):
        N.run_all(task, n_jobs=n_jobs)
        N.fold_table(task).to_csv(C.RESULTS / f'outer_folds_{task}.csv', index=False)
        N.master_table(task).to_csv(C.RESULTS / f'master_{task}.csv', index=False)
        print('Tabla maestra escrita:', C.RESULTS / f'master_{task}.csv', flush=True)
