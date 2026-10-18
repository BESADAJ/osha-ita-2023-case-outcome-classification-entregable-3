"""Repite la validación anidada de combinaciones seleccionadas con otras 3 semillas.

Uso: python scripts/run_seeds.py [n_jobs]   (requiere haber corrido run_nested.py)
"""
import sys
import warnings

import _bootstrap  # noqa: F401
from osha_e3 import config as C, seeds

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    C.set_global_seed()
    seeds.run_all(int(sys.argv[1]) if len(sys.argv) > 1 else None)
    seeds.table().to_csv(C.RESULTS / 'seeds.csv', index=False)
