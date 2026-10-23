"""Constantes globales del proyecto: rutas, semilla única y tamaños de muestra.

Toda aleatoriedad del proyecto parte de ``SEED``. Las semillas derivadas se
obtienen sumando un desplazamiento fijo y documentado (ver ``seed_for``).
"""
from pathlib import Path
import os
import random

import numpy as np

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'jbook' / 'figures'
DATA_FILE = 'ITA Case Detail Data 2023 through 12-31-2023OIICS.csv'

# Diseño de partición heredado del entregable 1 (no se cambia).
CUTOFF = '2023-10-01'
GAP_DAYS = 7
HOLDOUT_GROUP_FRACTION = 0.60

# Validación anidada.
OUTER_FOLDS = 5
INNER_FOLDS = 3
BUDGET = 16  # evaluaciones (configuraciones) por búsqueda; igual para los 4 métodos

# Submuestras por establecimiento para la búsqueda (ver capítulo de optimización computacional).
N_SUB_CLF = 30_000
N_SUB_REG = 20_000

CLASS_LABELS = [1, 2, 3, 4]
CLASS_NAMES = ['Fallecimiento', 'Días de ausencia', 'Traslado/restricción', 'Otro registrable']

OFFSETS = {'subsample': 1, 'outer': 2, 'inner': 3, 'search': 4, 'model': 5, 'bootstrap': 100}


def seed_for(kind, extra=0):
    """Semilla derivada de la semilla global.

    Parameters
    ----------
    kind : str
        Clave de ``OFFSETS`` (p. ej. ``'outer'``).
    extra : int
        Desplazamiento adicional (número de pliegue, réplica, etc.).

    Returns
    -------
    int
    """
    return SEED + OFFSETS[kind] * 1000 + int(extra)


def set_global_seed(seed=SEED):
    """Fija las semillas de ``random``, ``numpy`` y ``PYTHONHASHSEED``."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def data_path():
    """Ruta del CSV crudo de OSHA.

    Busca en ``$OSHA_DATA_DIR`` y, si no existe, en ``../../Proyecto_ML`` (estructura local del
    equipo). El CSV (655 MB) no se versiona.
    """
    candidates = []
    if os.environ.get('OSHA_DATA_DIR'):
        candidates.append(Path(os.environ['OSHA_DATA_DIR']) / DATA_FILE)
    candidates += [ROOT / DATA_FILE, ROOT.parent.parent / 'Proyecto_ML' / DATA_FILE]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError('No se encontró el CSV de OSHA. Fijar OSHA_DATA_DIR. Buscado en: '
                            + ', '.join(map(str, candidates)))
