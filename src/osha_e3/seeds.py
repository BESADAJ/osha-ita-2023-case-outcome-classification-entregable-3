"""Sensibilidad a la semilla aleatoria: repite la validación anidada completa de algunas combinaciones.

Se mantienen fijos los datos (submuestra y pliegues); cambian la semilla de los modelos (Random
Forest, XGBoost, sobremuestreo) y la de los métodos de búsqueda (Random, TPE, genético). Así se aísla
la variabilidad del algoritmo de la del muestreo de datos.
"""
import json
import os

import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from . import nested as N

SEEDS = [7, 123, 2024]


def selected_combos():
    """Combinaciones a repetir: mejor RF y mejor XGBoost de cada tarea, la mejor global de
    clasificación y, para ver la estabilidad del algoritmo genético, la versión genética de esa mejor."""
    out = []
    for task, metric, asc in [('clf', 'outer_recall_mean', False), ('reg', 'outer_rmse_mean', True)]:
        mt = N.master_table(task).sort_values(metric, ascending=asc)
        best = mt.iloc[0]
        out.append((task, best.model, best.balance, best.optimizer))
        out.append((task, best.model, best.balance, 'genetic'))
        for m in ['random_forest', 'xgboost']:
            r = mt[mt.model == m].iloc[0]
            out.append((task, r.model, r.balance, r.optimizer))
    return list(dict.fromkeys(out))


def path(task, model, balance, opt, seed, fold):
    return C.RESULTS / 'seeds' / task / f'{model}__{balance}__{opt}__s{seed}__f{fold}.json'


def run_all(n_jobs=None):
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    cells = [(t, m, b, o, f, s) for (t, m, b, o) in selected_combos() for s in SEEDS for f in range(C.OUTER_FOLDS)]
    todo = [c for c in cells if not path(c[0], c[1], c[2], c[3], c[5], c[4]).exists()]
    print(f'semillas: {len(cells)} celdas, {len(todo)} pendientes', flush=True)
    Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(N.run_cell)(t, m, b, o, f, seed=s, out=path(t, m, b, o, s, f)) for t, m, b, o, f, s in todo)


def table():
    """Una fila por (combinación, semilla, pliegue), incluida la semilla principal 42."""
    rows = []
    for (t, m, b, o) in selected_combos():
        for s in [C.SEED] + SEEDS:
            for f in range(C.OUTER_FOLDS):
                p = path(t, m, b, o, s, f) if s != C.SEED else N.cell_path(t, m, b, o, f)
                if p.exists():
                    r = json.loads(p.read_text(encoding='utf-8'))
                    rows.append({'task': t, 'model': m, 'balance': b, 'optimizer': o, 'seed': s, 'fold': f,
                                 'inner_score': r['inner_score'], 'params': json.dumps(r['best_params']),
                                 **{f'outer_{k}': v for k, v in r['outer'].items()}})
    return pd.DataFrame(rows)
