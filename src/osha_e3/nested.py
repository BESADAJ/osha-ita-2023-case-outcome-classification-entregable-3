"""Validación cruzada anidada para las 140 combinaciones (modelo × balanceo × optimizador).

* Bucle externo: ``OUTER_FOLDS`` pliegues por establecimiento (``StratifiedGroupKFold``). Solo
  estima el desempeño del modelo ya optimizado; nunca participa en la búsqueda.
* Bucle interno: dentro del entrenamiento externo, ``INNER_FOLDS`` pliegues por establecimiento
  donde cada método de búsqueda evalúa configuraciones.

Cada celda (combinación, pliegue externo) se guarda en un JSON propio, así la corrida puede
interrumpirse y reanudarse sin repetir trabajo. Las celdas se reparten entre procesos con joblib
(backend ``loky``); cada proceso usa un solo hilo de BLAS/OpenMP para no sobresuscribir la CPU.
"""
import json
import os
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from . import config as C
from . import data as D
from . import evaluation as E
from . import models as M
from . import search as S

HEAVY = {'random_forest': 0, 'xgboost': 1, 'knn': 2, 'logreg': 3, 'lasso': 3}


@lru_cache(maxsize=2)
def task_data(task):
    """Submuestra de desarrollo de la tarea: (X, y, grupos, estratos). y de clasificación va 0..3."""
    sub = (D.classification_sets if task == 'clf' else D.regression_sets)()[0]
    X = sub[D.FEATURES]
    if task == 'clf':
        y = sub.y_clf.to_numpy() - 1
        strata = y
    else:
        y = sub.y_reg.to_numpy()
        strata = pd.qcut(y, 5, labels=False, duplicates='drop')
    return X, y, sub.establishment_id.to_numpy(), np.asarray(strata)


def outer_folds(task):
    X, y, g, s = task_data(task)
    cv = StratifiedGroupKFold(n_splits=C.OUTER_FOLDS, shuffle=True, random_state=C.seed_for('outer'))
    return list(cv.split(X, s, g))


def inner_folds(task, train_idx, fold):
    X, y, g, s = task_data(task)
    cv = StratifiedGroupKFold(n_splits=C.INNER_FOLDS, shuffle=True, random_state=C.seed_for('inner', fold))
    return list(cv.split(train_idx, s[train_idx], g[train_idx]))


def cell_path(task, model, balance, opt, fold):
    return C.RESULTS / 'nested' / task / f'{model}__{balance}__{opt}__f{fold}.json'


def run_cell(task, model, balance, opt, fold, budget=C.BUDGET, seed=C.SEED, out=None):
    """Optimiza en el bucle interno y evalúa en el pliegue externo ``fold``.

    Returns
    -------
    dict
        Métricas externas, mejores hiperparámetros, puntaje interno, tiempos e historial.
    """
    path = out or cell_path(task, model, balance, opt, fold)
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    with threadpool_limits(limits=1):
        X, y, g, s = task_data(task)
        tr, te = outer_folds(task)[fold]
        Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]
        folds = inner_folds(task, tr, fold)
        ev = S.Evaluator(task, model, balance, Xtr, ytr, folds, seed, budget)
        t0 = time.perf_counter()
        params, inner, extra = S.run_search(opt, ev, task, model, C.seed_for('search', fold) + (seed - C.SEED))
        t_search = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipe = S.refit(task, model, params, balance, Xtr, ytr, seed)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = E.predict_output(task, pipe, X.iloc[te])
        t_pred = time.perf_counter() - t0
    res = {'task': task, 'model': model, 'balance': balance, 'optimizer': opt, 'fold': fold, 'seed': seed,
           'budget': budget, 'n_evals': len(ev.history), 'prepare_seconds': getattr(ev, 'prepare_seconds', 0.0), 'best_params': params, 'inner_score': inner,
           'outer': E.metrics_for(task, y[te], pred), 'search_seconds': t_search,
           'refit_seconds': t_fit, 'predict_seconds': t_pred, 'n_train': int(len(tr)), 'n_test': int(len(te)),
           'history': ev.history, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, default=_json_default), encoding='utf-8')
    return res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def run_all(task, n_jobs=None, verbose=5):
    """Ejecuta (o reanuda) todas las celdas de una tarea en paralelo."""
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    cells = [(task, m, b, o, f) for (m, b, o) in M.combos(task) for f in range(C.OUTER_FOLDS)]
    todo = [c for c in cells if not cell_path(*c).exists()]
    todo.sort(key=lambda c: (HEAVY.get(c[1], 9), c[2] != 'smote', c[2] != 'adasyn'))
    print(f'{task}: {len(cells)} celdas, {len(todo)} pendientes, n_jobs={n_jobs}', flush=True)
    Parallel(n_jobs=n_jobs, backend='loky', verbose=verbose)(delayed(run_cell)(*c) for c in todo)


def load_cells(task):
    """Lee todas las celdas guardadas como lista de diccionarios."""
    folder = C.RESULTS / 'nested' / task
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('*.json'))]


def fold_table(task):
    """Una fila por (combinación, pliegue externo) con las métricas externas y tiempos."""
    rows = []
    for r in load_cells(task):
        row = {k: r[k] for k in ['model', 'balance', 'optimizer', 'fold', 'seed', 'n_evals', 'inner_score',
                                 'search_seconds', 'refit_seconds', 'predict_seconds']}
        row.update({f'outer_{k}': v for k, v in r['outer'].items()})
        row['best_params'] = json.dumps(r['best_params'])
        row['eval_seconds_mean'] = float(np.mean([h['seconds'] for h in r['history']]))
        rows.append(row)
    return pd.DataFrame(rows)


def master_table(task):
    """Tabla maestra: una fila por combinación, métricas externas como media y desviación estándar.

    Los hiperparámetros «finales» de cada combinación son los del pliegue externo con mejor puntaje
    interno (se usan para reajustar sobre todo el desarrollo y evaluar en la prueba).
    """
    f = fold_table(task)
    keys = ['model', 'balance', 'optimizer']
    metrics = [c for c in f.columns if c.startswith('outer_')]
    agg = f.groupby(keys)[metrics].agg(['mean', 'std'])
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    times = f.groupby(keys).agg(n_folds=('fold', 'size'), n_evals_total=('n_evals', 'sum'),
                                search_seconds_total=('search_seconds', 'sum'),
                                eval_seconds_mean=('eval_seconds_mean', 'mean'),
                                inner_score_mean=('inner_score', 'mean'), seed=('seed', 'first'))
    best = f.loc[f.groupby(keys).inner_score.idxmax(), keys + ['best_params']].set_index(keys)
    out = agg.join(times).join(best.rename(columns={'best_params': 'final_params'})).reset_index()
    out['combo'] = out.model + '|' + out.balance + '|' + out.optimizer
    return out
