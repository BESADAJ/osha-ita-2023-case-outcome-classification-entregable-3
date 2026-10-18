"""Reajuste final de cada combinación y evaluación en la prueba (octubre-diciembre).

Cada combinación se reajusta sobre toda la submuestra de desarrollo con sus hiperparámetros finales
(los del pliegue externo con mejor puntaje interno; ver ``nested.master_table``) y se evalúa una
sola vez en la prueba: establecimientos y fechas que no participaron en nada de lo anterior.
"""
import json
import os
import time
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from . import config as C
from . import data as D
from . import evaluation as E
from . import nested as N
from . import search as S


@lru_cache(maxsize=2)
def test_data(task):
    sub, dev, test = (D.classification_sets if task == 'clf' else D.regression_sets)()
    y = test.y_clf.to_numpy() - 1 if task == 'clf' else test.y_reg.to_numpy()
    return test, y


def out_path(task, combo):
    return C.RESULTS / 'final' / task / (combo.replace('|', '__') + '.npz')


def run_one(task, model, balance, opt, params_json, seed=C.SEED):
    """Reajusta una combinación y guarda predicciones de prueba, métricas y tiempos."""
    combo = f'{model}|{balance}|{opt}'
    path = out_path(task, combo)
    if path.exists():
        return
    with threadpool_limits(limits=1):
        X, y, _, _ = N.task_data(task)
        test, yt = test_data(task)
        params = json.loads(params_json)
        t0 = time.perf_counter()
        pipe = S.refit(task, model, params, balance, X, y, seed)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = E.predict_output(task, pipe, test[D.FEATURES])
        pred_s = time.perf_counter() - t0
    m = E.metrics_for(task, yt, out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, out=out.astype(np.float32),
                        meta=json.dumps({'combo': combo, 'params': params, 'metrics': m,
                                         'fit_seconds': fit_s, 'predict_seconds': pred_s}))


def run_all(task, n_jobs=None):
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    mt = N.master_table(task)
    todo = [r for r in mt.itertuples() if not out_path(task, r.combo).exists()]
    print(f'{task}: {len(mt)} combinaciones, {len(todo)} pendientes', flush=True)
    Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(run_one)(task, r.model, r.balance, r.optimizer, r.final_params) for r in todo)


def load(task, combo):
    """(predicciones de prueba, metadatos) de una combinación."""
    z = np.load(out_path(task, combo))
    return z['out'], json.loads(str(z['meta']))


def test_table(task):
    """Métricas de prueba de todas las combinaciones."""
    rows = []
    for p in sorted((C.RESULTS / 'final' / task).glob('*.npz')):
        meta = json.loads(str(np.load(p)['meta']))
        m, b, o = meta['combo'].split('|')
        rows.append({'model': m, 'balance': b, 'optimizer': o, 'combo': meta['combo'],
                     **{f'test_{k}': v for k, v in meta['metrics'].items()},
                     'fit_seconds': meta['fit_seconds'], 'predict_seconds': meta['predict_seconds']})
    return pd.DataFrame(rows)
