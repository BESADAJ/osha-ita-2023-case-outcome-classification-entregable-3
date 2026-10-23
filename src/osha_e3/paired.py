"""Experimento pareado del modelo original (capítulos 19 y 20).

Para que la comparación del modelo original con los benchmarks sea válida (guía §5.3 y §6.2) todos los
modelos se evalúan con:

* las **mismas variables crudas** ``VPLUS``: las 9 del curso más los códigos OIICS de la lesión, NAICS de 6
  dígitos, SOC completo y el empleador (EIN y nombre de la empresa). Cada modelo aplica su propio
  preprocesamiento aprendido dentro del pliegue: el pipeline del curso (one-hot que agrupa categorías raras
  → balanceo → modelo) para los benchmarks y las vistas con codificación por objetivo fuera de pliegue para el
  modelo original;
* las **mismas particiones**: validación cruzada externa repetida 5×5 por establecimiento. La repetición 0
  es exactamente la partición externa de los entregables anteriores; las repeticiones 1-4 usan las semillas
  ``seed_for('outer', r)``, fijadas antes de correr;
* la **misma búsqueda interna** para los benchmarks (3 pliegues internos, presupuesto de 16
  configuraciones, el mismo optimizador y balanceo ganadores del capítulo 9);
* nada de la prueba (octubre-diciembre): todo sale del bucle externo.

Cada celda (modelo, repetición, pliegue) se guarda en ``results/pareado/`` y la corrida puede reanudarse.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from . import config as C
from . import data as D
from . import ensemble as ENS
from . import evaluation as E
from . import models as M
from . import nested as N
from . import search as S

OUT = C.RESULTS / 'pareado'
REPEATS = 5
VPLUS = D.FEATURES + ['nature', 'part', 'event', 'source', 'sec_source', 'naics6', 'soc6', 'ein', 'company']
METRIC = 'outer_recall_mean'  # balanced accuracy: la métrica principal de los entregables 2 y 3


def best_combos(metric=METRIC):
    """Mejor (balanceo, optimizador) de cada uno de los 7 modelos según la media del bucle externo."""
    mc = pd.read_csv(C.RESULTS / 'master_clf.csv')
    best = mc.loc[mc.groupby('model')[metric].idxmax()]
    return {r.model: (r.balance, r.optimizer) for r in best.itertuples()}


_DATA = {}


def paired_data():
    """(X con ``VPLUS`` y códigos completos, y 0..3, establecimientos): la misma submuestra de 30 030 casos."""
    if not _DATA:
        X, y, g, s = N.task_data('clf')
        sub = D.classification_sets()[0]
        Xf = ENS.add_full_codes(sub)
        assert (Xf.y_clf.to_numpy() - 1 == y).all() and (Xf.establishment_id.to_numpy() == g).all()
        _DATA['v'] = (Xf, y, g)
    return _DATA['v']


def outer_folds(rep):
    """Pliegues externos de la repetición ``rep`` (la 0 coincide con ``nested.outer_folds('clf')``)."""
    if rep == 0:
        return N.outer_folds('clf')
    X, y, g = paired_data()
    cv = StratifiedGroupKFold(C.OUTER_FOLDS, shuffle=True, random_state=C.seed_for('outer', rep))
    return list(cv.split(X, y, g))


def inner_folds(rep, fold, train_idx):
    X, y, g = paired_data()
    seed = C.seed_for('inner', fold if rep == 0 else 10 * rep + fold)
    cv = StratifiedGroupKFold(C.INNER_FOLDS, shuffle=True, random_state=seed)
    return list(cv.split(train_idx, y[train_idx], g[train_idx]))


def cell_path(model, rep, fold):
    return OUT / model / f'r{rep}_f{fold}.npz'


def run_benchmark(model, rep, fold, balance=None, opt=None):
    """Mejor combinación del curso para ``model`` con las variables ``VPLUS`` en un pliegue externo."""
    path = cell_path(model, rep, fold)
    if path.exists():
        return
    if balance is None:
        balance, opt = best_combos()[model]
    with threadpool_limits(limits=1):
        X, y, g = paired_data()
        tr, te = outer_folds(rep)[fold]
        Xtr, ytr = X[VPLUS].iloc[tr].reset_index(drop=True), y[tr]
        ev = S.Evaluator('clf', model, balance, Xtr, ytr, inner_folds(rep, fold, tr), C.SEED, C.BUDGET)
        t0 = time.perf_counter()
        params, inner, _ = S.run_search(opt, ev, 'clf', model, C.seed_for('search', fold) + 100 * rep)
        t_search = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipe = S.refit('clf', model, params, balance, Xtr, ytr, C.SEED)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = pipe.predict_proba(X[VPLUS].iloc[te])
        t_pred = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=p.astype(np.float32), y=y[te], idx=te, params=json.dumps(params, default=float),
                        inner=inner, search_s=t_search, fit_s=t_fit, pred_s=t_pred, balance=balance, opt=opt)


def run_original(rep, fold):
    """Ajusta el ensamble multivista (nivel 0 de G) en un pliegue externo y guarda todo lo que el combinador
    necesita: predicciones *fuera de pliegue internas* de cada modelo base (para elegir pesos e
    hiperparámetros del combinador sin mirar el pliegue externo) y sus probabilidades en el pliegue externo."""
    path = cell_path('original', rep, fold)
    if path.exists():
        return
    with threadpool_limits(limits=2):
        X, y, g = paired_data()
        tr, te = outer_folds(rep)[fold]
        Xtr = X.iloc[tr].reset_index(drop=True)
        t0 = time.perf_counter()
        ens = ENS.StackedEnsemble(ENS.default_views(full='diverso')).fit(Xtr, y[tr], g[tr])
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        P = ens.base_proba(X.iloc[te])
        t_pred = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y=y[te], idx=te, y_train=y[tr], idx_train=tr, fit_s=t_fit, pred_s=t_pred,
                        names=np.array(ens.names_),
                        **{f'oof_{n}': ens.oof_[n].astype(np.float32) for n in ens.names_},
                        **{f'out_{n}': P[n].astype(np.float32) for n in ens.names_})


def jobs(reps=range(REPEATS)):
    """Lista ordenada de celdas: primero la repetición 0 y, dentro de cada repetición, las más lentas."""
    order = ['original', 'knn', 'random_forest', 'svm', 'xgboost', 'logreg', 'tree', 'naive_bayes']
    return [(m, r, f) for r in reps for m in order for f in range(C.OUTER_FOLDS)]


def run(model, rep, fold):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return run_original(rep, fold) if model == 'original' else run_benchmark(model, rep, fold)


def load(model):
    """Celdas guardadas de ``model`` como lista de dicts ordenada por (rep, fold)."""
    out = []
    for r in range(REPEATS):
        for f in range(C.OUTER_FOLDS):
            p = cell_path(model, r, f)
            if p.exists():
                d = dict(np.load(p, allow_pickle=False))
                d['rep'], d['fold'] = r, f
                out.append(d)
    return out
