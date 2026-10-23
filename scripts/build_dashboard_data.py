"""Tablas livianas del dashboard (``dashboard_final/data``). Nada sale de la prueba (octubre-diciembre).

* ``eda.csv.gz``: la submuestra de desarrollo (30 030 casos, ene-sep 2023) sin identificadores.
* ``leaderboard.csv``: las 112 combinaciones (media ± d.e. del bucle externo) y la marca de la mejor por modelo.
* ``oof_<modelo>.csv.gz``: probabilidades fuera de pliegue externo de la mejor combinación de cada modelo; cada
  pliegue se reajusta con los hiperparámetros que ese pliegue eligió en la validación anidada.
* ``shap_global.csv``: |SHAP| medio (exacto) por variable y modelo sobre 150 casos del desarrollo.

Uso: python scripts/build_dashboard_data.py [eda] [leaderboard] [oof] [shap_global]
"""
import json
import sys
import warnings

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from osha_e3 import config as C, data as D, ensemble as ENS, nested as N, search as S

OUT = C.ROOT / 'dashboard_final' / 'data'
sys.path.insert(0, str(C.ROOT / 'dashboard_final'))
from src.explain import exact_shapley  # noqa: E402

TITLES = ['industry_description', 'soc_description', 'nature_title_pred', 'part_title_pred', 'event_title_pred',
          'source_title_pred', 'size', 'annual_average_employees', 'djtr_num_tr']


def eda():
    sub = D.classification_sets()[0]
    X = ENS.add_full_codes(sub)
    raw = pd.read_csv(C.data_path(), usecols=['id'] + TITLES, dtype=str, encoding='latin1')
    X = X.merge(raw, on='id', how='left', validate='one_to_one')
    out = pd.DataFrame({c: X[c] for c in D.FEATURES + ['naics6', 'soc6', 'nature', 'part', 'event', 'source']})
    out['naics2'] = X.naics3.str[:2]
    out['date'] = pd.to_datetime(X.date).dt.strftime('%Y-%m-%d')
    out['outcome'] = X.y_clf.map(dict(zip(C.CLASS_LABELS, C.CLASS_NAMES)))
    out['days_away'] = X.days_away
    out['days_restricted'] = pd.to_numeric(X.djtr_num_tr, errors='coerce')
    out['employees'] = pd.to_numeric(X.annual_average_employees, errors='coerce')
    out['size'] = X['size']
    for c in ['industry_description', 'soc_description', 'nature_title_pred', 'part_title_pred', 'event_title_pred',
              'source_title_pred']:
        out[c] = X[c].fillna('Unknown').str.slice(0, 60)
    out.to_csv(OUT / 'eda.csv.gz', index=False, compression='gzip')
    print('eda', out.shape, flush=True)


def leaderboard():
    mc = pd.read_csv(C.RESULTS / 'master_clf.csv')
    f = N.fold_table('clf')
    keys = ['model', 'balance', 'optimizer']
    f['ms_case'] = f.predict_seconds / 6006 * 1000  # ~6 006 casos por pliegue externo
    mc = mc.join(f.groupby(keys).ms_case.mean().rename('inference_ms_per_case'), on=keys)
    mc['best_of_model'] = mc.groupby('model')['outer_recall_mean'].transform('max').eq(mc.outer_recall_mean)
    cols = keys + ['best_of_model', 'inference_ms_per_case', 'search_seconds_total', 'final_params'] + \
        [c for c in mc.columns if c.startswith('outer_')]
    mc[cols].to_csv(OUT / 'leaderboard.csv', index=False)
    print('leaderboard', len(mc), flush=True)


def _oof_fold(model, balance, opt, fold):
    X, y, g, _ = N.task_data('clf')
    tr, te = N.outer_folds('clf')[fold]
    cell = json.loads(N.cell_path('clf', model, balance, opt, fold).read_text(encoding='utf-8'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipe = S.refit('clf', model, cell['best_params'], balance, X.iloc[tr], y[tr], C.SEED)
        return te, pipe.predict_proba(X.iloc[te])


def oof():
    X, y, g, _ = N.task_data('clf')
    mc = pd.read_csv(C.RESULTS / 'master_clf.csv')
    best = mc.loc[mc.groupby('model')['outer_recall_mean'].idxmax()]
    jobs = [(r.model, r.balance, r.optimizer, f) for r in best.itertuples() for f in range(C.OUTER_FOLDS)]
    res = Parallel(n_jobs=3, backend='loky')(delayed(_oof_fold)(*j) for j in jobs)
    for m in best.model:
        P = np.zeros((len(y), 4))
        fold = np.zeros(len(y), dtype=int)
        for (mm, _, _, f), (te, p) in zip(jobs, res):
            if mm == m:
                P[te], fold[te] = p, f
        df = pd.DataFrame(P.round(5), columns=[f'p{k}' for k in range(4)])
        df['y'], df['fold'] = y, fold
        df.to_csv(OUT / f'oof_{m}.csv.gz', index=False, compression='gzip')
        print('oof', m, flush=True)


def shap_global():
    X, y, g, _ = N.task_data('clf')
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(X), 150, replace=False)
    bg = X.iloc[rng.choice(len(X), 20, replace=False)]
    groups = {c: [c] for c in D.FEATURES}
    rows = []
    for path in sorted((C.ROOT / 'dashboard_final' / 'artifacts').glob('*.joblib')):
        if path.stem == 'original':
            continue
        est = joblib.load(path)
        for i in idx:
            x = X.iloc[[i]]
            cls = int(est.predict_proba(x).argmax())
            phi = exact_shapley(est.predict_proba, x, bg, groups, cls)
            rows.append({'model': path.stem, 'case': int(i), 'cls': cls, **phi.to_dict()})
        print('shap', path.stem, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'shap_cases.csv.gz', index=False, compression='gzip')
    df.groupby('model')[D.FEATURES].agg(lambda v: np.abs(v).mean()).to_csv(OUT / 'shap_global.csv')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for step in (sys.argv[1:] or ['eda', 'leaderboard', 'oof', 'shap_global']):
        globals()[step]()
