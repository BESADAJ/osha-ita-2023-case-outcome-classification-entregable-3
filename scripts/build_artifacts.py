"""Reentrena y serializa los artefactos del dashboard (guía §4.2), con metadata.json y prueba de ida y vuelta.

* Los 7 modelos del curso: mejor combinación (balanceo, optimizador) de cada uno según la balanced accuracy
  media del bucle externo (``results/master_clf.csv``), reajustada con sus hiperparámetros finales sobre la
  submuestra de desarrollo completa (30 030 casos, el mismo conjunto de entrenamiento de los capítulos 6 a 13).
* El modelo original (``OriginalModel``), ajustado sobre la misma submuestra con las variables ``VPLUS``.

La prueba de ida y vuelta usa 500 casos fijos del desarrollo que no están en la submuestra (nunca la prueba).
Uso: python scripts/build_artifacts.py [curso|original|todo]
"""
import json
import platform
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd

from osha_e3 import config as C, data as D, ensemble as ENS, nested as N, paired as PR, search as S

ART = C.ROOT / 'dashboard_final' / 'artifacts'
LIBS = ['numpy', 'pandas', 'scipy', 'sklearn', 'imblearn', 'xgboost', 'optuna', 'joblib']


def versions():
    out = {'python': platform.python_version()}
    for m in LIBS:
        out[m] = __import__(m).__version__
    return out


def roundtrip_rows(cols):
    sub, dev, _ = D.classification_sets()
    rest = dev.loc[~dev.id.isin(set(sub.id))]
    rows = rest.sample(500, random_state=C.SEED).reset_index(drop=True)
    return ENS.add_full_codes(rows) if any(c not in rows for c in cols) else rows


def schema(X, cols):
    out = []
    for c in cols:
        vc = X[c].astype(str).value_counts()
        out.append({'name': c, 'type': 'categórica (texto)', 'n_values': int(vc.size),
                    'values': vc.index[:400].tolist(), 'most_frequent': vc.index[0],
                    'note': 'valores fuera de la lista se tratan como categoría infrecuente/desconocida'})
    return out


def save(name, est, meta, X_rt, cols):
    path = ART / f'{name}.joblib'
    p_mem = est.predict_proba(X_rt[cols] if name != 'original' else X_rt)
    joblib.dump(est, path, compress=3)
    p_disk = joblib.load(path).predict_proba(X_rt[cols] if name != 'original' else X_rt)
    meta['roundtrip'] = {'n_rows': len(X_rt), 'identical': bool(np.array_equal(p_mem, p_disk)),
                         'max_abs_diff': float(np.abs(p_mem - p_disk).max()),
                         'source': '500 casos fijos del desarrollo fuera de la submuestra (semilla 42); nunca la prueba'}
    meta['size_mb'] = round(path.stat().st_size / 2**20, 2)
    assert meta['roundtrip']['identical'], name
    (ART / f'{name}.metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(name, meta['size_mb'], 'MB', meta['roundtrip'], flush=True)


def course():
    X, y, g, _ = N.task_data('clf')
    mc = pd.read_csv(C.RESULTS / 'master_clf.csv')
    best = mc.loc[mc.groupby('model')['outer_recall_mean'].idxmax()]
    X_rt = roundtrip_rows(D.FEATURES)
    for r in best.itertuples():
        params = json.loads(r.final_params)
        t0 = time.perf_counter()
        pipe = S.refit('clf', r.model, params, r.balance, X, y, C.SEED)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipe.predict_proba(X_rt[D.FEATURES])
        pred_ms = (time.perf_counter() - t0) / len(X_rt) * 1000
        outer = {k.replace('outer_', '').replace('_mean', ''): {'mean': getattr(r, k), 'sd': getattr(r, k.replace('_mean', '_std'))}
                 for k in mc.columns if k.startswith('outer_') and k.endswith('_mean')}
        meta = {'model': r.model, 'balance': r.balance, 'optimizer': r.optimizer, 'seed': C.SEED,
                'hyperparameters': params, 'selection': 'máxima balanced accuracy media del bucle externo (5 pliegues)',
                'outer_metrics': outer, 'trained_on': f'submuestra de desarrollo ({len(y)} casos, ene-sep 2023)',
                'fit_seconds': fit_s, 'inference_ms_per_case': pred_ms, 'versions': versions(),
                'input_schema': schema(X, D.FEATURES), 'classes': C.CLASS_NAMES}
        save(r.model, pipe, meta, X_rt, D.FEATURES)


def original():
    from osha_e3.original_final import OriginalModel
    X, y, g = PR.paired_data()
    X_rt = roundtrip_rows(PR.VPLUS)
    t0 = time.perf_counter()
    est = OriginalModel().fit(X, y, g)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    est.predict_proba(X_rt)
    pred_ms = (time.perf_counter() - t0) / len(X_rt) * 1000
    meta = {'model': 'original (ensamble multivista + SECR)', 'seed': C.SEED, 'hyperparameters': est.params_,
            'inner_score_ba': est.inner_score_, 'base_models': est.stack_.names_,
            'weights_global': dict(zip(est.stack_.names_, est.combiner_.weights_global_.round(4).tolist())),
            'n_regions_with_own_weights': len(est.combiner_.weights_region_),
            'trained_on': f'submuestra de desarrollo ({len(y)} casos, ene-sep 2023)', 'fit_seconds': fit_s,
            'inference_ms_per_case': pred_ms, 'versions': versions(),
            'input_schema': schema(X, PR.VPLUS), 'classes': C.CLASS_NAMES,
            'outer_metrics': 'ver results/pareado/resumen.json (validación anidada repetida 5x5)'}
    save('original', est, meta, X_rt, PR.VPLUS)


if __name__ == '__main__':
    ART.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else 'todo'
    if what in ('curso', 'todo'):
        course()
    if what in ('original', 'todo'):
        original()
