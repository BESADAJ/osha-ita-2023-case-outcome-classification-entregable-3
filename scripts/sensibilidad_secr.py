"""Sensibilidad y robustez de SECR (capítulo 20.7): BA en el pliegue externo al fijar cada hiperparámetro en una rejilla
(sin TPE), al cambiar la variable de compuerta y al reducir los datos con que se ajustan los pesos. Es un análisis
descriptivo: nada de esto se usa para elegir el modelo (eso lo hizo la búsqueda TPE interna).

Uso: python scripts/sensibilidad_secr.py  ->  results/pareado/sensibilidad_secr.csv
"""
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from osha_e3 import combinador as CB, config as C, original_eval as OE, paired as PR

warnings.filterwarnings('ignore')
threadpool_limits(1)
BASE = {'gate': 'naics2', 'lam': 100.0, 'n_steps': 30, 'min_region': 200, 'alpha': 1.0}
GRID = [('lam', v) for v in [1.0, 10.0, 100.0, 1000.0, 10000.0]] + \
       [('alpha', v) for v in [0.0, 0.25, 0.5, 0.75, 1.0]] + \
       [('n_steps', v) for v in [5, 10, 30, 100]] + \
       [('gate', v) for v in CB.GATES] + [('frac_datos', v) for v in [0.1, 0.25, 0.5, 1.0]]

X, _, _ = PR.paired_data()
rows = []
for d in PR.load('original'):
    names, P_in, P_out = OE.stacks(d)
    y_in, y_out = d['y_train'].astype(int), d['y'].astype(int)
    X_in, X_out = X.iloc[d['idx_train']].reset_index(drop=True), X.iloc[d['idx']].reset_index(drop=True)
    for k, v in GRID:
        params = dict(BASE)
        idx = np.arange(len(y_in))
        if k == 'frac_datos':
            rng = np.random.default_rng(C.SEED + d['rep'] * 10 + d['fold'])
            idx = np.sort(rng.choice(len(y_in), int(v * len(y_in)), replace=False))
        else:
            params[k] = v
        m, q = OE._fit_predict(params, P_in[idx], y_in[idx], X_in.iloc[idx].reset_index(drop=True), P_out, X_out)
        rows.append({'rep': d['rep'], 'fold': d['fold'], 'param': k, 'value': str(v),
                     'ba': balanced_accuracy_score(y_out, m.decide(q)),
                     'log_loss': float(-np.log(np.clip(q[np.arange(len(y_out)), y_out], 1e-12, 1)).mean())})
    print('celda', d['rep'], d['fold'], flush=True)
out = pd.DataFrame(rows)
out.to_csv(C.RESULTS / 'pareado' / 'sensibilidad_secr.csv', index=False)
print(out.groupby(['param', 'value'])[['ba', 'log_loss']].agg(['mean', 'std']).round(4).to_string())
