"""Complejidad empírica de SECR (capítulo 19): tiempo de ajuste e inferencia al variar n, M (modelos base) y el
número de regiones R, con datos sintéticos. Contrasta con O(T·n·M·(R+1)) para ajustar y O(M·K) por caso para predecir.
Uso: python scripts/complejidad_secr.py  ->  results/pareado/complejidad_secr.csv
"""
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from osha_e3 import combinador as CB, config as C

K, T = 4, 30
rows = []
rng = np.random.default_rng(C.SEED)


def data(n, M, R):
    y = rng.integers(0, K, n)
    P = rng.dirichlet(np.ones(K), (n, M))
    P[np.arange(n), :, y] += rng.uniform(0, 0.5, (n, M))
    P /= P.sum(axis=2, keepdims=True)
    r = rng.integers(1, R + 1, n) if R else np.zeros(n, dtype=int)
    return CB.meta_matrix(P, r), y


for n, M, R in [(n, 16, 10) for n in [2000, 4000, 8000, 16000, 32000, 64000]] + \
               [(16000, M, 10) for M in [2, 4, 8, 16, 32, 64]] + [(16000, 16, R) for R in [0, 5, 10, 20, 40]]:
    X, y = data(n, M, R)
    m = CB.ConditionalEnsembleSelection(n_steps=T, tol=0, min_region=1, gate=R > 0, lam=100)
    t0 = time.perf_counter(); m.fit(X, y); fit = time.perf_counter() - t0
    t0 = time.perf_counter(); m.predict_proba(X); pred = time.perf_counter() - t0
    rows.append({'n': n, 'M': M, 'R': R, 'T': T, 'fit_s': fit, 'pred_us_case': pred / n * 1e6})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(C.RESULTS / 'pareado' / 'complejidad_secr.csv', index=False)
