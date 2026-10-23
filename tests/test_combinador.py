"""Pruebas del combinador SECR (``src/osha_e3/combinador.py``), núcleo del modelo original.

Cubren lo que pide la guía (§5.4): (i) aprende la solución conocida en datos sintéticos, (ii) reproducibilidad,
(iii) entradas inválidas y (iv) compatibilidad con ``sklearn.base.clone``; y las propiedades demostradas en el
capítulo 19: pesos en el símplex, pérdida no creciente y brecha de Frank-Wolfe como cota del error.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from osha_e3 import combinador as CB  # noqa: E402

K = 4


def _synthetic(n=4000, seed=0):
    """Dos regiones. En la región 1 el modelo 0 es casi perfecto y el 1 es ruido; en la 2, al revés.
    El modelo 2 es mediocre en todas partes. Solución conocida: w_1 ≈ e_0, w_2 ≈ e_1."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    r = np.where(np.arange(n) < n // 2, 1, 2)
    good = np.full((n, K), 0.02)
    good[np.arange(n), y] = 0.94
    noise = rng.dirichlet(np.ones(K), n)
    mid = 0.5 * good + 0.5 * noise
    P = np.empty((n, 3, K))
    P[:, 0] = np.where((r == 1)[:, None], good, noise)
    P[:, 1] = np.where((r == 2)[:, None], good, noise)
    P[:, 2] = mid
    return CB.meta_matrix(P, r), y, P, r


def test_i_aprende_la_solucion_conocida():
    X, y, P, r = _synthetic()
    m = CB.ConditionalEnsembleSelection(lam=1.0, n_steps=200, min_region=10).fit(X, y)
    assert m.weights_region_[1][0] > 0.9 and m.weights_region_[2][1] > 0.9
    assert m.score(X, y) > 0.9
    glob = CB.ConditionalEnsembleSelection(gate=False, n_steps=200).fit(X, y)
    ll = lambda est: -np.log(est.predict_proba(X)[np.arange(len(y)), y]).mean()
    assert ll(m) < ll(glob) - 0.2  # las regiones aportan donde la solución es local


def test_ii_reproducible():
    X, y, _, _ = _synthetic(seed=3)
    a = CB.ConditionalEnsembleSelection(lam=50).fit(X, y).predict_proba(X)
    b = CB.ConditionalEnsembleSelection(lam=50).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)


def test_iii_entradas_invalidas():
    X, y, _, _ = _synthetic(n=200)
    with pytest.raises(ValueError):
        CB.ConditionalEnsembleSelection().fit(X[:, :-2], y)  # columnas que no forman bloques de K
    bad = X.copy()
    bad[0, 0] = 1.7
    with pytest.raises(ValueError):
        CB.ConditionalEnsembleSelection().fit(bad, y)
    with pytest.raises(ValueError):
        CB.ConditionalEnsembleSelection(alpha=2).fit(X, y)
    with pytest.raises(ValueError):
        CB.ConditionalEnsembleSelection().fit(X, y + 10)
    with pytest.raises(Exception):
        CB.ConditionalEnsembleSelection().predict(X)  # sin ajustar
    m = CB.ConditionalEnsembleSelection().fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(np.hstack([X[:, :K], X[:, -1:]]))  # otro número de modelos base


def test_iv_clone():
    m = CB.ConditionalEnsembleSelection(lam=7.0, alpha=0.3)
    c = clone(m)
    assert c.get_params() == m.get_params() and not hasattr(c, 'weights_global_')


def test_pesos_en_simplex_y_probabilidades_validas():
    X, y, _, _ = _synthetic(seed=5)
    m = CB.ConditionalEnsembleSelection(lam=20, min_region=10).fit(X, y)
    for w in [m.weights_global_, *m.weights_region_.values()]:
        assert np.all(w >= -1e-12) and abs(w.sum() - 1) < 1e-9
    p = m.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1) and np.all(p >= 0)


def test_perdida_no_creciente_y_brecha_certifica():
    rng = np.random.default_rng(1)
    n, M = 3000, 6
    T = np.clip(rng.beta(2, 3, (n, M)), CB.EPS, 1)
    a = np.full(n, 1 / n)
    w, h = CB.frank_wolfe(T, a, n_steps=300, tol=0)
    assert np.all(np.diff(h['loss']) <= 1e-12)
    # cota F(w_t) − F* ≤ G_t, con F* aproximado por muchas iteraciones más
    _, h2 = CB.frank_wolfe(T, a, n_steps=3000, tol=0)
    fstar = h2['loss'].min()
    assert np.all(h['loss'] - fstar <= h['gap'] + 1e-9)


def test_contraccion_extrema_da_pesos_globales():
    X, y, _, _ = _synthetic(seed=2)
    m = CB.ConditionalEnsembleSelection(lam=1e9, n_steps=200, min_region=10).fit(X, y)
    for w in m.weights_region_.values():
        assert np.allclose(w, m.weights_global_, atol=1e-3)


def test_regiones_nuevas_usan_pesos_globales():
    X, y, _, _ = _synthetic(seed=4)
    m = CB.ConditionalEnsembleSelection(min_region=10).fit(X, y)
    Xn = X.copy()
    Xn[:, -1] = 99
    P = Xn[:, :-1].reshape(len(Xn), -1, K)
    assert np.allclose(m.predict_proba(Xn), np.einsum('m,nmk->nk', m.weights_global_, P)
                       / np.einsum('m,nmk->nk', m.weights_global_, P).sum(1, keepdims=True))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
