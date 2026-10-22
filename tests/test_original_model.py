"""Pruebas unitarias del modelo original (PTBagging) y de sus iteraciones de diseño.

Ejecutar desde la raíz del repositorio:  pytest tests/test_original_model.py
Las cuatro pruebas que exige la guía (sección 5.4) para el modelo original están marcadas con (i)-(iv).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from osha_e3.original_model import (FrankHallXGB, HierarchicalMeta, MultiRandBal, PTBagging,  # noqa: E402
                                    focal_softmax_grad_hess, focal_softmax_loss, hafl_gamma)


def toy(n=3000, seed=0):
    """4 clases con etiquetas 1..4; la clase 1 es rara (~1 %). One-hot disperso de 8 variables con señal conocida."""
    rng = np.random.default_rng(seed)
    y = rng.choice([1, 2, 3, 4], size=n, p=[0.01, 0.33, 0.33, 0.33])
    cats = rng.integers(0, 10, size=(n, 8))
    cats[:, 0] = np.where(y == 1, 0, cats[:, 0])          # la clase rara siempre tiene la categoría 0 en la variable 0
    cats[:, 1] = (cats[:, 1] + y) % 10                     # señal para las demás clases
    rows = np.repeat(np.arange(n), 8)
    cols = (cats + np.arange(8) * 10).ravel()
    return sp.csr_matrix((np.ones(n * 8), (rows, cols)), shape=(n, 80)), y


# ---------------------------------------------------------------- PTBagging: pruebas de la sección 5.4 de la guía

def test_pt_i_aprende_solucion_conocida():
    """(i) Datos sintéticos con solución conocida: la clase depende de forma determinista de una variable."""
    rng = np.random.default_rng(1)
    x = rng.integers(0, 4, size=2000)
    X = np.eye(4)[x]                       # one-hot de una variable con 4 categorías
    y = x + 1                              # la clase es exactamente la categoría
    m = PTBagging(n_estimators=10, random_state=0).fit(X, y)
    assert np.mean(m.predict(X) == y) == 1.0
    # Con desbalance, el umbral = prior recupera la clase rara que argmax(P) pierde cuando sus hojas son mixtas.
    Xr, yr = toy(4000, seed=3)
    pt = PTBagging(n_estimators=30, min_samples_leaf=20, random_state=0).fit(Xr, yr)
    raw = PTBagging(n_estimators=30, min_samples_leaf=20, threshold=None, random_state=0).fit(Xr, yr)
    assert np.mean(pt.predict(Xr)[yr == 1] == 1) >= np.mean(raw.predict(Xr)[yr == 1] == 1)


def test_pt_ii_reproducible_con_semilla():
    """(ii) Misma semilla, mismas probabilidades; semillas distintas, bolsas distintas."""
    X, y = toy()
    p1 = PTBagging(n_estimators=15, random_state=42).fit(X, y).predict_proba(X)
    p2 = PTBagging(n_estimators=15, random_state=42).fit(X, y).predict_proba(X)
    p3 = PTBagging(n_estimators=15, random_state=7).fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, p3)
    # Entrenar en hilos no cambia el resultado (las bolsas y semillas se fijan antes).
    p4 = PTBagging(n_estimators=15, random_state=42, n_jobs=2).fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p4)


def test_pt_iii_entradas_invalidas():
    """(iii) Entradas inválidas: predecir sin ajustar, valores faltantes y número de columnas distinto."""
    X, y = toy(500)
    with pytest.raises(NotFittedError):
        PTBagging().predict(X)
    Xd = X.toarray()
    Xd[0, 0] = np.nan
    with pytest.raises(ValueError):
        PTBagging(n_estimators=3, random_state=0).fit(Xd, y)
    m = PTBagging(n_estimators=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :50])


def test_pt_iv_compatible_con_clone():
    """(iv) ``sklearn.base.clone`` copia los hiperparámetros sin el estado ajustado."""
    m = PTBagging(n_estimators=12, min_samples_leaf=5, threshold='prior', random_state=3)
    c = clone(m)
    assert c.get_params() == m.get_params()
    X, y = toy(500)
    m.fit(X, y)
    c2 = clone(m)
    assert not hasattr(c2, 'estimators_')
    assert np.array_equal(c2.fit(X, y).predict_proba(X), m.predict_proba(X))


def test_pt_regla_umbral_prior():
    """predict = argmax(P / prior) y la ablación sin umbral = argmax(P)."""
    X, y = toy()
    m = PTBagging(n_estimators=20, random_state=42).fit(X, y)
    raw, adj = m.predict_proba_raw(X), m.predict_proba(X)
    assert (m.predict(X) == m.classes_[(raw / m.prior_).argmax(axis=1)]).all()
    assert np.allclose(adj.sum(axis=1), 1)
    assert np.allclose(m.prior_, [np.mean(y == c) for c in [1, 2, 3, 4]])
    no_thr = PTBagging(n_estimators=20, threshold=None, random_state=42).fit(X, y)
    assert (no_thr.predict(X) == m.classes_[raw.argmax(axis=1)]).all()


# ---------------------------------------------------------------- Iteraciones de diseño

def test_multirandbal():
    X, y = toy()
    m = MultiRandBal(n_estimators=30, random_state=42).fit(X, y)
    t = m.targets_
    assert t.min() >= 2, 'piso de 2 casos'
    shares = t / t.sum(axis=1, keepdims=True)
    assert shares.std(axis=0).min() > 0.05, 'las proporciones por árbol deben variar'
    assert abs(t.sum(axis=1).mean() - len(y)) / len(y) < 0.01
    assert m.n_synthetic_ > 0 and list(m.classes_) == [1, 2, 3, 4]
    p = m.predict_proba(X[:50])
    assert np.allclose(p.sum(axis=1), 1) and (m.predict(X[:50]) == m.classes_[p.argmax(1)]).all()
    assert np.allclose(p, MultiRandBal(n_estimators=30, random_state=42).fit(X, y).predict_proba(X[:50]))
    assert MultiRandBal(n_estimators=10, smote=False, random_state=1).fit(X, y).n_synthetic_ == 0


def test_hierarchical():
    X, y = toy()
    m = HierarchicalMeta(n_estimators=20, random_state=42).fit(X, y, groups=np.arange(len(y)) // 3)
    p = m.predict_proba(X[:100])
    assert np.allclose(p.sum(axis=1), 1) and m.rare_class_ == 1
    assert m.oof_votes_.shape == (len(y), 7)


def test_frank_hall():
    X, y = toy(4000)
    m = FrankHallXGB(n_estimators=30, random_state=42).fit(X, y)
    p = m.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1) and (p >= 0).all() and list(m.classes_) == [1, 2, 3, 4]
    assert min(np.mean(m.predict(X)[y == c] == c) for c in (2, 3, 4)) > 0.25, 'columnas desalineadas'


@pytest.mark.parametrize('est', [MultiRandBal(n_estimators=10, random_state=42),
                                 PTBagging(n_estimators=10, random_state=42),
                                 HierarchicalMeta(n_estimators=10, random_state=42),
                                 FrankHallXGB(n_estimators=10, random_state=42)])
def test_sin_fuga(est):
    """La predicción sobre una mitad no depende de las etiquetas de esa mitad."""
    X, y = toy()
    a, b = np.arange(1500), np.arange(1500, 3000)
    y_alt = y.copy()
    y_alt[b] = np.random.default_rng(1).permutation(y_alt[b])
    assert np.allclose(clone(est).fit(X[a], y[a]).predict_proba(X[b]),
                       clone(est).fit(X[a], y_alt[a]).predict_proba(X[b]))


@pytest.mark.parametrize('gamma', [0.0, 0.03, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
@pytest.mark.parametrize('alpha', [0.25, 1.0])
def test_hafl_derivadas_diferencias_finitas(gamma, alpha):
    """Gradiente y hessiano diagonal de la pérdida focal frente a diferencias finitas (HAFL-XGBoost)."""
    h = 1e-4
    rng = np.random.default_rng(42)
    margins = rng.normal(scale=2.0, size=(400, 4))
    y = rng.integers(0, 4, size=400)
    grad, hess = focal_softmax_grad_hess(margins, y, np.full(400, gamma), alpha, hess_floor=-np.inf)
    f0 = focal_softmax_loss(margins, y, gamma, alpha)
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        fp, fm = focal_softmax_loss(margins + e, y, gamma, alpha), focal_softmax_loss(margins - e, y, gamma, alpha)
        assert np.abs(grad[:, j] - (fp - fm) / (2 * h)).max() < 1e-5
        assert np.abs(hess[:, j] - (fp - 2 * f0 + fm) / h ** 2).max() < 1e-3


def test_hafl_casos_limite():
    rng = np.random.default_rng(0)
    margins, y = rng.normal(size=(200, 4)), rng.integers(0, 4, size=200)
    p = np.exp(margins) / np.exp(margins).sum(axis=1, keepdims=True)
    grad, hess = focal_softmax_grad_hess(margins, y, 0.0, 1.0, hess_floor=-np.inf)
    assert np.allclose(grad, p - np.eye(4)[y]) and np.allclose(hess, p * (1 - p)), 'γ = 0 es la entropía cruzada'
    gx, hx = focal_softmax_grad_hess(np.array([[30.0, -30.0, 0, 0], [-30.0, 30.0, 0, 0]]), np.array([0, 0]),
                                     np.array([0.05, 0.05]))
    assert np.isfinite(gx).all() and np.isfinite(hx).all() and (hx > 0).all()
    gam = hafl_gamma(np.zeros((8, 4)), np.array([0, 1, 2, 3, 1, 2, 3, 3]), np.array([1., 2., 2., 3.]), 1, 3.0)
    assert np.allclose(gam, 3.0 * np.sqrt(np.array([1, 2, 2, 3]) / 3) * (np.log(2) + 0.5))
