"""Combinador del modelo original: selección de ensambles condicionada por región con contracción (SECR).

Problema
--------
Hay ``M`` modelos base (las vistas del ensamble multivista, ``ensemble.py``) que dan probabilidades
``P_m(x) ∈ Δ_K`` para las ``K = 4`` clases del desenlace. La selección de ensambles de Caruana et al. (2004)
elige **un único** vector de pesos convexos ``w ∈ Δ_M`` para todo el espacio de entrada. Pero la calidad
relativa de cada vista cambia según el tipo de caso: el código de la lesión informa mucho en unos sectores y
poco en otros, la vista del empleador solo ayuda donde hay empresas grandes, etc.

SECR aprende un vector de pesos **por región** ``w_g ∈ Δ_M`` (las regiones las define una variable de
compuerta de baja cardinalidad, p. ej. el sector NAICS de 2 dígitos) y lo **contrae** hacia los pesos
globales para que las regiones con pocos casos no sobreajusten. Para la región ``g`` se minimiza

    F_g(w) = 1/(n_g + λ) [ Σ_{i∈g} ℓ_i(w) + (λ/n) Σ_{i=1}^{n} ℓ_i(w) ],
    ℓ_i(w) = −log Σ_m w_m P_m(x_i)[y_i]                                            (ec. 1)

sobre el símplex ``Δ_M``. ``λ = 0`` da pesos puramente locales; ``λ → ∞`` da los pesos globales. ``F_g`` es la
log-loss de la mezcla con pesos por observación ``a_i ≥ 0`` (``Σ a_i = 1``), así que todas las regiones se
resuelven con el mismo algoritmo.

Algoritmo (Frank-Wolfe con búsqueda lineal exacta)
--------------------------------------------------
Entradas: matriz ``T[i, m] = P_m(x_i)[y_i]`` (probabilidad que el modelo m da a la clase verdadera), pesos
``a``, número máximo de pasos ``n_steps``, tolerancia ``tol``.

1. ``w⁰ = e_{m*}`` con ``m* = argmin_m F(e_m)`` (el mejor modelo individual).
2. Para ``t = 0, 1, …``: ``q = T w``; gradiente ``∇F = −Tᵀ(a / q)``; vértice ``s = argmin_m ∇F_m``;
   brecha de Frank-Wolfe ``G_t = ∇Fᵀ(w − e_s)``. Si ``G_t ≤ tol`` o ``t = n_steps``, parar.
3. ``γ_t = argmin_{γ∈[0,1]} F(w + γ(e_s − w))`` (función convexa de una variable: bisección sobre su derivada).
4. ``w ← w + γ_t (e_s − w)``.

Propiedades (demostradas en el capítulo 19): (i) ``F`` es convexa en ``Δ_M``; (ii) los iterados quedan en
``Δ_M`` y ``F(w^{t+1}) ≤ F(w^t)`` (la pérdida no aumenta); (iii) ``F(w^t) − F* ≤ G_t`` (la brecha es un
certificado de optimalidad) y ``F(w^t) − F* ≤ 2 C_F / (t + 2)`` con ``C_F`` finita porque ``T ≥ ε > 0``;
(iv) costo ``O(n_steps · n · M)`` por región y ``O(n·M·K)`` para armar ``T``; inferencia ``O(M·K)`` por caso.
La selección de Caruana con reemplazo es el caso particular sin regiones, con paso fijo ``γ_t = 1/(t+init+1)``
y oráculo que evalúa la pérdida real en lugar de la linealización.

Decisión: ``ŷ = argmax_k q_k / π_k^α`` con ``π`` las proporciones de clase del entrenamiento. ``α = 0`` es el
``argmax``; ``α = 1`` maximiza la balanced accuracy esperada si ``q`` está calibrada (Collell et al., 2018).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

EPS = 1e-6


def frank_wolfe(T, a, n_steps=50, tol=1e-6, w0=None):
    """Minimiza ``F(w) = −Σ_i a_i log(T_i · w)`` sobre el símplex con Frank-Wolfe y búsqueda lineal exacta.

    Parameters
    ----------
    T : ndarray (n, M)
        Probabilidad de la clase verdadera según cada modelo (``≥ EPS``).
    a : ndarray (n,)
        Pesos no negativos de las observaciones (suman 1).
    n_steps : int
        Máximo de iteraciones (criterio de parada 1).
    tol : float
        Tolerancia de la brecha de Frank-Wolfe (criterio de parada 2).
    w0 : ndarray (M,) or None
        Punto inicial en el símplex; por defecto, el mejor vértice.

    Returns
    -------
    w : ndarray (M,)
    history : dict
        ``loss`` y ``gap`` por iteración (curva de convergencia).
    """
    n, M = T.shape
    if w0 is None:
        w = np.zeros(M)
        w[np.argmin(-(a @ np.log(T)))] = 1.0
    else:
        w = np.asarray(w0, dtype=float).copy()
    loss, gaps = [], []
    for _ in range(int(n_steps) + 1):
        q = T @ w
        f = -float(a @ np.log(q))
        grad = -(T.T @ (a / q))
        s = int(np.argmin(grad))
        gap = float(grad @ w - grad[s])
        loss.append(f)
        gaps.append(gap)
        if gap <= tol or len(loss) > n_steps:
            break
        delta = T[:, s] - q  # dirección en el espacio de q: q(γ) = q + γ δ

        def dphi(g):
            return -float(a @ (delta / (q + g * delta)))

        if dphi(1.0) <= 0:  # la pérdida sigue bajando hasta el vértice
            gamma = 1.0
        else:  # φ'(0) = −gap < 0 < φ'(1): raíz única por convexidad
            lo, hi = 0.0, 1.0
            for _ in range(25):  # |γ − γ*| ≤ 2^-25
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if dphi(mid) < 0 else (lo, mid)
            gamma = 0.5 * (lo + hi)
        w = (1 - gamma) * w
        w[s] += gamma
    return w, {'loss': np.array(loss), 'gap': np.array(gaps)}


def true_class_matrix(P, y):
    """``T[i, m] = P[i, m, y_i]`` recortada a ``EPS``. ``P`` tiene forma (n, M, K)."""
    return np.clip(P[np.arange(len(y)), :, y], EPS, 1.0)


class ConditionalEnsembleSelection(BaseEstimator, ClassifierMixin):
    """Selección de ensambles condicionada por región con contracción (SECR).

    La entrada ``X`` es la matriz de meta-variables ``[P_1 | P_2 | … | P_M | región]``: las probabilidades de
    los ``M`` modelos base (``M·K`` columnas, en bloques de ``K``) y, al final, el código entero de la región
    (0, 1, …). Con ``gate=False`` la última columna se ignora y el modelo aprende pesos globales.

    Parameters
    ----------
    n_classes : int
        Número de clases ``K``.
    lam : float
        Fuerza de la contracción hacia los pesos globales (pseudo-observaciones, ec. 1).
    n_steps : int
        Máximo de iteraciones de Frank-Wolfe por región.
    min_region : int
        Regiones con menos casos de entrenamiento usan los pesos globales.
    alpha : float in [0, 1]
        Exponente de la corrección por prior en la decisión.
    gate : bool
        Si se usan regiones (``True``) o solo pesos globales (``False``).
    tol : float
        Tolerancia de la brecha de Frank-Wolfe.
    """

    def __init__(self, n_classes=4, lam=100.0, n_steps=50, min_region=200, alpha=1.0, gate=True, tol=1e-7):
        self.n_classes, self.lam, self.n_steps, self.min_region = n_classes, lam, n_steps, min_region
        self.alpha, self.gate, self.tol = alpha, gate, tol

    def _split(self, X):
        K = self.n_classes
        if (X.shape[1] - 1) % K or X.shape[1] < K + 1:
            raise ValueError(f'X debe tener M·{K} columnas de probabilidades más una de región; tiene {X.shape[1]}.')
        P = X[:, :-1].reshape(len(X), -1, K)
        if np.any(P < 0) or np.any(P > 1 + 1e-6):
            raise ValueError('Las probabilidades de los modelos base deben estar en [0, 1].')
        r = X[:, -1]
        if np.any(r < 0) or np.any(r != np.round(r)):
            raise ValueError('La región debe ser un entero no negativo.')
        return P, r.astype(int)

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        if self.lam < 0 or self.n_steps < 0 or not 0 <= self.alpha <= 1:
            raise ValueError('Se requiere lam >= 0, n_steps >= 0 y 0 <= alpha <= 1.')
        self.classes_ = np.arange(self.n_classes)
        y = y.astype(int)
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(f'y debe estar codificada 0..{self.n_classes - 1}.')
        P, r = self._split(X)
        self.n_models_ = P.shape[1]
        T = true_class_matrix(P, y)
        n = len(y)
        self.weights_global_, self.history_ = frank_wolfe(T, np.full(n, 1.0 / n), self.n_steps, self.tol)
        self.weights_region_ = {}
        if self.gate:
            for g in np.unique(r):
                m = r == g
                ng = int(m.sum())
                if ng < self.min_region:
                    continue
                a = np.full(n, self.lam / ((ng + self.lam) * n))
                a[m] += 1.0 / (ng + self.lam)
                self.weights_region_[int(g)], _ = frank_wolfe(T, a, self.n_steps, self.tol, self.weights_global_)
        self.prior_ = np.maximum(np.bincount(y, minlength=self.n_classes) / n, 1e-12)
        return self

    def region_weights(self, r):
        """Matriz (n, M) de pesos que corresponde a cada código de región."""
        W = np.tile(self.weights_global_, (len(r), 1))
        for g, w in self.weights_region_.items():
            W[r == g] = w
        return W

    def predict_proba(self, X):
        check_is_fitted(self, 'weights_global_')
        X = check_array(X, dtype=float)
        P, r = self._split(X)
        if P.shape[1] != self.n_models_:
            raise ValueError(f'Se esperaban {self.n_models_} modelos base; llegaron {P.shape[1]}.')
        q = np.einsum('nm,nmk->nk', self.region_weights(r), P)
        return q / q.sum(axis=1, keepdims=True)

    def decide(self, q):
        return (q / self.prior_ ** self.alpha).argmax(axis=1)

    def predict(self, X):
        return self.classes_[self.decide(self.predict_proba(X))]

    def score(self, X, y, sample_weight=None):
        """Balanced accuracy (la métrica principal del proyecto)."""
        return balanced_accuracy_score(y, self.predict(X), sample_weight=sample_weight)


# ------------------------------------------------------------------------------------------------
# Variables de compuerta (regiones) y reglas estándar de combinación (para la ablación).
# ------------------------------------------------------------------------------------------------
GATES = {
    'none': None,
    'type_of_incident': lambda X: X['type_of_incident'].astype(str),
    'establishment_type': lambda X: X['establishment_type'].astype(str),
    'naics2': lambda X: X['naics3'].astype(str).str[:2],
    'soc2': lambda X: X['soc6'].astype(str).str[:2],
    'nature1': lambda X: X['nature'].astype(str).str[:1],
    'event1': lambda X: X['event'].astype(str).str[:1],
    'part1': lambda X: X['part'].astype(str).str[:1],
}


def region_codes(X, gate, vocab=None):
    """Códigos enteros de región para ``gate``; ``vocab`` (del entrenamiento) manda los valores nuevos a 0."""
    if GATES.get(gate) is None:
        return np.zeros(len(X), dtype=int), {}
    v = GATES[gate](X).to_numpy()
    if vocab is None:
        vocab = {k: i + 1 for i, k in enumerate(sorted(set(v)))}
    return np.array([vocab.get(k, 0) for k in v], dtype=int), vocab


def meta_matrix(P, r):
    """Arma ``[P_1 | … | P_M | región]`` a partir de ``P`` (n, M, K) y ``r`` (n,)."""
    return np.hstack([P.reshape(len(P), -1), r[:, None].astype(float)])


def simple_average(P):
    return P.mean(axis=1)


def global_error_weights(P_oof, y):
    """Pesos por error global: ``w_m ∝ 1 / logloss_m`` (regla estándar de la guía, Cuadro 1)."""
    ll = -np.log(true_class_matrix(P_oof, y)).mean(axis=0)
    return (1 / ll) / (1 / ll).sum()


def caruana(P_oof, y, steps=40, init=3):
    """Selección de Caruana con reemplazo (modelo G del capítulo 18) sobre ``P`` (n, M, K)."""
    T = true_class_matrix(P_oof, y)
    M = T.shape[1]
    order = np.argsort(-(np.log(T).mean(axis=0)))  # de menor a mayor log-loss
    counts = np.zeros(M)
    counts[order[:init]] = 1
    total = T[:, order[:init]].sum(axis=1)
    for _ in range(steps):
        losses = [-np.log((total + T[:, m]) / (counts.sum() + 1)).mean() for m in range(M)]
        m = int(np.argmin(losses))
        counts[m] += 1
        total = total + T[:, m]
    return counts / counts.sum()
