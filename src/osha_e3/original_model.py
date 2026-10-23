"""Modelo original del entregable 3 y sus iteraciones de diseño (capítulo 17 del libro).

- **Modelo original: ``PTBagging``** (bagging con *threshold-moving*, umbral = prior de clase).
- Iteraciones de diseño (Figura 2 de la guía), con sus resultados en ``results/modelo_original/``:
  ``MultiRandBal`` (Ronda 1), ``HierarchicalMeta`` (Ronda 1), ``HAFLXGBoost`` (Ronda 2) y
  ``FrankHallXGB`` (Ronda 4).

Módulo independiente: solo usa numpy, scipy, scikit-learn, imbalanced-learn y, en las dos últimas
clases, xgboost; no depende de rutas locales ni de otros módulos del repositorio. Es una copia de
``disenio_temporal/osha3/original_models.py``, con el que se calcularon todas las cifras. Cada
estimador sigue la API de scikit-learn (``fit``, ``predict_proba``, ``predict``, ``classes_``) y
recibe la matriz ya codificada (one-hot disperso) que produce el pipeline.

En los tres, ``predict`` es el ``argmax`` de ``predict_proba``: así coincide con las métricas
del bucle externo, que se calculan a partir de las probabilidades.

Fuentes
-------
- MultiRandBal: Rodríguez, Díez-Pastor, Arnaiz-González y Kuncheva, "Random Balance ensembles
  for multiclass imbalance learning", Knowledge-Based Systems 193 (2020) 105434.
- PT-bagging: Collell, Prelec y Patil, "A simple plug-in bagging ensemble based on
  threshold-moving for classifying binary and multiclass imbalanced data",
  Neurocomputing 275 (2018) 330-340.
- Jerárquica: metaclasificación jerárquica con subconjuntos balanceados y Naive Bayes como
  meta-clasificador (BMC Medical Informatics and Decision Making, 2018,
  doi 10.1186/s12911-018-0675-x).
"""
import numpy as np
import scipy.sparse as sp
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


def _rows(X, idx):
    """Filas ``idx`` de una matriz densa o dispersa (CSR)."""
    return X[idx]


def _vstack(blocks):
    return sp.vstack(blocks, format='csr') if sp.issparse(blocks[0]) else np.vstack(blocks)


class _TreeEnsemble(BaseEstimator, ClassifierMixin):
    """Base común: entrena ``n_estimators`` árboles, cada uno sobre las filas que da ``_sample``,
    y promedia sus probabilidades."""

    def _tree(self, seed):
        return DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf, ccp_alpha=self.ccp_alpha,
                                      max_features=self.max_features, random_state=seed)

    def _fit_trees(self, X, y, samples, seeds):
        # Las muestras y semillas ya están fijadas, así que entrenar en hilos (n_jobs) no cambia el resultado.
        from joblib import Parallel, delayed

        def fit_one(idx, seed):
            return self._tree(int(seed)).fit(_rows(X, idx), y[idx])
        self.estimators_ = Parallel(n_jobs=getattr(self, 'n_jobs', 1), prefer='threads')(
            delayed(fit_one)(idx, seed) for idx, seed in zip(samples, seeds))
        return self

    def _mean_proba(self, X):
        check_is_fitted(self, 'estimators_')
        if sp.issparse(X):
            X = X.tocsr()
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.estimators_:
            cols = np.searchsorted(self.classes_, tree.classes_)
            proba[:, cols] += tree.predict_proba(X)
        return proba / len(self.estimators_)

    def predict(self, X):
        proba = self.predict_proba(X)          # primero: lanza NotFittedError si el modelo no está ajustado
        return self.classes_[proba.argmax(axis=1)]


class MultiRandBal(_TreeEnsemble):
    """Ensamble Random Balance multiclase (Algoritmos 1 y 2 del paper de KBS 2020).

    Para cada árbol: se sortea ``w_i ~ U(0, 1)`` por clase y el tamaño objetivo
    ``n'_i = max(round(n · w_i / Σw), 2)``. Si ``n'_i ≤ n_i`` se submuestrea sin reemplazo;
    si no, se conserva toda la clase y se añaden ``n'_i − n_i`` casos sintéticos.
    La predicción es el promedio de las probabilidades de los árboles.

    Desviación declarada: los sintéticos no se generan con SMOTE en cada árbol, sino de un
    **banco** que se genera una sola vez por llamada a ``fit`` (SMOTE con el ``k`` seguro) y
    del que cada árbol extrae, sin reemplazo, los que le tocan.

    Parameters
    ----------
    n_estimators : int, default=100
        Número de árboles ``L`` (el paper usa 100).
    smote : bool, default=True
        False = variante sin SMOTE: los casos que faltan se repiten con reemplazo.
    k_neighbors : int, default=5
        Vecinos de SMOTE; se usa ``min(k_neighbors, n_i − 1)`` para la clase más chica.
    criterion, max_depth, min_samples_leaf, ccp_alpha, max_features :
        Del árbol base. Por defecto, árbol sin podar con entropía (lo más cercano a J48 sin podar).
    random_state : int or None
        Semilla de los pesos ``w_i``, del banco SMOTE y de los árboles.
    """

    def __init__(self, n_estimators=100, smote=True, k_neighbors=5, criterion='entropy', max_depth=None,
                 min_samples_leaf=1, ccp_alpha=0.0, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.smote = smote
        self.k_neighbors = k_neighbors
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.max_features = max_features
        self.random_state = random_state

    def _targets(self, counts, rng):
        """Tamaños objetivo por árbol y clase (matriz L × K), con piso de 2 casos."""
        n = counts.sum()
        w = rng.uniform(size=(self.n_estimators, len(counts)))
        return np.maximum(np.round(n * w / w.sum(axis=1, keepdims=True)).astype(int), 2)

    def _bank(self, X, y, need, rng):
        """Banco de sintéticos por clase, generado una sola vez: {clase: (X_sint, y_sint)}."""
        need = {c: int(m) for c, m in need.items() if m > 0}
        if not need:
            return {}
        counts = {c: int(np.sum(y == c)) for c in self.classes_}
        k = min(self.k_neighbors, min(counts[c] for c in need) - 1)
        if not self.smote or k < 1:
            return None   # se repiten casos reales con reemplazo
        strategy = {c: counts[c] + m for c, m in need.items()}
        sampler = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=int(rng.integers(2**31 - 1)))
        X_res, y_res = sampler.fit_resample(X, y)
        X_new, y_new = X_res[len(y):], np.asarray(y_res)[len(y):]
        return {c: X_new[np.flatnonzero(y_new == c)] for c in need}

    def fit(self, X, y):
        y = np.asarray(y)
        X = X.tocsr() if sp.issparse(X) else np.asarray(X)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        by_class = [np.flatnonzero(y == c) for c in self.classes_]
        counts = np.array([len(i) for i in by_class])
        targets = self._targets(counts, rng)
        self.targets_ = targets                     # para verificar las proporciones por árbol
        need = dict(zip(self.classes_, (targets - counts).clip(min=0).max(axis=0)))
        bank = self._bank(X, y, need, rng)
        # Matriz de trabajo: reales + banco; cada árbol recibe índices de esta matriz.
        blocks, labels, offset, bank_idx = [X], [y], len(y), {}
        if bank:
            for c, Xc in bank.items():
                blocks.append(Xc)
                labels.append(np.full(Xc.shape[0], c))
                bank_idx[c] = np.arange(offset, offset + Xc.shape[0])
                offset += Xc.shape[0]
        X_all, y_all = _vstack(blocks) if len(blocks) > 1 else X, np.concatenate(labels)
        self.n_synthetic_ = offset - len(y)
        samples = []
        for t in range(self.n_estimators):
            parts = []
            for j, c in enumerate(self.classes_):
                real, target = by_class[j], targets[t, j]
                if target <= len(real):
                    parts.append(rng.choice(real, size=target, replace=False))
                else:
                    extra = target - len(real)
                    if bank:
                        parts += [real, rng.choice(bank_idx[c], size=extra, replace=False)]
                    else:
                        parts += [real, rng.choice(real, size=extra, replace=True)]
            samples.append(np.sort(np.concatenate(parts)))
        seeds = rng.integers(2**31 - 1, size=self.n_estimators)
        return self._fit_trees(X_all, y_all, samples, seeds)

    def predict_proba(self, X):
        return self._mean_proba(X)


class PTBagging(_TreeEnsemble):
    """Bagging con *threshold-moving* (PT-bagging, Neurocomputing 2018).

    Cada árbol se entrena con una muestra bootstrap simple (conserva la distribución natural de
    las clases). ``P(y=k|x)`` es el promedio de los árboles y la regla de decisión del paper es
    ``argmax_k P(y=k|x) / λ_k`` con ``λ_k`` = proporción de la clase k en el entrenamiento
    (Teorema 1: maximiza la macro-accuracy, es decir, la balanced accuracy).

    Para que ``predict`` = ``argmax(predict_proba)`` como en el resto del pipeline,
    ``predict_proba`` devuelve las probabilidades reponderadas y normalizadas
    ``(P_k / λ_k) / Σ_j (P_j / λ_j)`` (la posterior bajo un prior uniforme).
    Las probabilidades sin reponderar están en ``predict_proba_raw``.

    Parameters
    ----------
    n_estimators : int, default=100
        Número de árboles (el paper reporta resultados con 100).
    threshold : {'prior', None}, default='prior'
        None = ablación sin umbral (``argmax P``).
    criterion, max_depth, min_samples_leaf, ccp_alpha, max_features :
        Del árbol base. Por defecto, árbol sin podar con entropía (J48 sin podar ni Laplace).
    random_state : int or None
    """

    def __init__(self, n_estimators=100, threshold='prior', criterion='entropy', max_depth=None,
                 min_samples_leaf=1, ccp_alpha=0.0, max_features=None, random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y = np.asarray(y)
        X = X.tocsr() if sp.issparse(X) else np.asarray(X)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        self.prior_ = np.array([np.mean(y == c) for c in self.classes_])
        n = len(y)
        samples = [rng.integers(n, size=n) for _ in range(self.n_estimators)]
        seeds = rng.integers(2**31 - 1, size=self.n_estimators)
        return self._fit_trees(X, y, samples, seeds)

    def predict_proba_raw(self, X):
        return self._mean_proba(X)

    def predict_proba(self, X):
        proba = self._mean_proba(X)
        if self.threshold is None:
            return proba
        score = proba / self.prior_
        total = score.sum(axis=1, keepdims=True)
        return np.divide(score, total, out=np.full_like(score, 1 / len(self.classes_)), where=total > 0)


class HierarchicalMeta(BaseEstimator, ClassifierMixin):
    """Metaclasificación jerárquica (BMC Med. Inform. Decis. Mak. 2018), adaptada a OSHA.

    Nivel 1 (clase rara contra el resto): ``n_subsets`` subconjuntos balanceados; cada uno junta
    todos los casos de la clase rara con una muestra sin reemplazo, del mismo tamaño, del resto.
    Un Random Forest por subconjunto; cada caso queda representado por el vector de sus
    ``n_subsets`` etiquetas predichas y un Naive Bayes (Bernoulli) sobre ese vector da
    ``P(rara | x)``. El Naive Bayes se entrena con vectores **fuera de muestra**: validación
    cruzada interna de ``n_inner`` pliegues (por establecimiento si ``fit`` recibe ``groups``).
    Nivel 2: un Random Forest multiclase entrenado solo con las demás clases.

    Probabilidades: ``P(rara) = p``; ``P(k) = (1 − p) · P_2(k)`` para las demás.
    ``predict`` = ``argmax`` de esas probabilidades.

    Adaptación declarada: en el paper el nivel 1 separa la clase mayoritaria de las minoritarias;
    en OSHA las clases 2, 3 y 4 están casi balanceadas y la única rara es Fallecimiento, así que
    el nivel 1 es Fallecimiento contra {2, 3, 4} y el nivel 2 decide entre 2, 3 y 4.

    Parameters
    ----------
    n_subsets : int, default=7
    n_estimators : int, default=100
        Árboles de cada Random Forest (valor por defecto de scikit-learn).
    n_inner : int, default=3
        Pliegues internos para las predicciones fuera de muestra del Naive Bayes.
    rare_class : int or None
        Clase rara; None = la menos frecuente en ``y``.
    random_state : int or None
    """

    def __init__(self, n_subsets=7, n_estimators=100, n_inner=3, rare_class=None, random_state=None):
        self.n_subsets = n_subsets
        self.n_estimators = n_estimators
        self.n_inner = n_inner
        self.rare_class = rare_class
        self.random_state = random_state

    def _forest(self, rng):
        return RandomForestClassifier(n_estimators=self.n_estimators, random_state=int(rng.integers(2**31 - 1)))

    def _fit_level1(self, X, rare, rng):
        pos, neg = np.flatnonzero(rare), np.flatnonzero(~rare)
        forests = []
        for _ in range(self.n_subsets):
            idx = np.sort(np.concatenate([pos, rng.choice(neg, size=min(len(pos), len(neg)), replace=False)]))
            forests.append(self._forest(rng).fit(X[idx], rare[idx]))
        return forests

    @staticmethod
    def _votes(forests, X):
        return np.column_stack([f.predict(X) for f in forests]).astype(float)

    def fit(self, X, y, groups=None):
        y = np.asarray(y)
        X = X.tocsr() if sp.issparse(X) else np.asarray(X)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        counts = np.array([np.sum(y == c) for c in self.classes_])
        self.rare_class_ = self.classes_[counts.argmin()] if self.rare_class is None else self.rare_class
        rare = y == self.rare_class_
        seed = int(rng.integers(2**31 - 1))
        if groups is not None:
            cv = StratifiedGroupKFold(n_splits=self.n_inner, shuffle=True, random_state=seed)
            splits = cv.split(X, rare, groups)
        else:
            splits = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=seed).split(X, rare)
        votes = np.zeros((len(y), self.n_subsets))
        for a, b in splits:
            votes[b] = self._votes(self._fit_level1(X[a], rare[a], rng), X[b])
        self.meta_ = BernoulliNB().fit(votes, rare)
        self.oof_votes_ = votes
        self.level1_ = self._fit_level1(X, rare, rng)
        rest = np.flatnonzero(~rare)
        self.level2_ = self._forest(rng).fit(X[rest], y[rest])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'meta_')
        X = X.tocsr() if sp.issparse(X) else np.asarray(X)
        p = self.meta_.predict_proba(self._votes(self.level1_, X))[:, list(self.meta_.classes_).index(True)]
        proba = np.zeros((X.shape[0], len(self.classes_)))
        r = int(np.flatnonzero(self.classes_ == self.rare_class_)[0])
        proba[:, r] = p
        cols = np.searchsorted(self.classes_, self.level2_.classes_)
        proba[:, cols] = (1 - p)[:, None] * self.level2_.predict_proba(X)
        return proba

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------------------------
# Ronda 2: HAFL-XGBoost (Jiang, Wang, Meng, Saada y Cai, Neural Computing & Applications 35,
# 2023, doi 10.1007/s00521-023-08290-w), secciones 3.1 y 3.2. Sin la parte de ConvNeXt (red neuronal).
# ---------------------------------------------------------------------------------------------

def focal_softmax_grad_hess(margins, y_idx, gamma, alpha=0.25, hess_floor=1e-6):
    """Gradiente y hessiano diagonal de la pérdida focal con softmax respecto de los márgenes.

    Pérdida por muestra: ``L = -α (1 - p_t)^γ log p_t``, con ``p = softmax(z)`` y t la clase verdadera.
    Derivación propia (no se copian las ecuaciones 10 y 18 del paper). Con ``q = 1 - p_t``:
    - ``g(p_t) = α [γ p_t q^(γ-1) log p_t - q^γ]``; ``∂L/∂z_j = g(p_t) (δ_jt - p_j)``.
    - ``g'(p_t) = α [γ q^(γ-1) log p_t - γ(γ-1) p_t q^(γ-2) log p_t + 2γ q^(γ-1)]``.
    - ``∂²L/∂z_j² = g'(p_t) p_t (δ_jt - p_j)² - g(p_t) p_j (1 - p_j)``.
    Con γ = 0 se recupera la entropía cruzada: gradiente ``α (p - y)``, hessiano ``α p (1 - p)``.
    γ se trata como constante (no se deriva respecto de γ).

    Parameters
    ----------
    margins : ndarray (n, K)
    y_idx : ndarray (n,) de enteros 0..K-1
    gamma : ndarray (n,) o float
    alpha : float
    hess_floor : float
        Piso positivo del hessiano (XGBoost necesita hessianos > 0).

    Returns
    -------
    grad, hess : ndarray (n, K)
    """
    z = margins - margins.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    p = np.exp(logp)
    n = len(y_idx)
    rows = np.arange(n)
    pt, logpt = p[rows, y_idx], logp[rows, y_idx]
    others = p.copy()
    others[rows, y_idx] = 0.0
    q = np.clip(others.sum(axis=1), 1e-12, 1.0)          # 1 - p_t sin cancelación numérica
    g_ = np.broadcast_to(np.asarray(gamma, dtype=float), (n,))
    g = alpha * (g_ * pt * q ** (g_ - 1) * logpt - q ** g_)
    dg = alpha * (g_ * q ** (g_ - 1) * logpt - g_ * (g_ - 1) * pt * q ** (g_ - 2) * logpt + 2 * g_ * q ** (g_ - 1))
    delta = np.zeros_like(p)
    delta[rows, y_idx] = 1.0
    grad = g[:, None] * (delta - p)
    hess = dg[:, None] * pt[:, None] * (delta - p) ** 2 - g[:, None] * p * (1 - p)
    return grad, np.maximum(hess, hess_floor)


def focal_softmax_loss(margins, y_idx, gamma, alpha=0.25):
    """Pérdida focal por muestra (para la prueba con diferencias finitas)."""
    z = margins - margins.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    rows = np.arange(len(y_idx))
    pt, logpt = np.exp(logp[rows, y_idx]), logp[rows, y_idx]
    return -alpha * (1 - pt) ** np.asarray(gamma, dtype=float) * logpt


def hafl_gamma(margins, y_idx, counts, iteration, gamma_start):
    """γ adaptativo por clase (ecuaciones 20 a 23): γ_start · 1/√β · √(n_i/max n) · (log(prec_i/max prec + 1) + 0,5).

    ``prec_i`` = exactitud de entrenamiento de las muestras de la clase i en esta iteración
    (fracción con ``argmax`` correcto). Regla declarada para divisiones por cero: si ninguna clase
    acierta nada (max prec = 0) o todos los márgenes son iguales (primera iteración), el cociente
    ``prec_i / max prec`` se toma como 1 para todas las clases.
    """
    K = len(counts)
    pred = margins.argmax(axis=1)
    prec = np.array([np.mean(pred[y_idx == c] == c) if counts[c] else 0.0 for c in range(K)])
    flat = np.allclose(margins, margins[:, :1])
    ratio = np.ones(K) if (flat or prec.max() == 0) else prec / prec.max()
    g_epoch = 1.0 / np.sqrt(iteration)
    g_balance = np.sqrt(counts / counts.max())
    g_precision = np.log(ratio + 1) + 0.5
    return gamma_start * g_epoch * g_balance * g_precision


class HAFLXGBoost(BaseEstimator, ClassifierMixin):
    """XGBoost multiclase con pérdida focal adaptativa (HAFL) como objetivo propio.

    En cada ronda de boosting: (1) se calcula γ por clase con ``hafl_gamma`` a partir de los márgenes
    actuales; (2) cada muestra usa el γ de su clase verdadera; (3) gradiente y hessiano diagonal
    analíticos de ``focal_softmax_grad_hess``, multiplicados por el peso de la muestra.
    ``predict_proba`` = softmax de los márgenes; ``predict`` = argmax.

    Parameters
    ----------
    gamma_start : float
        γ inicial (el paper lo elige en [0, 6]); 0 = entropía cruzada con hessiano exacto.
    alpha : float, default=0.25
    n_estimators, learning_rate, max_depth, subsample, colsample_bytree, min_split_loss,
    min_child_weight, reg_alpha, reg_lambda :
        Hiperparámetros de XGBoost (``min_split_loss`` es el ``gamma`` de XGBoost).
    hess_floor : float, default=1e-6
    random_state, n_jobs : int
    """

    def __init__(self, gamma_start=3.0, alpha=0.25, n_estimators=100, learning_rate=0.3, max_depth=6,
                 subsample=1.0, colsample_bytree=1.0, min_split_loss=0.0, min_child_weight=1.0, reg_alpha=0.0,
                 reg_lambda=1.0, hess_floor=1e-6, random_state=None, n_jobs=1):
        self.gamma_start = gamma_start
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_split_loss = min_split_loss
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.hess_floor = hess_floor
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        import xgboost as xgb
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        K = len(self.classes_)
        counts = np.bincount(y_idx, minlength=K).astype(float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        dtrain = xgb.DMatrix(X, label=y_idx, weight=w)
        self.gamma_history_ = []

        def objective(predt, dmat):
            margins = predt.reshape(len(y_idx), K)
            gamma_c = hafl_gamma(margins, y_idx, counts, len(self.gamma_history_) + 1, self.gamma_start)
            self.gamma_history_.append(gamma_c.tolist())
            grad, hess = focal_softmax_grad_hess(margins, y_idx, gamma_c[y_idx], self.alpha, self.hess_floor)
            return grad * w[:, None], hess * w[:, None]

        params = {'num_class': K, 'disable_default_eval_metric': 1, 'tree_method': 'hist',
                  'eta': self.learning_rate, 'max_depth': self.max_depth, 'subsample': self.subsample,
                  'colsample_bytree': self.colsample_bytree, 'gamma': self.min_split_loss,
                  'min_child_weight': self.min_child_weight, 'alpha': self.reg_alpha, 'lambda': self.reg_lambda,
                  'seed': 0 if self.random_state is None else int(self.random_state), 'nthread': self.n_jobs,
                  'verbosity': 0}
        self.booster_ = xgb.train(params, dtrain, num_boost_round=self.n_estimators, obj=objective)
        return self

    def predict_proba(self, X):
        import xgboost as xgb
        check_is_fitted(self, 'booster_')
        m = self.booster_.predict(xgb.DMatrix(X), output_margin=True).reshape(X.shape[0], len(self.classes_))
        m = m - m.max(axis=1, keepdims=True)
        e = np.exp(m)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------------------------
# Ronda 4: clasificación ordinal de Frank y Hall (2001), "A simple approach to ordinal classification",
# ECML 2001, LNCS 2167, pp. 145-156, con XGBoost binario como clasificador base.
# ---------------------------------------------------------------------------------------------

class FrankHallXGB(BaseEstimator, ClassifierMixin):
    """Clasificación ordinal por descomposición de Frank y Hall con XGBoost binario.

    Con las clases ordenadas de menor a mayor gravedad (``order``), s = posición en el orden, se entrenan K − 1
    clasificadores binarios para P(s > k), k = 0..K−2, y se combinan:
    P(s=0) = 1 − P(s>0); P(s=k) = P(s>k−1) − P(s>k); P(s=K−1) = P(s>K−2).
    Los binarios se entrenan por separado y pueden no ser monótonos: las probabilidades negativas se recortan a 0 y
    se renormaliza (desviación declarada; ``negative_share_`` guarda la fracción de casos afectados en la última
    predicción). ``predict`` = argmax.

    Parameters
    ----------
    order : tuple
        Clases de menor a mayor gravedad. Por defecto OSHA: Otro (4) < Traslado (3) < Días de ausencia (2) < Muerte (1).
    n_estimators, learning_rate, max_depth, subsample, colsample_bytree, gamma, min_child_weight, reg_alpha,
    reg_lambda : hiperparámetros de cada XGBoost binario (los mismos para los K − 1).
    random_state, n_jobs : int
    """

    def __init__(self, order=(4, 3, 2, 1), n_estimators=100, learning_rate=0.3, max_depth=6, subsample=1.0,
                 colsample_bytree=1.0, gamma=0.0, min_child_weight=1.0, reg_alpha=0.0, reg_lambda=1.0,
                 random_state=None, n_jobs=1):
        self.order = order
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _binary(self):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth,
                             subsample=self.subsample, colsample_bytree=self.colsample_bytree, gamma=self.gamma,
                             min_child_weight=self.min_child_weight, reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
                             tree_method='hist', n_jobs=self.n_jobs, random_state=self.random_state,
                             eval_metric='logloss')

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        assert sorted(self.order) == list(self.classes_), 'order debe ser una permutación de las clases'
        position = np.array([list(self.order).index(c) for c in self.classes_])   # gravedad de cada clase
        rank = position[np.searchsorted(self.classes_, y)]
        self.estimators_ = [self._binary().fit(X, (rank > k).astype(int), sample_weight=sample_weight)
                            for k in range(len(self.order) - 1)]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'estimators_')
        G = np.column_stack([m.predict_proba(X)[:, 1] for m in self.estimators_])     # P(s > k)
        n, K = G.shape[0], len(self.order)
        P = np.empty((n, K))
        P[:, 0] = 1 - G[:, 0]
        for k in range(1, K - 1):
            P[:, k] = G[:, k - 1] - G[:, k]
        P[:, K - 1] = G[:, K - 2]
        self.negative_share_ = float(np.mean((P < 0).any(axis=1)))
        P = np.clip(P, 0, None)
        P /= P.sum(axis=1, keepdims=True)
        # Columnas en el orden de classes_ (1, 2, 3, 4).
        cols = [list(self.order).index(c) for c in self.classes_]
        return P[:, cols]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
