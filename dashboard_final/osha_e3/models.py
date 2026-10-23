"""Definición de modelos, técnicas de balanceo y espacios de búsqueda.

Cada modelo tiene:

* una fábrica ``make(params, balance, seed)`` que devuelve el estimador final;
* un espacio de búsqueda ``space`` (usado por Random, Optuna y el algoritmo genético);
* una rejilla ``grid`` con como máximo ``BUDGET`` puntos (Grid Search).

Así los cuatro métodos de optimización exploran el mismo espacio con el mismo presupuesto de
evaluaciones, y la comparación entre ellos es justa.
"""
import warnings

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from .data import FEATURES

BALANCES = ['none', 'smote', 'adasyn', 'class_weight']
OPTIMIZERS = ['grid', 'random', 'bayes', 'genetic']
CLF_MODELS = ['knn', 'naive_bayes', 'logreg', 'tree', 'random_forest', 'xgboost', 'svm']
REG_MODELS = ['knn', 'ridge', 'lasso', 'tree', 'random_forest', 'xgboost', 'svr']

PRETTY = {'knn': 'KNN', 'naive_bayes': 'Naive Bayes', 'logreg': 'Reg. logística', 'tree': 'Árbol',
          'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'svm': 'SVM', 'ridge': 'Ridge',
          'lasso': 'Lasso', 'svr': 'SVR', 'none': 'Sin balanceo', 'smote': 'SMOTE',
          'adasyn': 'ADASYN', 'class_weight': 'class_weight', 'grid': 'Grid', 'random': 'Random',
          'bayes': 'Bayesiana (TPE)', 'genetic': 'Genética'}


# --------------------------------------------------------------------------------------------
# Envoltorios para modelos sin ``class_weight`` nativo y para obtener probabilidades.
# --------------------------------------------------------------------------------------------
class WeightedBernoulliNB(BernoulliNB):
    """Naive Bayes de Bernoulli con pesos 'balanced' por observación (equivale a class_weight)."""

    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=compute_sample_weight('balanced', y))


class WeightedXGBClassifier(XGBClassifier):
    """XGBoost con pesos 'balanced' por observación (equivale a class_weight)."""

    def fit(self, X, y, **kw):
        kw['sample_weight'] = compute_sample_weight('balanced', y)
        return super().fit(X, y, **kw)


class PriorCorrectedKNN(KNeighborsClassifier):
    """KNN sensible al costo: divide la probabilidad vecinal por la prior de cada clase.

    KNN no acepta pesos por clase. Reponderar la probabilidad por ``1/prior`` y renormalizar es la
    regla de decisión que minimiza el error balanceado; es el análogo de ``class_weight='balanced'``.
    """

    def fit(self, X, y):
        super().fit(X, y)
        counts = np.bincount(np.searchsorted(self.classes_, y), minlength=len(self.classes_))
        self.prior_ = np.maximum(counts / counts.sum(), 1e-12)
        return self

    def predict_proba(self, X):
        p = super().predict_proba(X) / self.prior_
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class SoftmaxLinearSVC(LinearSVC):
    """LinearSVC con ``predict_proba`` = softmax de la función de decisión.

    Sirve para calcular AUC (solo importa el orden) sin el costo de Platt dentro de la búsqueda.
    No son probabilidades calibradas; la calibración se estudia aparte (escalado de Platt).
    """

    def predict_proba(self, X):
        z = self.decision_function(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class SafeSMOTE(SMOTE):
    """SMOTE que reduce ``k_neighbors`` cuando la clase minoritaria tiene pocos casos.

    Fallecimiento tiene ~5 casos en cada entrenamiento interno; con ``k_neighbors=5`` SMOTE falla.
    """

    def fit_resample(self, X, y):
        m = np.bincount(y).astype(float)
        m = int(m[m > 0].min())
        self.k_neighbors = max(1, min(5, m - 1))
        return super().fit_resample(X, y)


class SafeADASYN(ADASYN):
    """ADASYN (He et al., 2008) implementado clase por clase, válido con clases diminutas.

    La versión de imblearn falla en este problema por dos motivos: (1) usa ``n_neighbors + 1``
    vecinos *dentro* de la clase minoritaria, imposible con 4-5 fallecimientos, y (2) redondea por
    separado el número de muestras de cada punto, así que cuando hay que generar menos de 0,5 por
    punto (p. ej. 581 muestras entre 4.063 casos) todas redondean a 0 y lanza «No samples will be
    generated»; con ``sampling_strategy='auto'`` basta que falle una clase para que falle todo.

    Aquí, para cada clase c con menos casos que la mayoritaria:

    1. r_i = fracción de los k vecinos (en todos los datos) de cada caso de c que son de otra clase
       (dificultad); k = min(5, n_c − 1).
    2. Se reparten G = n_mayoritaria − n_c muestras proporcionalmente a r_i, con redondeo por restos
       mayores (la suma es exactamente G). Si todos los r_i son 0, el reparto es uniforme (= SMOTE).
    3. Cada muestra interpola entre el caso y uno de sus k vecinos de la misma clase.

    ``notes_`` deja constancia de lo hecho en cada clase.
    """

    def fit_resample(self, X, y):
        from scipy import sparse
        from sklearn.neighbors import NearestNeighbors
        rng = np.random.default_rng(self.random_state)
        is_sparse = sparse.issparse(X)
        X = X.tocsr() if is_sparse else np.asarray(X)
        y = np.asarray(y)
        counts = np.bincount(y)
        target = counts.max()
        nn_all = NearestNeighbors().fit(X)
        new_X, new_y, self.notes_ = [], [], []
        for c in np.flatnonzero((counts > 0) & (counts < target)):
            idx = np.flatnonzero(y == c)
            n_c, G = len(idx), int(target - counts[c])
            if n_c < 2:
                self.notes_.append(f'clase {c}: {n_c} caso, no se remuestrea')
                continue
            k = max(1, min(5, n_c - 1))
            Xc = X[idx]
            nb = nn_all.kneighbors(Xc, n_neighbors=k + 1, return_distance=False)[:, 1:]
            r = (y[nb] != c).mean(axis=1)
            uniform = r.sum() == 0
            r = np.full(n_c, 1.0 / n_c) if uniform else r / r.sum()
            raw = r * G
            g = np.floor(raw).astype(int)
            g[np.argsort(raw - g)[::-1][:G - g.sum()]] += 1
            within = NearestNeighbors(n_neighbors=k + 1).fit(Xc).kneighbors(Xc, return_distance=False)[:, 1:]
            rows = np.repeat(np.arange(n_c), g)
            pick = within[rows, rng.integers(0, k, len(rows))]
            lam = rng.random(len(rows))[:, None]
            if is_sparse:
                synth = Xc[rows] + Xc[pick].multiply(lam) - Xc[rows].multiply(lam)
            else:
                synth = Xc[rows] + lam * (Xc[pick] - Xc[rows])
            new_X.append(sparse.csr_matrix(synth) if is_sparse else synth)
            new_y.append(np.full(len(rows), c))
            self.notes_.append(f'clase {c}: n={n_c}, k={k}, {G} sintéticos' + (' (reparto uniforme)' if uniform else ''))
        if not new_X:
            return X, y
        stack = sparse.vstack if is_sparse else np.vstack
        return stack([X] + new_X).tocsr() if is_sparse else stack([X] + new_X), np.concatenate([y] + new_y)


# --------------------------------------------------------------------------------------------
# Espacios de búsqueda. Tipos: ('int', lo, hi, log) | ('float', lo, hi, log) | ('cat', [..]).
# Rangos fijados con valores habituales en la literatura y el tamaño de los datos (~20-30 mil
# filas, ~300 columnas binarias); C, lambda y learning rate en escala logarítmica.
# --------------------------------------------------------------------------------------------
SPACES = {
    'clf': {
        'knn': {'n_neighbors': ('int', 5, 801, True), 'weights': ('cat', ['uniform', 'distance'])},
        'naive_bayes': {'alpha': ('float', 1e-3, 10.0, True), 'fit_prior': ('cat', [True, False])},
        'logreg': {'C': ('float', 1e-3, 10.0, True), 'penalty': ('cat', ['l1', 'l2'])},
        'tree': {'max_depth': ('int', 3, 30, False), 'min_samples_leaf': ('int', 1, 500, True),
                 'criterion': ('cat', ['gini', 'entropy'])},
        'random_forest': {'n_estimators': ('int', 50, 250, False), 'max_depth': ('int', 6, 30, False),
                          'min_samples_leaf': ('int', 1, 50, True), 'max_features': ('float', 0.03, 0.3, True)},
        'xgboost': {'n_estimators': ('int', 100, 400, False), 'max_depth': ('int', 2, 10, False),
                    'learning_rate': ('float', 0.01, 0.3, True), 'subsample': ('float', 0.6, 1.0, False),
                    'colsample_bytree': ('float', 0.4, 1.0, False), 'min_child_weight': ('float', 1, 20, True)},
        'svm': {'C': ('float', 1e-3, 10.0, True), 'loss': ('cat', ['hinge', 'squared_hinge'])},
    },
    'reg': {
        'knn': {'n_neighbors': ('int', 5, 801, True), 'weights': ('cat', ['uniform', 'distance'])},
        'ridge': {'alpha': ('float', 1e-2, 1e3, True)},
        'lasso': {'alpha': ('float', 1e-5, 1e-1, True)},
        'tree': {'max_depth': ('int', 3, 30, False), 'min_samples_leaf': ('int', 1, 500, True),
                 'criterion': ('cat', ['squared_error', 'friedman_mse'])},
        'random_forest': {'n_estimators': ('int', 50, 250, False), 'max_depth': ('int', 6, 30, False),
                          'min_samples_leaf': ('int', 1, 50, True), 'max_features': ('float', 0.03, 0.3, True)},
        'xgboost': {'n_estimators': ('int', 100, 400, False), 'max_depth': ('int', 2, 10, False),
                    'learning_rate': ('float', 0.01, 0.3, True), 'subsample': ('float', 0.6, 1.0, False),
                    'colsample_bytree': ('float', 0.4, 1.0, False), 'min_child_weight': ('float', 1, 20, True)},
        'svr': {'C': ('float', 1e-3, 10.0, True), 'epsilon': ('float', 0.0, 1.5, False)},
    },
}

# Rejillas: <= 16 puntos, repartidas en los hiperparámetros más influyentes.
GRIDS = {
    'clf': {
        'knn': {'n_neighbors': [5, 21, 51, 101, 201, 301, 501, 801], 'weights': ['uniform', 'distance']},
        'naive_bayes': {'alpha': [1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0], 'fit_prior': [True, False]},
        'logreg': {'C': [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 10.0], 'penalty': ['l1', 'l2']},
        'tree': {'max_depth': [4, 8, 16, 30], 'min_samples_leaf': [1, 20, 100, 400]},
        'random_forest': {'n_estimators': [100, 200], 'max_depth': [10, 25],
                          'min_samples_leaf': [2, 20], 'max_features': [0.05, 0.2]},
        'xgboost': {'n_estimators': [150, 400], 'max_depth': [3, 8], 'learning_rate': [0.03, 0.15],
                    'subsample': [0.8], 'colsample_bytree': [0.5, 0.9], 'min_child_weight': [1]},
        'svm': {'C': [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 10.0], 'loss': ['hinge', 'squared_hinge']},
    },
    'reg': {
        'knn': {'n_neighbors': [5, 21, 51, 101, 201, 301, 501, 801], 'weights': ['uniform', 'distance']},
        'ridge': {'alpha': list(np.round(np.logspace(-2, 3, 16), 5))},
        'lasso': {'alpha': list(np.round(np.logspace(-5, -1, 16), 7))},
        'tree': {'max_depth': [4, 8, 16, 30], 'min_samples_leaf': [1, 20, 100, 400]},
        'random_forest': {'n_estimators': [100, 200], 'max_depth': [10, 25],
                          'min_samples_leaf': [2, 20], 'max_features': [0.05, 0.2]},
        'xgboost': {'n_estimators': [150, 400], 'max_depth': [3, 8], 'learning_rate': [0.03, 0.15],
                    'subsample': [0.8], 'colsample_bytree': [0.5, 0.9], 'min_child_weight': [1]},
        'svr': {'C': [1e-3, 1e-2, 0.1, 1.0], 'epsilon': [0.0, 0.3, 0.7, 1.2]},
    },
}


def make_estimator(task, name, params, balance='none', seed=42):
    """Crea el estimador final de un modelo con sus hiperparámetros.

    Parameters
    ----------
    task : {'clf', 'reg'}
    name : str
        Clave del modelo (``CLF_MODELS`` o ``REG_MODELS``).
    params : dict
        Hiperparámetros (claves del espacio de búsqueda).
    balance : str
        Solo importa ``'class_weight'``: activa pesos balanceados por la vía que el modelo admita.
    seed : int

    Returns
    -------
    estimator
    """
    p = dict(params)
    cw = 'balanced' if balance == 'class_weight' else None
    if task == 'clf':
        if name == 'knn':
            cls = PriorCorrectedKNN if cw else KNeighborsClassifier
            return cls(n_neighbors=int(p['n_neighbors']), weights=p['weights'], algorithm='brute', n_jobs=1)
        if name == 'naive_bayes':
            cls = WeightedBernoulliNB if cw else BernoulliNB
            return cls(alpha=p['alpha'], fit_prior=bool(p['fit_prior']))
        if name == 'logreg':
            return LogisticRegression(C=p['C'], penalty=p['penalty'], solver='saga', max_iter=300,
                                      tol=1e-3, class_weight=cw, random_state=seed)
        if name == 'tree':
            return DecisionTreeClassifier(max_depth=int(p['max_depth']), criterion=p.get('criterion', 'gini'),
                                          min_samples_leaf=int(p['min_samples_leaf']), class_weight=cw,
                                          random_state=seed)
        if name == 'random_forest':
            return RandomForestClassifier(n_estimators=int(p['n_estimators']), max_depth=int(p['max_depth']),
                                          min_samples_leaf=int(p['min_samples_leaf']),
                                          max_features=p['max_features'], class_weight=cw, n_jobs=1,
                                          random_state=seed)
        if name == 'xgboost':
            cls = WeightedXGBClassifier if cw else XGBClassifier
            return cls(n_estimators=int(p['n_estimators']), max_depth=int(p['max_depth']),
                       learning_rate=p['learning_rate'], subsample=p['subsample'],
                       colsample_bytree=p['colsample_bytree'], min_child_weight=p['min_child_weight'],
                       tree_method='hist', n_jobs=1, random_state=seed, verbosity=0)
        if name == 'svm':
            return SoftmaxLinearSVC(C=p['C'], loss=p['loss'], class_weight=cw, dual=True, max_iter=3000,
                                    tol=1e-3, random_state=seed)
    else:
        if name == 'knn':
            return KNeighborsRegressor(n_neighbors=int(p['n_neighbors']), weights=p['weights'],
                                       algorithm='brute', n_jobs=1)
        if name == 'ridge':
            return Ridge(alpha=p['alpha'], solver='sparse_cg', random_state=seed)
        if name == 'lasso':
            return Lasso(alpha=p['alpha'], max_iter=3000, tol=1e-3, random_state=seed)
        if name == 'tree':
            return DecisionTreeRegressor(max_depth=int(p['max_depth']), min_samples_leaf=int(p['min_samples_leaf']),
                                         criterion=p.get('criterion', 'squared_error'), random_state=seed)
        if name == 'random_forest':
            return RandomForestRegressor(n_estimators=int(p['n_estimators']), max_depth=int(p['max_depth']),
                                         min_samples_leaf=int(p['min_samples_leaf']),
                                         max_features=p['max_features'], n_jobs=1, random_state=seed)
        if name == 'xgboost':
            return XGBRegressor(n_estimators=int(p['n_estimators']), max_depth=int(p['max_depth']),
                                learning_rate=p['learning_rate'], subsample=p['subsample'],
                                colsample_bytree=p['colsample_bytree'], min_child_weight=p['min_child_weight'],
                                tree_method='hist', n_jobs=1, random_state=seed, verbosity=0)
        if name == 'svr':
            return LinearSVR(C=p['C'], epsilon=p['epsilon'], loss='epsilon_insensitive', dual=True,
                             max_iter=5000, tol=1e-3, random_state=seed)
    raise ValueError(f'Modelo desconocido: {task}/{name}')


def make_sampler(balance, seed):
    """Sobremuestreador de imblearn o ``None`` (sin balanceo o class_weight)."""
    if balance == 'smote':
        return SafeSMOTE(random_state=seed)
    if balance == 'adasyn':
        return SafeADASYN(random_state=seed)
    return None


def make_encoder():
    """One-hot que agrupa las categorías con menos de 20 casos del entrenamiento en «infrequent»."""
    return OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20, dtype=np.float32)


def make_pipeline(task, name, params, balance='none', seed=42):
    """Pipeline completo (codificación → balanceo → modelo) que se ajusta solo con datos de entrenamiento.

    Se usa el ``Pipeline`` de imblearn para que SMOTE/ADASYN se apliquen únicamente al ajustar
    (en ``predict`` el paso de remuestreo se omite), sin fuga hacia validación o prueba.
    """
    steps = [('onehot', make_encoder())]
    sampler = make_sampler(balance, seed) if task == 'clf' else None
    if sampler is not None:
        steps.append(('sampler', sampler))
    if name == 'random_forest':
        # sklearn construye los bosques mucho más rápido sobre matrices densas (≈4× aquí).
        steps.append(('dense', FunctionTransformer(_to_dense, accept_sparse=True)))
    steps.append(('model', make_estimator(task, name, params, balance, seed)))
    return Pipeline(steps)


def _to_dense(X):
    return X.toarray() if hasattr(X, 'toarray') else X


def combos(task):
    """Lista de combinaciones (modelo, balanceo, optimizador): 7×4×4 = 112 o 7×4 = 28."""
    if task == 'clf':
        return [(m, b, o) for m in CLF_MODELS for b in BALANCES for o in OPTIMIZERS]
    return [(m, 'none', o) for m in REG_MODELS for o in OPTIMIZERS]
