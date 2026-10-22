"""Modelo propuesto: *stacking* multivista de los modelos del curso.

Cada caso se describe desde varias **vistas** (grupos de variables con información distinta) y cada
modelo base se especializa en una vista:

* **Lugar de trabajo** (las 9 variables de los capítulos 6 a 13): logística, XGBoost, Random Forest,
  Naive Bayes y KNN, con los hiperparámetros que ya eligió la validación anidada.
* **Lesión**: códigos OIICS de naturaleza, parte del cuerpo, evento y fuente (asignados por OSHA a partir
  del relato del incidente) y el tipo de incidente: logística y XGBoost.
* **Códigos finos**: codificación por objetivo (*target encoding*) de NAICS de 6 dígitos, SOC completo e
  interacciones lugar × lesión, con un XGBoost encima.
* **Empleador**: codificación por objetivo de la empresa (EIN y nombre), calculada con *otros*
  establecimientos (los pliegues separan establecimientos), con una logística encima.

Un **meta-modelo** (regresión logística multinomial) combina las log-probabilidades *fuera de pliegue*
de todos los modelos base (Wolpert, 1992; Ting y Witten, 1999) y devuelve probabilidades calibradas. La
decisión puede ser ``argmax P`` o ``argmax P/λ`` (maximiza la balanced accuracy esperada si las
probabilidades están calibradas; Collell et al., 2018).
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline as sk_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config as C
from . import data as D
from . import models as M

K = 4
INJURY = ['nature', 'part', 'event', 'source', 'sec_source', 'type_of_incident']
FINE_TE = ['naics6', 'soc6', 'naics6*soc6', 'naics3*soc_minor', 'naics3*nature', 'naics3*event', 'soc_minor*nature',
           'nature*part', 'event*source', 'nature*event', 'state*naics3', 'state*nature', 'type_of_incident*naics3']
EMPLOYER_TE = ['ein', 'company', 'ein*nature', 'ein*type_of_incident']
EXTRA = ['naics6', 'soc6', 'nature', 'part', 'event', 'source', 'sec_source', 'ein', 'company']


def add_full_codes(df):
    """Añade códigos completos, OIICS y empleador (por ``id``) a un DataFrame preparado."""
    ext = pd.read_parquet(C.RESULTS / 'cache' / 'osha_2023_fullcodes.parquet')
    out = df.drop(columns=[c for c in EXTRA if c in df.columns]).merge(ext, on='id', how='left', validate='one_to_one')
    assert out[EXTRA].notna().all().all()
    return out


def _column(X, name):
    parts = name.split('*')
    col = X[parts[0]].astype(str)
    for p in parts[1:]:
        col = col + '|' + X[p].astype(str)
    return col


class MulticlassTargetEncoder:
    """Codificación por objetivo multiclase con suavizado: (n_ck + m·π_k)/(n_c + m).

    ``fit_transform`` devuelve codificaciones *fuera de pliegue* (pliegues por grupo): una fila nunca se
    codifica con su propia etiqueta ni con las de su establecimiento. ``unknown`` lista los valores que
    significan «sin dato» y se codifican con la proporción global. Se añade log(1 + n_c) como columna.
    """

    def __init__(self, columns, m=30.0, n_splits=5, seed=C.SEED, unknown=('Unknown',)):
        self.columns, self.m, self.n_splits, self.seed, self.unknown = columns, m, n_splits, seed, unknown

    def _stats(self, X, y):
        prior = np.bincount(y, minlength=K) / len(y)
        tables = {}
        onehot = pd.DataFrame(np.eye(K)[y], columns=range(K))
        for c in self.columns:
            onehot['key'] = _column(X, c).to_numpy()
            g = onehot.groupby('key')
            cnt, s = g.size(), g[list(range(K))].sum()
            t = s.add(self.m * prior, axis=1).div(cnt + self.m, axis=0)
            t['n'] = np.log1p(cnt)
            tables[c] = t
        return prior, tables

    def _apply(self, X, prior, tables):
        out = []
        for c in self.columns:
            key = _column(X, c)
            enc = tables[c].reindex(key.to_numpy()).to_numpy()
            miss = np.isnan(enc[:, 0]) | key.str.split('|').map(lambda p: any(v in self.unknown for v in p)).to_numpy()
            enc[miss, :K] = prior
            enc[miss, K] = 0.0
            out.append(enc)
        return np.hstack(out).astype(np.float32)

    def fit(self, X, y):
        self.prior_, self.tables_ = self._stats(X, y)
        return self

    def transform(self, X):
        return self._apply(X, self.prior_, self.tables_)

    def fit_transform(self, X, y, groups):
        self.fit(X, y)
        out = np.zeros((len(X), (K + 1) * len(self.columns)), dtype=np.float32)
        cv = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed)
        for a, b in cv.split(X, y, groups):
            prior, tables = self._stats(X.iloc[a], y[a])
            out[b] = self._apply(X.iloc[b], prior, tables)
        return out


class ViewModel(BaseEstimator, ClassifierMixin):
    """Un modelo base especializado en una vista.

    kind:
      'course:<modelo>'  pipeline del curso sobre las 9 variables (hiperparámetros de la validación anidada)
      'onehot_lr' / 'onehot_xgb'  one-hot de ``columns`` con logística o XGBoost
      'te_xgb' / 'te_lr' / 'te_rf'  codificación por objetivo de ``columns`` con XGBoost, logística o
                                    Random Forest
    """

    def __init__(self, kind, columns=None, params=None, seed=C.SEED):
        self.kind, self.columns, self.params, self.seed = kind, columns, params, seed

    def fit(self, X, y, groups):
        p = dict(self.params or {})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if self.kind.startswith('course:'):
                self.est_ = M.make_pipeline('clf', self.kind.split(':')[1], p, 'none', self.seed)
                self.est_.fit(X[D.FEATURES], y)
            elif self.kind.startswith('onehot'):
                enc = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20)
                if self.kind == 'onehot_lr':
                    clf = LogisticRegression(C=p.get('C', 0.3), max_iter=500, solver='lbfgs', random_state=self.seed)
                else:
                    clf = XGBClassifier(n_estimators=p.get('n_estimators', 300), max_depth=p.get('max_depth', 5),
                                        learning_rate=p.get('learning_rate', 0.05), subsample=0.8, colsample_bytree=0.8,
                                        min_child_weight=p.get('min_child_weight', 3), tree_method='hist', n_jobs=4,
                                        random_state=self.seed, verbosity=0)
                self.est_ = sk_pipeline(enc, clf).fit(X[self.columns], y)
            else:
                self.enc_ = MulticlassTargetEncoder(self.columns, m=p.get('m', 30.0), seed=self.seed)
                Z = self.enc_.fit_transform(X, y, groups)
                if self.kind == 'te_xgb':
                    clf = XGBClassifier(n_estimators=p.get('n_estimators', 300), max_depth=p.get('max_depth', 4),
                                        learning_rate=p.get('learning_rate', 0.05), subsample=p.get('subsample', 0.8),
                                        colsample_bytree=p.get('colsample_bytree', 0.8),
                                        min_child_weight=p.get('min_child_weight', 5), tree_method='hist', n_jobs=4,
                                        random_state=self.seed, verbosity=0)
                elif self.kind == 'te_rf':
                    from sklearn.ensemble import RandomForestClassifier
                    clf = RandomForestClassifier(n_estimators=p.get('n_estimators', 300),
                                                 min_samples_leaf=p.get('min_samples_leaf', 20),
                                                 max_features=p.get('max_features', 0.3), n_jobs=2,
                                                 random_state=self.seed)
                else:
                    clf = sk_pipeline(StandardScaler(), LogisticRegression(C=p.get('C', 0.3), max_iter=1000,
                                                                           random_state=self.seed))
                self.est_ = clf.fit(Z, y)
        return self

    def predict_proba(self, X):
        if self.kind.startswith('course:'):
            p = self.est_.predict_proba(X[D.FEATURES])
            cls = self.est_.classes_
        elif self.kind.startswith('onehot'):
            p = self.est_.predict_proba(X[self.columns])
            cls = self.est_.classes_
        else:
            p = self.est_.predict_proba(self.enc_.transform(X))
            cls = self.est_.classes_
        out = np.full((len(X), K), 1e-6)
        out[:, np.asarray(cls, dtype=int)] = p
        return np.clip(out, 1e-6, 1)


def course_params(metric='outer_log_loss_mean'):
    """Hiperparámetros de los modelos del curso: la combinación *sin balanceo* con mejor log-loss externo
    en la validación anidada (no se mira la prueba). Sin balanceo para que las probabilidades reflejen
    las frecuencias reales; el desbalance se trata al decidir."""
    mc = pd.read_csv(C.RESULTS / 'master_clf.csv')
    mc = mc[mc.balance.eq('none')]
    best = mc.loc[mc.groupby('model')[metric].idxmin()].set_index('model')
    return {m: json.loads(best.loc[m, 'final_params']) for m in best.index}


ALL_TE = list(dict.fromkeys(FINE_TE + EMPLOYER_TE + D.FEATURES + INJURY))


def default_views(course=('logreg', 'xgboost', 'random_forest', 'naive_bayes', 'knn'), injury=True, fine=True,
                  employer=True, full=False):
    """Lista (nombre, ViewModel) del nivel 0.

    ``full=True`` añade tres modelos que ven *toda* la información con sesgos inductivos distintos
    (árboles sobre codificación por objetivo, logística sobre codificación por objetivo y logística
    sobre one-hot de las variables originales y de la lesión): la diversidad de algoritmos es lo que
    el meta-modelo aprovecha cuando todos los modelos ven las mismas variables.
    """
    cp = course_params()
    views = [(f'curso_{m}', ViewModel(f'course:{m}', params=cp[m])) for m in course]
    if injury:
        views += [('lesion_lr', ViewModel('onehot_lr', INJURY, {'C': 0.3})),
                  ('lesion_xgb', ViewModel('onehot_xgb', INJURY))]
    if fine:
        views += [('finos_te_xgb', ViewModel('te_xgb', FINE_TE))]
    if employer:
        views += [('empleador_te_lr', ViewModel('te_lr', EMPLOYER_TE, {'C': 0.3, 'm': 10.0}))]
    if full:
        views += [('todo_te_xgb', ViewModel('te_xgb', ALL_TE)),
                  ('todo_te_lr', ViewModel('te_lr', ALL_TE, {'C': 0.3})),
                  ('todo_onehot_lr', ViewModel('onehot_lr', D.FEATURES + INJURY, {'C': 0.3}))]
    if full == 'diverso':
        views += [('todo_te_xgb_profundo', ViewModel('te_xgb', ALL_TE, {'max_depth': 8, 'learning_rate': 0.03,
                                                                         'n_estimators': 500, 'min_child_weight': 10,
                                                                         'colsample_bytree': 0.5}, seed=C.SEED + 1)),
                  ('todo_te_xgb_semilla2', ViewModel('te_xgb', ALL_TE, {'subsample': 0.7, 'colsample_bytree': 0.6},
                                                     seed=C.SEED + 2)),
                  ('todo_te_rf', ViewModel('te_rf', ALL_TE, seed=C.SEED + 3)),
                  ('todo_onehot_xgb', ViewModel('onehot_xgb', D.FEATURES + INJURY, {'max_depth': 6, 'n_estimators': 400}))]
    return views


def _logloss(y, p):
    p = np.clip(p, 1e-12, 1)
    return -np.mean(np.log(p[np.arange(len(y)), y] / p.sum(axis=1)))


def caruana_selection(P, y, steps=40, init=3):
    """Selección de ensamble de Caruana et al. (2004), con reemplazo.

    Empieza con los ``init`` mejores modelos (log-loss fuera de pliegue) y en cada paso añade, con
    reemplazo, el modelo que más baja la log-loss del promedio de probabilidades. Devuelve los pesos
    (fracción de veces que se eligió cada modelo). Elegir con reemplazo equivale a aprender pesos no
    negativos que suman 1; empezar con varios modelos evita sobreajustar el primer paso.
    """
    names = sorted(P, key=lambda n: _logloss(y, P[n]))
    chosen = names[:init]
    total = sum(P[n] for n in chosen)
    for _ in range(steps):
        best = min(names, key=lambda n: _logloss(y, (total + P[n]) / (len(chosen) + 1)))
        chosen.append(best)
        total = total + P[best]
    return (pd.Series(chosen).value_counts() / len(chosen)).to_dict()


class StackedEnsemble(BaseEstimator, ClassifierMixin):
    """Stacking multivista con meta-modelo logístico multinomial.

    Parameters
    ----------
    views : list of (str, ViewModel)
        Modelos base. Por defecto, ``default_views()``.
    meta_C : float
        Inverso de la regularización L2 del meta-modelo.
    n_splits : int
        Pliegues (por establecimiento) para las predicciones fuera de pliegue del nivel 0.
    """

    def __init__(self, views=None, meta_C=1.0, n_splits=5, seed=C.SEED, combiner='logistic', caruana_steps=40):
        self.views, self.meta_C, self.n_splits, self.seed = views, meta_C, n_splits, seed
        self.combiner, self.caruana_steps = combiner, caruana_steps

    @staticmethod
    def _meta_X(P):
        return np.hstack([np.log(p) for p in P])

    def fit(self, X, y, groups):
        from sklearn.base import clone
        views = self.views if self.views is not None else default_views()
        self.names_ = [n for n, _ in views]
        self.oof_ = {n: np.zeros((len(y), K)) for n in self.names_}
        cv = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed + 7)
        for a, b in cv.split(X, y, groups):
            Xa, Xb = X.iloc[a], X.iloc[b]
            for n, v in views:
                self.oof_[n][b] = clone(v).fit(Xa, y[a], groups[a]).predict_proba(Xb)
        self.meta_ = LogisticRegression(C=self.meta_C, max_iter=3000, random_state=self.seed)
        self.meta_.fit(self._meta_X([self.oof_[n] for n in self.names_]), y)
        self.weights_ = caruana_selection(self.oof_, y, steps=self.caruana_steps)
        self.models_ = {n: clone(v).fit(X, y, groups) for n, v in views}
        self.prior_ = np.bincount(y, minlength=K) / len(y)
        self.classes_ = np.arange(K)
        return self

    def base_proba(self, X):
        return {n: self.models_[n].predict_proba(X) for n in self.names_}

    def predict_proba(self, X, base=None, combiner=None):
        P = base or self.base_proba(X)
        if (combiner or self.combiner) == 'caruana':
            return sum(w * P[n] for n, w in self.weights_.items())
        q = self.meta_.predict_proba(self._meta_X([P[n] for n in self.names_]))
        out = np.zeros((len(q), K))  # siempre K columnas, aunque falte una clase en el entrenamiento
        out[:, self.meta_.classes_.astype(int)] = q
        return out

    def decide(self, p, rule='argmax'):
        return (p / self.prior_).argmax(axis=1) if rule == 'prior' else p.argmax(axis=1)

    def predict(self, X):
        return self.decide(self.predict_proba(X))
