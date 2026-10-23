"""Modelo original completo (capítulo 19): ensamble multivista + combinador SECR, compatible con scikit-learn.

``OriginalModel.fit(X, y, groups)`` hace, solo con los datos que recibe:

1. ajusta los 16 modelos base del ensamble multivista y obtiene sus predicciones *fuera de pliegue*
   (5 pliegues por establecimiento, ``ensemble.StackedEnsemble``);
2. optimiza los hiperparámetros de SECR con TPE sobre esas predicciones (``original_eval.tune``: 16
   configuraciones, 3 pliegues internos, balanced accuracy);
3. ajusta SECR con los mejores hiperparámetros.

``predict_proba`` combina las probabilidades de los modelos base con los pesos de la región del caso.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from . import combinador as CB
from . import config as C
from . import ensemble as ENS


class OriginalModel(BaseEstimator, ClassifierMixin):
    """Ensamble multivista con selección de ensambles condicionada por región (SECR).

    Parameters
    ----------
    n_trials : int
        Presupuesto de la búsqueda TPE de los hiperparámetros de SECR.
    seed : int
    """

    def __init__(self, n_trials=C.BUDGET, seed=C.SEED):
        self.n_trials, self.seed = n_trials, seed

    def fit(self, X, y, groups):
        from . import original_eval as OE
        self.stack_ = ENS.StackedEnsemble(ENS.default_views(full='diverso'), seed=self.seed).fit(X, y, groups)
        names = self.stack_.names_
        P = np.stack([self.stack_.oof_[n] for n in names], axis=1)
        X = X.reset_index(drop=True)
        self.params_, self.inner_score_, self.history_ = OE.tune(P, y, groups, X, self.n_trials, self.seed)
        r, self.vocab_ = CB.region_codes(X, self.params_['gate'])
        p = self.params_
        self.combiner_ = CB.ConditionalEnsembleSelection(lam=p['lam'], n_steps=p['n_steps'], min_region=p['min_region'],
                                                         alpha=p['alpha'], gate=p['gate'] != 'none')
        self.combiner_.fit(CB.meta_matrix(P, r), y)
        self.stack_.oof_ = None  # no se necesitan para predecir (ahorra memoria en el artefacto)
        self.classes_ = np.arange(CB.ConditionalEnsembleSelection().n_classes)
        return self

    def base_proba(self, X):
        check_is_fitted(self, 'combiner_')
        P = self.stack_.base_proba(X)
        return np.stack([P[n] for n in self.stack_.names_], axis=1)

    def predict_proba(self, X, P=None):
        P = self.base_proba(X) if P is None else P
        r, _ = CB.region_codes(X.reset_index(drop=True), self.params_['gate'], self.vocab_)
        return self.combiner_.predict_proba(CB.meta_matrix(P, r))

    def predict(self, X):
        return self.combiner_.decide(self.predict_proba(X))
