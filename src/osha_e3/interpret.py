"""Interpretabilidad con SHAP y LIME para un pipeline ya ajustado."""
import contextlib
import io

import numpy as np
import pandas as pd
import shap

from . import data as D

TREE = {'tree', 'random_forest', 'xgboost'}
LINEAR = {'logreg', 'ridge', 'lasso', 'svm', 'svr'}


def readable(names):
    """Nombres legibles de las columnas one-hot (``naics3_336`` → ``naics3=336``)."""
    out = []
    for n in names:
        for f in D.FEATURES:
            if n.startswith(f + '_'):
                n = f + '=' + n[len(f) + 1:]
                break
        out.append(n.replace('infrequent_sklearn', 'raras'))
    return out


def transformed(pipe, X, model_name=None):
    """Matriz densa que ve el modelo y nombres de sus columnas.

    XGBoost se entrenó con matrices dispersas, donde un cero no almacenado significa *faltante*; para
    que la matriz densa se enrute igual por los árboles, sus ceros se pasan a ``NaN``.
    """
    enc = pipe.named_steps['onehot']
    A = enc.transform(X)
    A = (A.toarray() if hasattr(A, 'toarray') else A).astype(np.float32)
    if model_name == 'xgboost':
        A[A == 0] = np.nan
    return A, readable(enc.get_feature_names_out(D.FEATURES))


def shap_values(task, model_name, pipe, X_background, X_explain, seed=42):
    """Valores SHAP en el espacio one-hot.

    * Árbol y Random Forest: ``TreeExplainer`` *interventional* respecto de un fondo de 200 casos.
    * XGBoost: ``TreeExplainer`` *tree_path_dependent* (usa las coberturas de los propios árboles; el
      modo interventional devuelve ceros con esta versión de XGBoost y matrices con faltantes).
    * Lineales: ``LinearExplainer`` (exactos bajo independencia de variables).
    * Otros (KNN, Naive Bayes): ``KernelExplainer`` con fondo resumido por k-means (aproximado).

    Returns
    -------
    values : ndarray
        (n, p) en regresión; (n, p, K) en clasificación (escala de salida cruda del modelo).
    base : ndarray o float
    A : ndarray
        Matriz explicada.
    names : list of str
    """
    model = pipe.named_steps['model']
    B, names = transformed(pipe, X_background, model_name)
    A, _ = transformed(pipe, X_explain, model_name)
    with contextlib.redirect_stderr(io.StringIO()):
        if model_name == 'xgboost':
            ex = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
            vals = ex.shap_values(A)
        elif model_name in TREE:
            ex = shap.TreeExplainer(model, data=B[:200], feature_perturbation='interventional', model_output='raw')
            vals = ex.shap_values(A, check_additivity=False)
        elif model_name in LINEAR:
            ex = shap.LinearExplainer(model, B)
            vals = ex.shap_values(A)
        else:
            f = model.predict_proba if task == 'clf' else model.predict
            ex = shap.KernelExplainer(f, shap.kmeans(B, 20))
            vals = ex.shap_values(A, nsamples=300, silent=True)
    vals = np.asarray(vals)
    if task == 'clf' and vals.ndim == 3 and vals.shape[0] != A.shape[0]:
        vals = np.moveaxis(vals, 0, -1)  # (K, n, p) → (n, p, K)
    return vals, np.asarray(ex.expected_value), A, names


def group_by_variable(values, names):
    """Suma los SHAP de las categorías de cada variable original.

    Por aditividad, la suma de los valores de Shapley de un bloque de columnas es la contribución
    del bloque: la variable original completa.
    """
    var = np.array([n.split('=')[0] for n in names])
    return pd.DataFrame({v: values[..., var == v].sum(axis=-1) for v in D.FEATURES})


def lime_explain(pipe, X_train, x_row, class_index, num_features=9, seed=42, num_samples=5000):
    """LIME sobre las 9 variables originales (categóricas), pasando por el pipeline completo.

    Cada perturbación cambia la categoría de algunas variables por otra observada en el entrenamiento
    (con su frecuencia), así que todas las filas perturbadas son casos posibles; el pipeline las
    codifica igual que en el entrenamiento. LIME ajusta luego un modelo lineal ponderado por cercanía
    sobre la probabilidad de la clase ``class_index``.

    Returns
    -------
    list of (str, float)
        ``'variable=categoría'`` y su peso local.
    """
    from lime.lime_tabular import LimeTabularExplainer
    cats = {f: sorted(X_train[f].astype(str).unique()) for f in D.FEATURES}
    enc = lambda df: np.column_stack([pd.Categorical(df[f].astype(str), categories=cats[f]).codes for f in D.FEATURES])
    Xe = enc(X_train)
    xr = enc(x_row)[0]

    def predict(Z):
        df = pd.DataFrame({f: np.asarray(cats[f], dtype=object)[np.clip(Z[:, i].astype(int), 0, len(cats[f]) - 1)]
                           for i, f in enumerate(D.FEATURES)})
        return pipe.predict_proba(df)

    ex = LimeTabularExplainer(Xe, feature_names=D.FEATURES, categorical_features=list(range(len(D.FEATURES))),
                              categorical_names={i: cats[f] for i, f in enumerate(D.FEATURES)},
                              discretize_continuous=False, random_state=seed, mode='classification')
    exp = ex.explain_instance(xr, predict, labels=[class_index], num_features=num_features, num_samples=num_samples)
    return exp.as_list(label=class_index)
