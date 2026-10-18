"""Interpretabilidad con SHAP y LIME para un pipeline ya ajustado."""
import numpy as np
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


def transformed(pipe, X):
    """Matriz densa que ve el modelo y nombres de sus columnas."""
    enc = pipe.named_steps['onehot']
    A = enc.transform(X)
    A = A.toarray() if hasattr(A, 'toarray') else A
    return A.astype(np.float32), readable(enc.get_feature_names_out(D.FEATURES))


def shap_values(task, model_name, pipe, X_background, X_explain, seed=42):
    """Valores SHAP en el espacio one-hot.

    * Árboles (árbol, RF, XGBoost): ``TreeExplainer`` (SHAP exactos para árboles, *interventional*
      respecto del fondo).
    * Lineales: ``LinearExplainer`` (exactos bajo independencia de variables).
    * Otros (KNN, Naive Bayes): ``KernelExplainer`` con fondo resumido por k-means (aproximado).

    Returns
    -------
    values : ndarray
        (n, p) en regresión; (n, p, K) en clasificación (log-odds o probabilidad según el modelo).
    base : ndarray o float
    A : ndarray
        Matriz explicada.
    names : list of str
    """
    model = pipe.named_steps['model']
    B, names = transformed(pipe, X_background)
    A, _ = transformed(pipe, X_explain)
    if model_name in TREE:
        ex = shap.TreeExplainer(model, data=B[:200], feature_perturbation='interventional',
                                model_output='raw')
        vals = ex.shap_values(A, check_additivity=False)
        base = ex.expected_value
    elif model_name in LINEAR:
        ex = shap.LinearExplainer(model, B)
        vals = ex.shap_values(A)
        base = ex.expected_value
    else:
        bg = shap.kmeans(B, 20)
        f = model.predict_proba if task == 'clf' else model.predict
        ex = shap.KernelExplainer(f, bg)
        vals = ex.shap_values(A, nsamples=300, silent=True)
        base = ex.expected_value
    vals = np.asarray(vals)
    if task == 'clf' and vals.ndim == 3 and vals.shape[0] != A.shape[0]:
        vals = np.moveaxis(vals, 0, -1)  # (K, n, p) → (n, p, K)
    return vals, np.asarray(base), A, names


def lime_explain(pipe, X_train, X_row, class_index, num_features=10, seed=42):
    """Explicación LIME (modelo lineal local ponderado) de una fila, para la clase ``class_index``."""
    from lime.lime_tabular import LimeTabularExplainer
    B, names = transformed(pipe, X_train)
    a, _ = transformed(pipe, X_row)
    model = pipe.named_steps['model']
    ex = LimeTabularExplainer(B, feature_names=names, categorical_features=list(range(B.shape[1])),
                              discretize_continuous=False, random_state=seed, mode='classification')
    exp = ex.explain_instance(a[0], model.predict_proba, labels=[class_index], num_features=num_features,
                              num_samples=3000)
    return exp.as_list(label=class_index), exp
