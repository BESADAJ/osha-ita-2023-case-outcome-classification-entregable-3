"""Métricas de clasificación (incluida calibración) y de regresión."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             roc_auc_score)

K = 4


def expected_calibration_error(y, proba, bins=15):
    """ECE de la clase predicha (confianza top-1) con ``bins`` intervalos de igual ancho.

    ECE = sum_b (n_b / n) |acc_b - conf_b|.
    """
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_multiclass(y, proba):
    """Brier multiclase: media de sum_k (p_k - 1[y=k])^2 (0 = perfecto, 2 = peor)."""
    onehot = np.eye(proba.shape[1])[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def classification_metrics(y, proba):
    """Accuracy, precision/recall/F1 macro, AUC macro OvR, Brier, log-loss y ECE.

    ``y`` va codificado 0..3 (0 = fallecimiento). El recall macro es el balanced accuracy.
    """
    y = np.asarray(y)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-12, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    labels = list(range(K))
    present = np.unique(y)
    try:
        auc = roc_auc_score(y, proba, labels=labels, multi_class='ovr', average='macro') \
            if len(present) == K else roc_auc_score(
                (y[:, None] == present).astype(int), proba[:, present], average='macro')
    except ValueError:
        auc = np.nan
    return {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, labels=labels, average='macro', zero_division=0),
        'recall': balanced_accuracy_score(y, pred),
        'f1': f1_score(y, pred, labels=labels, average='macro', zero_division=0),
        'auc': float(auc),
        'brier': brier_multiclass(y, proba),
        'log_loss': float(log_loss(y, proba, labels=labels)),
        'ece': expected_calibration_error(y, proba),
    }


def regression_metrics(y, pred):
    """RMSE, MAE y R² (en la escala log1p(días))."""
    return {'rmse': float(np.sqrt(mean_squared_error(y, pred))), 'mae': float(mean_absolute_error(y, pred)),
            'r2': float(r2_score(y, pred))}


def predict_output(task, pipe, X):
    """Probabilidades (clasificación) o predicción puntual (regresión)."""
    return pipe.predict_proba(X) if task == 'clf' else pipe.predict(X)


def metrics_for(task, y, out):
    return classification_metrics(y, out) if task == 'clf' else regression_metrics(y, out)
