"""Utilidades de los experimentos del modelo propuesto (no versionado todavía)."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from osha_e3 import nested as N, evaluation as E, ensemble as ENS, data as D, config as C

def outer_data():
    X, y, g, s = N.task_data('clf')
    sub = (D.classification_sets)()[0]
    Xf = ENS.add_full_codes(sub)
    assert (Xf.y_clf.to_numpy() - 1 == y).all()
    return Xf, y, g

def report(y, p, pred=None):
    m = E.classification_metrics(y, p)
    pred = p.argmax(1) if pred is None else pred
    rec = [((pred == k) & (y == k)).sum() / max((y == k).sum(), 1) for k in range(4)]
    present = [k for k in range(4) if (y == k).any()]
    m['ba_decision'] = float(np.mean([rec[k] for k in present]))
    m['ba3'] = float(np.mean(rec[1:]))
    m['death_hits'] = int(((pred == 0) & (y == 0)).sum()); m['death_flagged'] = int((pred == 0).sum())
    return m

from sklearn.metrics import roc_auc_score, log_loss
def per_class(y, p):
    out = {}
    for k in range(4):
        if (y == k).any() and (y != k).any():
            out[f'auc{k}'] = round(roc_auc_score(y == k, p[:, k]), 4)
    out['aucF'] = round(np.mean([out[f'auc{k}'] for k in (1, 2, 3)]), 4)
    keep = y > 0
    q = np.clip(p[keep][:, 1:], 1e-12, None); q = q / q.sum(1, keepdims=True)
    out['ll3'] = round(log_loss(y[keep] - 1, q, labels=[0, 1, 2]), 4)
    return out
