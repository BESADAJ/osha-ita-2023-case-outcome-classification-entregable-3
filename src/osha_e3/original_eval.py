"""Evaluación del modelo original (ensamble multivista + combinador SECR) en el experimento pareado 5×5.

Para cada celda (repetición, pliegue externo) ya guardada por ``paired.run_original``:

1. Se arman las predicciones *fuera de pliegue internas* de los 16 modelos base (solo filas del entrenamiento
   externo) y sus probabilidades en el pliegue externo.
2. Los hiperparámetros del combinador (compuerta, λ, pasos, tamaño mínimo de región y α de la decisión) se
   optimizan con **búsqueda bayesiana TPE (Optuna)**, el mismo presupuesto de 16 configuraciones, la misma
   métrica (balanced accuracy) y 3 pliegues internos por establecimiento, como los benchmarks. El pliegue
   externo no participa.
3. Con los mejores hiperparámetros se ajusta SECR sobre todas las predicciones internas y se evalúa en el
   pliegue externo. Las mismas predicciones sirven para la **ablación**: promedio simple, pesos por error
   global, *stacking* lineal, Caruana global (modelo G) y SECR sin regiones o sin corrección por prior.
"""
import json
import time
import warnings

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from . import combinador as CB
from . import config as C
from . import evaluation as E
from . import paired as PR

optuna.logging.set_verbosity(optuna.logging.WARNING)
K = 4
OUT = PR.OUT / 'original_eval'
GATE_CHOICES = list(CB.GATES)


def stacks(d):
    names = [str(n) for n in d['names']]
    P_in = np.stack([d[f'oof_{n}'].astype(float) for n in names], axis=1)
    P_out = np.stack([d[f'out_{n}'].astype(float) for n in names], axis=1)
    return names, P_in, P_out


def _fit_predict(params, P_a, y_a, X_a, P_b, X_b):
    r_a, vocab = CB.region_codes(X_a, params['gate'])
    r_b, _ = CB.region_codes(X_b, params['gate'], vocab)
    m = CB.ConditionalEnsembleSelection(lam=params['lam'], n_steps=params['n_steps'],
                                        min_region=params['min_region'], alpha=params['alpha'],
                                        gate=params['gate'] != 'none')
    m.fit(CB.meta_matrix(P_a, r_a), y_a)
    return m, m.predict_proba(CB.meta_matrix(P_b, r_b))


def tune(P_in, y, groups, X, n_trials=C.BUDGET, seed=C.SEED):
    """Búsqueda TPE de los hiperparámetros de SECR con 3 pliegues internos por establecimiento."""
    folds = list(StratifiedGroupKFold(C.INNER_FOLDS, shuffle=True, random_state=seed).split(P_in, y, groups))
    history = []

    def objective(trial):
        params = {'gate': trial.suggest_categorical('gate', GATE_CHOICES),
                  'lam': trial.suggest_float('lam', 1.0, 1e4, log=True),
                  'n_steps': trial.suggest_int('n_steps', 5, 200, log=True),
                  'min_region': trial.suggest_int('min_region', 50, 3000, log=True),
                  'alpha': trial.suggest_float('alpha', 0.0, 1.0)}
        scores = []
        for a, b in folds:
            m, q = _fit_predict(params, P_in[a], y[a], X.iloc[a], P_in[b], X.iloc[b])
            scores.append(balanced_accuracy_score(y[b], m.decide(q)))
        history.append({**params, 'score': float(np.mean(scores))})
        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value, history


def _decision(q, prior, alpha):
    return (q / prior ** alpha).argmax(axis=1)


def evaluate_cell(d, X):
    """Modelo original y variantes de la ablación en una celda. Devuelve dict con probabilidades y metadatos."""
    names, P_in, P_out = stacks(d)
    y_in, y_out = d['y_train'].astype(int), d['y'].astype(int)
    X_in, X_out = X.iloc[d['idx_train']].reset_index(drop=True), X.iloc[d['idx']].reset_index(drop=True)
    g_in = X_in.establishment_id.to_numpy()
    prior = np.maximum(np.bincount(y_in, minlength=K) / len(y_in), 1e-12)
    seed = C.seed_for('search', int(d['fold'])) + 100 * int(d['rep'])
    t0 = time.perf_counter()
    best, inner, hist = tune(P_in, y_in, g_in, X_in, seed=seed)
    t_tune = time.perf_counter() - t0
    t0 = time.perf_counter()
    model, q = _fit_predict(best, P_in, y_in, X_in, P_out, X_out)
    t_fit = time.perf_counter() - t0
    variants = {'original': (q, model.decide(q))}
    # Ablación 1: sin regiones (mismos λ, pasos y α)
    _, q_glob = _fit_predict({**best, 'gate': 'none'}, P_in, y_in, X_in, P_out, X_out)
    variants['sin_regiones'] = (q_glob, _decision(q_glob, prior, best['alpha']))
    # Ablación 2: sin corrección por prior (α = 0)
    variants['sin_prior'] = (q, q.argmax(axis=1))
    # Reglas estándar de combinación (Cuadro 1 de la guía), con la misma decisión del original
    for name, w in [('promedio_simple', np.full(len(names), 1 / len(names))),
                    ('pesos_error_global', CB.global_error_weights(P_in, y_in)),
                    ('caruana_G', CB.caruana(P_in, y_in))]:
        qq = np.einsum('m,nmk->nk', w, P_out)
        variants[name] = (qq, _decision(qq, prior, best['alpha']))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(np.log(np.clip(P_in, 1e-6, 1)).reshape(len(P_in), -1), y_in)
    qs = np.zeros((len(P_out), K))
    qs[:, lr.classes_] = lr.predict_proba(np.log(np.clip(P_out, 1e-6, 1)).reshape(len(P_out), -1))
    variants['stacking_lineal'] = (qs, _decision(qs, prior, best['alpha']))
    # Cada modelo base por separado (mejor vista individual)
    best_single = int(np.argmin(-np.log(CB.true_class_matrix(P_in, y_in)).mean(axis=0)))  # menor log-loss
    qb = P_out[:, best_single]
    variants['mejor_base'] = (qb, _decision(qb, prior, best['alpha']))
    return {'variants': variants, 'y': y_out, 'params': best, 'inner': inner, 'history': hist,
            'weights_global': model.weights_global_, 'weights_region': model.weights_region_,
            'fw_loss': model.history_['loss'], 'fw_gap': model.history_['gap'], 'names': names,
            'tune_s': t_tune, 'fit_s': t_fit + float(d['fit_s']), 'pred_s': float(d['pred_s']),
            'best_single': names[best_single]}


def metrics(y, q, pred):
    m = E.classification_metrics(y, q)
    m['recall'] = balanced_accuracy_score(y, pred)  # BA con la regla de decisión del modelo
    m['accuracy'] = float((pred == y).mean())
    return m


def run_all():
    """Evalúa todas las celdas disponibles del original y guarda un JSON por celda."""
    X, _, _ = PR.paired_data()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in PR.load('original'):
        path = OUT / f"r{d['rep']}_f{d['fold']}.json"
        if path.exists():
            continue
        res = evaluate_cell(d, X)
        rec = {'rep': d['rep'], 'fold': d['fold'], 'params': res['params'], 'inner': res['inner'],
               'history': res['history'], 'names': res['names'], 'best_single': res['best_single'],
               'weights_global': res['weights_global'].tolist(),
               'weights_region': {str(k): v.tolist() for k, v in res['weights_region'].items()},
               'fw_loss': res['fw_loss'].tolist(), 'fw_gap': res['fw_gap'].tolist(),
               'tune_s': res['tune_s'], 'fit_s': res['fit_s'], 'pred_s': res['pred_s'],
               'metrics': {k: metrics(res['y'], q, p) for k, (q, p) in res['variants'].items()}}
        np.savez_compressed(OUT / f"r{d['rep']}_f{d['fold']}.npz", y=res['y'], idx=d['idx'],
                            **{f'q_{k}': q.astype(np.float32) for k, (q, _) in res['variants'].items()},
                            **{f'pred_{k}': p for k, (_, p) in res['variants'].items()})
        path.write_text(json.dumps(rec, default=float), encoding='utf-8')
        print('celda', d['rep'], d['fold'], rec['metrics']['original']['recall'], flush=True)
