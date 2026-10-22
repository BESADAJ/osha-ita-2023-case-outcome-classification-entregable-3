"""Ronda 2 del modelo original: cribado de HAFL-XGBoost en los 3 pliegues temporales.

Por pliegue externo k:
1. Hiperparámetros de XGBoost = los de la referencia (results/runs/xgboost__class_weight__bayes/fold<k>.json),
   mismo preprocesamiento y mismos pesos class_weight (compute_sample_weight('balanced')).
2. γ_start ∈ {1, 3, 5} se elige con la validación interna agrupada de la corrida base (muestra de búsqueda del
   25 %, StratifiedGroupKFold(3) por establecimiento), maximizando balanced accuracy. α = 0,25 fijo.
3. Reajuste con el 100 % del entrenamiento externo y predicción de la validación externa.
Salidas en results/original_r2/ (reanudable): cribado/<modelo>/fold<k>.json y _pred.npz, cribado_r2.csv y
decision_r2.md (regla de regla_r2.md).

python cribado_r2.py --workers 2
python cribado_r2.py --time-one            # mide un ajuste real (pliegue 3, γ_start = 3) y sale
python cribado_r2.py --table-only
"""
import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, balanced_accuracy_score, cohen_kappa_score
from threadpoolctl import threadpool_limits

from osha3.data import LABELS, RAW_FEATURES, SEED, inner_splits, read_fold
from osha3.metrics import classification_metrics, wilson_interval
from osha3.models import REGISTRY
from osha3.original_models import HAFLXGBoost
from osha3.pipeline import build_pipeline

ROOT = Path(__file__).resolve().parent / 'results'
OUT = ROOT / 'original_r2'
CRIB = OUT / 'cribado'
REF = 'xgboost__class_weight__bayes'
GAMMAS = (1.0, 3.0, 5.0)
ALPHA = 0.25


def log(msg):
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {msg}'
    print(line, flush=True)
    with (OUT / 'cribado_r2_log.txt').open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def ref_params(fold):
    return json.loads((ROOT / 'runs' / REF / f'fold{fold}.json').read_text(encoding='utf-8'))['best_params']


def make_pipe(params, gamma_start, y_fit):
    """Pipeline de la referencia (mismo one-hot y mismos pesos) con el clasificador cambiado por HAFL-XGBoost."""
    pipe, fit_params, _ = build_pipeline(REGISTRY['xgboost'], params, 'class_weight', y_fit)
    p = {**REGISTRY['xgboost'].defaults, **params}
    pipe.steps[-1] = ('clf', HAFLXGBoost(
        gamma_start=gamma_start, alpha=ALPHA, n_estimators=p['n_estimators'], learning_rate=p['learning_rate'],
        max_depth=p['max_depth'], subsample=p['subsample'], colsample_bytree=p['colsample_bytree'],
        min_split_loss=p['gamma'], min_child_weight=p['min_child_weight'], reg_alpha=p['reg_alpha'],
        reg_lambda=p['reg_lambda'], random_state=SEED, n_jobs=1))
    return pipe, fit_params


def run_fold(fold, model='hafl_xgboost', gammas=GAMMAS):
    out = CRIB / model
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'fold{fold}.json'
    if path.exists():
        return model, fold, 'skip', 0.0
    start = time.perf_counter()
    params = ref_params(fold)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sample, train, valid = read_fold(fold, 'search'), read_fold(fold, 'train'), read_fold(fold, 'valid')
        assert not set(train.establishment_id) & set(valid.establishment_id)
        assert set(sample.id) <= set(train.id)
        ys = sample.incident_outcome.astype(int).to_numpy()
        inner = {}
        if len(gammas) > 1:   # selección de γ_start solo con datos de entrenamiento
            for g in gammas:
                scores = []
                for a, b in inner_splits(sample):
                    pipe, fp = make_pipe(params, g, ys[a])
                    pipe.fit(sample[RAW_FEATURES].iloc[a], ys[a], **fp)
                    scores.append(float(balanced_accuracy_score(ys[b], pipe.predict(sample[RAW_FEATURES].iloc[b]))))
                inner[g] = scores
            best = max(gammas, key=lambda g: (np.mean(inner[g]), -g))     # empate: el γ menor
        else:
            best = gammas[0]
        y_train = train.incident_outcome.astype(int).to_numpy()
        y_valid = valid.incident_outcome.astype(int).to_numpy()
        pipe, fp = make_pipe(params, best, y_train)
        t = time.perf_counter()
        pipe.fit(train[RAW_FEATURES], y_train, **fp)
        refit = time.perf_counter() - t
        proba = pipe.predict_proba(valid[RAW_FEATURES])
        assert list(pipe.classes_) == LABELS and np.isfinite(proba).all()
    np.savez_compressed(out / f'fold{fold}_pred.npz', proba=proba.astype(np.float32), y=y_valid.astype(np.int8),
                        establishment_id=valid.establishment_id.to_numpy(dtype=str),
                        case_id=valid.id.to_numpy(dtype=str), fold=np.full(len(valid), fold, dtype=np.int8))
    record = {'model': model, 'outer_fold': fold, 'seed': SEED, 'alpha': ALPHA, 'gamma_grid': list(gammas),
              'inner_scores': {str(g): s for g, s in inner.items()}, 'gamma_start': best,
              'xgb_params_from': f'results/runs/{REF}/fold{fold}.json', 'xgb_params': params,
              'metrics': classification_metrics(y_valid, proba),
              'gamma_last_iteration': pipe.named_steps['clf'].gamma_history_[-1],
              'seconds_refit': refit, 'seconds_total': time.perf_counter() - start,
              'train_rows': len(train), 'valid_rows': len(valid), 'train_deaths': int((y_train == 1).sum()),
              'valid_deaths': int((y_valid == 1).sum())}
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    tmp.replace(path)
    return model, fold, 'ok', time.perf_counter() - start


def topk_metrics(y, proba, k):
    """Decisión a casos marcados iguales: los k de mayor P(Fall.) son Fallecimiento; el resto, argmax en 2-4."""
    pred = np.asarray(LABELS)[1:][proba[:, 1:].argmax(axis=1)]
    if k > 0:
        pred[np.argsort(-proba[:, 0], kind='stable')[:k]] = 1
    return pred


def summarize(y, proba, pred):
    m = classification_metrics(y, proba)
    deaths, hits = int((y == 1).sum()), int(((y == 1) & (pred == 1)).sum())
    lo, hi = wilson_interval(hits, deaths)
    rec = [np.mean(pred[y == c] == c) for c in (2, 3, 4)]
    return {'muertes': deaths, 'muertes_acertadas': hits, 'casos_marcados': int((pred == 1).sum()),
            'precision_fall': hits / max(1, int((pred == 1).sum())), 'wilson_low': lo, 'wilson_high': hi,
            'aucpr_fall': float(average_precision_score(y == 1, proba[:, 0])),
            'recall_2_4': float(np.mean(rec)), 'auc_macro': m['roc_auc_macro_ovr'],
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'accuracy': float(np.mean(pred == y)),
            'kappa': float(cohen_kappa_score(y, pred)), 'log_loss': m['log_loss'], 'n': len(y)}


def build_table():
    ref = [np.load(ROOT / 'runs' / REF / f'fold{k}_pred.npz') for k in (1, 2, 3)]
    ks = [int((r['proba'].argmax(axis=1) == 0).sum()) for r in ref]
    rows, pooled = [], {}
    models = [('referencia', REF, ref)]
    for name in ('hafl_xgboost', 'hafl_xgboost_gamma0'):
        if all((CRIB / name / f'fold{k}.json').exists() for k in (1, 2, 3)):
            models.append(('candidato' if name == 'hafl_xgboost' else 'ablación', name,
                           [np.load(CRIB / name / f'fold{k}_pred.npz') for k in (1, 2, 3)]))
    for kind, name, preds in models:
        for rule in ('argmax', 'k_igual_ref'):
            ys, ps, ds = [], [], []
            for k, z in enumerate(preds):
                y, pr = z['y'].astype(int), z['proba'].astype(float)
                d = np.asarray(LABELS)[pr.argmax(axis=1)] if rule == 'argmax' else topk_metrics(y, pr, ks[k])
                rows.append({'tipo': kind, 'modelo': name, 'decision': rule, 'pliegue': str(k + 1), **summarize(y, pr, d)})
                ys.append(y), ps.append(pr), ds.append(d)
            row = {'tipo': kind, 'modelo': name, 'decision': rule, 'pliegue': 'todos',
                   **summarize(np.concatenate(ys), np.vstack(ps), np.concatenate(ds))}
            rows.append(row)
            pooled[(name, rule)] = row
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'cribado_r2.csv', index=False)
    return table, pooled, ks


def decide(pooled, ks):
    r = pooled[(REF, 'argmax')]
    lines = ['# Decisión de la Ronda 2 (HAFL-XGBoost)', '',
             f'Generado {datetime.now():%Y-%m-%d %H:%M}. Regla: `regla_r2.md` (fijada antes de ver resultados).',
             f'Referencia `{REF}`: H = {r["muertes_acertadas"]}/{r["muertes"]}, casos marcados por pliegue {ks}, '
             f'AUC-PR = {r["aucpr_fall"]:.4f}, R234 = {r["recall_2_4"]:.3f}, AUC = {r["auc_macro"]:.3f}.', '']
    if ('hafl_xgboost', 'k_igual_ref') not in pooled:
        lines.append('Sin resultados del candidato.')
    else:
        c = pooled[('hafl_xgboost', 'k_igual_ref')]
        c1 = c['muertes_acertadas'] > r['muertes_acertadas']
        c2 = c['aucpr_fall'] > r['aucpr_fall']
        c3 = c['recall_2_4'] >= r['recall_2_4'] and c['auc_macro'] >= r['auc_macro'] - 0.005
        lines += ['| Criterio | Candidato | Referencia | Cumple |', '|---|---|---|---|',
                  f'| 1. Muertes acertadas (mismos casos marcados) | {c["muertes_acertadas"]}/{c["muertes"]} '
                  f'(Wilson {c["wilson_low"]:.2f}–{c["wilson_high"]:.2f}) | {r["muertes_acertadas"]}/{r["muertes"]} | {"sí" if c1 else "no"} |',
                  f'| 2. AUC-PR de Fallecimiento | {c["aucpr_fall"]:.4f} | {r["aucpr_fall"]:.4f} | {"sí" if c2 else "no"} |',
                  f'| 3a. Recall medio 2–4 | {c["recall_2_4"]:.3f} | {r["recall_2_4"]:.3f} | {"sí" if c["recall_2_4"] >= r["recall_2_4"] else "no"} |',
                  f'| 3b. AUC macro (tolerancia 0,005) | {c["auc_macro"]:.3f} | {r["auc_macro"]:.3f} | '
                  f'{"sí" if c["auc_macro"] >= r["auc_macro"] - 0.005 else "no"} |', '']
        passed = c1 and c2 and c3
        lines.append('**PASA** la regla: avisar a Ivan antes de lanzar nada más.' if passed else
                     '**NO PASA** la regla: se detiene aquí. Se probaron cuatro candidatos y ninguno mejoró.')
        lines.append('Efecto de selección: cuatro candidatos probados; con 12 muertes, una diferencia puede ser azar.')
    (OUT / 'decision_r2.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--folds', nargs='+', type=int, default=[3, 2, 1])
    ap.add_argument('--time-one', action='store_true')
    ap.add_argument('--ablation', action='store_true', help='γ_start = 0 (informativa, fuera de la regla)')
    ap.add_argument('--table-only', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.time_one:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train = read_fold(3, 'train')
            y = train.incident_outcome.astype(int).to_numpy()
            pipe, fp = make_pipe(ref_params(3), 3.0, y)
            t = time.perf_counter()
            pipe.fit(train[RAW_FEATURES], y, **fp)
            log(f'Medición: un ajuste de HAFL-XGBoost en el pliegue 3 (100 %, {len(y)} filas, '
                f'{ref_params(3)["n_estimators"]} rondas): {time.perf_counter() - t:.0f} s')
        return
    if not args.table_only:
        tasks = [(f, 'hafl_xgboost', GAMMAS) for f in args.folds]
        if args.ablation:
            tasks += [(f, 'hafl_xgboost_gamma0', (0.0,)) for f in args.folds]
        log(f'Ronda 2: {len(tasks)} tareas, {args.workers} procesos')
        for m, f, status, sec in Parallel(n_jobs=args.workers, return_as='generator_unordered')(
                delayed(run_fold)(f, m, g) for f, m, g in tasks):
            log(f'{m} fold{f}: {status} ({sec:.0f} s)')
    table, pooled, ks = build_table()
    decide(pooled, ks)


if __name__ == '__main__':
    main()
