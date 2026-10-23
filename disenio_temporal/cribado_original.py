"""Fase A del modelo original: cribado de los tres candidatos con los hiperparámetros del paper.

Para cada candidato y pliegue externo temporal (1, 2, 3): un ajuste sobre el 100 % del
entrenamiento externo (sin optimizador, sin bucle interno) y predicción sobre la validación
externa. Mismo preprocesamiento que la corrida base (prepare_features → imputación → one-hot).
Las referencias (Random Forest y XGBoost con class_weight) se leen de results/runs/ sin reajustar.

Salidas en results/original/:
- cribado/<candidato>/fold<k>.json y fold<k>_pred.npz (mismo formato que results/runs/; reanudable).
- cribado.csv: una fila por modelo × pliegue y una fila «todos» con los 3 pliegues juntos.
- decision_fase_a.md: aplicación de la regla de regla_decision.md.

python cribado_original.py --workers 3            # ajusta lo que falte y construye la tabla
python cribado_original.py --table-only           # solo la tabla y la decisión
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
from sklearn.metrics import average_precision_score, cohen_kappa_score
from threadpoolctl import threadpool_limits

from osha3.data import LABELS, RAW_FEATURES, SEED, read_fold
from osha3.metrics import classification_metrics, wilson_interval
from osha3.models import ORIGINAL
from osha3.pipeline import build_pipeline

ROOT = Path(__file__).resolve().parent / 'results'
OUT = ROOT / 'original'
CRIB = OUT / 'cribado'
CANDIDATES = ['pt_bagging', 'multirandbal', 'hierarchical']        # orden fijo de desempate
REFERENCES = [f'{m}__class_weight__{o}' for m in ('xgboost', 'random_forest')
              for o in ('grid', 'random', 'bayes', 'genetic')]
PRETTY = {'pt_bagging': 'PT-bagging', 'multirandbal': 'MultiRandBal', 'hierarchical': 'Jerárquica'}


def log(msg):
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {msg}'
    print(line, flush=True)
    with (OUT / 'cribado_log.txt').open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def fit_one(model, fold, params=None):
    """Ajusta un candidato en un pliegue externo y guarda JSON + predicciones (se salta si ya existe)."""
    out = CRIB / model
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'fold{fold}.json'
    if path.exists():
        return model, fold, 'skip', 0.0
    start = time.perf_counter()
    spec = ORIGINAL[model]
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train, valid = read_fold(fold, 'train'), read_fold(fold, 'valid')
        assert not set(train.establishment_id) & set(valid.establishment_id)     # control de fuga
        y_train = train.incident_outcome.astype(int).to_numpy()
        y_valid = valid.incident_outcome.astype(int).to_numpy()
        pipe, fit_params, _ = build_pipeline(spec, params or {}, 'none', y_train)
        if model == 'hierarchical':      # votos fuera de muestra agrupados por establecimiento
            fit_params['clf__groups'] = train.establishment_id.to_numpy()
        fit_start = time.perf_counter()
        pipe.fit(train[RAW_FEATURES], y_train, **fit_params)
        fit_seconds = time.perf_counter() - fit_start
        assert list(pipe.classes_) == LABELS
        proba = pipe.predict_proba(valid[RAW_FEATURES])
        extra = {}
        if model == 'pt_bagging':        # probabilidades sin el umbral, para la ablación A
            extra['proba_raw'] = pipe[:-1].transform(valid[RAW_FEATURES])
            extra['proba_raw'] = pipe.named_steps['clf'].predict_proba_raw(extra['proba_raw']).astype(np.float32)
    np.savez_compressed(out / f'fold{fold}_pred.npz', proba=proba.astype(np.float32), y=y_valid.astype(np.int8),
                        establishment_id=valid.establishment_id.to_numpy(dtype=str),
                        case_id=valid.id.to_numpy(dtype=str), fold=np.full(len(valid), fold, dtype=np.int8), **extra)
    clf = pipe.named_steps['clf']
    record = {'model': model, 'balance': 'none', 'optimizer': 'none (hiperparámetros del paper)', 'outer_fold': fold,
              'seed': SEED, 'params': {**spec.defaults, **(params or {})}, 'metrics': classification_metrics(y_valid, proba),
              'seconds_fit': fit_seconds, 'seconds_total': time.perf_counter() - start,
              'train_rows': len(train), 'valid_rows': len(valid), 'train_deaths': int((y_train == 1).sum()),
              'valid_deaths': int((y_valid == 1).sum()),
              'n_encoded_features': int(len(pipe.named_steps['onehot'].get_feature_names_out())),
              'n_synthetic': int(getattr(clf, 'n_synthetic_', 0))}
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    tmp.replace(path)
    return model, fold, 'ok', time.perf_counter() - start


def row_metrics(y, proba):
    """Métricas de la tabla de cribado; la decisión es argmax(proba), como en osha3/metrics.py."""
    m = classification_metrics(y, proba)
    pred = np.asarray(LABELS)[proba.argmax(axis=1)]
    deaths, hits, flagged = int((y == 1).sum()), int(((y == 1) & (pred == 1)).sum()), int((pred == 1).sum())
    lo, hi = wilson_interval(hits, deaths)
    return {'muertes': deaths, 'muertes_acertadas': hits, 'casos_marcados': flagged,
            'precision_fall': hits / flagged if flagged else 0.0,
            'aucpr_fall': float(average_precision_score(y == 1, proba[:, 0])) if deaths else np.nan,
            'recall_fall': hits / deaths if deaths else np.nan, 'wilson_low': lo, 'wilson_high': hi,
            'recall_2_4': float(np.mean([m['recall_days_away'], m['recall_transfer'], m['recall_other']])),
            'auc_macro': m['roc_auc_macro_ovr'], 'balanced_accuracy': m['balanced_accuracy'],
            'f1_macro': m['f1_macro'], 'accuracy': m['accuracy'],
            'kappa': float(cohen_kappa_score(y, pred)), 'log_loss': m['log_loss'], 'n': len(y)}


def load_preds(kind, name):
    base = CRIB / name if kind == 'candidato' else ROOT / 'runs' / name
    return [np.load(base / f'fold{k}_pred.npz') for k in (1, 2, 3)]


def build_table():
    rows = []
    entries = [('candidato', c) for c in CANDIDATES if all((CRIB / c / f'fold{k}.json').exists() for k in (1, 2, 3))]
    entries += [('referencia', r) for r in REFERENCES]
    for kind, name in entries:
        preds = load_preds(kind, name)
        variants = [(name, 'proba')]
        if name == 'pt_bagging':
            variants.append(('pt_bagging_sin_umbral', 'proba_raw'))      # ablación A: argmax(P)
        for label, key in variants:
            ys = [p['y'].astype(int) for p in preds]
            ps = [p[key].astype(float) for p in preds]
            for k, (y, pr) in enumerate(zip(ys, ps), 1):
                rows.append({'tipo': kind if label == name else 'ablación', 'modelo': label, 'pliegue': str(k),
                             **row_metrics(y, pr)})
            rows.append({'tipo': kind if label == name else 'ablación', 'modelo': label, 'pliegue': 'todos',
                         **row_metrics(np.concatenate(ys), np.vstack(ps))})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'cribado.csv', index=False)
    return table


def best_reference():
    """Referencia de la regla: la combinación con class_weight de mayor BA media en master_registry.csv."""
    reg = pd.read_csv(ROOT / 'master_registry.csv')
    reg = reg[reg.model.isin(['xgboost', 'random_forest']) & reg.balance.eq('class_weight')]
    mean = reg.groupby(['model', 'balance', 'optimizer']).outer_balanced_accuracy.mean().sort_values(ascending=False)
    return '__'.join(mean.index[0]), float(mean.iloc[0])


def decide(table):
    """Aplica la regla fijada en results/original/regla_decision.md (pliegues juntos)."""
    ref_name, ref_ba = best_reference()
    pooled = table[table.pliegue.eq('todos')].set_index('modelo')
    r = pooled.loc[ref_name]
    lines = ['# Decisión de la Fase A', '',
             f'Generado {datetime.now():%Y-%m-%d %H:%M}. Regla: `regla_decision.md` (fijada antes de ver resultados).',
             f'Referencia: `{ref_name}` (BA media por pliegue {ref_ba:.3f} en `master_registry.csv`).',
             f'Referencia, pliegues juntos: H = {r.muertes_acertadas}/{r.muertes}, F = {r.casos_marcados}, '
             f'R234 = {r.recall_2_4:.3f}, AUC = {r.auc_macro:.3f}, LL = {r.log_loss:.3f}.', '',
             '| Candidato | H | F | a1 | a2 | R234 | AUC | LL | (b) | Mejora |', '|---|---|---|---|---|---|---|---|---|---|']
    status = []
    for c in CANDIDATES:
        if c not in pooled.index:
            lines.append(f'| {PRETTY[c]} | sin resultados | | | | | | | | |')
            continue
        x = pooled.loc[c]
        a1 = x.muertes_acertadas > r.muertes_acertadas and x.casos_marcados <= 1.10 * r.casos_marcados
        a2 = x.muertes_acertadas >= r.muertes_acertadas and x.casos_marcados < r.casos_marcados
        b = (x.recall_2_4 >= r.recall_2_4 - 0.005 and x.auc_macro >= r.auc_macro - 0.01
             and x.log_loss <= 1.10 * r.log_loss)
        ok = (a1 or a2) and b
        status.append((c, ok, int(x.muertes_acertadas), int(x.casos_marcados)))
        lines.append(f'| {PRETTY[c]} | {x.muertes_acertadas}/{x.muertes} | {x.casos_marcados} | {"sí" if a1 else "no"} | '
                     f'{"sí" if a2 else "no"} | {x.recall_2_4:.3f} | {x.auc_macro:.3f} | {x.log_loss:.3f} | '
                     f'{"sí" if b else "no"} | **{"sí" if ok else "no"}** |')
    if status:
        passing = [s for s in status if s[1]]
        pool = passing or status
        order = sorted(pool, key=lambda s: (-s[2], s[3], CANDIDATES.index(s[0])))
        winner = order[0][0]
        verdict = (f'Cumple la regla: {", ".join(PRETTY[s[0]] for s in passing)}. Pasa a la Fase B: **{PRETTY[winner]}**.'
                   if passing else f'Ningún candidato cumple la regla. Pasa a la Fase B el que más muertes acierta con '
                                   f'menos casos marcados: **{PRETTY[winner]}**; el informe dirá que **no mejoró**.')
        lines += ['', verdict, '', 'Sesgo declarado: elegir el mejor de 3 candidatos sobrestima su ventaja.']
    (OUT / 'decision_fase_a.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=CANDIDATES)
    ap.add_argument('--folds', nargs='+', type=int, default=[3, 2, 1])
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--table-only', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.table_only:
        tasks = [(m, f) for f in args.folds for m in args.models]
        log(f'Fase A: {len(tasks)} ajustes, {args.workers} procesos')
        jobs = (delayed(fit_one)(m, f) for m, f in tasks)
        for m, f, status, sec in Parallel(n_jobs=args.workers, return_as='generator_unordered')(jobs):
            log(f'{m} fold{f}: {status} ({sec:.0f} s)')
    decide(build_table())


if __name__ == '__main__':
    main()
