"""Ronda 4 (exploratoria): cribado de Frank–Hall ordinal con XGBoost (regla: results/original_r4/regla_r4.md).

Por pliegue: hiperparámetros, preprocesamiento y pesos de la mejor base de la Ronda 3 (xgboost__class_weight__random);
solo cambia la descomposición ordinal. Control: el mismo pipeline con XGBoost multiclase debe reproducir las
predicciones guardadas de esa base. Comparación en el subproblema 2–4 con bootstrap pareado por establecimiento.

python cribado_r4.py --workers 3
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
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from cribado_original import row_metrics
from osha3.data import LABELS, RAW_FEATURES, SEED, read_fold
from osha3.models import REGISTRY
from osha3.original_models import FrankHallXGB
from osha3.pipeline import build_pipeline
from ronda3_2_4 import from_confusion

ROOT = Path(__file__).resolve().parent / 'results'
OUT = ROOT / 'original_r4'
BASE = 'xgboost__class_weight__random'
B, CLS = 2000, np.array([2, 3, 4])


def log(msg):
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {msg}'
    print(line, flush=True)
    with (OUT / 'cribado_r4_log.txt').open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def run_fold(fold, kind):
    """kind = 'frank_hall' (candidato) o 'control' (XGBoost multiclase: debe reproducir la base guardada)."""
    out = OUT / 'cribado' / kind
    out.mkdir(parents=True, exist_ok=True)
    if (out / f'fold{fold}.json').exists():
        return kind, fold, 'skip', 0.0
    start = time.perf_counter()
    params = json.loads((ROOT / 'runs' / BASE / f'fold{fold}.json').read_text(encoding='utf-8'))['best_params']
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train, valid = read_fold(fold, 'train'), read_fold(fold, 'valid')
        assert not set(train.establishment_id) & set(valid.establishment_id)
        y = train.incident_outcome.astype(int).to_numpy()
        pipe, fit_params, _ = build_pipeline(REGISTRY['xgboost'], params, 'class_weight', y)
        if kind == 'frank_hall':
            p = {**REGISTRY['xgboost'].defaults, **params}
            pipe.steps[-1] = ('clf', FrankHallXGB(
                n_estimators=p['n_estimators'], learning_rate=p['learning_rate'], max_depth=p['max_depth'],
                subsample=p['subsample'], colsample_bytree=p['colsample_bytree'], gamma=p['gamma'],
                min_child_weight=p['min_child_weight'], reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
                random_state=SEED, n_jobs=1))
        pipe.fit(train[RAW_FEATURES], y, **fit_params)
        proba = pipe.predict_proba(valid[RAW_FEATURES])
        assert list(pipe.classes_) == LABELS
    y_valid = valid.incident_outcome.astype(int).to_numpy()
    np.savez_compressed(out / f'fold{fold}_pred.npz', proba=proba.astype(np.float32), y=y_valid.astype(np.int8),
                        establishment_id=valid.establishment_id.to_numpy(dtype=str),
                        case_id=valid.id.to_numpy(dtype=str), fold=np.full(len(valid), fold, dtype=np.int8))
    rec = {'model': kind, 'outer_fold': fold, 'params_from': f'results/runs/{BASE}/fold{fold}.json', 'params': params,
           'negative_share': getattr(pipe.named_steps['clf'], 'negative_share_', None),
           'seconds': time.perf_counter() - start}
    (out / f'fold{fold}.json').write_text(json.dumps(rec, indent=1, default=str), encoding='utf-8')
    return kind, fold, 'ok', time.perf_counter() - start


def load(path):
    zs = [np.load(path / f'fold{k}_pred.npz') for k in (1, 2, 3)]
    return {k: np.concatenate([z[k] for z in zs]) for k in ('proba', 'y', 'establishment_id', 'case_id')}


def evaluate():
    base = load(ROOT / 'runs' / BASE)
    ctrl = load(OUT / 'cribado' / 'control')
    cand = load(OUT / 'cribado' / 'frank_hall')
    assert np.array_equal(base['case_id'], cand['case_id']) and np.array_equal(base['case_id'], ctrl['case_id'])
    repro = float(np.abs(ctrl['proba'] - base['proba']).max())
    lines = ['# Ronda 4 (exploratoria): Frank–Hall ordinal con XGBoost', '',
             f'Generado {datetime.now():%Y-%m-%d %H:%M}. Regla: `regla_r4.md` (escrita antes de calcular).',
             f'Control de reproducción (XGBoost multiclase reajustado vs predicciones guardadas de `{BASE}`): '
             f'diferencia máxima de probabilidades {repro:.2e} → {"OK" if repro < 1e-4 else "NO REPRODUCE"}.', '']
    # Subproblema 2–4 (igual que la Ronda 3).
    mask = np.isin(base['y'], CLS)
    y = base['y'][mask].astype(int)
    yi = np.searchsorted(CLS, y)
    g_idx, groups = pd.factorize(base['establishment_id'][mask])
    G = len(groups)
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(G, size=G), minlength=G) for _ in range(B)]).astype(np.float32)

    def sub(proba):
        p3 = proba[mask][:, 1:].astype(float)
        s = p3.sum(axis=1, keepdims=True)
        return np.where(s > 0, p3 / np.where(s > 0, s, 1), 1 / 3).argmax(axis=1)

    res = {}
    for name, z in (('base', base), ('frank_hall', cand), ('control', ctrl)):
        pi = sub(z['proba'])
        C = np.bincount(g_idx * 9 + yi * 3 + pi, minlength=G * 9).reshape(G, 9).astype(np.float32)
        conf = C.sum(axis=0).reshape(3, 3)
        res[name] = {'pi': pi, 'point': from_confusion(conf), 'boot': from_confusion((W @ C).reshape(B, 3, 3)),
                     'conf': conf}
    diff = res['frank_hall']['boot']['ba_2_4'] - res['base']['boot']['ba_2_4']
    d0 = float(res['frank_hall']['point']['ba_2_4'] - res['base']['point']['ba_2_4'])
    lo, hi = np.percentile(diff, [2.5, 97.5])
    p_boot = float(min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))
    cb, cc = res['base']['pi'] == yi, res['frank_hall']['pi'] == yi
    b10, b01 = int((cc & ~cb).sum()), int((~cc & cb).sum())
    p_mc = float(binomtest(b10, b10 + b01, 0.5).pvalue)
    passed = d0 > 0 and lo > 0
    rows, data = [], {'base': base, 'frank_hall': cand, 'control': ctrl}
    for name in ('base', 'frank_hall', 'control'):
        r, c = res[name], res[name]['conf']
        nonadj, err24 = c[0, 2] + c[2, 0], c[0, 1] + c[0, 2] + c[2, 0] + c[2, 1]
        full = row_metrics(data[name]['y'].astype(int), data[name]['proba'].astype(float))
        rows.append({'modelo': name, **{f'{k}_2_4' if k != 'ba_2_4' else k: float(v) for k, v in r['point'].items()},
                     **{f'{k}_low': float(np.percentile(r['boot'][k], 2.5)) for k in r['point']},
                     **{f'{k}_high': float(np.percentile(r['boot'][k], 97.5)) for k in r['point']},
                     'frac_no_adyacente_2_4': float(nonadj / err24),
                     **{f'4clases_{k}': v for k, v in full.items()}})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'tabla_r4.csv', index=False)
    t = table.set_index('modelo')
    for name in ('base', 'frank_hall'):
        r = t.loc[name]
        lines.append(f'- **{name}**: BA 2–4 {r.ba_2_4:.4f} ({r.ba_2_4_low:.4f}–{r.ba_2_4_high:.4f}); accuracy 2–4 '
                     f'{r.accuracy_2_4:.3f}; kappa {r.kappa_2_4:.3f}; kappa cuadrático {r.kappa_cuadratico_2_4:.3f}; '
                     f'fracción de errores no adyacentes {r.frac_no_adyacente_2_4:.3f}. 4 clases: BA '
                     f'{r["4clases_balanced_accuracy"]:.3f}, muertes {int(r["4clases_muertes_acertadas"])}/12 marcando '
                     f'{int(r["4clases_casos_marcados"])}, AUC macro {r["4clases_auc_macro"]:.3f}, '
                     f'log-loss {r["4clases_log_loss"]:.3f}.')
    lines += ['', f'ΔBA 2–4 (Frank–Hall − base) = {d0:+.4f}, IC 95 % bootstrap pareado {lo:+.4f} a {hi:+.4f}, '
              f'p = {p_boot:.4f}. McNemar: {b10} a favor / {b01} en contra, p = {p_mc:.2g}.', '',
              ('**PASA** el cribado → etapa 2 (validación anidada, 12 celdas, Holm sobre 4).' if passed else
               '**NO PASA** el cribado: se detiene. Cinco candidatos probados; ninguno mejoró.'),
              'Exploratorio y posterior (método elegido tras la Ronda 3); efecto de selección declarado.']
    pd.DataFrame(res['frank_hall']['conf'], index=CLS, columns=CLS).to_csv(OUT / 'confusion_frank_hall_2_4.csv')
    (OUT / 'decision_r4.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('Confusión Frank–Hall (filas verdad 2,3,4):\n', res['frank_hall']['conf'].astype(int))
    return passed, repro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(f, k) for f in (3, 2, 1) for k in ('frank_hall', 'control')]
    log(f'Ronda 4: {len(tasks)} ajustes, {args.workers} procesos')
    for kind, f, status, sec in Parallel(n_jobs=args.workers, return_as='generator_unordered')(
            delayed(run_fold)(f, k) for f, k in tasks):
        log(f'{kind} fold{f}: {status} ({sec:.0f} s)')
    evaluate()


if __name__ == '__main__':
    main()
