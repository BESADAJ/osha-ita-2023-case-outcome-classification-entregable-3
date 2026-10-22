"""Ronda 3 (exploratoria): subproblema de las clases 2 a 4 con predicciones guardadas (regla: results/original_r3/regla_r3.md).

No reajusta nada. Por modelo: descarta P(Fallecimiento), renormaliza P(2..4), predice el argmax entre 2, 3 y 4 y
calcula las métricas sobre los casos con clase verdadera 2, 3 o 4 (3 pliegues juntos). El bootstrap por establecimiento
se hace sobre matrices de confusión por establecimiento: las mismas 2.000 remuestras para todos (pareado).

python ronda3_2_4.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent / 'results'
OUT = ROOT / 'original_r3'
B, SEED, CLS = 2000, 42, np.array([2, 3, 4])
CANDIDATES = {'pt_bagging': ROOT / 'original' / 'cribado' / 'pt_bagging',
              'multirandbal': ROOT / 'original' / 'cribado' / 'multirandbal',
              'hierarchical': ROOT / 'original' / 'cribado' / 'hierarchical',
              'hafl_xgboost': ROOT / 'original_r2' / 'cribado' / 'hafl_xgboost'}
ABLATIONS = {'hafl_xgboost_gamma0': ROOT / 'original_r2' / 'cribado' / 'hafl_xgboost_gamma0'}
COURSE = ('knn', 'naive_bayes', 'logreg', 'tree', 'random_forest', 'xgboost', 'svm')


def load(path):
    zs = [np.load(path / f'fold{k}_pred.npz') for k in (1, 2, 3)]
    return {key: np.concatenate([z[key] for z in zs]) for key in ('proba', 'y', 'establishment_id', 'case_id')}


def sources():
    out = {}
    for d in sorted((ROOT / 'runs').iterdir()):
        if all((d / f'fold{k}_pred.npz').exists() for k in (1, 2, 3)):
            out[d.name] = ('base' if d.name.split('__')[0] in COURSE else 'referencia', d)
    for name, d in CANDIDATES.items():
        out[name] = ('candidato', d)
    for name, d in ABLATIONS.items():
        out[name] = ('ablación', d)
    phase_b = ROOT / 'original' / 'runs'          # Fase B: solo optimizadores con los 3 pliegues
    for d in sorted(phase_b.glob('pt_bagging__none__*')):
        if all((d / f'fold{k}.json').exists() for k in (1, 2, 3)):
            out[f'faseB_{d.name}'] = ('candidato', d)
    return out


def from_confusion(c):
    """Métricas a partir de matrices de confusión (..., 3, 3): filas = verdad 2,3,4; columnas = predicción."""
    c = c.astype(float)
    n = c.sum(axis=(-2, -1))
    rows, cols = c.sum(axis=-1), c.sum(axis=-2)
    diag = np.diagonal(c, axis1=-2, axis2=-1)
    ba = np.mean(diag / rows, axis=-1)
    acc = diag.sum(axis=-1) / n
    pe = (rows * cols).sum(axis=-1) / n ** 2
    kappa = (acc - pe) / (1 - pe)
    i, j = np.meshgrid(np.arange(3), np.arange(3), indexing='ij')
    w = (i - j) ** 2 / 4.0
    expected = rows[..., :, None] * cols[..., None, :] / n[..., None, None]
    qwk = 1 - (w * c).sum(axis=(-2, -1)) / (w * expected).sum(axis=(-2, -1))
    return {'ba_2_4': ba, 'accuracy': acc, 'kappa': kappa, 'kappa_cuadratico': qwk}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = sources()
    first = load(next(iter(src.values()))[1])
    mask = np.isin(first['y'], CLS)
    y = first['y'][mask].astype(int)
    case = first['case_id'][mask]
    est = first['establishment_id'][mask]
    g_idx, groups = pd.factorize(est)
    G = len(groups)
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(G, size=G), minlength=G) for _ in range(B)]).astype(np.float32)  # B × G
    yi = np.searchsorted(CLS, y)

    preds, rows, boots = {}, [], {}
    for name, (kind, path) in src.items():
        z = load(path)
        assert np.array_equal(z['case_id'][mask], case), f'{name}: casos distintos'
        p3 = z['proba'][mask][:, 1:].astype(float)
        s = p3.sum(axis=1, keepdims=True)
        p3 = np.where(s > 0, p3 / np.where(s > 0, s, 1), 1 / 3)
        pi = p3.argmax(axis=1)
        preds[name] = pi
        cell = g_idx * 9 + yi * 3 + pi                       # confusión por establecimiento
        C = np.bincount(cell, minlength=G * 9).reshape(G, 9).astype(np.float32)
        conf_b = (W @ C).reshape(B, 3, 3)
        conf = C.sum(axis=0).reshape(3, 3)
        point = from_confusion(conf)
        boot = from_confusion(conf_b)
        boots[name] = boot['ba_2_4']
        row = {'tipo': kind, 'modelo': name, 'n': len(y)}
        for k in point:
            row[k] = float(point[k])
            row[f'{k}_low'], row[f'{k}_high'] = (float(v) for v in np.percentile(boot[k], [2.5, 97.5]))
        row['auc_macro_2_4'] = float(roc_auc_score(y, p3, labels=CLS, multi_class='ovr', average='macro'))
        row['log_loss_2_4'] = float(log_loss(y, np.clip(p3, 1e-15, 1), labels=CLS))
        rows.append(row)
        if name in ('dummy__none__grid',):
            continue
    table = pd.DataFrame(rows).sort_values('ba_2_4', ascending=False)
    base = table[table.tipo.eq('base')].iloc[0]
    best = base.modelo
    table['mejor_base'] = table.modelo.eq(best)

    # Comparaciones pareadas de los candidatos contra el mejor base.
    correct_best = preds[best] == yi
    comps = []
    for name in [n for n, (k, _) in src.items() if k in ('candidato', 'ablación')]:
        diff = boots[name] - boots[best]
        d0 = float(table.set_index('modelo').loc[name, 'ba_2_4'] - base.ba_2_4)
        p_boot = float(min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))
        c = preds[name] == yi
        b01, b10 = int((~c & correct_best).sum()), int((c & ~correct_best).sum())
        p_mc = float(binomtest(b10, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0
        comps.append({'modelo': name, 'tipo': src[name][0], 'delta_ba_2_4': d0,
                      'delta_low': float(np.percentile(diff, 2.5)), 'delta_high': float(np.percentile(diff, 97.5)),
                      'p_bootstrap': p_boot, 'mcnemar_candidato_acierta_base_falla': b10,
                      'mcnemar_base_acierta_candidato_falla': b01, 'p_mcnemar': p_mc, '_diff': diff})
    cand = [c for c in comps if c['tipo'] == 'candidato']
    m = len(cand)
    for key in ('p_bootstrap', 'p_mcnemar'):
        order = sorted(range(m), key=lambda i: cand[i][key])
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * cand[i][key]))
            cand[i][f'{key}_holm'] = running
            if key == 'p_bootstrap':
                level = 0.05 / (m - rank)
                lo, hi = np.percentile(cand[i]['_diff'], [100 * level / 2, 100 * (1 - level / 2)])
                cand[i]['nivel_holm'] = 1 - level
                cand[i]['delta_low_holm'], cand[i]['delta_high_holm'] = float(lo), float(hi)
    for c in cand:
        c['mejora'] = bool(c['delta_ba_2_4'] > 0 and c['delta_low_holm'] > 0)
    comp = pd.DataFrame([{k: v for k, v in c.items() if k != '_diff'} for c in comps])

    # Estructura ordinal del mejor base.
    pb = preds[best]
    conf = pd.crosstab(pd.Series(y, name='verdad'), pd.Series(CLS[pb], name='prediccion')).reindex(
        index=CLS, columns=CLS, fill_value=0)
    conf.to_csv(OUT / 'confusion_mejor_base_2_4.csv')
    cell = g_idx * 9 + yi * 3 + pb
    Cb = (W @ np.bincount(cell, minlength=G * 9).reshape(G, 9).astype(np.float32)).reshape(B, 3, 3)
    c0 = conf.to_numpy()
    nonadj, err24 = c0[0, 2] + c0[2, 0], c0[0, 1] + c0[0, 2] + c0[2, 0] + c0[2, 1]
    frac = nonadj / err24
    frac_b = (Cb[:, 0, 2] + Cb[:, 2, 0]) / (Cb[:, 0, 1] + Cb[:, 0, 2] + Cb[:, 2, 0] + Cb[:, 2, 1])
    f_lo, f_hi = np.percentile(frac_b, [2.5, 97.5])
    errors = c0.sum() - np.trace(c0)
    adj = c0[0, 1] + c0[1, 0] + c0[1, 2] + c0[2, 1]

    table.to_csv(OUT / 'tabla_2_4.csv', index=False)
    comp.to_csv(OUT / 'comparaciones_2_4.csv', index=False)
    pretty = lambda r, k: f'{r[k]:.3f} ({r[k + "_low"]:.3f}–{r[k + "_high"]:.3f})'
    t = table.set_index('modelo')
    lines = ['# Ronda 3 (exploratoria): subproblema de las clases 2 a 4', '',
             f'Generado {datetime.now():%Y-%m-%d %H:%M}. Regla: `regla_r3.md` (escrita antes de calcular). '
             f'{len(y)} casos (clases 2–4, 3 pliegues), {G} establecimientos, {B} remuestras por establecimiento.',
             f'Exploratorio, decidido después de ver que Fallecimiento no es evaluable ni mejorable con 12 muertes.', '',
             f'**Mejor base:** `{best}` — BA 2–4 {pretty(t.loc[best], "ba_2_4")}, accuracy {pretty(t.loc[best], "accuracy")}, '
             f'kappa {pretty(t.loc[best], "kappa")}, kappa cuadrático {pretty(t.loc[best], "kappa_cuadratico")}, '
             f'AUC macro {t.loc[best, "auc_macro_2_4"]:.3f}, log-loss {t.loc[best, "log_loss_2_4"]:.3f}. '
             'Sesgo por elegir el máximo de 108: pequeño con ~61.600 casos, pero existe.', '',
             '| Modelo | BA 2–4 | Δ vs mejor base (IC 95 %) | IC al nivel de Holm | p bootstrap (Holm) | McNemar a favor / en contra | p McNemar (Holm) | ¿Mejora? |',
             '|---|---|---|---|---|---|---|---|']
    for c in comps:
        holm = (f'{c["delta_low_holm"]:+.4f} a {c["delta_high_holm"]:+.4f} ({100 * c["nivel_holm"]:.2f} %)'
                if 'nivel_holm' in c else 'no cuenta (ablación)')
        lines.append(f'| {c["modelo"]} | {t.loc[c["modelo"], "ba_2_4"]:.3f} | {c["delta_ba_2_4"]:+.4f} '
                     f'({c["delta_low"]:+.4f} a {c["delta_high"]:+.4f}) | {holm} | '
                     f'{c["p_bootstrap"]:.4f} ({c.get("p_bootstrap_holm", float("nan")):.4f}) | '
                     f'{c["mcnemar_candidato_acierta_base_falla"]} / {c["mcnemar_base_acierta_candidato_falla"]} | '
                     f'{c["p_mcnemar"]:.2g} ({c.get("p_mcnemar_holm", float("nan")):.2g}) | '
                     f'{"sí" if c.get("mejora") else "no"} |')
    any_better = any(c.get('mejora') for c in cand)
    lines += ['', ('Algún candidato **mejora** (exploratorio): ' + ', '.join(c['modelo'] for c in cand if c['mejora']))
              if any_better else '**Ningún candidato mejora** al mejor base en el subproblema 2–4.', '',
              '## Estructura ordinal (mejor base)',
              f'Errores: {errors} de {c0.sum()}; adyacentes (2↔3, 3↔4) {adj} ({adj / errors:.1%}); '
              f'no adyacentes (2↔4) {nonadj} ({nonadj / errors:.1%}).',
              f'Entre los errores de los casos de clase 2 y 4, fracción que cae en la clase no adyacente: {frac:.3f} '
              f'(IC 95 % por establecimiento {f_lo:.3f}–{f_hi:.3f}; sin orden sería ~0,5).',
              ('→ **Hay estructura ordinal** (intervalo entero por debajo de 0,5).' if f_hi < 0.5 else
               '→ **No hay evidencia de estructura ordinal** con el criterio fijado (el intervalo no queda por debajo de 0,5).'),
              '', 'Matriz de confusión del mejor base (filas = verdad, columnas = predicción): `confusion_mejor_base_2_4.csv`.',
              '', 'Fase B de PT-bagging: no entra (ningún optimizador tenía los 3 pliegues al calcular).'
              if not any(n.startswith('faseB_') for n in src) else '']
    (OUT / 'decision_r3.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(conf.to_string())


if __name__ == '__main__':
    main()
