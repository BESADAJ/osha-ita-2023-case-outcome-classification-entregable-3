"""Análisis del experimento pareado (R3): tablas, pruebas de hipótesis, ablación y salidas para el libro y el dashboard.

1. Evalúa el modelo original en cada celda disponible (``original_eval.run_all``: TPE anidado + SECR + ablación).
2. Arma la tabla por pliegue (modelo × repetición × pliegue) con BA, F1, AUC, log-loss y tiempos.
3. Pruebas (todas sobre el bucle externo; la prueba oct-dic está bloqueada):
   * Friedman (Iman-Davenport) + Nemenyi con el original en el universo: (a) los 8 modelos en los 5 pliegues de la
     repetición 0; (b) los modelos con todas las celdas (15).
   * t corregida de Nadeau-Bengio, Wilcoxon de rangos con signo y prueba bayesiana con ROPE: original frente al mejor
     benchmark, con las celdas comunes.
   * Wilcoxon original frente a cada uno de los 7 benchmarks, con Holm.
   * Tamaño del efecto: δ de Cliff con IC bootstrap; IC BCa de la diferencia media.
4. Guarda ``results/pareado/resumen.json``, ``results/pareado/pliegues.csv``, ``dashboard_final/data/original_results.json``,
   ``dashboard_final/data/oof_original.csv.gz`` y las figuras del capítulo 20 en ``jbook/figures``.

Uso: python scripts/analyze_paired.py [n_procesos]
"""
import json
import sys
import warnings

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

from osha_e3 import config as C, evaluation as E, original_eval as OE, paired as PR, stats_tests as T  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

warnings.filterwarnings('ignore')
BENCH = ['svm', 'tree', 'logreg', 'xgboost', 'knn', 'random_forest', 'naive_bayes']
PRETTY = {'original': 'Original (SECR)', 'knn': 'KNN', 'naive_bayes': 'Naive Bayes', 'logreg': 'Reg. logística',
          'tree': 'Árbol', 'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'svm': 'SVM'}
ABL = {'original': 'SECR completo (original)', 'sin_regiones': 'SECR sin regiones (solo pesos globales)',
       'sin_prior': 'SECR sin corrección por prior (α = 0)', 'caruana_G': 'Caruana global (modelo G)',
       'stacking_lineal': 'Stacking lineal (logística multinomial)', 'pesos_error_global': 'Pesos por error global',
       'promedio_simple': 'Promedio simple', 'mejor_base': 'Mejor modelo base individual'}
FIG = C.ROOT / 'jbook' / 'figures'
DASH = C.ROOT / 'dashboard_final' / 'data'


def _eval_cell(rep, fold):
    X, _, _ = PR.paired_data()
    d = [c for c in PR.load('original') if c['rep'] == rep and c['fold'] == fold][0]
    path = OE.OUT / f'r{rep}_f{fold}.json'
    if path.exists():
        return
    res = OE.evaluate_cell(d, X)
    rec = {'rep': rep, 'fold': fold, 'params': res['params'], 'inner': res['inner'], 'history': res['history'],
           'names': res['names'], 'best_single': res['best_single'], 'weights_global': res['weights_global'].tolist(),
           'weights_region': {str(k): v.tolist() for k, v in res['weights_region'].items()},
           'fw_loss': res['fw_loss'].tolist(), 'fw_gap': res['fw_gap'].tolist(), 'tune_s': res['tune_s'],
           'fit_s': res['fit_s'], 'pred_s': res['pred_s'], 'n_test': int(len(res['y'])),
           'metrics': {k: OE.metrics(res['y'], q, p) for k, (q, p) in res['variants'].items()}}
    OE.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OE.OUT / f'r{rep}_f{fold}.npz', y=res['y'], idx=d['idx'],
                        **{f'q_{k}': q.astype(np.float32) for k, (q, _) in res['variants'].items()},
                        **{f'pred_{k}': p for k, (_, p) in res['variants'].items()})
    path.write_text(json.dumps(rec, default=float), encoding='utf-8')


def fold_table():
    rows = []
    for m in BENCH:
        for d in PR.load(m):
            y, p = d['y'].astype(int), d['p'].astype(float)
            mm = E.classification_metrics(y, p)
            rows.append({'model': m, 'rep': d['rep'], 'fold': d['fold'], 'ba': mm['recall'], 'f1': mm['f1'],
                         'auc': mm['auc'], 'log_loss': mm['log_loss'], 'accuracy': mm['accuracy'],
                         'train_s': float(d['search_s'] + d['fit_s']), 'refit_s': float(d['fit_s']),
                         'ms_case': float(d['pred_s']) / len(y) * 1000, 'n_test': len(y),
                         'balance': str(d['balance']), 'optimizer': str(d['opt'])})
    abl = []
    for path in sorted(OE.OUT.glob('*.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        o = r['metrics']['original']
        rows.append({'model': 'original', 'rep': r['rep'], 'fold': r['fold'], 'ba': o['recall'], 'f1': o['f1'],
                     'auc': o['auc'], 'log_loss': o['log_loss'], 'accuracy': o['accuracy'],
                     'train_s': r['fit_s'] + r['tune_s'], 'refit_s': r['fit_s'], 'ms_case': r['pred_s'] / r['n_test'] * 1000,
                     'n_test': r['n_test'], 'balance': 'interno (α)', 'optimizer': 'bayes (TPE) del combinador'})
        for k, v in r['metrics'].items():
            abl.append({'variant': k, 'rep': r['rep'], 'fold': r['fold'], 'ba': v['recall'], 'log_loss': v['log_loss'],
                        'f1': v['f1'], 'auc': v['auc']})
    return pd.DataFrame(rows), pd.DataFrame(abl)


def nemenyi_p(r1, r2, k, n):
    z = abs(r1 - r2) / np.sqrt(k * (k + 1) / (6.0 * n))
    return float(stats.studentized_range.sf(z * np.sqrt(2), k, np.inf))


def friedman_block(f, models, reps):
    sub = f[f.model.isin(models) & f.rep.isin(reps)]
    piv = sub.pivot_table(index=['rep', 'fold'], columns='model', values='ba').dropna()
    piv = piv[models]
    fr = T.friedman(piv.to_numpy())
    cd = T.nemenyi_cd(len(models), len(piv))
    ranks = dict(zip(models, fr['avg_ranks']))
    return piv, fr, cd, ranks


def main(n_jobs):
    cells = [(c['rep'], c['fold']) for c in PR.load('original')]
    todo = [c for c in cells if not (OE.OUT / f'r{c[0]}_f{c[1]}.json').exists()]
    print(f'original: {len(cells)} celdas, {len(todo)} por evaluar', flush=True)
    Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(delayed(_eval_cell)(*c) for c in todo)

    f, abl = fold_table()
    f.to_csv(PR.OUT / 'pliegues.csv', index=False)
    abl.to_csv(PR.OUT / 'ablacion.csv', index=False)
    count = f.groupby('model').size()
    print(count.to_string(), flush=True)
    full = [m for m in ['original'] + BENCH if count.get(m, 0) == count.get('original', 0) and m in count]
    reps_full = sorted(f[f.model.eq('original')].rep.unique())
    rep0 = [m for m in ['original'] + BENCH if f[(f.model == m) & (f.rep == 0)].shape[0] == C.OUTER_FOLDS]

    # ---------------------------------------------------------------- tabla resumen
    summ = []
    for m, g in f.groupby('model'):
        lo, hi = T.bca_mean_ci(g.ba.to_numpy(), reps=5000) if len(g) > 2 else (np.nan, np.nan)
        g0 = g[g.rep == 0]
        summ.append({'model': m, 'n_pliegues': len(g), 'ba_mean': g.ba.mean(), 'ba_sd': g.ba.std(), 'ba_ci_lo': lo,
                     'ba_ci_hi': hi, 'ba_rep0_mean': g0.ba.mean(), 'f1_mean': g.f1.mean(), 'auc_mean': g.auc.mean(),
                     'log_loss_mean': g.log_loss.mean(), 'train_s_mean': g.train_s.mean(), 'ms_case_mean': g.ms_case.mean(),
                     'balance': g.balance.iloc[0], 'optimizer': g.optimizer.iloc[0]})
    summ = pd.DataFrame(summ).sort_values('ba_mean', ascending=False)
    summ.to_csv(PR.OUT / 'resumen_modelos.csv', index=False)
    print(summ.round(4).to_string(), flush=True)

    # ---------------------------------------------------------------- pruebas
    bench_full = [m for m in full if m != 'original']
    piv25, fr25, cd25, ranks25 = friedman_block(f, full, reps_full)
    piv5, fr5, cd5, ranks5 = friedman_block(f, rep0, [0])
    best = max(bench_full, key=lambda m: piv25[m].mean())
    diff = (piv25['original'] - piv25[best]).to_numpy()
    n_tr = float(np.mean(f[f.model == 'original'].n_test)) * (C.OUTER_FOLDS - 1)
    n_te = float(np.mean(f[f.model == 'original'].n_test))
    nb = T.nadeau_bengio(diff, n_tr, n_te)
    wil = T.wilcoxon_signed_rank(diff)
    bay = T.bayesian_signed_rank(diff, rope=0.01, samples=20000)
    delta, dlo, dhi = T.cliffs_delta_ci(piv25['original'].to_numpy(), piv25[best].to_numpy(), reps=3000)
    mlo, mhi = T.bca_mean_ci(diff, reps=5000)
    nem_p = nemenyi_p(ranks25['original'], ranks25[best], len(full), len(piv25))
    shapiro = stats.shapiro(diff)
    tests = [('Friedman (Iman-Davenport), universo de 15 pliegues (3 × 5)', 'Todos los modelos tienen el mismo rango medio',
              f'bloques = pares (rep., pliegue) independientes entre sí en la asignación; k = {len(full)}, N = {len(piv25)}',
              f"F = {fr25['iman_davenport_F']:.3f}", fr25['p_value_F'], 'global'),
             (f'Nemenyi post hoc: original vs {PRETTY[best]}', 'Igual rango medio',
              f'tras Friedman; CD = {cd25:.3f}', f"ΔR = {ranks25[best] - ranks25['original']:.3f}", nem_p, 'pareada'),
             (f't corregida de Nadeau-Bengio: original vs {PRETTY[best]}', 'Diferencia media de BA ≤ 0',
              f'normalidad de diferencias (Shapiro p = {shapiro.pvalue:.3f}); corrección n_test/n_train = {n_te / n_tr:.2f}',
              f"t({nb['df']}) = {nb['t']:.3f}", nb['p_one_sided'], 'pareada'),
             (f'Wilcoxon de rangos con signo: original vs {PRETTY[best]}', 'Mediana de la diferencia ≤ 0',
              'diferencias simétricas, pareadas por pliegue', f"W = {wil['W']:.1f}", wil['p_one_sided'], 'pareada')]
    raw = [t[4] for t in tests]
    adj = T.adjust(raw, 'holm')
    rows = []
    for (name, h0, sup, stat, p, kind), pa in zip(tests, adj):
        rows.append({'Prueba': name, 'H0': h0, 'Supuestos': sup, 'Estadístico': stat, 'p crudo': f'{p:.2e}',
                     'p ajustado (Holm, 4)': f'{pa:.2e}', 'Efecto': f'δ de Cliff = {delta:.3f} [{dlo:.3f}, {dhi:.3f}] ({T.cliffs_label(delta)})',
                     'Conclusión': 'Se rechaza H0' if pa < 0.05 else 'No se rechaza H0'})
    rows.append({'Prueba': f'Bayesiana de rangos con signo (ROPE ±0,01 BA): original vs {PRETTY[best]}',
                 'H0': '— (probabilidades posteriores)', 'Supuestos': 'proceso de Dirichlet, pseudo-observación en 0',
                 'Estadístico': f"P(original mejor) = {bay['p_right']:.3f}; P(ROPE) = {bay['p_rope']:.3f}; "
                                f"P(peor) = {bay['p_left']:.3f}", 'p crudo': '—', 'p ajustado (Holm, 4)': '—',
                 'Efecto': f'Δ BA media = {diff.mean():.4f}, IC BCa 95 % [{mlo:.4f}, {mhi:.4f}]',
                 'Conclusión': 'Original mejor con probabilidad ≥ 0,95' if bay['p_right'] >= 0.95 else 'No concluyente'})
    # Wilcoxon contra cada benchmark (Holm sobre 7)
    vs = []
    for m in BENCH:
        common = f[f.model.isin(['original', m])].pivot_table(index=['rep', 'fold'], columns='model', values='ba').dropna()
        if m not in common or len(common) < 5:
            continue
        dd = (common['original'] - common[m]).to_numpy()
        w = T.wilcoxon_signed_rank(dd)
        cdl = T.cliffs_delta(common['original'].to_numpy(), common[m].to_numpy())
        vs.append({'benchmark': m, 'n_pliegues': len(dd), 'ba_original': common['original'].mean(),
                   'ba_benchmark': common[m].mean(), 'diff_mean': dd.mean(), 'gana_en': int((dd > 0).sum()),
                   'W': w['W'], 'p_crudo': w['p_one_sided'], 'cliff': cdl, 'cliff_label': T.cliffs_label(cdl)})
    vs = pd.DataFrame(vs)
    vs['p_holm'] = T.adjust(vs.p_crudo.to_numpy(), 'holm')
    vs.to_csv(PR.OUT / 'wilcoxon_vs_cada.csv', index=False)
    print(vs.round(4).to_string(), flush=True)

    # ---------------------------------------------------------------- veredicto (guía §6.1)
    means = summ.set_index('model')
    beats_all = all(means.loc['original', 'ba_mean'] > means.loc[m, 'ba_mean'] for m in means.index if m != 'original')
    beats_rep0 = all(means.loc['original', 'ba_rep0_mean'] > means.loc[m, 'ba_rep0_mean'] for m in means.index if m != 'original')
    n_sig = int(sum(pa < 0.05 for pa in adj))
    global_sig = adj[0] < 0.05
    effect_ok = abs(delta) >= 0.147
    superior = bool(beats_all and beats_rep0 and n_sig >= 3 and global_sig and effect_ok and diff.mean() > 0)
    verdict = (f"Veredicto R3: {'SUPERIOR' if superior else 'NO SUPERIOR'}. BA media del original "
               f"{means.loc['original', 'ba_mean']:.3f} frente a {means.loc[best, 'ba_mean']:.3f} del mejor benchmark pareado "
               f"({PRETTY[best]}); {n_sig} de 4 pruebas frecuentistas rechazan H0 con p ajustado por Holm < 0,05 "
               f"(la global de Friedman {'sí' if global_sig else 'no'}); δ de Cliff = {delta:.3f} ({T.cliffs_label(delta)}). "
               f"Media mayor que la de los 7 benchmarks: {'sí' if beats_all and beats_rep0 else 'no'}.")
    print(verdict, flush=True)

    # ---------------------------------------------------------------- ablación
    a = abl.pivot_table(index=['rep', 'fold'], columns='variant', values='ba')
    abl_rows = []
    for k in ABL:
        if k not in a:
            continue
        dd = (a['original'] - a[k]).to_numpy()
        p = T.wilcoxon_signed_rank(dd)['p_one_sided'] if k != 'original' and np.any(dd != 0) else np.nan
        ll = abl[abl.variant == k].log_loss.mean()
        abl_rows.append({'variante': ABL[k], 'ba_mean': a[k].mean(), 'ba_sd': a[k].std(), 'log_loss': ll,
                         'diff_vs_original': -dd.mean(), 'p_wilcoxon': p})
    abl_t = pd.DataFrame(abl_rows)
    ok = abl_t.p_wilcoxon.notna()
    abl_t.loc[ok, 'p_holm'] = T.adjust(abl_t.loc[ok, 'p_wilcoxon'].to_numpy(), 'holm')
    abl_t.to_csv(PR.OUT / 'ablacion_resumen.csv', index=False)
    print(abl_t.round(4).to_string(), flush=True)

    # ---------------------------------------------------------------- figuras
    FIG.mkdir(exist_ok=True)
    for tag, names, rk, cd, n in [('25', full, ranks25, cd25, len(piv25)), ('rep0', rep0, ranks5, cd5, len(piv5))]:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        T.cd_diagram([PRETTY[m] for m in names], np.array([rk[m] for m in names]), cd, ax,
                     title=f'Diagrama de diferencia crítica (Nemenyi, α = 0,05), BA, N = {n} pliegues')
        fig.savefig(FIG / f'e3_cd_pareado_{tag}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    order = summ.model.tolist()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([f[f.model == m].ba for m in order], labels=[PRETTY[m] for m in order], showmeans=True)
    ax.set_ylabel('Balanced accuracy por pliegue externo')
    ax.set_title('BA por pliegue en el experimento pareado (mismas variables y particiones)')
    plt.xticks(rotation=20)
    fig.savefig(FIG / 'e3_ba_pliegues_pareado.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    r0 = json.loads((OE.OUT / 'r0_f0.json').read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(r0['fw_loss'], marker='o', ms=3, label='Log-loss F(wᵗ)')
    ax2 = ax.twinx()
    ax2.semilogy(np.maximum(r0['fw_gap'], 1e-12), color='C3', marker='s', ms=3, label='Brecha Gᵗ')
    ax.set_xlabel('Iteración t')
    ax.set_ylabel('Log-loss fuera de pliegue')
    ax2.set_ylabel('Brecha de Frank-Wolfe (log)')
    ax.set_title('Convergencia de Frank-Wolfe (pesos globales, rep. 0, pliegue 0)')
    fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.85))
    fig.savefig(FIG / 'e3_convergencia_fw.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------------------------------------------------------------- salidas
    out = {'superior': superior, 'verdict': verdict, 'best_benchmark': best, 'n_cells': count.to_dict(),
           'universe_25': full, 'universe_rep0': rep0,
           'friedman25': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in fr25.items()}, 'cd25': cd25,
           'friedman_rep0': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in fr5.items()}, 'cd_rep0': cd5,
           'nadeau_bengio': nb, 'wilcoxon_best': wil, 'bayes': bay, 'cliff': [delta, dlo, dhi], 'diff_ci_bca': [mlo, mhi],
           'shapiro_p': float(shapiro.pvalue), 'holm_adjusted': adj.tolist(), 'raw': raw, 'n_tests_holm': len(raw),
           'tests': rows, 'vs_each': vs.to_dict('records'), 'ablation': abl_t.to_dict('records'),
           'summary_raw': summ.to_dict('records')}
    (PR.OUT / 'resumen.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    dash_summary = [{'Modelo': PRETTY[r.model], 'Balanceo': r.balance, 'Optimizador': r.optimizer,
                     'Pliegues': int(r.n_pliegues), 'BA (media ± d.e.)': f'{r.ba_mean:.3f} ± {r.ba_sd:.3f}',
                     'IC BCa 95 % de la media': f'[{r.ba_ci_lo:.3f}, {r.ba_ci_hi:.3f}]',
                     'F1 macro': round(r.f1_mean, 3), 'AUC': round(r.auc_mean, 3),
                     'Entrenamiento (s/pliegue)': round(r.train_s_mean, 1), 'Inferencia (ms/caso)': round(r.ms_case_mean, 3)}
                    for r in summ.itertuples()]
    dash = {'superior': superior, 'verdict': verdict, 'summary': dash_summary, 'tests': rows,
            'cd': {'names': [PRETTY[m] for m in full], 'ranks': [ranks25[m] for m in full], 'cd': cd25,
                   'insight': f"Con N = {len(piv25)} pliegues y k = {len(full)} modelos, la diferencia crítica de Nemenyi es "
                              f"{cd25:.2f}. Rango medio del original: {ranks25['original']:.2f}; del mejor benchmark "
                              f"({PRETTY[best]}): {ranks25[best]:.2f}. Friedman p = {fr25['p_value_F']:.2e}."},
            'convergence': {'loss': r0['fw_loss'], 'gap': r0['fw_gap']},
            'ablation': abl_t.to_dict('records')}
    (DASH / 'original_results.json').write_text(json.dumps(dash, ensure_ascii=False, indent=1, default=float),
                                                encoding='utf-8')
    oo = []
    for fo in range(C.OUTER_FOLDS):
        z = np.load(OE.OUT / f'r0_f{fo}.npz')
        df = pd.DataFrame(z['q_original'].astype(float).round(5), columns=[f'p{k}' for k in range(4)])
        df['y'], df['fold'], df['idx'] = z['y'], fo, z['idx']
        oo.append(df)
    pd.concat(oo).sort_values('idx').drop(columns='idx').to_csv(DASH / 'oof_original.csv.gz', index=False, compression='gzip')
    print('listo', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
