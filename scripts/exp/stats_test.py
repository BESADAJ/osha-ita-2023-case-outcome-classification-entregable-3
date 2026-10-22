"""Pruebas estadísticas del ensamble frente a los modelos del curso en la prueba (oct-dic)."""
import sys, json
from common import *
from scipy import stats
from osha_e3 import final as F, stats_tests as T, models as M

name = sys.argv[1] if len(sys.argv) > 1 else 'stack_full'
d = np.load(C.RESULTS / 'ensemble' / name / 'foldtest.npz')
p_ens, y = (d['p_caruana'] if 'p_caruana' in d.files else d['p']).astype(float), d['y']
test, yt = F.test_data('clf')
assert (yt == y).all()
groups = test.establishment_id.to_numpy()

mc = pd.read_csv(C.RESULTS / 'master_clf.csv')
refs = {'mejor BA anidada': mc.loc[mc.outer_recall_mean.idxmax(), 'combo'],
        'mejor AUC anidada': mc.loc[mc.outer_auc_mean.idxmax(), 'combo'],
        'mejor accuracy anidada': mc.loc[mc.outer_accuracy_mean.idxmax(), 'combo'],
        'mejor log-loss anidada': mc.loc[mc.outer_log_loss_mean.idxmin(), 'combo']}
P = {k: F.load('clf', c)[0].astype(float) for k, c in refs.items()}
P['un solo XGBoost con todas las variables'] = np.load(C.RESULTS / 'ensemble' / 'unico_todo' / 'foldtest.npz')['p'].astype(float)
P['mejor componente del ensamble (XGBoost, todo)'] = d['base_todo_te_xgb'].astype(float) if 'base_todo_te_xgb' in d.files else d['base_finos_te_xgb'].astype(float)
print('Referencias:', refs)

# --- Métricas por caso para el bootstrap pareado por establecimiento
codes, uniq = pd.factorize(groups)
G = len(uniq)
rng = np.random.default_rng(C.seed_for('bootstrap', 50))
W = [np.bincount(rng.integers(0, G, G), minlength=G)[codes] for _ in range(1000)]


def case_stats(p):
    pred = p.argmax(1)
    q = np.clip(p, 1e-12, 1); q = q / q.sum(1, keepdims=True)
    return pred, -np.log(q[np.arange(len(y)), y])


def wmetrics(pred, ll, w):
    acc = np.sum(w * (pred == y)) / w.sum()
    rec = [np.sum(w * ((pred == k) & (y == k))) / np.sum(w * (y == k)) for k in range(4)]
    return acc, np.mean(rec), np.mean(rec[1:]), np.sum(w * ll) / w.sum()


pe, le = case_stats(p_ens)
rows = []
for ref, p in P.items():
    pr, lr = case_stats(p)
    base = wmetrics(pe, le, np.ones(len(y))), wmetrics(pr, lr, np.ones(len(y)))
    boot = np.array([np.subtract(wmetrics(pe, le, w), wmetrics(pr, lr, w)) for w in W])
    for i, lab in enumerate(['accuracy', 'balanced acc. (4 clases)', 'balanced acc. (3 clases)', 'log-loss']):
        diff = base[0][i] - base[1][i]
        lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
        better = diff < 0 if lab == 'log-loss' else diff > 0
        p_boot = 2 * min((boot[:, i] <= 0).mean(), (boot[:, i] >= 0).mean())
        rows.append({'referencia': ref, 'métrica': lab, 'ensamble': base[0][i], 'referencia_valor': base[1][i],
                     'diferencia': diff, 'IC 95 % (bootstrap por establecimiento)': f'[{lo:.4f}, {hi:.4f}]',
                     'p bootstrap': max(p_boot, 1 / len(W)), 'mejor el ensamble': better})
    # McNemar (aciertos)
    a, b = ((pe == y) & (pr != y)).sum(), ((pe != y) & (pr == y)).sum()
    chi = (abs(a - b) - 1) ** 2 / (a + b)
    rows.append({'referencia': ref, 'métrica': 'McNemar (aciertos solo ensamble / solo referencia)',
                 'ensamble': a, 'referencia_valor': b, 'diferencia': a - b, 'IC 95 % (bootstrap por establecimiento)': '',
                 'p bootstrap': stats.chi2.sf(chi, 1), 'mejor el ensamble': a > b})
tab = pd.DataFrame(rows)
tab['p Holm'] = T.adjust(tab['p bootstrap'].to_numpy())
pd.set_option('display.width', 250)
print(tab.round(4).to_string())

# --- DeLong por clase frente a cada referencia
rows = []
for ref, p in P.items():
    for k in range(4):
        r = T.delong_test(y == k, p_ens[:, k], p[:, k])
        rows.append({'referencia': ref, 'clase': C.CLASS_NAMES[k], 'AUC ensamble': r['auc1'], 'AUC referencia': r['auc2'],
                     'diferencia': r['diff'], 'z': r['z'], 'p': r['p_value']})
dl = pd.DataFrame(rows); dl['p Holm'] = T.adjust(dl.p.to_numpy())
print(dl.round(4).to_string())

# --- Friedman/Nemenyi semanal: ensamble frente a la mejor combinación de cada modelo del curso
best = mc.loc[mc.groupby('model').outer_recall_mean.idxmax()].set_index('model')
week = pd.to_datetime(test.date).dt.to_period('W').astype(str).to_numpy()
weeks = sorted(set(week))
cols = {'Ensamble multivista': p_ens, **{M.PRETTY[m]: F.load('clf', best.loc[m, 'combo'])[0].astype(float) for m in best.index}}
for metric in ['accuracy', 'balanced']:
    Wm = pd.DataFrame({n: [((p[week == w].argmax(1) == y[week == w]).mean() if metric == 'accuracy'
                            else np.mean([((p[week == w].argmax(1) == k) & (y[week == w] == k)).sum() / max((y[week == w] == k).sum(), 1)
                                          for k in range(4) if (y[week == w] == k).any()]))
                           for w in weeks] for n, p in cols.items()}, index=weeks)
    fr = T.friedman(Wm.to_numpy()); cd = T.nemenyi_cd(fr['k'], fr['n_blocks'])
    ranks = pd.Series(fr['avg_ranks'], index=Wm.columns).sort_values()
    print(f'\nFriedman ({metric} semanal): chi2={fr["chi2"]:.1f} p={fr["p_value"]:.2g} CD={cd:.2f}')
    print(ranks.round(2).to_dict())
    print('Cliff ensamble vs cada modelo:', {c: round(T.cliffs_delta(Wm['Ensamble multivista'], Wm[c]), 3) for c in Wm.columns[1:]})
    Wm.to_csv(C.RESULTS / 'ensemble' / name / f'semanal_{metric}.csv')
tab.to_csv(C.RESULTS / 'ensemble' / name / 'pruebas_bootstrap.csv', index=False)
dl.to_csv(C.RESULTS / 'ensemble' / name / 'pruebas_delong.csv', index=False)
