"""Resume las variantes del ensamble frente a los 112 modelos del curso (pliegues y prueba)."""
import sys, glob
from common import *
from osha_e3 import final as F, stats_tests as T

OUT = C.RESULTS / 'ensemble'
names = sys.argv[1:] or sorted(p.name for p in OUT.iterdir() if p.is_dir())


def metrics_all(y, p, rule='argmax'):
    m = report(y, p, p.argmax(1) if rule == 'argmax' else None)
    m.update(per_class(y, p))
    return m


test, yt = F.test_data('clf')
course = {}
for f in sorted((C.RESULTS / 'final' / 'clf').glob('*.npz')):
    combo = f.stem.replace('__', '|')
    course[combo] = metrics_all(yt, np.load(f)['out'].astype(float))
cdf = pd.DataFrame(course).T
best_course = {'accuracy': cdf.accuracy.max(), 'recall': cdf.recall.max(), 'f1': cdf.f1.max(), 'auc': cdf.auc.max(),
               'aucF': cdf.aucF.max(), 'log_loss': cdf.log_loss.min(), 'll3': cdf.ll3.min(), 'ba3': cdf.ba3.max(),
               'brier': cdf.brier.min(), 'ece': cdf.ece.min()}
rows = [{'modelo': 'Mejor de los 112 del curso (por métrica)', **best_course}]
for n in names:
    z = OUT / n / 'foldtest.npz'
    if z.exists():
        d = np.load(z)
        rows.append({'modelo': n, **{k: v for k, v in metrics_all(d['y'], d['p']).items() if k in best_course}})
        if 'p_caruana' in d.files:
            rows.append({'modelo': n + ' [Caruana]', **{k: v for k, v in metrics_all(d['y'], d['p_caruana'].astype(float)).items() if k in best_course}})
        for k in d.files:
            if k.startswith('base_') and n == names[0]:
                rows.append({'modelo': '  base ' + k[5:], **{kk: v for kk, v in metrics_all(d['y'], d[k].astype(float)).items() if kk in best_course}})
print('=== PRUEBA (oct-dic, 127.391 casos) ===')
print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())

f = pd.read_csv(C.RESULTS / 'outer_folds_clf.csv')
print('\n=== PLIEGUES EXTERNOS (media de 5) ===')
rows = []
for col, lab, better in [('outer_recall', 'BA', max), ('outer_auc', 'AUC', max), ('outer_accuracy', 'accuracy', max),
                         ('outer_f1', 'F1', max), ('outer_log_loss', 'log-loss', min)]:
    per = f.groupby(['model', 'balance', 'optimizer'])[col].mean()
    rows.append({'métrica': lab, 'mejor combinación del curso': per.idxmax() if better is max else per.idxmin(),
                 'valor del curso': per.max() if better is max else per.min()})
summ = pd.DataFrame(rows).set_index('métrica')
for n in names:
    vals = {'BA': [], 'AUC': [], 'accuracy': [], 'F1': [], 'log-loss': [], 'aucF': [], 'ba3': []}
    ok = True
    for k in range(5):
        z = OUT / n / f'fold{k}.npz'
        if not z.exists():
            ok = False; break
        d = np.load(z); m = metrics_all(d['y'], d['p'].astype(float))
        for a, b in [('BA', 'recall'), ('AUC', 'auc'), ('accuracy', 'accuracy'), ('F1', 'f1'), ('log-loss', 'log_loss'), ('aucF', 'aucF'), ('ba3', 'ba3')]:
            vals[a].append(m[b])
    if ok:
        summ[n] = [np.mean(vals[k]) for k in ['BA', 'AUC', 'accuracy', 'F1', 'log-loss']]
        if 'p_caruana' in np.load(OUT / n / 'fold0.npz').files:
            mm = [metrics_all(np.load(OUT / n / f'fold{k}.npz')['y'], np.load(OUT / n / f'fold{k}.npz')['p_caruana'].astype(float)) for k in range(5)]
            summ[n + ' [Caruana]'] = [np.mean([m[b] for m in mm]) for b in ['recall', 'auc', 'accuracy', 'f1', 'log_loss']]
            print(n, '[Caruana] aucF', round(np.mean([m['aucF'] for m in mm]), 4), 'ba3', round(np.mean([m['ba3'] for m in mm]), 4))
        print(n, 'aucF', np.round(np.mean(vals['aucF']), 4), 'ba3', np.round(np.mean(vals['ba3']), 4), '| BA por pliegue', np.round(vals['BA'], 3))
print(summ.round(4).to_string())
