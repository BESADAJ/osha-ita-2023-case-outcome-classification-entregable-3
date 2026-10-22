"""Combinadores alternativos aprendidos solo con las predicciones fuera de pliegue del entrenamiento.

Se evalúan en la prueba usando las predicciones de los modelos base reajustados con todo el
entrenamiento (las mismas que usa el meta-modelo original).
"""
import sys
from common import *
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

v = sys.argv[1] if len(sys.argv) > 1 else 'stack_v2'
oof = np.load(C.RESULTS / 'ensemble' / v / 'oof_train.npz')
te = np.load(C.RESULTS / 'ensemble' / v / 'foldtest.npz')
names = [k for k in oof.files if k != 'y']
ytr, yte = oof['y'], te['y']
O = {n: np.clip(oof[n].astype(float), 1e-6, 1) for n in names}
Tt = {n: np.clip(te['base_' + n].astype(float), 1e-6, 1) for n in names}


def ll(y, p):
    p = np.clip(p, 1e-12, 1); p = p / p.sum(1, keepdims=True)
    return -np.mean(np.log(p[np.arange(len(y)), y]))


def show(lab, p):
    m = report(yte, p); m.update(per_class(yte, p))
    print(lab.ljust(42), {k: round(m[k], 4) for k in ['accuracy', 'ba3', 'f1', 'aucF', 'log_loss', 'ece']})


print('OOF log-loss por modelo:', {n: round(ll(ytr, O[n]), 4) for n in names})
show('referencia: todo_te_xgb solo', Tt['todo_te_xgb'] if 'todo_te_xgb' in Tt else Tt[names[0]])
# 1) meta logístico con distintos C (sobre log-probas)
for Cc in [0.003, 0.01, 0.03, 0.1, 1.0]:
    meta = LogisticRegression(C=Cc, max_iter=3000).fit(np.hstack([np.log(O[n]) for n in names]), ytr)
    show(f'meta logístico C={Cc}', meta.predict_proba(np.hstack([np.log(Tt[n]) for n in names])))


# 2) selección de ensamble de Caruana (con reemplazo) minimizando log-loss OOF
def caruana(steps=40, init=3):
    order = sorted(names, key=lambda n: ll(ytr, O[n]))
    chosen = order[:init]
    for _ in range(steps):
        cur = np.mean([O[c] for c in chosen], axis=0)
        best = min(names, key=lambda n: ll(ytr, (cur * len(chosen) + O[n]) / (len(chosen) + 1)))
        chosen.append(best)
    return pd.Series(chosen).value_counts() / len(chosen)
w = caruana()
print('pesos Caruana:', w.round(3).to_dict())
show('Caruana (promedio de probabilidades)', sum(w[n] * Tt[n] for n in w.index))


# 3) promedio geométrico ponderado óptimo (pesos ≥ 0 que suman 1) minimizando log-loss OOF
def geo(Pd, wts):
    z = sum(wi * np.log(Pd[n]) for wi, n in zip(wts, names))
    z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)
res = minimize(lambda a: ll(ytr, geo(O, np.exp(a) / np.exp(a).sum())), np.zeros(len(names)), method='Nelder-Mead',
               options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-6})
wg = np.exp(res.x) / np.exp(res.x).sum()
print('pesos geométricos:', dict(zip(names, np.round(wg, 3))))
show('promedio geométrico ponderado', geo(Tt, wg))
# 4) meta logístico + recalibración por clase (temperature) — sin cambios de pesos
