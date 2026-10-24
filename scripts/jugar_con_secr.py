"""Experimentar con SECR en segundos, sin reentrenar los 16 modelos base (sustentación y estudio).

Reutiliza las predicciones ya guardadas de cada pliegue del experimento pareado (``results/pareado/original/*.npz``):
ajusta SECR con los hiperparámetros que se le pasen sobre las predicciones *internas* del pliegue y mide la BA en el
pliegue externo. Imprime la BA de cada pliegue, la media y la comparación con lo que eligió la TPE y con KNN.

Es un análisis exploratorio: el modelo reportado es el elegido por la TPE interna; usar estas cifras para elegir
hiperparámetros sería mirar los pliegues del veredicto (capítulo 19.9).

Ejemplos:
    python scripts/jugar_con_secr.py                                  # configuración por defecto
    python scripts/jugar_con_secr.py --gate none                      # sin regiones (pesos globales)
    python scripts/jugar_con_secr.py --gate naics2 --lam 0            # regiones por sector, sin contracción
    python scripts/jugar_con_secr.py --alpha 0                        # decisión argmax (sin corrección por prior)
    python scripts/jugar_con_secr.py --steps 1                        # casi solo el mejor modelo individual
    python scripts/jugar_con_secr.py --rep 0 --fold 1 --pesos         # un pliegue y los pesos aprendidos
"""
import argparse
import json
import warnings

import _bootstrap  # noqa: F401
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from osha_e3 import combinador as CB, original_eval as OE, paired as PR

warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('--gate', default='type_of_incident', choices=list(CB.GATES))
ap.add_argument('--lam', type=float, default=10.0)
ap.add_argument('--steps', type=int, default=15)
ap.add_argument('--min-region', type=int, default=200)
ap.add_argument('--alpha', type=float, default=0.8)
ap.add_argument('--rep', type=int)
ap.add_argument('--fold', type=int)
ap.add_argument('--pesos', action='store_true', help='mostrar los pesos globales y por región')
a = ap.parse_args()
params = {'gate': a.gate, 'lam': a.lam, 'n_steps': a.steps, 'min_region': a.min_region, 'alpha': a.alpha}

X, _, _ = PR.paired_data()
knn = {(d['rep'], d['fold']): d for d in PR.load('knn')}
rows = []
for d in PR.load('original'):
    if (a.rep is not None and d['rep'] != a.rep) or (a.fold is not None and d['fold'] != a.fold):
        continue
    names, P_in, P_out = OE.stacks(d)
    X_in, X_out = X.iloc[d['idx_train']].reset_index(drop=True), X.iloc[d['idx']].reset_index(drop=True)
    m, q = OE._fit_predict(params, P_in, d['y_train'].astype(int), X_in, P_out, X_out)
    y = d['y'].astype(int)
    ba = balanced_accuracy_score(y, m.decide(q))
    tpe = json.loads((OE.OUT / f"r{d['rep']}_f{d['fold']}.json").read_text(encoding='utf-8'))['metrics']['original']['recall']
    k = knn[(d['rep'], d['fold'])]
    ba_knn = balanced_accuracy_score(k['y'], np.asarray(k['p']).argmax(1))
    rows.append((d['rep'], d['fold'], ba, tpe, ba_knn, int((y == 0).sum())))
    print(f"rep {d['rep']} pliegue {d['fold']}: BA = {ba:.3f} | con la TPE {tpe:.3f} | KNN {ba_knn:.3f} | "
          f"fallecimientos {int((y == 0).sum())}")
    if a.pesos:
        show = lambda w: {n: round(float(x), 3) for n, x in zip(names, w) if x > 0.01}  # noqa: E731
        print('   pesos globales:', show(m.weights_global_))
        for g, w in m.weights_region_.items():
            print(f'   región {g}:', show(w))
r = np.array([x[2:5] for x in rows])
print(f'\nConfiguración {params}')
print(f'BA media: {r[:, 0].mean():.3f} (TPE {r[:, 1].mean():.3f}, KNN {r[:, 2].mean():.3f}) en {len(rows)} pliegues; '
      f'gana a KNN en {int((r[:, 0] > r[:, 2]).sum())}')
