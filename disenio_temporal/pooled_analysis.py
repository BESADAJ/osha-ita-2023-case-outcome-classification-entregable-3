"""Resumen agrupado de la corrida: métricas con los 3 pliegues externos juntos (61.649 casos, 12 muertes).

Lee ``results/runs/*/fold*_pred.npz`` (descomprimir antes ``entregable3_predicciones.zip``) y escribe
``results/pooled_summary.csv``. Ejecutar desde la carpeta ``entregable3/``.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from osha3.metrics import wilson_interval  # noqa: E402

rows = []
for d in sorted(glob.glob('results/runs/*__*__*')):
    m, b, o = os.path.basename(d).split('__')
    zs = [np.load(os.path.join(d, f'fold{f}_pred.npz')) for f in (1, 2, 3)]
    y = np.concatenate([z['y'] for z in zs]); p = np.concatenate([z['proba'] for z in zs]); pred = p.argmax(1) + 1
    dm = y == 1; hits = int((pred[dm] == 1).sum()); npd = int((pred == 1).sum())
    rec = [(pred[y == k] == k).mean() for k in (1, 2, 3, 4)]
    rows.append(dict(model=m, balance=b, optimizer=o, deaths=int(dm.sum()), hits=hits, pred_death=npd,
                     prec_death=hits / npd if npd else np.nan, rec2=rec[1], rec3=rec[2], rec4=rec[3],
                     BA_pooled=float(np.mean(rec))))
r = pd.DataFrame(rows); r.to_csv('results/pooled_summary.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 30)
main = r[~r.model.isin(['dummy', 'logreg_e1'])]
print('Muertes totales en validacion externa:', int(main.deaths.iloc[0]))
print('\nPor balanceo (media de combinaciones):')
print(main.groupby('balance')[['hits', 'pred_death', 'prec_death', 'rec2', 'rec3', 'rec4', 'BA_pooled']].mean().round(3))
print('\nTop 8 por BA agrupada:'); print(main.sort_values('BA_pooled', ascending=False).head(8).round(3).to_string(index=False))
top = main.sort_values('BA_pooled', ascending=False).iloc[0]
print('\nWilson 95% recall Death de la mejor:', [round(x, 3) for x in wilson_interval(int(top.hits), 12)])
print('Combinaciones con >=1 acierto de Death:', int((main.hits >= 1).sum()), 'de', len(main))
