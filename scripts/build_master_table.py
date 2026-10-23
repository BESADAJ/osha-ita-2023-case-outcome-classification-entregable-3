"""Tabla maestra de experimentos (guía §10.5): las 140 combinaciones del entregable 2 (bucle externo, 5 pliegues), los
7 benchmarks del experimento pareado y el modelo original (bucle externo repetido 3×5), con métricas, tiempos y semillas.

Uso: python scripts/build_master_table.py  ->  results/tabla_maestra_experimentos.csv
"""
import _bootstrap  # noqa: F401
import pandas as pd

from osha_e3 import config as C

rows = []
for task, path in [('clasificación', 'master_clf.csv'), ('regresión', 'master_reg.csv')]:
    m = pd.read_csv(C.RESULTS / path)
    for r in m.to_dict('records'):
        rows.append({'experimento': 'entregable 2 (9 variables)', 'tarea': task, 'modelo': r['model'],
                     'balanceo': r['balance'], 'optimizador': r['optimizer'], 'n_pliegues_externos': r['n_folds'],
                     'semilla': r['seed'], 'tiempo_busqueda_total_s': r['search_seconds_total'],
                     'hiperparametros_finales': r['final_params'],
                     **{k: v for k, v in r.items() if k.startswith('outer_')}})

f = pd.read_csv(C.RESULTS / 'pareado' / 'pliegues.csv')
for (model, bal, opt), g in f.groupby(['model', 'balance', 'optimizer']):
    rows.append({'experimento': 'pareado (mismas variables VPLUS, validación externa repetida 3×5)', 'tarea': 'clasificación',
                 'modelo': 'original (multivista + SECR)' if model == 'original' else model, 'balanceo': bal,
                 'optimizador': opt, 'n_pliegues_externos': len(g), 'semilla': C.SEED,
                 'tiempo_busqueda_total_s': g.train_s.sum(), 'hiperparametros_finales': 'por pliegue (ver results/pareado)',
                 'outer_recall_mean': g.ba.mean(), 'outer_recall_std': g.ba.std(), 'outer_f1_mean': g.f1.mean(),
                 'outer_f1_std': g.f1.std(), 'outer_auc_mean': g.auc.mean(), 'outer_auc_std': g.auc.std(),
                 'outer_log_loss_mean': g.log_loss.mean(), 'outer_log_loss_std': g.log_loss.std(),
                 'outer_accuracy_mean': g.accuracy.mean(), 'outer_accuracy_std': g.accuracy.std(),
                 'inferencia_ms_por_caso': g.ms_case.mean()})
out = pd.DataFrame(rows)
out.to_csv(C.RESULTS / 'tabla_maestra_experimentos.csv', index=False)
print(out.groupby('experimento').size())
