"""Análisis de las corridas de búsqueda: curvas *anytime*, eficiencia y estabilidad."""
import numpy as np
import pandas as pd

from . import nested as N


def anytime_table(task):
    """Mejor puntaje interno hallado hasta la evaluación t, para cada celda y método.

    Para comparar entre modelos se normaliza dentro de cada (modelo, balanceo, pliegue): 0 = peor
    puntaje observado por cualquier método en ese grupo, 1 = mejor. Se agrega también el tiempo
    acumulado hasta la evaluación t.
    """
    rows = []
    for r in N.load_cells(task):
        best, elapsed = -np.inf, 0.0
        for h in r['history']:
            best = max(best, h['score'])
            elapsed += h['seconds']
            rows.append({'model': r['model'], 'balance': r['balance'], 'optimizer': r['optimizer'],
                         'fold': r['fold'], 't': h['eval'], 'score': h['score'], 'best_so_far': best,
                         'cum_seconds': elapsed})
    d = pd.DataFrame(rows)
    g = d.groupby(['model', 'balance', 'fold'])
    lo, hi = g.score.transform('min'), g.score.transform('max')
    d['best_norm'] = (d.best_so_far - lo) / (hi - lo).replace(0, np.nan)
    return d


def budget_efficiency(any_tab, budgets=(4, 8, 12, 16)):
    """Calidad normalizada alcanzada con t evaluaciones y área bajo la curva anytime por método."""
    d = any_tab.dropna(subset=['best_norm'])
    full = d.pivot_table(index=['model', 'balance', 'fold', 'optimizer'], columns='t', values='best_norm')
    full = full.reindex(columns=range(1, 17)).ffill(axis=1)
    rows = []
    for opt, x in full.groupby(level='optimizer'):
        arr = x.to_numpy()
        row = {'optimizer': opt, 'celdas': len(arr), 'AUC anytime (0-1)': float(np.nanmean(arr.mean(axis=1)))}
        for b in budgets:
            row[f'calidad con {b} evals'] = float(np.nanmean(arr[:, b - 1]))
        reach = np.argmax(arr >= 0.99 * arr[:, [-1]], axis=1) + 1
        row['evals para 99 % del final (mediana)'] = float(np.median(reach))
        rows.append(row)
    return pd.DataFrame(rows).set_index('optimizer'), full


def paired_auc(full):
    """AUC anytime por celda (modelo, balanceo, pliegue) en columnas por método, para pruebas pareadas."""
    return full.mean(axis=1).unstack('optimizer').dropna()


def ga_diversity(task):
    """Diversidad genética y mejor aptitud por generación de todas las búsquedas genéticas."""
    rows = []
    for r in N.load_cells(task):
        for g in r.get('ga_log', []):
            rows.append({'model': r['model'], 'balance': r['balance'], 'fold': r['fold'], **g})
    return pd.DataFrame(rows)


def selection_bias(task, metric):
    """Diferencia entre el mejor puntaje interno y la métrica externa del mismo pliegue."""
    f = N.fold_table(task)
    outer = f[f'outer_{metric}'] if task == 'clf' else -f[f'outer_{metric}']
    f['optimismo'] = f.inner_score - outer
    return f
