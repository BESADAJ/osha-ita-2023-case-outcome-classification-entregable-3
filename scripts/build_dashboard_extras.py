"""Tablas livianas del proyecto final para el dashboard (``dashboard_final/data/proyecto_final.json``).

Solo lee resultados ya versionados; no entrena ni usa la prueba:

* ``results/pareado/pliegues.csv``: BA por pliegue del original y de KNN, y diferencia media por repetición (sesgo de
  selección, sección 20.6);
* ``results/pareado/sensibilidad_secr.csv``: media de BA por hiperparámetro (sección 20.7);
* ``results/modelo_original/cribado*.csv`` y ``tabla_r4.csv``: candidatos del capítulo 17;
* ``results/ensemble/<variante>/fold*.npz``: BA (argmax) y accuracy de las variantes A–G en los 5 pliegues externos de
  la repetición 0 (misma regla que el capítulo 18).

Uso: python scripts/build_dashboard_extras.py
"""
import json
import re

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from osha_e3 import config as C

R = C.RESULTS
OUT = C.ROOT / 'dashboard_final' / 'data' / 'proyecto_final.json'
PRETTY = {'original': 'Original (SECR)', 'knn': 'KNN', 'svm': 'SVM', 'logreg': 'Reg. logística', 'xgboost': 'XGBoost',
          'random_forest': 'Random Forest', 'naive_bayes': 'Naive Bayes', 'tree': 'Árbol'}
VARIANTS = [('stack_curso', 'A', 'Stacking con los 5 modelos del curso (9 variables)'),
            ('stack_sin_empleador', 'B', 'Stacking multivista sin la vista del empleador'),
            ('stack_full', 'C', 'Stacking multivista (4 vistas), meta-modelo logístico'),
            ('unico_todo', 'D', 'Un solo XGBoost con todas las variables'),
            ('stack_v2', 'E', 'C + 3 modelos de información completa, meta-modelo logístico'),
            ('stack_v3:logistic', 'F', '16 modelos base, meta-modelo logístico'),
            ('stack_v3', 'G', '16 modelos base + selección de Caruana')]


def folds():
    f = pd.read_csv(R / 'pareado' / 'pliegues.csv')
    p = f.pivot_table(index=['rep', 'fold'], columns='model', values='ba').reset_index()
    per_fold = [{'rep': int(r.rep), 'fold': int(r.fold), 'original': float(r.original), 'knn': float(r.knn)}
                for r in p.itertuples()]
    by_rep = (p.original - p.knn).groupby(p.rep).mean()
    return per_fold, {int(k): float(v) for k, v in by_rep.items()}


def sensitivity():
    s = pd.read_csv(R / 'pareado' / 'sensibilidad_secr.csv')
    g = s.groupby(['param', 'value'], sort=False).ba.agg(['mean', 'std']).reset_index()
    return [{'param': r.param, 'value': str(r.value), 'ba_mean': float(r['mean']), 'ba_sd': float(r['std'])}
            for _, r in g.iterrows()]


def variant_metrics(name):
    d, _, comb = name.partition(':')
    ba, acc = [], []
    for k in range(C.OUTER_FOLDS):
        z = np.load(R / 'ensemble' / d / f'fold{k}.npz')
        p = z['p_caruana' if ('p_caruana' in z.files and comb != 'logistic') else 'p']
        y, pred = z['y'], p.argmax(1)
        ba.append(np.mean([((pred == c) & (y == c)).sum() / (y == c).sum() for c in range(4) if (y == c).any()]))
        acc.append((pred == y).mean())
    return float(np.mean(ba)), float(np.mean(acc))


def candidates():
    mo = R / 'modelo_original'
    cr = pd.read_csv(mo / 'cribado.csv')
    cr = cr[cr.pliegue.astype(str) == 'todos'].set_index('modelo')
    r2 = pd.read_csv(mo / 'cribado_r2.csv')
    r2 = r2[(r2.pliegue.astype(str) == 'todos') & (r2.decision == 'argmax')].set_index('modelo')
    r4 = pd.read_csv(mo / 'tabla_r4.csv').set_index('modelo')
    ref = cr.loc['xgboost__class_weight__bayes']
    temporal = 'Pliegues temporales (3; diseño del cap. 17)'
    rows = [
        ('17', 'PT-bagging', temporal, f"BA {cr.loc['pt_bagging'].balanced_accuracy:.3f} vs {ref.balanced_accuracy:.3f} "
         f"de la referencia; {int(cr.loc['pt_bagging'].muertes_acertadas)} de 12 muertes", 'cribado.csv'),
        ('17', 'MultiRandBal', temporal, f"BA {cr.loc['multirandbal'].balanced_accuracy:.3f}; "
         f"{int(cr.loc['multirandbal'].muertes_acertadas)} de 12 muertes", 'cribado.csv'),
        ('17', 'Metaclasificación jerárquica', temporal, f"BA {cr.loc['hierarchical'].balanced_accuracy:.3f}; "
         f"{int(cr.loc['hierarchical'].muertes_acertadas)} de 12 muertes", 'cribado.csv'),
        ('17', 'HAFL-XGBoost', temporal, f"BA {r2.loc['hafl_xgboost'].balanced_accuracy:.3f}, igual que sin pérdida focal "
         f"({r2.loc['hafl_xgboost_gamma0'].balanced_accuracy:.3f}): el mecanismo no aporta", 'cribado_r2.csv'),
        ('17', 'Frank–Hall ordinal (clases 2–4)', temporal, f"BA 2–4 {r4.loc['frank_hall'].ba_2_4:.3f} vs "
         f"{r4.loc['base'].ba_2_4:.3f} del mejor base", 'tabla_r4.csv'),
    ]
    out = [dict(zip(['cap', 'diseno', 'datos', 'resultado', 'fuente'], r)) for r in rows]
    for name, letter, desc in VARIANTS:
        ba, acc = variant_metrics(name)
        out.append({'cap': '18', 'diseno': f'{letter}. {desc}', 'datos': 'Pliegues externos (rep. 0) y prueba',
                    'resultado': f'BA {ba:.3f}; accuracy {acc:.3f}' + (' · elegido como nivel 0' if letter == 'G' else ''),
                    'fuente': 'results/ensemble/'})
    s = pd.read_csv(R / 'pareado' / 'resumen_modelos.csv').set_index('model')
    out.append({'cap': '19–20', 'diseno': 'SECR sobre los 16 modelos de G (modelo final)',
                'datos': 'Pliegues internos (TPE dentro de cada pliegue externo)',
                'resultado': f"BA {s.loc['original'].ba_mean:.3f} ± {s.loc['original'].ba_sd:.3f} en 15 pliegues; "
                             f"no superior a KNN ({s.loc['knn'].ba_mean:.3f})", 'fuente': 'resumen_modelos.csv'})
    for r in out:  # decimales con coma, como el resto del dashboard
        r['resultado'] = re.sub(r'(\d)\.(\d)', lambda m: m.group(1) + ',' + m.group(2), r['resultado'])
    return out


if __name__ == '__main__':
    per_fold, by_rep = folds()
    rec = {'per_fold': per_fold, 'diff_by_rep': by_rep, 'sensitivity': sensitivity(), 'candidates': candidates()}
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    print('escrito', OUT, '| candidatos:', len(rec['candidates']), '| diferencias por repetición:',
          {k: round(v, 4) for k, v in by_rep.items()})
