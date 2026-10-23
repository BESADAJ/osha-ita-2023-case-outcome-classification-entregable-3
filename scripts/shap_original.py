"""SHAP exacto por grupos de variables del modelo original frente a la mejor combinación del curso (capítulo 20.8).

Mismos 25 casos del desarrollo y mismo fondo de 6 casos para los dos modelos; 10 grupos (2^10 coaliciones).
Uso: python scripts/shap_original.py  ->  results/pareado/shap_original_vs_benchmark.csv y su figura
"""
import sys

import _bootstrap  # noqa: F401
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from osha_e3 import config as C, data as D, paired as PR  # noqa: E402

sys.path.insert(0, str(C.ROOT / 'dashboard_final'))
from src.explain import exact_shapley  # noqa: E402

threadpool_limits(2)
GROUPS = {'Estado': ['state'], 'Actividad (NAICS)': ['naics3', 'naics6'], 'Tipo de establecimiento': ['establishment_type'],
          'Ocupación (SOC)': ['soc_minor', 'soc6'], 'Tipo de incidente': ['type_of_incident'],
          'Horario': ['time_unknown', 'start_hour', 'incident_hour', 'hours_since_start'],
          'Naturaleza de la lesión': ['nature'], 'Parte del cuerpo': ['part'],
          'Evento y fuente': ['event', 'source', 'sec_source'], 'Empleador': ['ein', 'company']}
X, y, g = PR.paired_data()
rng = np.random.default_rng(C.SEED)
cases = X.iloc[rng.choice(len(X), 25, replace=False)][PR.VPLUS].reset_index(drop=True)
bg = X.iloc[rng.choice(len(X), 6, replace=False)][PR.VPLUS].reset_index(drop=True)
art = C.ROOT / 'dashboard_final' / 'artifacts'
models = {'Original (SECR)': joblib.load(art / 'original.joblib'), 'SVM + SMOTE (mejor del curso)': joblib.load(art / 'svm.joblib')}
out = {}
for name, est in models.items():
    f = (lambda Z, e=est: e.predict_proba(Z[PR.VPLUS])) if 'Original' in name else (lambda Z, e=est: e.predict_proba(Z[D.FEATURES]))
    phis = []
    for i in range(len(cases)):
        x = cases.iloc[[i]]
        cls = int(f(x).argmax())
        phis.append(exact_shapley(f, x, bg, GROUPS, cls).abs())
        print(name, i, flush=True)
    out[name] = pd.concat(phis, axis=1).mean(axis=1)
tab = pd.DataFrame(out).sort_values('Original (SECR)', ascending=False)
tab.to_csv(C.RESULTS / 'pareado' / 'shap_original_vs_benchmark.csv')
fig, ax = plt.subplots(figsize=(8, 4.5))
tab.iloc[::-1].mul(100).plot.barh(ax=ax, color=['#1f3b73', '#e9a23b'])
ax.set_xlabel('|SHAP| medio (puntos porcentuales de la clase predicha)')
ax.set_title('Importancia por grupo de variables: original frente al mejor benchmark')
fig.tight_layout()
fig.savefig(C.ROOT / 'jbook' / 'figures' / 'e3_shap_original_vs_benchmark.png', dpi=150)
print(tab.round(4))
