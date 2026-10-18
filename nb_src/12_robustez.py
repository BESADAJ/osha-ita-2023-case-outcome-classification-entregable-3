# %% [markdown]
# # 12. Robustez y estabilidad de las métricas
#
# Un número suelto («balanced accuracy = 0,39») esconde dos fuentes de variación: **los datos** (qué
# establecimientos caen en cada pliegue) y **el algoritmo** (semillas del bosque, del boosting, del
# sobremuestreo y de la búsqueda). Este capítulo mide las dos y las compara con las diferencias entre
# modelos.

# %%
import sys, json, warnings
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from osha_e3 import config as C, models as M, nested as N, seeds as SD, plots as P
P.setup()
mc, mr = N.master_table('clf'), N.master_table('reg')
fc, fr = N.fold_table('clf'), N.fold_table('reg')
st = SD.table()
print('Combinaciones repetidas con otras semillas:', SD.selected_combos())
print('Semillas:', [C.SEED] + SD.SEEDS)

# %% [markdown]
# ## 12.1 Media ± desviación estándar en los pliegues externos
#
# Todas las combinaciones (112 + 28) están en las tablas maestras con media y desviación de cada
# métrica. Resumen por modelo (mejor combinación de cada uno):

# %%
def best_rows(mt, metric, asc):
    b = mt.loc[mt.groupby('model')[f'outer_{metric}_mean'].idxmin() if asc else mt.groupby('model')[f'outer_{metric}_mean'].idxmax()]
    return b.set_index('model')
bc = best_rows(mc, 'recall', False).reindex(P.MODEL_ORDER_CLF)
br = best_rows(mr, 'rmse', True).reindex(P.MODEL_ORDER_REG)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
for ax, b, metric, lab, f in [(axes[0], bc, 'recall', 'balanced accuracy externa', fc), (axes[1], br, 'rmse', 'RMSE externo (log1p días)', fr)]:
    for i, (m, r) in enumerate(b.iterrows()):
        vals = f[(f.model == m) & (f.balance == r.balance) & (f.optimizer == r.optimizer)][f'outer_{metric}']
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, color=P.MODEL_COLORS[m], s=18, alpha=0.7)
        ax.errorbar(i + 0.25, r[f'outer_{metric}_mean'], yerr=r[f'outer_{metric}_std'], fmt='o', color=P.INK, ms=5, capsize=4)
    ax.set_xticks(range(len(b)), [M.PRETTY[m] for m in b.index], rotation=30, ha='right', fontsize=8.5)
    ax.set_ylabel(lab); ax.grid(axis='x', visible=False)
axes[0].set_title('Clasificación: 5 pliegues (puntos) y media ± d.e.'); axes[1].set_title('Regresión: 5 pliegues (puntos) y media ± d.e.')
fig.tight_layout(); P.save(fig, 'e3_robustez_pliegues'); plt.show()
pd.DataFrame({'Clasificación: combinación': [f'{M.PRETTY[r.balance]} · {M.PRETTY[r.optimizer]}' for _, r in bc.iterrows()],
              'balanced acc. (media ± d.e.)': [f'{r.outer_recall_mean:.3f} ± {r.outer_recall_std:.3f}' for _, r in bc.iterrows()],
              'AUC (media ± d.e.)': [f'{r.outer_auc_mean:.3f} ± {r.outer_auc_std:.3f}' for _, r in bc.iterrows()]},
             index=[M.PRETTY[m] for m in bc.index])

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 12.2 Sensibilidad a la semilla aleatoria
#
# Para las combinaciones seleccionadas se repitió la validación anidada completa (búsqueda incluida)
# con las semillas 7, 123 y 2024, además de la 42. Los datos y los pliegues no cambian: solo la
# aleatoriedad del algoritmo. Se compara la desviación **entre semillas** (de la media de los 5
# pliegues) con la desviación **entre pliegues**.

# %%
metric = {'clf': 'outer_recall', 'reg': 'outer_rmse'}
rows = []
for (t, m, b, o), d in st.groupby(['task', 'model', 'balance', 'optimizer']):
    k = metric[t]
    per_seed = d.groupby('seed')[k].mean()
    rows.append({'tarea': 'Clasificación' if t == 'clf' else 'Regresión',
                 'combinación': f'{M.PRETTY[m]} · {M.PRETTY[b]} · {M.PRETTY[o]}',
                 'semillas': len(per_seed), 'media': per_seed.mean(),
                 'd.e. entre semillas': per_seed.std(ddof=1), 'rango entre semillas': per_seed.max() - per_seed.min(),
                 'd.e. entre pliegues (semilla 42)': d[d.seed == C.SEED][k].std(ddof=1),
                 'hiperparámetros distintos elegidos (de 20 búsquedas)': d.params.nunique()})
seed_tab = pd.DataFrame(rows).set_index(['tarea', 'combinación'])
seed_tab['razón semillas / pliegues'] = seed_tab['d.e. entre semillas'] / seed_tab['d.e. entre pliegues (semilla 42)']
seed_tab.round(4)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
for ax, t, lab in [(axes[0], 'clf', 'balanced accuracy externa'), (axes[1], 'reg', 'RMSE externo')]:
    d = st[st.task == t]
    combos = list(d.groupby(['model', 'balance', 'optimizer']).groups)
    for i, cmb in enumerate(combos):
        x = d[(d.model == cmb[0]) & (d.balance == cmb[1]) & (d.optimizer == cmb[2])]
        for j, s in enumerate([C.SEED] + SD.SEEDS):
            v = x[x.seed == s][metric[t]]
            ax.scatter(np.full(len(v), i + (j - 1.5) * 0.15), v, s=12, color=P.SERIES[j], label=f'semilla {s}' if i == 0 else None, alpha=0.8)
    ax.set_xticks(range(len(combos)), [f'{M.PRETTY[c[0]]}\n{M.PRETTY[c[1]]}\n{M.PRETTY[c[2]]}' for c in combos], fontsize=7)
    ax.set_ylabel(lab); ax.grid(axis='x', visible=False)
axes[0].legend(fontsize=7.5, ncol=2); axes[0].set_title('Clasificación'); axes[1].set_title('Regresión')
fig.tight_layout(); P.save(fig, 'e3_robustez_semillas'); plt.show()

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 12.3 ¿Las diferencias entre modelos superan la variabilidad de cada modelo?
#
# Se compara la diferencia entre la mejor combinación y cada una de las demás (en el mismo pliegue,
# pareada) con la desviación estándar entre pliegues de cada una. Si la diferencia media es menor que
# esa desviación, la diferencia observada no se distingue del ruido de muestreo con estos datos; eso
# motiva las pruebas formales del capítulo 13.

# %%
def diff_vs_noise(f, mt, metric, higher):
    top = mt.sort_values(f'outer_{metric}_mean', ascending=not higher).iloc[0]
    ref = f[(f.model == top.model) & (f.balance == top.balance) & (f.optimizer == top.optimizer)].set_index('fold')[f'outer_{metric}']
    best = mt.loc[mt.groupby('model')[f'outer_{metric}_mean'].idxmax() if higher else mt.groupby('model')[f'outer_{metric}_mean'].idxmin()]
    rows = []
    for _, r in best.iterrows():
        v = f[(f.model == r.model) & (f.balance == r.balance) & (f.optimizer == r.optimizer)].set_index('fold')[f'outer_{metric}']
        d = (ref - v) if higher else (v - ref)
        rows.append({'modelo': M.PRETTY[r.model], 'diferencia media con el mejor (pareada)': d.mean(),
                     'd.e. de la diferencia': d.std(ddof=1), 'd.e. entre pliegues del modelo': v.std(ddof=1),
                     '¿diferencia > d.e. del modelo?': bool(d.mean() > v.std(ddof=1)), 'pliegues donde el mejor gana': int((d > 0).sum())})
    return pd.DataFrame(rows).set_index('modelo').sort_values('diferencia media con el mejor (pareada)')
print('Clasificación (balanced accuracy):'); display(diff_vs_noise(fc, mc, 'recall', True).round(4))
print('Regresión (RMSE):'); display(diff_vs_noise(fr, mr, 'rmse', False).round(4))

# %% [markdown]
# **Interpretación.** TODO
