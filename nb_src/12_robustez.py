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

# %% [markdown]
# ## 12.4 Corrección del entregable 1: la brecha entre entrenamiento y validación
#
# El profesor observó que la brecha del entregable 1 (balanced accuracy de 0,65 en entrenamiento
# frente a 0,39 en validación) no se analizó a fondo. Como el balanced accuracy es el promedio del
# recall de las 4 clases, la brecha se puede **descomponer por clase**:
# brecha = ¼ Σₖ (recallₖ entrenamiento − recallₖ validación).
#
# **(a) La logística del entregable 1.** La corrida con diseño temporal de Iván (sección 12.6)
# reprodujo esa logística (C = 1, L2, `class_weight='balanced'`, 1.433 columnas) en los 3 pliegues
# hacia adelante del entregable 1 y guardó el recall de Fallecimiento en entrenamiento y validación.

# %%
iv = pd.read_csv(C.RESULTS / 'disenio_temporal' / 'master_registry.csv')
e1 = iv[iv.model == 'logreg_e1'][['outer_fold', 'train_deaths', 'valid_deaths', 'train_balanced_accuracy', 'outer_balanced_accuracy',
                                  'train_recall_death', 'outer_recall_death']].copy()
e1['brecha total'] = e1.train_balanced_accuracy - e1.outer_balanced_accuracy
e1['aporte de Fallecimiento (¼ Δ recall)'] = (e1.train_recall_death - e1.outer_recall_death) / 4
e1['aporte de las otras 3 clases'] = e1['brecha total'] - e1['aporte de Fallecimiento (¼ Δ recall)']
e1.set_index('outer_fold').round(3)

# %% [markdown]
# **(b) Los mejores modelos de este entregable.** Para la mejor combinación de cada modelo se
# reajusta en el entrenamiento de cada pliegue externo y se mide el recall por clase en ese mismo
# entrenamiento y en el pliegue externo.

# %%
import json, warnings
from osha_e3 import search as S, data as D, models as MM, evaluation as E, cache
def run_gap():
    X, y, g, s = N.task_data('clf')
    rows = []
    for m, r in bc.iterrows():
        for f, (tr, te) in enumerate(N.outer_folds('clf')):
            pipe = S.refit('clf', m, json.loads(r.final_params), r.balance, X.iloc[tr], y[tr], C.SEED)
            for part, idx in [('entrenamiento', tr), ('validación', te)]:
                pred = pipe.predict_proba(X.iloc[idx]).argmax(1)
                rec = [((pred == k) & (y[idx] == k)).sum() / (y[idx] == k).sum() if (y[idx] == k).any() else np.nan for k in range(4)]
                rows.append({'model': m, 'fold': f, 'parte': part, **{f'recall_{k}': rec[k] for k in range(4)}})
    return pd.DataFrame(rows)
gap = cache.cached('robustez_gap', run_gap)
g = gap.groupby(['model', 'parte'])[[f'recall_{k}' for k in range(4)]].mean().unstack('parte')
out = pd.DataFrame(index=[M.PRETTY[m] for m in P.MODEL_ORDER_CLF])
for k, name in enumerate(C.CLASS_NAMES):
    out[f'recall entren. {name}'] = g[(f'recall_{k}', 'entrenamiento')].reindex(P.MODEL_ORDER_CLF).to_numpy()
    out[f'recall valid. {name}'] = g[(f'recall_{k}', 'validación')].reindex(P.MODEL_ORDER_CLF).to_numpy()
deltas = pd.DataFrame({f'Δ {n}': out[f'recall entren. {n}'] - out[f'recall valid. {n}'] for n in C.CLASS_NAMES})
out['brecha balanced acc.'] = deltas.mean(axis=1)
out['aporte de Fallecimiento'] = deltas[f'Δ {C.CLASS_NAMES[0]}'] / 4
out['aporte de las 3 clases frecuentes'] = out['brecha balanced acc.'] - out['aporte de Fallecimiento']
out.round(3)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 12.5 Corrección del entregable 1: ¿qué pasa si se quita `time_unknown`?
#
# `time_unknown` supera el umbral de colinealidad (sección 4.6). Se reajusta la mejor combinación de
# cada modelo **sin esa variable** en los 5 pliegues externos (mismos hiperparámetros) y se compara,
# pliegue a pliegue, con el resultado original.

# %%
def run_ablation():
    X, y, g, s = N.task_data('clf')
    Xd = X.drop(columns=['time_unknown'])
    rows = []
    for m, r in bc.iterrows():
        for f, (tr, te) in enumerate(N.outer_folds('clf')):
            for label, XX in [('con time_unknown', X), ('sin time_unknown', Xd)]:
                pipe = MM.make_pipeline('clf', m, json.loads(r.final_params), r.balance, C.SEED)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pipe.fit(XX.iloc[tr], y[tr])
                met = E.classification_metrics(y[te], pipe.predict_proba(XX.iloc[te]))
                rows.append({'model': m, 'fold': f, 'variante': label, **met})
    return pd.DataFrame(rows)
abl = cache.cached('robustez_ablation_time_unknown', run_ablation)
rows = []
for m in P.MODEL_ORDER_CLF:
    a = abl[(abl.model == m) & (abl.variante == 'con time_unknown')].set_index('fold')
    b = abl[(abl.model == m) & (abl.variante == 'sin time_unknown')].set_index('fold')
    rows.append({'modelo': M.PRETTY[m], **{f'Δ {k} (sin − con)': (b[k] - a[k]).mean() for k in ['recall', 'f1', 'auc', 'log_loss']},
                 'd.e. entre pliegues (balanced acc.)': a.recall.std(ddof=1)})
pd.DataFrame(rows).set_index('modelo').round(4)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 12.6 Diseño alternativo: validación anidada con pliegues temporales (corrida de Iván Prada)
#
# En paralelo, Iván corrió una validación anidada con otro diseño (tablas en
# `results/disenio_temporal/`): 3 pliegues externos **hacia adelante en el tiempo** (los del
# entregable 1), NAICS y SOC completos con el agrupamiento de categorías raras como hiperparámetro,
# búsqueda en el 25 % del entrenamiento externo (con todas las muertes) y reentrenamiento con el
# 100 %, 20 evaluaciones por optimizador y SMOTE/ADASYN solo sobre Fallecimiento. KNN ×
# `class_weight` no se corrió (108 combinaciones). Comparar los dos diseños indica qué conclusiones
# dependen de cómo se valida.

# %%
from scipy.stats import spearmanr
pooled = pd.read_csv(C.RESULTS / 'disenio_temporal' / 'pooled_summary.csv')
ivm = iv[~iv.model.isin(['dummy', 'logreg_e1'])].groupby(['model', 'balance', 'optimizer']).agg(
    ba_temporal=('outer_balanced_accuracy', 'mean'), auc_temporal=('outer_roc_auc_macro_ovr', 'mean'),
    f1_temporal=('outer_f1_macro', 'mean')).reset_index()
both = mc.merge(ivm, on=['model', 'balance', 'optimizer'])
print('Combinaciones comunes:', len(both))
for a, b, lab in [('outer_recall_mean', 'ba_temporal', 'balanced accuracy'), ('outer_auc_mean', 'auc_temporal', 'AUC macro'),
                  ('outer_f1_mean', 'f1_temporal', 'F1 macro')]:
    rho, p = spearmanr(both[a], both[b])
    print(f'Spearman entre diseños ({lab}): rho = {rho:.2f} (p = {p:.2g})')
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
for ax, a, b, lab in [(axes[0], 'outer_recall_mean', 'ba_temporal', 'balanced accuracy'), (axes[1], 'outer_auc_mean', 'auc_temporal', 'AUC macro')]:
    for m in P.MODEL_ORDER_CLF:
        x = both[both.model == m]
        ax.scatter(x[a], x[b], color=P.MODEL_COLORS[m], s=28, label=M.PRETTY[m], edgecolor='white', linewidth=0.5)
    lo = min(both[a].min(), both[b].min()); hi = max(both[a].max(), both[b].max())
    ax.plot([lo, hi], [lo, hi], color=P.INK2, ls='--', lw=0.8)
    ax.set_xlabel(f'{lab}: este diseño (5 pliegues por establecimiento)'); ax.set_ylabel(f'{lab}: diseño temporal (3 pliegues)')
axes[1].legend(fontsize=7.5, loc='lower right')
fig.tight_layout(); P.save(fig, 'e3_robustez_disenios'); plt.show()
top_iv = pooled[~pooled.model.isin(['dummy'])].sort_values('BA_pooled', ascending=False).head(6)
top_iv[['model', 'balance', 'optimizer', 'deaths', 'hits', 'pred_death', 'prec_death', 'BA_pooled']].round(4)

# %% [markdown]
# **Interpretación.** TODO
