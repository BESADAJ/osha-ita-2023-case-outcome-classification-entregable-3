# %% [markdown]
# # 13. Comparación estadística de los modelos
#
# Con 140 modelos hay 9.730 pares posibles. Probar todos los pares con α = 0,05 daría unos 487 falsos
# positivos solo por azar. Por eso la comparación es jerárquica:
#
# - **Clasificación:** Friedman (¿algún modelo difiere?) → Nemenyi y diagrama de Diferencia Crítica →
#   DeLong solo entre los finalistas, con corrección de Holm → tamaño del efecto (delta de Cliff).
# - **Regresión:** Model Confidence Set sobre las 28 combinaciones → Giacomini-White o Clark-West
#   entre los finalistas → Diebold-Mariano (HLN) entre el primero y el segundo, con bootstrap
#   estacionario → tamaño del efecto (d de Cohen). Como control de *data snooping*, SPA de Hansen.
# - Intervalos de confianza bootstrap (BCa cuando la métrica es una razón de sumas; percentiles para
#   AUC) remuestreando **establecimientos completos**.

# %%
import sys, json, warnings
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from osha_e3 import config as C, models as M, nested as N, final as F, evaluation as E, stats_tests as T, plots as P, cache
P.setup()
fc, fr = N.fold_table('clf'), N.fold_table('reg')
mc, mr = N.master_table('clf'), N.master_table('reg')
test_c, yt_c = F.test_data('clf'); test_r, yt_r = F.test_data('reg')
MODELS_C, MODELS_R = P.MODEL_ORDER_CLF, P.MODEL_ORDER_REG

# %% [markdown]
# ## 13.1 Clasificación, etapa 1: prueba ómnibus de Friedman
#
# **(a) Las 112 combinaciones** sobre los 5 pliegues externos (bloques), con la balanced accuracy
# externa. **(b) Los 7 modelos** (mejor combinación de cada uno según la validación anidada) sobre
# 13 bloques semanales de la prueba: con solo 5 bloques Nemenyi tiene muy poca potencia, y las
# semanas de octubre a diciembre son bloques naturales (independientes del ajuste y ordenados en
# el tiempo). **(c) Los 4 optimizadores y las 4 técnicas de balanceo**, con bloques = (modelo ×
# balanceo × pliegue) y (modelo × optimizador × pliegue).

# %%
pv112 = fc.pivot_table(index='fold', columns=['model', 'balance', 'optimizer'], values='outer_recall')
fr112 = T.friedman(pv112.to_numpy())
best_c = mc.loc[mc.groupby('model').outer_recall_mean.idxmax()].set_index('model').reindex(MODELS_C)
probs = {m: F.load('clf', best_c.loc[m, 'combo'])[0] for m in MODELS_C}
week = pd.to_datetime(test_c.date).dt.to_period('W').astype(str).to_numpy()
weeks = sorted(set(week))
W = pd.DataFrame({M.PRETTY[m]: [balanced_accuracy_score(yt_c[week == w], probs[m][week == w].argmax(1)) for w in weeks] for m in MODELS_C}, index=weeks)
fr7 = T.friedman(W.to_numpy())
pv_opt = fc.pivot_table(index=['model', 'balance', 'fold'], columns='optimizer', values='outer_recall')[M.OPTIMIZERS]
pv_bal = fc.pivot_table(index=['model', 'optimizer', 'fold'], columns='balance', values='outer_recall')[M.BALANCES]
fr_opt, fr_bal = T.friedman(pv_opt.to_numpy()), T.friedman(pv_bal.to_numpy())
pd.DataFrame([
    {'Comparación': '112 combinaciones (bloques: 5 pliegues externos)', 'χ²_F': fr112['chi2'], 'g.l.': fr112['df'], 'p-valor': fr112['p_value'], 'N bloques': fr112['n_blocks']},
    {'Comparación': '7 modelos (bloques: 13 semanas de la prueba)', 'χ²_F': fr7['chi2'], 'g.l.': fr7['df'], 'p-valor': fr7['p_value'], 'N bloques': fr7['n_blocks']},
    {'Comparación': '4 optimizadores (bloques: modelo × balanceo × pliegue)', 'χ²_F': fr_opt['chi2'], 'g.l.': fr_opt['df'], 'p-valor': fr_opt['p_value'], 'N bloques': fr_opt['n_blocks']},
    {'Comparación': '4 balanceos (bloques: modelo × optimizador × pliegue)', 'χ²_F': fr_bal['chi2'], 'g.l.': fr_bal['df'], 'p-valor': fr_bal['p_value'], 'N bloques': fr_bal['n_blocks']},
]).set_index('Comparación').round(5)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 13.2 Clasificación, etapa 2: Nemenyi y diagramas de Diferencia Crítica
#
# Rango promedio: 1 = mejor. Dos modelos unidos por una barra horizontal no difieren
# significativamente según Nemenyi (α = 0,05).

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 9.5))
cd7 = T.nemenyi_cd(fr7['k'], fr7['n_blocks'])
T.cd_diagram(list(W.columns), fr7['avg_ranks'], cd7, axes[0], 'Mejor combinación de cada modelo (13 semanas de prueba)')
cdo = T.nemenyi_cd(fr_opt['k'], fr_opt['n_blocks'])
T.cd_diagram([M.PRETTY[o] for o in M.OPTIMIZERS], fr_opt['avg_ranks'], cdo, axes[1], f'Optimizadores ({fr_opt["n_blocks"]} bloques)')
cdb = T.nemenyi_cd(fr_bal['k'], fr_bal['n_blocks'])
T.cd_diagram([M.PRETTY[b] for b in M.BALANCES], fr_bal['avg_ranks'], cdb, axes[2], f'Técnicas de balanceo ({fr_bal["n_blocks"]} bloques)')
fig.tight_layout(); P.save(fig, 'e3_stats_cd'); plt.show()
cd112 = T.nemenyi_cd(fr112['k'], fr112['n_blocks'])
r112 = pd.Series(fr112['avg_ranks'], index=pv112.columns).sort_values()
print(f'112 combinaciones: CD = {cd112:.1f} rangos (sobre 112). Mejor rango promedio {r112.iloc[0]:.1f}; '
      f'combinaciones cuyo rango no difiere del mejor: {(r112 - r112.iloc[0] < cd112).sum()} de 112.')
print('Rangos promedio, 7 modelos:', dict(zip(W.columns, np.round(fr7['avg_ranks'], 2))), '| CD =', round(cd7, 2))

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 13.3 Clasificación: DeLong entre los finalistas, con Holm
#
# Se comparan los tres mejores modelos según el rango de Friedman en la prueba (3 pares). DeLong
# compara AUC binarias, así que se aplica a cada clase una contra el resto (4 clases × 3 pares = 12
# pruebas) y se ajustan los 12 p-valores con Holm-Bonferroni (y Benjamini-Hochberg como
# referencia). El intervalo de la diferencia de AUC es un bootstrap de establecimientos (200 réplicas).

# %%
rank7 = pd.Series(fr7['avg_ranks'], index=MODELS_C).sort_values()
final3 = list(rank7.index[:3])
groups = test_c.establishment_id.to_numpy()
codes, uniq = pd.factorize(groups)
def run_delong():
    rows = []
    rng = np.random.default_rng(C.seed_for('bootstrap', 1))
    boots = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes] for _ in range(200)]
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = final3[i], final3[j]
            for k in range(4):
                yk = (yt_c == k)
                d = T.delong_test(yk, probs[a][:, k], probs[b][:, k])
                diffs = [roc_auc_score(yk, probs[a][:, k], sample_weight=w) - roc_auc_score(yk, probs[b][:, k], sample_weight=w) for w in boots]
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                rows.append({'Modelo A': M.PRETTY[a], 'Modelo B': M.PRETTY[b], 'Clase': C.CLASS_NAMES[k], 'AUC A': d['auc1'], 'AUC B': d['auc2'],
                             'Diferencia de AUC': d['diff'], 'IC 95 % bootstrap': f'[{lo:.4f}, {hi:.4f}]', 'z': d['z'], 'p crudo': d['p_value']})
    out = pd.DataFrame(rows)
    out['p Holm'] = T.adjust(out['p crudo'].to_numpy(), 'holm'); out['p BH'] = T.adjust(out['p crudo'].to_numpy(), 'fdr_bh')
    return out
dl = cache.cached('stats_delong', run_delong)
dl.round(4)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 13.4 Clasificación: tamaño del efecto (delta de Cliff)
#
# Sobre las balanced accuracy semanales (13 semanas) entre todos los pares de los 7 modelos, y entre
# los optimizadores y las técnicas de balanceo sobre sus bloques. |δ| < 0,147 despreciable,
# < 0,33 pequeño, < 0,474 mediano, si no grande. Se reportan todos los pares, favorables o no.

# %%
rows = []
for i, a in enumerate(W.columns):
    for b in W.columns[i + 1:]:
        d = T.cliffs_delta(W[a], W[b])
        rows.append({'A': a, 'B': b, 'media A − B (balanced acc.)': (W[a] - W[b]).mean(), 'δ de Cliff': d, 'magnitud': T.cliffs_label(d)})
cl7 = pd.DataFrame(rows)
for tab, lab in [(pv_opt, 'optimizador'), (pv_bal, 'balanceo')]:
    for i, a in enumerate(tab.columns):
        for b in tab.columns[i + 1:]:
            d = T.cliffs_delta(tab[a], tab[b])
            rows.append({'A': M.PRETTY[a], 'B': M.PRETTY[b], 'media A − B (balanced acc.)': (tab[a] - tab[b]).mean(), 'δ de Cliff': d, 'magnitud': T.cliffs_label(d)})
pd.DataFrame(rows).round(4)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 13.5 Regresión, paso 1: Model Confidence Set sobre las 28 combinaciones
#
# Pérdida: error cuadrático en la prueba, promediado por día (92 días de octubre a diciembre) para
# tener una serie temporal; el bootstrap estacionario (bloques de longitud media 7 días) respeta el
# patrón semanal. MCS con estadístico de rango (*range*), α = 0,10, 5.000 réplicas.

# %%
preds_r = {}
for c in mr.combo:
    preds_r[c] = F.load('reg', c)[0].astype(float)
dates = pd.to_datetime(test_r.date).to_numpy()
sq = pd.DataFrame({c: (yt_r - p) ** 2 for c, p in preds_r.items()}, index=dates)
daily = sq.groupby(level=0).mean()
daily.columns = [c.replace('|none|', ' · ') for c in daily.columns]
mcs = cache.cached('stats_mcs', lambda: T.model_confidence_set(daily, alpha=0.10, reps=5000, block=7, seed=C.seed_for('bootstrap', 2)))
mcs['RMSE prueba'] = np.sqrt(mcs.avg_loss)
mcs.round(4)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 13.6 Regresión, pasos 2 y 3: comparaciones dirigidas entre finalistas
#
# - **Clark-West** cuando un modelo está anidado en otro. La referencia «media del entrenamiento»
#   (modelo con solo intercepto) está anidada en Ridge y en Lasso; se contrasta si las variables
#   aportan capacidad predictiva real.
# - **Giacomini-White** (habilidad predictiva condicional, instrumentos = constante y diferencial de
#   pérdida del día anterior) entre finalistas no anidados.
# - **Diebold-Mariano con corrección HLN** entre el mejor y el segundo del MCS, con IC del diferencial
#   medio por bootstrap estacionario y d de Cohen.
#
# Los p-valores de este bloque se ajustan con Holm.

# %%
order_mcs = list(mcs.index)
finalists = list(mcs[mcs.in_mcs].index) if mcs.in_mcs.sum() >= 2 else order_mcs[:3]
finalists = finalists[:4]
key = {c.replace('|none|', ' · '): c for c in mr.combo}
rows = []
yd = pd.Series(yt_r, index=dates)
mean_pred = np.full_like(yt_r, N.task_data('reg')[1].mean())
for name in [n for n in order_mcs if n.startswith('ridge') or n.startswith('lasso')][:2]:
    cw = T.clark_west(yt_r, mean_pred, preds_r[key[name]])
    rows.append({'prueba': 'Clark-West (anidados)', 'A': 'media del entrenamiento', 'B': name, 'estadístico': cw['t'], 'p crudo': cw['p_value'],
                 'diferencial medio': cw['mean_adj_diff'], 'unidad': 'por caso'})
for i in range(len(finalists)):
    for j in range(i + 1, len(finalists)):
        a, b = finalists[i], finalists[j]
        gw = T.giacomini_white(daily[a].to_numpy(), daily[b].to_numpy())
        rows.append({'prueba': 'Giacomini-White', 'A': a, 'B': b, 'estadístico': gw['stat'], 'p crudo': gw['p_value'],
                     'diferencial medio': gw['mean_diff'], 'unidad': 'diaria'})
a, b = order_mcs[0], order_mcs[1]
dm = T.diebold_mariano(daily[a].to_numpy(), daily[b].to_numpy(), h=1)
lo, hi = T.stationary_bootstrap_ci(daily[a] - daily[b], reps=5000, block=7, seed=C.seed_for('bootstrap', 3))
rows.append({'prueba': 'Diebold-Mariano (HLN)', 'A': a, 'B': b, 'estadístico': dm['dm_hln'], 'p crudo': dm['p_value'],
             'diferencial medio': dm['mean_diff'], 'unidad': 'diaria'})
dir_tab = pd.DataFrame(rows)
dir_tab['p Holm'] = T.adjust(dir_tab['p crudo'].to_numpy())
print(f'DM {a} vs {b}: diferencial medio diario {dm["mean_diff"]:.5f}, IC 95 % bootstrap estacionario [{lo:.5f}, {hi:.5f}], '
      f'd de Cohen {T.cohens_d_paired(daily[a] - daily[b]):.3f}')
dir_tab.round(5)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 13.7 Regresión: control de *data snooping* (SPA de Hansen y Reality Check)
#
# ¿Alguna de las 28 combinaciones supera de verdad a una referencia simple, o la aparente mejora se
# debe a haber probado muchas? Referencia: la media del entrenamiento. H0: ninguna combinación es
# mejor que la referencia. Se reportan los p-valores inferior, consistente y superior de SPA (el
# superior equivale al Reality Check de White).

# %%
bench = pd.Series((yt_r - mean_pred) ** 2, index=dates).groupby(level=0).mean()
spa = cache.cached('stats_spa', lambda: T.spa_test(bench.to_numpy(), daily.to_numpy(), reps=5000, block=7, seed=C.seed_for('bootstrap', 4)))
pd.Series(spa, name='p-valor SPA').to_frame().round(4)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 13.8 Intervalos de confianza al 95 % (bootstrap por establecimiento)
#
# - Accuracy (clasificación) y RMSE/MAE (regresión): **BCa** sobre establecimientos, con el
#   estadístico como razón de sumas (aciertos / casos; errores cuadráticos / casos).
# - AUC macro: percentiles de 200 réplicas de establecimientos (BCa requiere el jackknife, que con
#   34.000 establecimientos y AUC es demasiado costoso).

# %%
def run_ci():
    rows = []
    gc = test_c.establishment_id.to_numpy(); gr = test_r.establishment_id.to_numpy()
    rng = np.random.default_rng(C.seed_for('bootstrap', 5))
    codes, uniq = pd.factorize(gc)
    boots = [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes] for _ in range(200)]
    for m in MODELS_C:
        p = probs[m]; correct = (p.argmax(1) == yt_c).astype(float)
        lo, hi = T.group_bca_ci(correct, np.ones_like(correct), gc, reps=2000, seed=C.SEED)
        rows.append({'Tarea': 'Clasificación', 'Modelo': M.PRETTY[m], 'Métrica': 'Accuracy', 'Estimación': correct.mean(), 'IC inferior': lo, 'IC superior': hi, 'Método': 'BCa'})
        aucs = [roc_auc_score(yt_c, p, multi_class='ovr', average='macro', sample_weight=w) for w in boots]
        rows.append({'Tarea': 'Clasificación', 'Modelo': M.PRETTY[m], 'Métrica': 'AUC macro', 'Estimación': roc_auc_score(yt_c, p, multi_class='ovr', average='macro'),
                     'IC inferior': np.percentile(aucs, 2.5), 'IC superior': np.percentile(aucs, 97.5), 'Método': 'percentil (200)'})
    best_r = mr.loc[mr.groupby('model').outer_rmse_mean.idxmin()].set_index('model').reindex(MODELS_R)
    for m in MODELS_R:
        p = preds_r[best_r.loc[m, 'combo']]; se = (yt_r - p) ** 2
        lo, hi = T.group_bca_ci(se, np.ones_like(se), gr, reps=2000, seed=C.SEED)
        rows.append({'Tarea': 'Regresión', 'Modelo': M.PRETTY[m], 'Métrica': 'RMSE', 'Estimación': np.sqrt(se.mean()), 'IC inferior': np.sqrt(lo), 'IC superior': np.sqrt(hi), 'Método': 'BCa'})
        ae = np.abs(yt_r - p); lo, hi = T.group_bca_ci(ae, np.ones_like(ae), gr, reps=2000, seed=C.SEED)
        rows.append({'Tarea': 'Regresión', 'Modelo': M.PRETTY[m], 'Métrica': 'MAE', 'Estimación': ae.mean(), 'IC inferior': lo, 'IC superior': hi, 'Método': 'BCa'})
    return pd.DataFrame(rows)
ci = cache.cached('stats_ci', run_ci)
ci.set_index(['Tarea', 'Modelo', 'Métrica']).round(4)

# %% [markdown]
# **Interpretación.** TODO
