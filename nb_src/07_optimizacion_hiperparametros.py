# %% [markdown]
# # 7. Optimización de hiperparámetros
#
# Cuatro métodos, el mismo espacio y el mismo presupuesto (16 configuraciones por búsqueda, cada una
# evaluada con 3 pliegues internos), dentro de la validación anidada del capítulo 6. Toda métrica
# de desempeño de este capítulo viene del **bucle externo**, salvo cuando se dice explícitamente
# que es el puntaje interno (el que guía la búsqueda).

# %%
import sys, json, warnings
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from osha_e3 import config as C, models as M, nested as N, analysis as A, plots as P, search as S, cache
P.setup()
OPTS = ['grid', 'random', 'bayes', 'genetic']
fc, fr = N.fold_table('clf'), N.fold_table('reg')
mc, mr = N.master_table('clf'), N.master_table('reg')
print('Celdas de clasificación:', len(fc), '| de regresión:', len(fr))
print('Evaluaciones por celda (mín., máx.):', fc.n_evals.min(), fc.n_evals.max(), fr.n_evals.min(), fr.n_evals.max())

# %% [markdown]
# ## 7.1 Los cuatro métodos
#
# **Grid Search.** Recorre una rejilla fija de 16 puntos dentro del espacio (tabla 6.6). Es
# exhaustiva en la rejilla pero ciega fuera de ella: si el óptimo cae entre dos valores, no lo ve, y
# con varios hiperparámetros la rejilla solo puede probar 2 valores de cada uno.
#
# **Random Search.** 16 puntos uniformes en el espacio codificado (escala log incluida). Con el
# mismo número de puntos prueba 16 valores distintos de *cada* hiperparámetro, lo que ayuda
# cuando solo unos pocos importan (Bergstra y Bengio, 2012).
#
# **Optimización bayesiana (Optuna).**
# - *Modelo sustituto:* **Tree-structured Parzen Estimator (TPE)**. Separa las configuraciones ya
#   evaluadas en «buenas» (cuantil superior γ del puntaje) y «malas», y estima con núcleos de Parzen
#   las densidades l(x) = p(x | bueno) y g(x) = p(x | malo). Se usa la versión multivariada, que
#   modela los hiperparámetros conjuntamente.
# - *Función de adquisición:* **Mejora Esperada (EI)**. En TPE, maximizar EI equivale a elegir el
#   candidato con mayor razón l(x)/g(x) entre 24 candidatos muestreados de l(x).
# - *Exploración-explotación:* las 5 primeras configuraciones son aleatorias (exploración pura); luego
#   TPE explota las regiones con alta l/g, pero como los candidatos se muestrean de l(x) (una
#   mezcla de núcleos con ancho finito) sigue explorando alrededor. EI favorece más la explotación
#   que UCB con κ grande; con solo 16 evaluaciones es una elección razonable.
#
# **Algoritmo genético (DEAP).** Cada individuo es un vector de genes en [0, 1] (uno por
# hiperparámetro), decodificado igual que en Random Search.
# - *Selección:* torneo de tamaño 3: presión moderada, insensible a la escala del puntaje (solo
#   importa el orden), a diferencia de la ruleta.
# - *Cruce:* mezcla BLX-α (α = 0,5) con probabilidad 0,8: los hijos pueden caer algo fuera del
#   segmento entre los padres, lo que mantiene exploración en genes reales.
# - *Mutación:* gaussiana (σ = 0,15 en la escala [0, 1], probabilidad por gen 0,5), aplicada al 30 %
#   de los hijos.
# - *Elitismo y población:* 6 individuos; el mejor pasa intacto a la siguiente generación. Con 16
#   evaluaciones, una población pequeña es la única forma de tener varias generaciones (6 iniciales +
#   2 generaciones de 5 hijos); una población grande sería solo una búsqueda aleatoria.
# - Las configuraciones repetidas no se reevalúan ni consumen presupuesto.

# %% [markdown]
# ## 7.2 Desempeño externo por método

# %%
def summary(f, metric, higher=True):
    g = f.groupby('optimizer')
    out = pd.DataFrame({
        f'{metric} externo (media de celdas)': g[f'outer_{metric}'].mean(),
        'tiempo de búsqueda total (h·núcleo)': g.search_seconds.sum() / 3600,
        'tiempo medio por evaluación (s)': g.eval_seconds_mean.mean(),
        'evaluaciones totales': g.n_evals.sum(),
        'ajustes totales (× 3 pliegues internos)': g.n_evals.sum() * C.INNER_FOLDS,
    }).reindex(OPTS)
    return out
print('Clasificación (balanced accuracy externa):')
display(summary(fc, 'recall').round(4))
print('Regresión (RMSE externo, log1p días):')
display(summary(fr, 'rmse').round(4))

# %%
pv = mc.pivot_table(index='model', columns='optimizer', values='outer_recall_mean', aggfunc='max').reindex(columns=OPTS)
pv.index = [M.PRETTY[i] for i in pv.index]
pr = mr.pivot_table(index='model', columns='optimizer', values='outer_rmse_mean', aggfunc='min').reindex(columns=OPTS)
pr.index = [M.PRETTY[i] for i in pr.index]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, tab, title, lab in [(axes[0], pv, 'Clasificación: mejor balanced accuracy externa', 'balanced acc.'),
                            (axes[1], pr, 'Regresión: RMSE externo (menor es mejor)', 'RMSE')]:
    x = np.arange(len(tab)); w = 0.2
    for j, o in enumerate(OPTS):
        ax.bar(x + (j - 1.5) * w, tab[o], width=w - 0.02, color=P.OPT_COLORS[o], label=M.PRETTY[o])
    ax.set_xticks(x, tab.index, rotation=30, ha='right', fontsize=8); ax.set_title(title); ax.set_ylabel(lab)
    lo = np.nanmin(tab.to_numpy()); hi = np.nanmax(tab.to_numpy()); ax.set_ylim(lo - (hi - lo) * 0.6, hi + (hi - lo) * 0.3)
    ax.grid(axis='x', visible=False)
axes[0].legend(ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(1.1, 1.22))
P.save(fig, 'e3_hpo_por_modelo'); plt.show()

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 7.3 Convergencia: curvas de desempeño en cualquier momento (*anytime*)
#
# Para cada celda se toma el mejor puntaje **interno** encontrado hasta la evaluación t. Para poder
# promediar entre modelos con escalas distintas se normaliza dentro de cada (modelo, balanceo,
# pliegue externo): 0 es el peor puntaje que obtuvo cualquier método y 1 el mejor. En Grid Search
# la curva depende del orden en que se recorre la rejilla (aquí, el orden de `ParameterGrid`).

# %%
any_c, any_r = A.anytime_table('clf'), A.anytime_table('reg')
eff_c, full_c = A.budget_efficiency(any_c)
eff_r, full_r = A.budget_efficiency(any_r)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, full, title in [(axes[0], full_c, 'Clasificación (112 combinaciones × 5 pliegues)'),
                        (axes[1], full_r, 'Regresión (28 combinaciones × 5 pliegues)')]:
    for o in OPTS:
        x = full.xs(o, level='optimizer')
        m = x.mean(); se = x.std() / np.sqrt(len(x))
        ax.plot(m.index, m.values, color=P.OPT_COLORS[o], label=M.PRETTY[o])
        ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=P.OPT_COLORS[o], alpha=0.15, lw=0)
    ax.set_xlabel('evaluaciones (t)'); ax.set_title(title, fontsize=10)
axes[0].set_ylabel('mejor puntaje interno hasta t (normalizado)')
axes[0].legend(fontsize=8, loc='lower right')
P.save(fig, 'e3_hpo_anytime'); plt.show()

# %%
print('Clasificación'); display(eff_c.reindex(OPTS).round(3))
print('Regresión'); display(eff_r.reindex(OPTS).round(3))

# %% [markdown]
# ## 7.4 Evidencia cuantitativa: calidad por unidad de presupuesto
#
# La comparación visual no basta. Para cada celda (modelo × balanceo × pliegue externo) se calcula
# el área bajo la curva anytime normalizada (promedio de la curva en t = 1…16; 1 sería encontrar
# el mejor puntaje en la primera evaluación). Las celdas son pareadas (mismos datos para los cuatro
# métodos), así que se comparan con Friedman y luego con Wilcoxon de rangos con signo por pares,
# con corrección de Holm. También se compara la calidad alcanzada **a mitad de presupuesto
# (t = 8)** y el puntaje externo final.

# %%
from osha_e3 import stats_tests as T
def budget_tests(full, f, metric, higher):
    auc = A.paired_auc(full)[OPTS]
    half = full[8].unstack('optimizer')[OPTS].dropna()
    fr_auc = T.friedman(auc.to_numpy())
    outer = f.pivot_table(index=['model', 'balance', 'fold'], columns='optimizer', values=f'outer_{metric}')[OPTS].dropna()
    rows = []
    for a, b in [('bayes', 'grid'), ('bayes', 'random'), ('genetic', 'grid'), ('genetic', 'random'), ('bayes', 'genetic'), ('random', 'grid')]:
        for label, tab, sign in [('AUC anytime', auc, 1), ('calidad en t=8', half, 1), (f'{metric} externo', outer, 1 if higher else -1)]:
            d = (tab[a] - tab[b]) * sign
            p = stats.wilcoxon(tab[a], tab[b]).pvalue if (d != 0).any() else 1.0
            rows.append({'comparación': f'{M.PRETTY[a]} vs {M.PRETTY[b]}', 'medida': label, 'diferencia media (a − b, + = a mejor)': d.mean(),
                         'a gana en (% celdas)': (d > 0).mean() * 100, 'p Wilcoxon': p,
                         'delta de Cliff': T.cliffs_delta(tab[a] * sign, tab[b] * sign)})
    out = pd.DataFrame(rows)
    for m_ in out['medida'].unique():
        idx = out['medida'] == m_
        out.loc[idx, 'p ajustado (Holm)'] = T.adjust(out.loc[idx, 'p Wilcoxon'].to_numpy())
    return fr_auc, out
fr_c, bt_c = budget_tests(full_c, fc, 'recall', True)
fr_r, bt_r = budget_tests(full_r, fr, 'rmse', False)
print(f"Friedman sobre el AUC anytime — clasificación: chi2 = {fr_c['chi2']:.1f}, p = {fr_c['p_value']:.2g}; "
      f"rangos {dict(zip(OPTS, np.round(fr_c['avg_ranks'], 2)))}")
print(f"Friedman sobre el AUC anytime — regresión: chi2 = {fr_r['chi2']:.1f}, p = {fr_r['p_value']:.2g}; "
      f"rangos {dict(zip(OPTS, np.round(fr_r['avg_ranks'], 2)))}")
bt_c.round(4)

# %%
bt_r.round(4)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 7.5 Tiempo de reloj
#
# El presupuesto se fijó en evaluaciones, no en tiempo. El costo por evaluación depende de la
# configuración elegida (más árboles o más vecinos cuestan más), así que los métodos que exploran
# zonas caras gastan más reloj con el mismo número de evaluaciones.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, anyt, title in [(axes[0], any_c, 'Clasificación'), (axes[1], any_r, 'Regresión')]:
    d = anyt.dropna(subset=['best_norm']).copy()
    d['frac_time'] = d.cum_seconds / d.groupby(['model', 'balance', 'fold']).cum_seconds.transform('max')
    grid_t = np.linspace(0, 1, 41)
    for o in OPTS:
        curves = []
        for _, x in d[d.optimizer == o].groupby(['model', 'balance', 'fold']):
            curves.append(np.interp(grid_t, x.frac_time, x.best_norm, left=0))
        ax.plot(grid_t, np.mean(curves, axis=0), color=P.OPT_COLORS[o], label=M.PRETTY[o])
    ax.set_xlabel('fracción del mayor tiempo de búsqueda de la celda'); ax.set_title(title)
axes[0].set_ylabel('mejor puntaje interno (normalizado)'); axes[0].legend(fontsize=8, loc='lower right')
P.save(fig, 'e3_hpo_anytime_tiempo'); plt.show()

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 7.6 Optuna: historia de una búsqueda y superficie explorada
#
# Historia de la búsqueda bayesiana de XGBoost (sin balanceo, pliegue externo 1) y todas las
# configuraciones de XGBoost evaluadas por los cuatro métodos en ese pliegue, en el plano
# learning rate × profundidad (color = balanced accuracy interna).

# %%
cells = {(r['optimizer']): r for r in N.load_cells('clf') if r['model'] == 'xgboost' and r['balance'] == 'none' and r['fold'] == 0}
h = pd.DataFrame(cells['bayes']['history'])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].scatter(h['eval'], h.score, color=P.OPT_COLORS['bayes'], s=30, zorder=3, label='configuración evaluada')
axes[0].step(h['eval'], h.score.cummax(), where='post', color=P.INK, lw=1.5, label='mejor hasta t')
axes[0].axvline(5.5, color=P.INK2, ls='--', lw=1); axes[0].text(5.6, h.score.min(), ' fin de la fase aleatoria', fontsize=8, color=P.INK2)
axes[0].set_xlabel('prueba (t)'); axes[0].set_ylabel('balanced accuracy interna'); axes[0].set_title('Optuna (TPE): historia de la búsqueda')
axes[0].legend(fontsize=8, loc='lower right')
markers = {'grid': 's', 'random': 'o', 'bayes': '^', 'genetic': 'D'}
allh = pd.concat([pd.DataFrame(c['history']).assign(optimizer=o) for o, c in cells.items()])
allh['lr'] = allh.params.map(lambda p: p['learning_rate']); allh['depth'] = allh.params.map(lambda p: p['max_depth'])
sc = None
for o in OPTS:
    x = allh[allh.optimizer == o]
    sc = axes[1].scatter(x.lr, x.depth + (OPTS.index(o) - 1.5) * 0.12, c=x.score, cmap='Blues', vmin=allh.score.min(), vmax=allh.score.max(),
                         marker=markers[o], s=45, edgecolor=P.INK2, linewidth=0.5, label=M.PRETTY[o])
axes[1].set_xscale('log'); axes[1].set_xlabel('learning rate (log)'); axes[1].set_ylabel('max_depth')
axes[1].set_title('Configuraciones de XGBoost evaluadas (pliegue 1)'); axes[1].legend(fontsize=7.5, loc='upper right')
plt.colorbar(sc, ax=axes[1], label='balanced accuracy interna')
P.save(fig, 'e3_hpo_optuna'); plt.show()

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 7.7 Algoritmo genético: diversidad de la población
#
# Diversidad = distancia euclidiana media entre pares de individuos en el espacio de genes,
# dividida por √d (0 = todos iguales, ~0,4 = población uniforme al azar). Una caída rápida a cero
# indicaría convergencia prematura.

# %%
gd = pd.concat([A.ga_diversity('clf').assign(tarea='Clasificación'), A.ga_diversity('reg').assign(tarea='Regresión')])
gsum = gd.groupby(['tarea', 'generation']).agg(diversidad=('diversity', 'mean'), sd=('diversity', 'std'),
                                                 n=('diversity', 'size'), evals=('evals', 'mean')).reset_index()
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for i, (t, x) in enumerate(gsum.groupby('tarea')):
    ax.errorbar(x.generation, x.diversidad, yerr=x.sd, marker='o', ms=6, capsize=3, color=P.SERIES[i], label=t)
ax.set_xlabel('generación'); ax.set_ylabel('diversidad genética (media ± d.e.)')
ax.set_title('Diversidad de la población del algoritmo genético'); ax.legend(fontsize=8); ax.set_ylim(0, None)
ax.set_xticks(sorted(gsum.generation.unique()))
P.save(fig, 'e3_hpo_ga_diversidad'); plt.show()
gsum.round(3)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 7.8 Sesgo de selección: puntaje interno frente a externo
#
# El mejor puntaje interno es el máximo de 16 estimaciones ruidosas, así que tiende a ser
# optimista. La validación anidada existe para no reportarlo. La diferencia interno − externo mide
# cuánto se habría sobreestimado el desempeño.

# %%
sb_c = A.selection_bias('clf', 'recall'); sb_r = A.selection_bias('reg', 'rmse')
tab = pd.DataFrame({
    'Clasificación: interno − externo (balanced acc.)': sb_c.groupby('optimizer').optimismo.mean(),
    'Regresión: externo − interno (RMSE)': sb_r.groupby('optimizer').optimismo.mean(),
}).reindex(OPTS)
print('Media global clasificación:', round(sb_c.optimismo.mean(), 4), '| % de celdas con interno > externo:', round((sb_c.optimismo > 0).mean() * 100, 1))
print('Media global regresión:', round(sb_r.optimismo.mean(), 4), '| % de celdas con RMSE externo > interno:', round((sb_r.optimismo > 0).mean() * 100, 1))
tab.round(4)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 7.9 Optimización multi-fidelidad: Successive Halving
#
# Se compara, en el pliegue externo 1 y para XGBoost sin balanceo, una búsqueda aleatoria de 27
# configuraciones evaluadas con presupuesto completo (400 árboles) contra **Successive Halving**
# (`HalvingRandomSearchCV`) con las mismas 27 configuraciones iniciales:
#
# - **Fidelidad = número de árboles.** Es el recurso natural del boosting: los primeros árboles
#   ya ordenan bien las configuraciones (un learning rate o una profundidad malos se notan pronto) y
#   el costo crece linealmente con él.
# - **Factor de reducción η = 3.** En cada ronda sobrevive el mejor tercio y recibe el triple de
#   árboles: 27 configuraciones con ~15 árboles → 9 con ~44 → 3 con ~133 → 1 con 400. Con η = 3 hay
#   4 rondas, un equilibrio entre descartar demasiado pronto (η grande) y gastar mucho en las malas
#   (η pequeño).

# %%
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import time
def run_halving():
    X, y, g, s = N.task_data('clf')
    tr, te = N.outer_folds('clf')[0]
    folds = N.inner_folds('clf', tr, 0)
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]
    pipe = Pipeline([('onehot', M.make_encoder()),
                     ('model', XGBClassifier(n_estimators=400, tree_method='hist', n_jobs=1, random_state=C.SEED, verbosity=0))])
    dist = {'model__max_depth': randint(2, 11), 'model__learning_rate': loguniform(0.01, 0.3),
            'model__subsample': uniform(0.6, 0.4), 'model__colsample_bytree': uniform(0.4, 0.6),
            'model__min_child_weight': loguniform(1, 20)}
    out = {}
    for name, search in [
        ('Random (27 × 400 árboles)', RandomizedSearchCV(pipe, dist, n_iter=27, cv=folds, scoring='balanced_accuracy',
                                                       random_state=C.SEED, n_jobs=11, refit=True)),
        ('Successive Halving (η = 3)', HalvingRandomSearchCV(pipe, dist, n_candidates=27, factor=3, resource='model__n_estimators',
                                                           min_resources=15, max_resources=400, cv=folds, scoring='balanced_accuracy',
                                                           random_state=C.SEED, n_jobs=11, refit=True))]:
        t0 = time.perf_counter(); search.fit(Xtr, ytr); sec = time.perf_counter() - t0
        res = pd.DataFrame(search.cv_results_)
        trees = res['param_model__n_estimators'].astype(float) if 'param_model__n_estimators' in res else pd.Series(400.0, index=res.index)
        out[name] = {'segundos (reloj, 11 procesos)': sec, 'árboles ajustados (suma × 3 pliegues)': float(trees.sum() * 3),
                     'configuraciones evaluadas': len(res), 'mejor interno': search.best_score_,
                     'balanced acc. externa': S.score_predictions('clf', y[te], search.best_estimator_, X.iloc[te]),
                     'rondas': int(getattr(search, 'n_iterations_', 1))}
    return pd.DataFrame(out).T
halv = cache.cached('hpo_halving', run_halving)
halv

# %% [markdown]
# **Interpretación.** TODO
