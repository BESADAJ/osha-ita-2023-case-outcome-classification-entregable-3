# %% [markdown]
# # 11. Interpretabilidad global y local (SHAP y LIME)
#
# Los valores de Shapley reparten la predicción de un caso entre sus variables según la teoría de
# juegos cooperativos: la contribución de una variable es su aporte marginal promedio sobre todos
# los órdenes posibles de incorporación. Cumplen **eficiencia** (las contribuciones suman la
# diferencia entre la predicción y el valor base), **simetría** (dos variables que aportan lo mismo
# reciben lo mismo) y **aditividad** (en un ensamble, los SHAP del ensamble son la suma de los de sus
# partes). Se calculan en el espacio que ve el modelo (columnas one-hot), así que cada barra es una
# categoría concreta, por ejemplo `soc_minor=29-1` (profesionales de la salud).
#
# Se explica el mejor modelo de cada tarea (según la validación anidada) y, como pide la guía, el
# mejor XGBoost de clasificación con SHAP y LIME.

# %%
import sys, json, warnings
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
from osha_e3 import config as C, data as D, models as M, nested as N, final as F, plots as P, search as S, interpret as I, cache
P.setup()
mc, mr = N.master_table('clf'), N.master_table('reg')
bc = mc.loc[mc.outer_recall_mean.idxmax()]; br = mr.loc[mr.outer_rmse_mean.idxmin()]
bx = mc[mc.model == 'xgboost'].sort_values('outer_recall_mean').iloc[-1]
print('Mejor clasificación:', bc.combo, '| mejor regresión:', br.combo, '| mejor XGBoost:', bx.combo)
Xc, yc, _, _ = N.task_data('clf'); Xr, yr, _, _ = N.task_data('reg')
test_c, yt_c = F.test_data('clf'); test_r, yt_r = F.test_data('reg')
rng = np.random.default_rng(C.SEED)
ic = rng.choice(len(test_c), 1500, replace=False); ir = rng.choice(len(test_r), 1500, replace=False)
CLASS = 1  # «días de ausencia»: la clase más frecuente y la de mayor costo laboral

# %%
def explain(task, row, Xtr, ytr, Xex):
    pipe = S.refit(task, row.model, json.loads(row.final_params), row.balance, Xtr, ytr, C.SEED)
    vals, base, A, names = I.shap_values(task, row.model, pipe, Xtr.sample(1000, random_state=C.SEED), Xex)
    return pipe, vals, base, A, names
pipe_c, sv_c, base_c, A_c, names = cache.cached('shap_clf', lambda: explain('clf', bc, Xc, yc, test_c[D.FEATURES].iloc[ic]))
pipe_r, sv_r, base_r, A_r, names_r = cache.cached('shap_reg', lambda: explain('reg', br, Xr, yr, test_r[D.FEATURES].iloc[ir]))
print('SHAP clasificación:', sv_c.shape, '| regresión:', sv_r.shape)

# %% [markdown]
# ## 11.1 Clasificación: interpretabilidad global
#
# Gráfico resumen para la clase «días de ausencia» (salida del modelo antes de softmax, en log-odds
# si el modelo es de árboles o lineal). Cada punto es un caso de la prueba; el color indica si la
# categoría está presente (1, rojo) o no (0, azul).

# %%
sv1 = sv_c[:, :, CLASS] if sv_c.ndim == 3 else sv_c
plt.figure()
shap.summary_plot(sv1, A_c, feature_names=names, max_display=15, show=False, plot_size=(8, 5.5))
plt.title(f'SHAP, clase «días de ausencia» — {M.PRETTY[bc.model]} · {M.PRETTY[bc.balance]} · {M.PRETTY[bc.optimizer]}', fontsize=10)
P.save(plt.gcf(), 'e3_shap_clf_summary'); plt.show()

# %%
imp = pd.DataFrame({C.CLASS_NAMES[k]: np.abs(sv_c[:, :, k]).mean(0) for k in range(sv_c.shape[2])}, index=names) if sv_c.ndim == 3 \
    else pd.DataFrame({'|SHAP| medio': np.abs(sv_c).mean(0)}, index=names)
imp['total'] = imp.sum(axis=1)
by_var = imp.groupby(lambda n: n.split('=')[0]).sum().sort_values('total', ascending=False)
print('Importancia por variable original (suma de |SHAP| medio de sus categorías):')
display(by_var.round(4))
imp.sort_values('total', ascending=False).head(12).round(4)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 11.2 Clasificación: interpretabilidad local
#
# Dos casos de la muestra explicada: uno bien clasificado como «días de ausencia» con alta
# probabilidad, y uno cuya clase real es «días de ausencia» pero que el modelo asignó con
# confianza a otra clase (error elevado).

# %%
pc = pipe_c.predict_proba(test_c[D.FEATURES].iloc[ic]); yy = yt_c[ic]
ok = np.where((pc.argmax(1) == CLASS) & (yy == CLASS))[0]; ok = ok[np.argmax(pc[ok, CLASS])]
bad = np.where((pc.argmax(1) != CLASS) & (yy == CLASS))[0]; bad = bad[np.argmin(pc[bad, CLASS])]
fig = plt.figure(figsize=(13, 5))
for j, (i, title) in enumerate([(ok, 'bien clasificado'), (bad, 'error elevado')]):
    ax = fig.add_subplot(1, 2, j + 1); plt.sca(ax)
    b = base_c[CLASS] if np.ndim(base_c) else base_c
    shap.plots.waterfall(shap.Explanation(sv1[i], b, A_c[i], feature_names=names), max_display=10, show=False)
    ax.set_title(f'Caso {title}: P(días de ausencia) = {pc[i, CLASS]:.2f}, real = {C.CLASS_NAMES[yy[i]]}, '
                 f'predicho = {C.CLASS_NAMES[pc[i].argmax()]}', fontsize=9)
plt.tight_layout(); P.save(plt.gcf(), 'e3_shap_clf_waterfall'); plt.show()
pd.DataFrame({'bien clasificado': test_c[D.FEATURES].iloc[ic].iloc[ok], 'error elevado': test_c[D.FEATURES].iloc[ic].iloc[bad]})

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 11.3 Regresión: interpretabilidad global y local

# %%
plt.figure()
shap.summary_plot(sv_r, A_r, feature_names=names_r, max_display=15, show=False, plot_size=(8, 5.5))
plt.title(f'SHAP, log(1 + días) — {M.PRETTY[br.model]} · {M.PRETTY[br.optimizer]}', fontsize=10)
P.save(plt.gcf(), 'e3_shap_reg_summary'); plt.show()
imp_r = pd.Series(np.abs(sv_r).mean(0), index=names_r)
imp_r.groupby(lambda n: n.split('=')[0]).sum().sort_values(ascending=False).round(4).to_frame('suma |SHAP| medio')

# %%
pr = pipe_r.predict(test_r[D.FEATURES].iloc[ir]); er = np.abs(yt_r[ir] - pr)
good, worst = int(np.argmin(er)), int(np.argmax(er))
fig = plt.figure(figsize=(13, 5))
for j, (i, title) in enumerate([(good, 'bien predicho'), (worst, 'error elevado')]):
    ax = fig.add_subplot(1, 2, j + 1); plt.sca(ax)
    shap.plots.waterfall(shap.Explanation(sv_r[i], float(np.ravel(base_r)[0]), A_r[i], feature_names=names_r), max_display=10, show=False)
    ax.set_title(f'Caso {title}: real {np.expm1(yt_r[ir][i]):.0f} días, predicho {np.expm1(pr[i]):.0f} días', fontsize=9.5)
plt.tight_layout(); P.save(plt.gcf(), 'e3_shap_reg_waterfall'); plt.show()

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 11.4 XGBoost: SHAP frente a LIME
#
# Se explica el mismo caso con las dos técnicas, para la clase «días de ausencia».
#
# - **SHAP (TreeExplainer)** calcula valores de Shapley exactos para el ensamble de árboles,
#   relativos a una distribución de fondo. Las contribuciones suman exactamente la predicción
#   (log-odds) menos el valor base y son consistentes entre casos.
# - **LIME** perturba el caso (aquí, apaga y prende categorías al azar), pondera las perturbaciones
#   por cercanía y ajusta un modelo **lineal local** sobre la probabilidad. Sus coeficientes
#   dependen del núcleo de proximidad, del muestreo y de la semilla, y no tienen por qué sumar la
#   predicción.

# %%
def run_xgb_explain():
    pipe = S.refit('clf', 'xgboost', json.loads(bx.final_params), bx.balance, Xc, yc, C.SEED)
    vals, base, A, nm = I.shap_values('clf', 'xgboost', pipe, Xc.sample(1000, random_state=C.SEED), test_c[D.FEATURES].iloc[ic[:300]])
    return pipe, vals, base, A, nm
pipe_x, sv_x, base_x, A_x, names_x = cache.cached('shap_xgb', run_xgb_explain)
px = pipe_x.predict_proba(test_c[D.FEATURES].iloc[ic[:300]])
case = int(np.argmax(px[:, CLASS]))
def run_lime():
    out = []
    for seed in [C.SEED, C.SEED + 1, C.SEED + 2]:
        lst, _ = I.lime_explain(pipe_x, Xc.sample(5000, random_state=C.SEED), test_c[D.FEATURES].iloc[ic[:300]].iloc[[case]],
                                CLASS, num_features=10, seed=seed)
        out.append(dict(lst))
    return out
lime_runs = cache.cached('lime_xgb', run_lime)
shap_case = pd.Series(sv_x[case, :, CLASS], index=names_x)
present = [n for n, a in zip(names_x, A_x[case]) if a > 0]
top_shap = shap_case.abs().sort_values(ascending=False).head(10).index
lime_df = pd.DataFrame(lime_runs).T
lime_df.columns = ['LIME semilla 42', 'LIME semilla 43', 'LIME semilla 44']
comp = pd.DataFrame({'SHAP (log-odds)': shap_case[top_shap], 'presente en el caso': [n in present for n in top_shap]})
print(f'Caso {case}: P(días de ausencia) = {px[case, CLASS]:.3f}')
print('Categorías presentes:', present)
display(comp.round(4))
lime_df.round(4)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
s = shap_case[top_shap][::-1]
axes[0].barh(range(len(s)), s.values, color=[P.SERIES[7] if v > 0 else P.SERIES[0] for v in s.values])
axes[0].set_yticks(range(len(s)), s.index, fontsize=8); axes[0].axvline(0, color=P.INK2, lw=0.8)
axes[0].set_title('SHAP (TreeExplainer): 10 mayores contribuciones', fontsize=10); axes[0].set_xlabel('contribución a los log-odds')
l = lime_df.iloc[:, 0].dropna().sort_values(key=np.abs).tail(10)
axes[1].barh(range(len(l)), l.values, color=[P.SERIES[7] if v > 0 else P.SERIES[0] for v in l.values])
axes[1].set_yticks(range(len(l)), l.index, fontsize=8); axes[1].axvline(0, color=P.INK2, lw=0.8)
axes[1].set_title('LIME (semilla 42): 10 mayores coeficientes locales', fontsize=10); axes[1].set_xlabel('coeficiente sobre la probabilidad')
for a in axes: a.grid(axis='y', visible=False)
fig.tight_layout(); P.save(fig, 'e3_shap_vs_lime'); plt.show()
overlap = len({k.rsplit('=', 1)[0] for k in lime_df.iloc[:, 0].dropna().index} & set(top_shap))
stab = [len(set(lime_df.iloc[:, a].dropna().index) & set(lime_df.iloc[:, b].dropna().index)) for a, b in [(0, 1), (0, 2), (1, 2)]]
print('Variables en común entre el top 10 de SHAP y el de LIME:', overlap, '| coincidencias del top 10 de LIME entre semillas:', stab)

# %% [markdown]
# **Interpretación.** TODO
#
# **Cuándo divergen.** (1) Con variables correlacionadas (aquí, sector y ocupación describen el
# mismo lugar de trabajo), LIME perturba cada categoría por separado y genera combinaciones
# irreales; SHAP intervencional también las usa, pero promedia sobre todos los órdenes. (2) Si el
# modelo es muy no lineal cerca del caso, la aproximación lineal de LIME es pobre. (3) LIME explica
# la *probabilidad* y SHAP aquí explica los *log-odds*: los signos coinciden pero las magnitudes no
# son comparables. (4) LIME depende de la semilla y del ancho del núcleo; SHAP para árboles es
# determinista.
