# %% [markdown]
# # 9. Evaluación de los modelos de clasificación
#
# Dos niveles de evaluación:
#
# 1. **Validación anidada (bucle externo).** Las 112 combinaciones, media ± desviación estándar en los
#    5 pliegues externos. Con ella se elige la mejor combinación de cada modelo (sin mirar la prueba).
# 2. **Prueba (octubre–diciembre, establecimientos no vistos, 127.391 casos).** Cada combinación se
#    reajusta en toda la submuestra de desarrollo con sus hiperparámetros finales y se evalúa una vez.
#
# La prueba ya se había usado en los entregables 1 y 2 para reportar la regresión logística; no es
# una prueba virgen (ver límites en el capítulo de conclusiones).

# %%
import sys, json, warnings
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from osha_e3 import config as C, data as D, models as M, nested as N, final as F, evaluation as E, plots as P, search as S, cache
P.setup()
mc = N.master_table('clf'); tc = F.test_table('clf')
test, yt = F.test_data('clf')
OPTS, BALS, MODELS = M.OPTIMIZERS, M.BALANCES, P.MODEL_ORDER_CLF
mc['label'] = mc.model.map(M.PRETTY) + ' · ' + mc.balance.map(M.PRETTY) + ' · ' + mc.optimizer.map(M.PRETTY)
fmt = lambda m, s: f'{m:.3f} ± {s:.3f}'

# %% [markdown]
# ## 9.1 Las 112 combinaciones en la validación anidada
#
# Tabla maestra completa: `results/master_clf.csv` (una fila por combinación, con hiperparámetros
# finales, métricas externas, tiempos y semilla). Aquí, las 15 mejores por balanced accuracy externa.

# %%
cols = ['recall', 'f1', 'auc', 'accuracy', 'precision', 'brier', 'ece']
top = mc.sort_values('outer_recall_mean', ascending=False).head(15)
pd.DataFrame({'Combinación': top.label,
              **{c: [fmt(a, b) for a, b in zip(top[f'outer_{c}_mean'], top[f'outer_{c}_std'])] for c in cols}}
             ).rename(columns={'recall': 'Balanced acc. (recall macro)', 'f1': 'F1 macro', 'auc': 'AUC macro',
                               'accuracy': 'Accuracy', 'precision': 'Precision macro', 'brier': 'Brier', 'ece': 'ECE'}).set_index('Combinación')

# %%
heat = mc.pivot_table(index='model', columns='balance', values='outer_recall_mean', aggfunc='max').reindex(index=MODELS, columns=BALS)
fig, ax = plt.subplots(figsize=(6.8, 4.2))
im = ax.imshow(heat.to_numpy(), cmap='Blues', aspect='auto')
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.iat[i, j]
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8.5,
                color='white' if v > np.nanpercentile(heat.to_numpy(), 70) else P.INK)
ax.set_xticks(range(len(BALS)), [M.PRETTY[b] for b in BALS]); ax.set_yticks(range(len(MODELS)), [M.PRETTY[m] for m in MODELS])
ax.grid(False); ax.set_title('Balanced accuracy externa (mejor optimizador de cada celda)')
plt.colorbar(im, ax=ax, fraction=0.04)
P.save(fig, 'e3_clf_heatmap'); plt.show()

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 9.2 Mejor combinación de cada modelo en la prueba
#
# Para cada uno de los 7 modelos se toma la combinación (balanceo × optimizador) con mejor
# balanced accuracy externa **media** en la validación anidada; la prueba no interviene en la
# elección. Además se incluye la regresión logística del entregable 2 (C = 1, L2, sin búsqueda)
# como referencia.

# %%
best = mc.loc[mc.groupby('model').outer_recall_mean.idxmax()].set_index('model').reindex(MODELS)
best_combo = best.combo.to_dict()
tt = tc.set_index('combo')
res = pd.DataFrame({
    'Combinación elegida': [f'{M.PRETTY[best.loc[m, "balance"]]} · {M.PRETTY[best.loc[m, "optimizer"]]}' for m in MODELS],
    'Balanced acc. anidada': [fmt(best.loc[m, 'outer_recall_mean'], best.loc[m, 'outer_recall_std']) for m in MODELS],
    **{k: [tt.loc[best_combo[m], f'test_{v}'] for m in MODELS] for k, v in
       [('Accuracy', 'accuracy'), ('Precision', 'precision'), ('Recall (balanced acc.)', 'recall'),
        ('F1-score', 'f1'), ('AUC', 'auc'), ('Brier', 'brier'), ('ECE', 'ece')]}
}, index=[M.PRETTY[m] for m in MODELS])
res.round(3)

# %% [markdown]
# Precision, recall y F1 son promedios macro de las 4 clases; AUC es el promedio macro de las
# curvas una-contra-el-resto. Como referencia, el entregable 2 reportó para su regresión logística
# (C = 1, L2, `class_weight='balanced'`, 1.433 columnas, entrenada con todo el desarrollo) un
# balanced accuracy de 0,40, F1 macro de 0,40 y AUC macro de 0,69 en esta misma prueba.
#
# **Interpretación.** TODO

# %% [markdown]
# ## 9.3 Matrices de confusión

# %%
probs = {m: F.load('clf', best_combo[m])[0] for m in MODELS}
fig, axes = plt.subplots(2, 4, figsize=(14, 6.8))
short = ['Fallec.', 'Ausencia', 'Trasl./restr.', 'Otro']
for ax, m in zip(axes.flat, MODELS):
    cm = confusion_matrix(yt, probs[m].argmax(1), labels=range(4))
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{pct[i, j]:.0f}%\n({cm[i, j]:,})', ha='center', va='center', fontsize=7,
                    color='white' if pct[i, j] > 55 else P.INK)
    ax.set_xticks(range(4), short, fontsize=7.5); ax.set_yticks(range(4), short, fontsize=7.5)
    ax.set_title(M.PRETTY[m], fontsize=10); ax.grid(False)
axes[1, 3].axis('off')
fig.supxlabel('clase predicha'); fig.supylabel('clase real (cada fila suma 100 %)')
fig.tight_layout(); P.save(fig, 'e3_clf_confusion'); plt.show()

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 9.4 Curvas ROC
#
# Una curva por clase (una contra el resto) para cada modelo. La de Fallecimiento se basa en 39
# casos y es muy irregular.

# %%
fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), sharex=True, sharey=True)
for ax, m in zip(axes.flat, MODELS):
    for k in range(4):
        fpr, tpr, _ = roc_curve(yt == k, probs[m][:, k])
        ax.plot(fpr, tpr, color=P.CLASS_COLORS[k], lw=1.6, label=f'{short[k]} ({roc_auc_score(yt == k, probs[m][:, k]):.2f})')
    ax.plot([0, 1], [0, 1], color=P.INK2, lw=0.8, ls='--')
    ax.set_title(f'{M.PRETTY[m]} (AUC macro {tt.loc[best_combo[m], "test_auc"]:.3f})', fontsize=9.5)
    ax.legend(fontsize=7, loc='lower right')
axes[1, 3].axis('off')
fig.supxlabel('tasa de falsos positivos'); fig.supylabel('tasa de verdaderos positivos (recall)')
fig.tight_layout(); P.save(fig, 'e3_clf_roc'); plt.show()

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 9.5 Balanceo de clases: con y sin
#
# Diferencia en la validación anidada entre cada técnica y «sin balanceo», con el mismo modelo y el
# mismo optimizador (pareado), promediada sobre los 4 optimizadores. Además, el recall por clase
# en la prueba del mejor optimizador de cada (modelo, balanceo).

# %%
fcl = N.fold_table('clf')
rows = []
for met, lab in [('recall', 'Balanced acc.'), ('f1', 'F1 macro'), ('auc', 'AUC macro'), ('accuracy', 'Accuracy')]:
    pv = fcl.pivot_table(index=['model', 'optimizer', 'fold'], columns='balance', values=f'outer_{met}')
    for b in ['smote', 'adasyn', 'class_weight']:
        d = (pv[b] - pv['none']).groupby(level='model').mean()
        for m, v in d.items():
            rows.append({'métrica': lab, 'modelo': M.PRETTY[m], 'balanceo': M.PRETTY[b], 'Δ vs sin balanceo': v})
delta = pd.DataFrame(rows).pivot_table(index='modelo', columns=['métrica', 'balanceo'], values='Δ vs sin balanceo')
delta = delta.reindex([M.PRETTY[m] for m in MODELS])
delta[['Balanced acc.', 'F1 macro', 'AUC macro', 'Accuracy']].round(3)

# %%
rec = []
for m in MODELS:
    for b in BALS:
        sub = mc[(mc.model == m) & (mc.balance == b)]
        combo = sub.loc[sub.outer_recall_mean.idxmax(), 'combo']
        p = F.load('clf', combo)[0].argmax(1)
        r = classification_report(yt, p, labels=range(4), output_dict=True, zero_division=0)
        rec.append({'modelo': M.PRETTY[m], 'balanceo': M.PRETTY[b], **{f'recall {short[k]}': r[str(k)]['recall'] for k in range(4)},
                    'fallecimientos detectados (de 39)': int(((p == 0) & (yt == 0)).sum()),
                    'casos predichos como fallecimiento': int((p == 0).sum())})
pd.DataFrame(rec).set_index(['modelo', 'balanceo']).round(3)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 9.6 Tratamiento de la clase Fallecimiento
#
# Corrección del entregable 1: el profesor señaló que la clase Fallecimiento quedó sin tratamiento.
# En este entregable se trata con las tres técnicas de balanceo de la guía (SMOTE, ADASYN y
# `class_weight`) y se mide su efecto con intervalos, porque con 39 casos en la prueba cualquier
# recall es muy incierto. Intervalo de Wilson al 95 % para el recall (aciertos de 39) y para la
# precisión (aciertos entre los casos predichos como fallecimiento).
#
# Además, el balanced accuracy promedia el recall de las 4 clases: acertar **un** fallecimiento más de
# 39 sube la métrica 0,25/39 ≈ 0,006, y en un pliegue externo con 1 fallecimiento la sube 0,25. Por eso
# se reporta también el balanced accuracy de las **3 clases frecuentes** (sin Fallecimiento), que es
# mucho más estable.

# %%
def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
rows = []
for m in MODELS:
    for b in BALS:
        sub = mc[(mc.model == m) & (mc.balance == b)]
        combo = sub.loc[sub.outer_recall_mean.idxmax(), 'combo']
        pr = F.load('clf', combo)[0]; pred = pr.argmax(1)
        hits, n_pred, n_true = int(((pred == 0) & (yt == 0)).sum()), int((pred == 0).sum()), int((yt == 0).sum())
        rlo, rhi = wilson(hits, n_true); plo, phi = wilson(hits, n_pred)
        rec = [((pred == k) & (yt == k)).sum() / (yt == k).sum() for k in range(4)]
        rows.append({'modelo': M.PRETTY[m], 'balanceo': M.PRETTY[b], 'aciertos (de 39)': hits, 'predichos como fallec.': n_pred,
                     'recall fallec.': hits / n_true, 'IC Wilson recall': f'[{rlo:.3f}, {rhi:.3f}]',
                     'precisión fallec.': hits / n_pred if n_pred else np.nan,
                     'IC Wilson precisión': f'[{plo:.4f}, {phi:.4f}]' if n_pred else '—',
                     'AUC fallec. (1 vs resto)': roc_auc_score(yt == 0, pr[:, 0]),
                     'balanced acc. (4 clases)': np.mean(rec), 'balanced acc. (3 clases frecuentes)': np.mean(rec[1:])})
death = pd.DataFrame(rows).set_index(['modelo', 'balanceo'])
death.round(4)

# %%
ba3 = death['balanced acc. (3 clases frecuentes)'].unstack('balanceo')[[M.PRETTY[b] for b in BALS]]
ba4 = death['balanced acc. (4 clases)'].unstack('balanceo')[[M.PRETTY[b] for b in BALS]]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=False)
for ax, tab, title in [(axes[0], ba4, 'Balanced accuracy, 4 clases (prueba)'), (axes[1], ba3, 'Balanced accuracy, 3 clases frecuentes (prueba)')]:
    x = np.arange(len(tab)); w = 0.2
    for j, b in enumerate(BALS):
        ax.bar(x + (j - 1.5) * w, tab[M.PRETTY[b]], width=w - 0.02, color=P.BAL_COLORS[b], label=M.PRETTY[b])
    ax.set_xticks(x, tab.index, rotation=30, ha='right', fontsize=8); ax.set_title(title)
    lo, hi = np.nanmin(tab.to_numpy()), np.nanmax(tab.to_numpy()); ax.set_ylim(lo - (hi - lo) * 0.5, hi + (hi - lo) * 0.2)
    ax.grid(axis='x', visible=False)
axes[0].legend(fontsize=8, ncol=2)
fig.tight_layout(); P.save(fig, 'e3_clf_death_ba3'); plt.show()

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 9.7 Calibración de probabilidades
#
# - **Brier multiclase:** media de Σₖ (pₖ − 1[y = k])²; 0 es perfecto. El Dummy que predice las
#   frecuencias de la prueba obtendría ≈ 0,664 (entregable 2).
# - **ECE:** con 15 intervalos de la confianza de la clase predicha, promedio ponderado de
#   |accuracy − confianza|.
# - **Diagrama de confiabilidad:** confianza predicha frente a la frecuencia observada de acierto,
#   y para la clase «días de ausencia», probabilidad predicha frente a la frecuencia real.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for i, m in enumerate(MODELS):
    p = probs[m]
    conf = p.max(1); correct = (p.argmax(1) == yt)
    fo, mp = calibration_curve(correct, conf, n_bins=10, strategy='quantile')
    axes[0].plot(mp, fo, marker='o', ms=4, color=P.MODEL_COLORS[m], label=M.PRETTY[m])
    fo2, mp2 = calibration_curve(yt == 1, p[:, 1], n_bins=10, strategy='quantile')
    axes[1].plot(mp2, fo2, marker='o', ms=4, color=P.MODEL_COLORS[m], label=M.PRETTY[m])
for ax, t in zip(axes, ['Confianza de la clase predicha', 'Clase «días de ausencia»']):
    ax.plot([0, 1], [0, 1], ls='--', color=P.INK2, lw=0.8)
    ax.set_xlabel('probabilidad predicha (promedio del intervalo)'); ax.set_ylabel('frecuencia observada'); ax.set_title(t)
axes[0].set_xlim(0.2, 1); axes[0].set_ylim(0.2, 1); axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=7.5, loc='upper left')
fig.tight_layout(); P.save(fig, 'e3_clf_calibracion'); plt.show()

# %% [markdown]
# ### Recalibración (Platt e isotónica)
#
# Para cada uno de los 7 modelos se reajusta el pipeline dentro de `CalibratedClassifierCV` con 3
# pliegues por establecimiento de la submuestra de desarrollo: en cada pliegue se entrena el modelo
# con dos tercios y se ajusta el calibrador (sigmoide de Platt o regresión isotónica, una contra el
# resto y luego normalizada) con el tercio restante. Así el calibrador nunca ve la prueba.

# %%
from sklearn.model_selection import StratifiedGroupKFold
def run_recal():
    X, y, g, s = N.task_data('clf')
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=C.seed_for('inner', 99)).split(X, y, g))
    rows, curves = [], {}
    for m in MODELS:
        r = best.loc[m]
        params = json.loads(r.final_params)
        rows.append({'modelo': M.PRETTY[m], 'método': 'sin recalibrar', **E.classification_metrics(yt, probs[m])})
        for method in ['sigmoid', 'isotonic']:
            pipe = M.make_pipeline('clf', m, params, r.balance, C.SEED)
            cal = CalibratedClassifierCV(pipe, method=method, cv=folds, n_jobs=3)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); cal.fit(X, y)
            p = cal.predict_proba(test[D.FEATURES])
            rows.append({'modelo': M.PRETTY[m], 'método': 'Platt (sigmoide)' if method == 'sigmoid' else 'isotónica',
                         **E.classification_metrics(yt, p)})
            curves[(m, method)] = calibration_curve(p.argmax(1) == yt, p.max(1), n_bins=10, strategy='quantile')
    return pd.DataFrame(rows), curves
recal, recal_curves = cache.cached('clf_recalibration', run_recal)
recal.set_index(['modelo', 'método'])[['brier', 'ece', 'log_loss', 'recall', 'auc']].round(4)

# %% [markdown]
# **Interpretación.** TODO

# %% [markdown]
# ## 9.8 Segunda prueba: mismos establecimientos, meses nuevos
#
# Corrección del entregable 1 (costo del diseño; ver sección 4.7). Los 83.912 casos de octubre a
# diciembre de los establecimientos de **desarrollo** quedaban descartados. Aquí se usan como una
# segunda prueba que solo separa por tiempo: el modelo ya vio esos establecimientos (en enero a
# septiembre), pero no esos meses. La diferencia con la prueba principal (establecimientos nuevos)
# mide cuánto cuesta, en desempeño, exigir que el modelo generalice a lugares que no conoce. Se usan
# los mismos 7 modelos, reajustados sobre la misma submuestra de desarrollo.

# %%
from osha_e3 import data as D
df_all = D.load_prepared()
dev_est = set(df_all.loc[df_all.split.eq('development'), 'establishment_id'])
temporal = df_all[df_all.establishment_id.isin(dev_est) & df_all.date.ge('2023-10-01')].reset_index(drop=True)
y_tmp = temporal.y_clf.to_numpy() - 1
print(f'Segunda prueba: {len(temporal):,} casos, {temporal.establishment_id.nunique():,} establecimientos, '
      f'{(y_tmp == 0).sum()} fallecimientos')
def run_temporal():
    X, y, _, _ = N.task_data('clf')
    out = {}
    for m in MODELS:
        r = best.loc[m]
        pipe = S.refit('clf', m, json.loads(r.final_params), r.balance, X, y, C.SEED)
        out[m] = pipe.predict_proba(temporal[D.FEATURES])
    return out
p_tmp = cache.cached('clf_temporal_only', run_temporal)
rows = []
for m in MODELS:
    a = E.classification_metrics(yt, probs[m]); b = E.classification_metrics(y_tmp, p_tmp[m])
    rows.append({'modelo': M.PRETTY[m], 'balanced acc. (establ. nuevos)': a['recall'], 'balanced acc. (mismos establ.)': b['recall'],
                 'AUC (establ. nuevos)': a['auc'], 'AUC (mismos establ.)': b['auc'],
                 'accuracy (establ. nuevos)': a['accuracy'], 'accuracy (mismos establ.)': b['accuracy']})
tmp_tab = pd.DataFrame(rows).set_index('modelo')
tmp_tab['Δ AUC (mismos − nuevos)'] = tmp_tab['AUC (mismos establ.)'] - tmp_tab['AUC (establ. nuevos)']
tmp_tab.round(4)

# %% [markdown]
# **Interpretación.** TODO
