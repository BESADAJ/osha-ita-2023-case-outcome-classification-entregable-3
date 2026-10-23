# %% [markdown]
# # 20. Validación estadística del modelo original (R3)
#
# Este capítulo verifica si el modelo original (ensamble multivista + SECR, capítulo 19) **supera** a los benchmarks
# según la definición operativa de la guía (§6.1):
#
# 1. su BA media del bucle externo es mayor que la del mejor benchmark y que la de la mejor combinación de cada uno de
#    los 7 modelos;
# 2. al menos tres pruebas de hipótesis distintas rechazan la igualdad, con p ajustado por Holm < 0,05, y al menos una
#    es global;
# 3. el tamaño del efecto no es despreciable: |δ de Cliff| ≥ 0,147.
#
# **Protocolo común (guía §5.3 y §6.2).**
#
# - **Mismas variables.** Los 7 benchmarks (la mejor combinación de balanceo y optimizador de cada modelo, capítulo 13b)
#   se reentrenan con las mismas variables crudas que usa el original: las 9 del curso, los códigos OIICS de la lesión,
#   NAICS-6, SOC-6 y el empleador. Cada uno con su propio preprocesamiento aprendido dentro del pliegue.
# - **Mismas particiones.** Validación externa repetida 5×5 por establecimiento. La repetición 0 es la partición de los
#   capítulos 6 a 13; las repeticiones 1 a 4 usan semillas `seed_for('outer', r)` fijadas antes de correr.
# - **Mismo presupuesto.** Búsqueda interna de 16 configuraciones con 3 pliegues internos para todos. Los benchmarks
#   buscan con su optimizador ganador; el original busca sus hiperparámetros del combinador con TPE.
# - **La prueba de octubre a diciembre no se usa** (está bloqueada por código): todas las pruebas son sobre pliegues
#   externos. Por eso McNemar y DeLong «sobre el conjunto de prueba» (Figura 4 de la guía) se reemplazan por pruebas
#   sobre pliegues que el Cuadro 3 admite: Nadeau–Bengio, Wilcoxon–Holm y la prueba bayesiana con ROPE.
#
# Código: `src/osha_e3/paired.py` (corrida), `src/osha_e3/original_eval.py` (TPE anidado y ablación),
# `src/osha_e3/stats_tests.py` (pruebas) y `scripts/analyze_paired.py` (tablas). Las corridas cuestan horas y sus
# resultados están versionados en `results/pareado/`; este capítulo los lee.

# %%
import sys, json
sys.path.insert(0, '../src')
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
from osha_e3 import config as C
R = C.RESULTS / 'pareado'
res = json.loads((R / 'resumen.json').read_text(encoding='utf-8'))
f = pd.read_csv(R / 'pliegues.csv')
PRETTY = {'original': 'Original (SECR)', 'knn': 'KNN', 'naive_bayes': 'Naive Bayes', 'logreg': 'Reg. logística',
          'tree': 'Árbol', 'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'svm': 'SVM'}
print('Celdas (repetición × pliegue) por modelo:', res['n_cells'])
print('Universo con las 25 celdas:', [PRETTY[m] for m in res['universe_25']])

# %% [markdown]
# ## 20.1 Tabla resumen
#
# Media ± desviación estándar de la balanced accuracy en los pliegues externos, intervalo bootstrap BCa del 95 % de la
# media, otras métricas y tiempos (entrenamiento = búsqueda + reajuste por pliegue; inferencia por caso). La columna
# «BA rep. 0» usa solo los 5 pliegues de la repetición 0, que todos los modelos tienen.

# %%
s = pd.DataFrame(res['summary_raw'])
tab = pd.DataFrame({
    'Modelo': s.model.map(PRETTY), 'Balanceo': s.balance, 'Optimizador': s.optimizer, 'Pliegues': s.n_pliegues,
    'BA media ± d.e.': [f'{m:.4f} ± {d:.4f}' for m, d in zip(s.ba_mean, s.ba_sd)],
    'IC BCa 95 %': [f'[{a:.4f}, {b:.4f}]' for a, b in zip(s.ba_ci_lo, s.ba_ci_hi)],
    'BA rep. 0': s.ba_rep0_mean.round(4), 'F1 macro': s.f1_mean.round(4), 'AUC macro': s.auc_mean.round(4),
    'Log-loss': s.log_loss_mean.round(4), 'Entrenamiento (s/pliegue)': s.train_s_mean.round(0),
    'Inferencia (ms/caso)': s.ms_case_mean.round(3)})
tab

# %% [markdown]
# ## 20.2 Prueba global: Friedman y diagrama de diferencia crítica
#
# Friedman (con la corrección F de Iman y Davenport) compara todos los modelos a la vez, con los pliegues como bloques
# y **el modelo original incluido en el universo**. Si rechaza, Nemenyi indica qué pares difieren: dos modelos cuya
# distancia de rangos medios es menor que la diferencia crítica (CD) no se distinguen. Se hace con (a) los modelos que
# tienen las 25 celdas y (b) los 8 modelos en los 5 pliegues de la repetición 0.

# %%
for tag, key, cd in [('25 pliegues', 'friedman25', 'cd25'), ('repetición 0 (5 pliegues)', 'friedman_rep0', 'cd_rep0')]:
    fr = res[key]
    print(f"{tag}: k = {fr['k']}, N = {fr['n_blocks']}, χ² = {fr['chi2']:.2f}, p = {fr['p_value']:.2e}; "
          f"F de Iman-Davenport = {fr['iman_davenport_F']:.2f}, p = {fr['p_value_F']:.2e}; CD = {res[cd]:.3f}")
display(Image('figures/e3_cd_pareado_25.png'))
display(Image('figures/e3_cd_pareado_rep0.png'))

# %% [markdown]
# ## 20.3 Pruebas pareadas frente al mejor benchmark y a cada benchmark
#
# Tabla obligatoria de la guía (§6.3): prueba, hipótesis nula, supuestos verificados, estadístico, p crudo, p ajustado,
# tamaño del efecto y conclusión. Las 4 pruebas frecuentistas forman una familia y se ajustan con Holm (4 comparaciones).
# La bayesiana no da un p-valor: da probabilidades posteriores de que el original sea peor, equivalente (dentro de la
# ROPE de ±0,01 de BA) o mejor.

# %%
pd.DataFrame(res['tests'])

# %%
vs = pd.DataFrame(res['vs_each'])
vs['benchmark'] = vs.benchmark.map(PRETTY)
print('Wilcoxon de rangos con signo (unilateral, H1: el original es mejor) contra cada benchmark, Holm sobre',
      len(vs), 'comparaciones:')
vs.round(4)

# %%
display(Image('figures/e3_ba_pliegues_pareado.png'))
print('Supuestos de Nadeau-Bengio: Shapiro-Wilk sobre las diferencias, p =', round(res['shapiro_p'], 3))
print('IC BCa 95 % de la diferencia media de BA (original − mejor):', np.round(res['diff_ci_bca'], 4))
print('δ de Cliff [IC 95 %]:', np.round(res['cliff'], 3))

# %% [markdown]
# ## 20.4 Ablación: ¿qué aporta cada componente?
#
# Mismos modelos base, mismas predicciones y mismos 25 pliegues. Se retira un componente cada vez, o la regla de
# combinación se reemplaza por una regla estándar (Cuadro 1 de la guía: promedio simple, pesos por error global,
# stacking lineal). La columna «p (Holm)» es el Wilcoxon unilateral de «SECR completo es mejor que la variante».

# %%
a = pd.DataFrame(res['ablation'])
a.round(4)

# %% [markdown]
# ## 20.5 Convergencia de Frank-Wolfe
#
# La curva muestra la log-loss fuera de pliegue y la brecha de Frank-Wolfe por iteración (pesos globales,
# repetición 0, pliegue 0). La pérdida nunca sube (proposición 2) y la brecha, que acota la distancia al óptimo
# (proposición 3), cae varios órdenes de magnitud.

# %%
display(Image('figures/e3_convergencia_fw.png'))

# %% [markdown]
# ## 20.6 Veredicto

# %%
display(Markdown('> **' + res['verdict'] + '**'))
