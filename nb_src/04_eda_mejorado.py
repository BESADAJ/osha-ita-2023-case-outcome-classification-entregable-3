# %% [markdown]
# # 4. EDA mejorado para el entregable 3
#
# Este capítulo amplía el EDA de los entregables 1 y 2 con lo que hizo falta para entrenar 140
# modelos: (1) agrupar los códigos de alta cardinalidad, (2) una variable nueva de horas trabajadas
# antes del incidente y (3) el objetivo de regresión. Todo se calcula **solo con el desarrollo**
# (enero a septiembre, 40 % de los establecimientos); la prueba no se mira aquí.
#
# :::{note}
# Las correcciones que el profesor pidió sobre el entregable 1 se listan en el capítulo 3
# (sección 3.8) a medida que se incorporan.
# :::

# %%
import sys, warnings
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from osha_e3 import config as C, data as D, plots as P
P.setup()
df = D.load_prepared()
dev = df[df.split.eq('development')].copy()
raw_codes = D.load_raw()[['naics_code', 'soc_code']].loc[df.index[df.split.eq('development')]] \
    if False else None  # los códigos completos se resumen abajo desde el CSV
print(f'Desarrollo: {len(dev):,} casos, {dev.establishment_id.nunique():,} establecimientos')

# %% [markdown]
# ## 4.1 Agrupar NAICS y SOC por jerarquía
#
# En los entregables anteriores NAICS (715 códigos) y SOC (598) entraban completos al one-hot:
# 1.433 columnas, 700 códigos NAICS con menos del 1 % de los casos. Con 140 modelos (entre ellos KNN,
# SVM y SMOTE, cuyo costo crece con el número de columnas) eso no es viable, y los propios
# entregables dejaron anotado como siguiente paso agrupar por jerarquía. Se usa el **subsector
# NAICS de 3 dígitos** y el **grupo menor SOC** (primeros 4 caracteres, p. ej. `51-2`). Además,
# dentro del pipeline el one-hot agrupa en «infrequent» las categorías con menos de 20 casos del
# entrenamiento de cada pliegue.

# %%
raw = D.load_raw()
raw = raw.loc[raw.id.isin(dev.id)]
def cramers_v(x, y):
    """V de Cramér con corrección de sesgo de Bergsma (2013)."""
    t = pd.crosstab(x, y).to_numpy()
    n = t.sum(); chi2 = stats.chi2_contingency(t, correction=False)[0]
    phi2 = chi2 / n; r, k = t.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1)))

y = dev.y_clf.to_numpy()
soc_full = raw.soc_code.replace({'9999': np.nan, '0000': np.nan}).fillna('Unknown').to_numpy()
comp = pd.DataFrame([
    ('NAICS', raw.naics_code.nunique(), cramers_v(raw.naics_code.to_numpy(), y),
     dev.naics3.nunique(), cramers_v(dev.naics3, y)),
    ('SOC', pd.Series(soc_full).nunique(), cramers_v(soc_full, y),
     dev.soc_minor.nunique(), cramers_v(dev.soc_minor, y)),
], columns=['Variable', 'Categorías (código completo)', 'V de Cramér (completo)',
            'Categorías (agrupado)', 'V de Cramér (agrupado)'])
comp.round(3)

# %% [markdown]
# **Interpretación.** En el desarrollo, agrupar reduce las categorías de NAICS de 715 a 97 y las de
# SOC de 599 a 98 (unas 6 veces menos). La V de Cramér con el resultado baja alrededor de un 10 %
# (0,236 → 0,213 en NAICS; 0,206 → 0,182 en SOC). La mayor
# parte de la señal está en el nivel de subsector y grupo ocupacional; el detalle fino aporta
# poco y, con pocos casos por código, sobre todo varianza. Con el one-hot agrupado el modelo pasa
# de 1.433 a unas 210 columnas.

# %% [markdown]
# ## 4.2 Asociación de cada variable con el resultado

# %%
assoc = pd.Series({c: cramers_v(dev[c], y) for c in D.FEATURES}).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.barh([c for c in assoc.index], assoc.values, color=P.SERIES[0], height=0.6)
for i, v in enumerate(assoc.values):
    ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=8, color=P.INK2)
ax.set_xlabel('V de Cramér con el resultado (corregida)')
ax.set_title('Asociación de cada variable con el resultado (desarrollo)')
ax.grid(axis='y', visible=False)
P.save(fig, 'e3_eda_cramer'); plt.show()

# %% [markdown]
# **Interpretación.** El orden coincide con los entregables anteriores: el subsector y el grupo
# ocupacional son lo más asociado con el resultado, seguidos por el tipo de incidente y el estado;
# las variables de hora casi no aportan. Son asociaciones de a una variable: no dicen cuánto aporta
# cada una cuando las demás ya están en el modelo (eso lo mide SHAP en el capítulo 11).

# %% [markdown]
# ## 4.3 Variable nueva: horas desde el inicio de la jornada
#
# Es la diferencia (módulo 24) entre la hora del incidente y la de inicio de la jornada, en
# tramos. Los entregables anteriores la dejaron como siguiente paso. Es faltante cuando falta
# cualquiera de las dos horas o la hora del incidente se reportó como desconocida.

# %%
order = ['0-1', '2-3', '4-5', '6-7', '8-9', '10-12', '13+', 'Unknown']
tab = pd.crosstab(dev.hours_since_start, dev.y_clf, normalize='index').reindex(order) * 100
counts = dev.hours_since_start.value_counts().reindex(order)
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(order)); w = 0.27
for j, k in enumerate([2, 3, 4]):
    ax.bar(x + (j - 1) * w, tab[k], width=w - 0.02, color=P.SERIES[j + 1], label=C.CLASS_NAMES[k - 1])
ax.set_xticks(x, [f'{o}\n(n={counts[o]:,})' for o in order], fontsize=8)
ax.set_ylabel('% de los casos del tramo'); ax.set_xlabel('Horas desde el inicio de la jornada')
ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15)); ax.grid(axis='x', visible=False)
P.save(fig, 'e3_eda_horas_desde_inicio'); plt.show()
print('V de Cramér:', round(cramers_v(dev.hours_since_start, y), 3))

# %% [markdown]
# **Interpretación.** El 63 % de los incidentes con hora conocida ocurre en las primeras 8 horas de
# la jornada. Hasta la hora 9 la mezcla de resultados casi no cambia (días de ausencia entre 35 % y
# 37 %). En jornadas largas (10 horas o más) y cuando falta la hora, sube la proporción de días de
# ausencia (39 % a 41 %) y baja la de traslado o restricción (24 % a 28 %). Es una asociación
# débil en conjunto (V = 0,043) y la categoría «Unknown» vuelve a mostrar que los faltantes de
# hora no son al azar, por eso se conservan como categoría propia. Fallecimiento no se grafica
# (69 casos en el desarrollo).

# %% [markdown]
# ## 4.4 Objetivo de regresión: días de ausencia
#
# La guía exige también 7 modelos de regresión. El único objetivo numérico del archivo que tiene
# sentido con las mismas variables es **el número de días de ausencia** (`dafw_num_away`) de los
# casos cuyo resultado es «días de ausencia». En clasificación esta columna se excluía porque
# revela el resultado; en regresión es el objetivo, y se usa solo dentro de esa clase, así que no
# hay fuga: la pregunta es *dado que el caso terminó en días de ausencia, cuántos días fueron*.

# %%
r = dev[dev.y_reg.notna()]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].hist(r.days_away.clip(upper=365), bins=80, color=P.SERIES[0])
axes[0].set_xlabel('Días de ausencia (recortado en 365 para graficar)'); axes[0].set_ylabel('Casos')
axes[0].set_title('Escala original: asimetría fuerte')
axes[1].hist(r.y_reg, bins=60, color=P.SERIES[2])
axes[1].set_xlabel('log(1 + días)'); axes[1].set_title('Escala logarítmica (objetivo del modelo)')
for a in axes: a.grid(axis='x', visible=False)
P.save(fig, 'e3_eda_dias'); plt.show()
desc = r.days_away.describe(percentiles=[.25, .5, .75, .9, .99])
print(desc.round(1).to_string())
print('Asimetría original:', round(stats.skew(r.days_away), 2), '| asimetría log1p:', round(stats.skew(r.y_reg), 2))
print('Casos con más de 180 días:', int((r.days_away > 180).sum()), f'({(r.days_away > 180).mean():.1%})')

# %% [markdown]
# **Interpretación.** Los días de ausencia son muy asimétricos (mediana 11, media 40, máximo 587).
# La regla de registro de OSHA permite dejar de contar a los 180 días, pero el archivo tiene casos
# por encima: no todas las empresas aplican el tope. Por la asimetría, el modelo predice
# `log(1 + días)` (asimetría 1,87 → 0,34): los errores se miden en esa escala (un RMSE de 1
# equivale a equivocarse por un factor de unas 2,7 veces en días). Dos rasgos del registro se ven
# en la gráfica: los valores pequeños son enteros (1, 2, 3... días, por eso las barras sueltas a la
# izquierda) y hay un pico en 180 días, el tope de la regla de OSHA (1,2 % de los casos supera ese
# tope).

# %%
by_type = r.groupby('type_of_incident').days_away.agg(['count', 'median', 'mean']).round(1)
names = {'1': 'Lesión', '2': 'Trastorno de piel', '3': 'Respiratoria', '4': 'Envenenamiento',
         '5': 'Pérdida auditiva', '6': 'Otra enfermedad'}
by_type.index = [names.get(i, i) for i in by_type.index]
by_type.sort_values('count', ascending=False)

# %% [markdown]
# **Interpretación.** Las lesiones son el 85 % de los casos con días de ausencia (mediana 14 días).
# Las enfermedades respiratorias tienen ausencias mucho más cortas y homogéneas (mediana 6, media
# 7,8). Es una inferencia, no comprobada con este archivo, que se trate sobre todo de COVID-19
# registrado como caso laboral. Encaja con el entregable 2, donde las respiratorias se asociaban
# con muchas más posibilidades de terminar en días de ausencia: muchas ausencias, pero cortas.

# %% [markdown]
# ## 4.5 Del EDA a la configuración del entregable 3
#
# | Hallazgo | Decisión |
# |---|---|
# | NAICS/SOC completos: 1.433 columnas, mucha varianza | Subsector NAICS-3 y grupo menor SOC; categorías con < 20 casos → «infrequent» dentro del pipeline |
# | Horas sin relación monótona; faltantes no aleatorios | Horas y tramos de horas desde el inicio como categorías con «Unknown» |
# | Fallecimiento = 0,026 % | 4 clases; balanced accuracy como métrica de validación; SMOTE/ADASYN con vecinos ajustados a la clase minoritaria |
# | Días de ausencia muy asimétricos | Objetivo de regresión `log1p(días)` |
# | Casos agrupados por establecimiento | Todos los pliegues (externos e internos) separan establecimientos |
