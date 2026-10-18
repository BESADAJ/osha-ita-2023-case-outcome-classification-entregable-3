# %% [markdown]
# # 6. Diseño del pipeline combinatorio
#
# El entregable 3 no entrena un modelo sino **140**: cada modelo base se combina con cada técnica
# de balanceo (solo en clasificación) y cada combinación se optimiza con cada uno de los cuatro
# métodos de búsqueda.
#
# | Tarea | Modelos | Balanceo | Optimización | Total |
# |---|---|---|---|---|
# | Clasificación (resultado del caso, 4 clases) | 7 | 4 | 4 | 7 × 4 × 4 = **112** |
# | Regresión (log de los días de ausencia) | 7 | — | 4 | 7 × 4 = **28** |
# | **Total** | | | | **140** |
#
# Todo el código vive en el paquete `src/osha_e3/`, separado por responsabilidad:
#
# | Módulo | Qué hace |
# |---|---|
# | `config.py` | Semilla global única (42), semillas derivadas, rutas, tamaños de muestra |
# | `data.py` | Carga del CSV, partición del entregable 1, preparación de variables, submuestras |
# | `models.py` | Los 14 modelos, el balanceo, los espacios de búsqueda y las rejillas |
# | `search.py` | Evaluador común y los 4 métodos (Grid, Random, Optuna-TPE, DEAP) |
# | `nested.py` | Validación cruzada anidada, registro de cada corrida y tabla maestra |
# | `final.py` | Reajuste de cada combinación y evaluación en la prueba |
# | `evaluation.py` | Métricas de clasificación (incl. calibración) y de regresión |
# | `stats_tests.py` | Friedman, Nemenyi, DeLong, Diebold-Mariano, MCS, GW, CW, SPA, bootstrap |
# | `complexity.py` | Experimentos de costo computacional y perfilamiento |
#
# Las corridas largas se lanzan con `scripts/run_nested.py` y `scripts/run_final.py`; los capítulos
# leen los resultados guardados en `results/`.

# %%
import sys, json, warnings
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from osha_e3 import config as C, data as D, models as M, nested as N
print('Combinaciones de clasificación:', len(M.combos('clf')), '| de regresión:', len(M.combos('reg')))

# %% [markdown]
# ## 6.1 Datos de cada tarea
#
# - **Clasificación.** El objetivo son los 4 resultados del caso, igual que en los entregables 1 y 2.
#   Se mantiene la partición del entregable 1: la **prueba** son los casos de octubre a diciembre de
#   un 60 % de los establecimientos, que no se usa en nada de la búsqueda.
# - **Regresión.** `log(1 + días de ausencia)` en los casos que terminaron en días de ausencia,
#   con la misma partición.
#
# **Submuestra para la búsqueda.** El desarrollo tiene 269.516 casos. Entrenar 140 combinaciones ×
# 5 pliegues externos × 16 configuraciones × 3 pliegues internos (más de 33.000 ajustes) sobre todo
# el desarrollo no cabe en el tiempo disponible (con KNN y SMOTE el costo crece con n² o n·p por
# consulta; ver capítulo 8). Se trabaja con una submuestra de **establecimientos completos**
# (~30.000 casos en clasificación, ~20.000 en regresión) sorteada con la semilla global. La prueba
# se usa completa.

# %%
sub_c, dev_c, test_c = D.classification_sets()
sub_r, dev_r, test_r = D.regression_sets()
resumen = pd.DataFrame({
    'Desarrollo completo': [len(dev_c), len(dev_r)],
    'Submuestra (búsqueda)': [len(sub_c), len(sub_r)],
    'Establecimientos submuestra': [sub_c.establishment_id.nunique(), sub_r.establishment_id.nunique()],
    'Prueba (oct-dic)': [len(test_c), len(test_r)],
}, index=['Clasificación', 'Regresión'])
resumen

# %%
dist = pd.DataFrame({'Desarrollo completo (%)': dev_c.y_clf.value_counts(normalize=True).sort_index() * 100,
                     'Submuestra (%)': sub_c.y_clf.value_counts(normalize=True).sort_index() * 100,
                     'Submuestra (casos)': sub_c.y_clf.value_counts().sort_index(),
                     'Prueba (%)': test_c.y_clf.value_counts(normalize=True).sort_index() * 100})
dist.index = C.CLASS_NAMES
dist.round(3)

# %% [markdown]
# **Interpretación.** La submuestra conserva la mezcla de clases del desarrollo (diferencias de
# décimas de punto). Fallecimiento queda con solo 9 casos: con esa cantidad ninguna métrica de esa
# clase es estable, y SMOTE/ADASYN deben usar menos vecinos que su valor por defecto (5) porque en
# cada entrenamiento interno quedan unos 5 fallecimientos.

# %% [markdown]
# ## 6.2 Validación cruzada anidada
#
# - **Bucle externo: 5 pliegues** por establecimiento (`StratifiedGroupKFold`). Cada pliegue
#   externo deja entre 5.200 y 7.300 casos fuera para medir el desempeño del modelo *ya optimizado*. 5 pliegues
#   dan 5 mediciones independientes para la media ± desviación y para las pruebas de Friedman sin
#   dejar entrenamientos demasiado pequeños.
# - **Bucle interno: 3 pliegues** por establecimiento dentro del entrenamiento externo (23.000 a 25.000 casos;
#   cada entrenamiento interno tiene unos 16.000). Es donde cada método de búsqueda evalúa configuraciones. 3 pliegues bastan para ordenar
#   configuraciones y mantienen el costo: cada configuración cuesta 3 ajustes.
# - **Por qué agrupar:** los casos de un establecimiento comparten prácticas de registro; si un
#   establecimiento quedara a ambos lados, la validación sería optimista (lo mismo que motivó la
#   partición del entregable 1).
# - **Métrica reportada:** siempre la del bucle externo. El mejor valor del bucle interno se guarda
#   solo para comparar (capítulo 7, sesgo de selección).

# %%
X, y, g, s = N.task_data('clf')
rows = []
for f, (tr, te) in enumerate(N.outer_folds('clf')):
    inner = N.inner_folds('clf', tr, f)
    rows.append({'Pliegue externo': f + 1, 'Entrenamiento externo': len(tr), 'Evaluación externa': len(te),
                 'Fallecimientos (evaluación)': int((y[te] == 0).sum()),
                 'Establecimientos compartidos': len(set(g[tr]) & set(g[te])),
                 'Tamaño medio entrenamiento interno': int(np.mean([len(a) for a, b in inner])),
                 'Fallecimientos por entrenamiento interno': ', '.join(str(int((y[tr][a] == 0).sum())) for a, b in inner)})
pd.DataFrame(rows).set_index('Pliegue externo')

# %% [markdown]
# **Interpretación.** Ningún establecimiento se comparte entre entrenamiento y evaluación de un
# mismo pliegue. Los pliegues externos tienen entre 0 y 4 fallecimientos (el pliegue 3, ninguno): la AUC de esa
# clase en cada pliegue es muy ruidosa, y por eso la métrica de validación no puede depender de
# ella sola.

# %% [markdown]
# ## 6.3 Métrica de validación
#
# **Clasificación: balanced accuracy** (el recall promedio de las 4 clases). En este problema el
# error más costoso es no reconocer un caso grave (un fallecimiento o unos días de ausencia
# clasificados como «otro caso registrable»): es un falso negativo de la clase grave. El recall por
# clase mide exactamente eso, y promediarlo da a cada clase el mismo peso sin importar su tamaño.
# El accuracy simple, en cambio, premia acertar la clase mayoritaria. Se reportan además precision,
# F1 y AUC macro, y Brier/ECE para la calibración.
#
# **Regresión: RMSE** en escala `log1p`. Penaliza más los errores grandes (predecir 3 días para
# una ausencia de 6 meses), que son los costosos para planificar reemplazos o reintegro. Se
# reporta también el MAE.

# %% [markdown]
# ## 6.4 Pipeline y ausencia de fuga de información
#
# La preparación de `data.prepare_features` es determinista (cortar códigos, extraer la hora): no
# aprende nada de los datos y puede aplicarse antes de partir. Todo lo que aprende de los datos está
# dentro de un `Pipeline` de imblearn que se ajusta **solo con la parte de entrenamiento** de cada
# pliegue:
#
# 1. `OneHotEncoder(min_frequency=20, handle_unknown='infrequent_if_exist')`: qué categorías son
#    raras se decide con el entrenamiento del pliegue.
# 2. `SMOTE` o `ADASYN` (si corresponde): genera casos sintéticos solo en el entrenamiento; al
#    predecir, el paso se omite.
# 3. El modelo.
#
# Para Optuna y DEAP, que no integran `Pipeline`, el evaluador (`search.Evaluator`) ajusta
# manualmente codificador → balanceo → modelo en cada pliegue interno. Como la codificación y el
# remuestreo no dependen de los hiperparámetros, se calculan una vez por pliegue y se reutilizan
# en las 16 configuraciones (el mismo cálculo, sin repetirlo).

# %%
pipe = M.make_pipeline('clf', 'random_forest',
                       {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 2, 'max_features': 0.1},
                       balance='smote')
pipe

# %% [markdown]
# ## 6.5 Balanceo de clases
#
# | Técnica | Qué hace | Detalle de implementación |
# |---|---|---|
# | Sin balanceo | Línea base | — |
# | SMOTE | Interpola casos sintéticos entre un caso minoritario y sus vecinos de la misma clase | Lleva todas las clases al tamaño de la mayoritaria; `k_neighbors = min(5, n_min − 1)` |
# | ADASYN | Como SMOTE, pero genera más casos donde la minoritaria está rodeada de otras clases (más difícil) | `n_neighbors = min(5, n_min − 1)`; si no puede generar, no remuestrea |
# | `class_weight='balanced'` | Pondera los errores por la inversa de la frecuencia de la clase | Nativo en logística, árbol, RF y SVM; pesos por observación en Naive Bayes y XGBoost; en KNN, división de la probabilidad por la prior (regla de decisión equivalente) |
#
# Una advertencia: las variables son todas binarias (one-hot). SMOTE y ADASYN interpolan y producen
# valores fraccionarios entre 0 y 1 que no corresponden a ninguna categoría real. SMOTE-NC está
# pensado para categóricas, pero la guía pide SMOTE y ADASYN; el efecto se mide, no se supone.

# %% [markdown]
# ## 6.6 Espacios de búsqueda
#
# Los cuatro métodos exploran el **mismo espacio** con el **mismo presupuesto: 16 configuraciones**
# por búsqueda. Los parámetros de escala (C, alpha, learning rate, min_child_weight, número de
# vecinos, min_samples_leaf) se exploran en escala logarítmica. Los rangos siguen valores
# habituales y el tamaño de los datos (~16.000 filas por entrenamiento interno, ~210 columnas):
#
# - **KNN:** 5 a 801 vecinos (con pocas variables informativas hacen falta muchos vecinos para
#   suavizar; una primera corrida con tope 201 eligió el tope en la mitad de las búsquedas de
#   regresión, así que el rango se amplió), pesos uniformes o por distancia.
# - **Naive Bayes (Bernoulli):** suavizado alpha entre 0,001 y 10; con o sin prior. Se usa la
#   versión de Bernoulli porque todas las variables son binarias.
# - **Logística:** C de 0,001 a 10, L1 o L2 (solver SAGA, válido para ambas).
# - **Árbol:** profundidad 3 a 30, hojas de 1 a 500 casos (ampliado desde 200 por la misma razón).
# - **Random Forest:** 50 a 250 árboles, profundidad 6 a 30, fracción de variables por división
#   de 3 % a 30 % (≈ √210 / 210 = 7 % es el valor por defecto).
# - **XGBoost:** 100 a 400 árboles, profundidad 2 a 10, learning rate 0,01 a 0,3, submuestreo de
#   filas 0,6 a 1 y de columnas 0,4 a 1, min_child_weight 1 a 20.
# - **SVM (LinearSVC) / SVR (LinearSVR):** C de 0,001 a 10; pérdida bisagra o bisagra cuadrática
#   (SVM) y epsilon de 0 a 1,5 (SVR, en escala log1p de días; ampliado desde 1).
# - **Ridge:** alpha 0,01 a 1.000; **Lasso:** alpha 0,00001 a 0,1.

# %%
def fmt(spec):
    if spec[0] == 'cat':
        return ' / '.join(map(str, spec[1]))
    return f'{spec[1]} – {spec[2]}' + (' (log)' if spec[3] else '')
rows = [{'Tarea': 'Clasificación' if t == 'clf' else 'Regresión', 'Modelo': M.PRETTY[m], 'Hiperparámetro': k,
         'Rango (Random / Optuna / genético)': fmt(v), 'Valores en la rejilla (Grid)': ', '.join(map(str, M.GRIDS[t][m].get(k, ['por defecto'])))}
        for t in ['clf', 'reg'] for m, sp in M.SPACES[t].items() for k, v in sp.items()]
pd.DataFrame(rows).set_index(['Tarea', 'Modelo'])

# %% [markdown]
# **Por qué SVM lineal.** El SVM con núcleo (RBF) cuesta entre O(n²) y O(n³) en entrenamiento.
# Con 16.000 a 25.000 filas por ajuste y 245 ajustes por combinación no es viable (capítulo 8 lo
# mide). La guía misma recomienda `LinearSVC`/SGD para muestras grandes. Para AUC se usa el
# softmax de la función de decisión (solo importa el orden); la calibración con Platt se estudia
# en el capítulo 9.

# %% [markdown]
# ## 6.7 Registro de experimentos y semillas
#
# Cada celda (combinación × pliegue externo) se guarda en `results/nested/<tarea>/*.json` con los
# hiperparámetros elegidos, la métrica interna, las métricas externas, el tiempo de búsqueda, de
# reajuste y de predicción, la semilla y el historial completo de las 16 evaluaciones. Con ellas
# se construye la **tabla maestra** (`results/master_clf.csv`, `results/master_reg.csv`).
#
# Semilla global: **42** (`config.SEED`). Las semillas derivadas se calculan como
# `42 + desplazamiento × 1000 + índice` (`config.seed_for`): submuestra, pliegues externos,
# internos, búsqueda (Random, TPE, genético) y bootstrap. La misma semilla se pasa a numpy,
# scikit-learn, XGBoost, imblearn, Optuna y DEAP.

# %%
pd.Series({k: C.seed_for(k) for k in C.OFFSETS}, name='semilla (índice 0)').to_frame()
