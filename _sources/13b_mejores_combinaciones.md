# 13b. Mejores combinaciones, reentrenamiento final y artefactos

Este capítulo cumple el Bloque B de la guía (§4). De las 112 combinaciones de clasificación elige la mejor de cada uno
de los 7 modelos con un criterio estadístico. Luego las reentrena y las serializa para el dashboard, y verifica que el
modelo cargado desde disco prediga exactamente lo mismo que el modelo en memoria.

## 13b.1 Criterio de selección

- **Métrica:** la **balanced accuracy (BA) media del bucle externo** de la validación anidada (5 pliegues por
  establecimiento). Es la métrica principal del proyecto (capítulo 6) y la misma que optimizó la búsqueda interna.
- **Por modelo:** para cada uno de los 7 modelos se elige la combinación (balanceo, optimizador) con mayor BA media.
  Los empates se resuelven por menor desviación estándar (no hubo empates).
- **Nunca se usa la prueba (octubre–diciembre) para elegir.**

| Modelo | Balanceo | Optimizador | BA externa (media ± d.e.) | Motivo de la selección |
|---|---|---|---|---|
| SVM lineal | SMOTE | Random | **0,469 ± 0,101** | Mayor BA media de las 16 combinaciones de SVM; **mejor combinación global** |
| Árbol de decisión | `class_weight` | Bayesiana (TPE) | 0,461 ± 0,084 | Mayor BA media de las 16 del árbol |
| Regresión logística | ADASYN | Bayesiana (TPE) | 0,427 ± 0,067 | Mayor BA media de las 16 de la logística |
| XGBoost | `class_weight` | Bayesiana (TPE) | 0,425 ± 0,069 | Mayor BA media de las 16 de XGBoost |
| KNN | SMOTE | Genética | 0,415 ± 0,107 | Mayor BA media de las 16 de KNN |
| Random Forest | SMOTE | Bayesiana (TPE) | 0,407 ± 0,060 | Mayor BA media de las 16 de Random Forest |
| Naive Bayes | ADASYN | Genética | 0,389 ± 0,060 | Mayor BA media de las 16 de Naive Bayes |

Fuente: `results/master_clf.csv`, columna `outer_recall_mean` (la balanced accuracy es el recall macro). Las 112 filas
con todas las métricas están en la tabla de líderes del dashboard y en `dashboard_final/data/leaderboard.csv`.

**Pertenencia de la mejor global al conjunto de finalistas.** En el capítulo 13 (sección 13.2), Friedman rechaza la
igualdad de las 112 combinaciones (p = 0,0001). Sin embargo, con 5 bloques la diferencia crítica de Nemenyi es de 89
rangos, y **ninguna combinación se distingue de la mejor**: todas pertenecen al conjunto de finalistas, también
SVM + SMOTE + Random. Entre los 7 mejores (14 semanas de bloques), SVM tiene el segundo mejor rango (3,00) y queda unida
por la barra de Nemenyi a Random Forest (2,64), que encabeza. La mejor global por BA media está, entonces, en el grupo de
cabeza que no se puede separar estadísticamente. Las diferencias entre las 7 mejores son pequeñas frente a su
variabilidad por pliegue (d.e. de 0,06 a 0,11), que viene sobre todo de la clase Fallecimiento: 0 a 4 casos por pliegue.

## 13b.2 Reentrenamiento final y serialización

Cada combinación elegida se **reentrena** con sus hiperparámetros finales (los del pliegue externo con mejor puntaje
interno, `nested.master_table`) sobre el **conjunto de entrenamiento completo**: la submuestra de desarrollo de 30 030
casos (ene–sep 2023), el mismo conjunto con el que se ajustaron los modelos de los capítulos 6 a 13. El pipeline
completo (one-hot que agrupa las categorías raras → SMOTE/ADASYN o pesos → modelo) se guarda con
`joblib.dump(..., compress=3)` en `dashboard_final/artifacts/`. Lo hace `scripts/build_artifacts.py curso`. El dashboard
**nunca reentrena ni optimiza**: solo carga estos archivos.

Junto a cada modelo se guarda `<modelo>.metadata.json` con:

- las versiones de Python y de cada librería;
- la semilla (42);
- los hiperparámetros finales;
- el **esquema de entrada** (nombre, tipo, número de valores y lista de valores válidos de cada variable);
- las métricas del bucle externo (media y d.e.);
- el tiempo de ajuste, el tiempo de inferencia por caso y el resultado de la prueba de ida y vuelta.

| Artefacto | Tamaño (MB) | Ajuste (s) | Inferencia (ms/caso) | Ida y vuelta |
|---|---|---|---|---|
| `svm.joblib` | 0,08 | 4,1 | 0,013 | idéntica |
| `tree.joblib` | 0,01 | 0,5 | 0,010 | idéntica |
| `logreg.joblib` | 0,01 | 34,6 | 0,012 | idéntica |
| `xgboost.joblib` | 0,68 | 3,8 | 0,053 | idéntica |
| `knn.joblib` | 0,57 | 4,2 | 1,299 | idéntica |
| `random_forest.joblib` | 2,59 | 24,9 | 0,110 | idéntica |
| `naive_bayes.joblib` | 0,01 | 14,5 | 0,011 | idéntica |
| `original.joblib` (capítulo 19) | ver `original.metadata.json` | — | — | idéntica |

**Prueba de ida y vuelta.** Para cada artefacto se predicen las probabilidades de **500 casos fijos del desarrollo que
no están en la submuestra de entrenamiento** (semilla 42), primero con el modelo en memoria y luego con el modelo
recién cargado desde disco. El script exige igualdad exacta (`np.array_equal`) y se detiene si falla. Los 8 artefactos
dan diferencia máxima 0. La prueba de octubre a diciembre no se usa (está bloqueada; capítulo 00, fila B1).

**Tamaño y memoria en Render.** Todos los artefactos suman unos 4 MB más el del modelo original. El Random Forest
(212 árboles de profundidad 25) pesa 2,6 MB comprimido, así que no hizo falta podar árboles ni usar Git LFS. Las
versiones de `numpy`, `pandas`, `scipy`, `scikit-learn`, `imbalanced-learn`, `xgboost` y `joblib` del dashboard
(`dashboard_final/requirements.txt`) son las mismas con las que se entrenaron los artefactos, para que `joblib` los cargue
sin diferencias.
