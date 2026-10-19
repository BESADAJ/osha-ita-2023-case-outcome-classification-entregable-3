# Entregable 3 – OSHA ITA 2023 (Machine Learning, EST 1055-01)

Autores: Jaime Andrés Besada e Iván Prada.

Continuación del [entregable 2](https://github.com/sak-buks/osha-ita-2023-case-outcome-classification-segundo-entregable)
(este repositorio parte de su historial). Sobre el archivo de casos de OSHA (ITA 2023) se entrenan, optimizan y
evalúan **140 modelos**:

- **Clasificación** del resultado del caso (4 clases): 7 modelos (KNN, Naive Bayes, regresión logística L1/L2,
  árbol, Random Forest, XGBoost, SVM) × 4 balanceos (ninguno, SMOTE, ADASYN, `class_weight`) × 4 optimizadores
  (Grid, Random, bayesiano con Optuna-TPE, genético con DEAP) = 112.
- **Regresión** de `log(1 + días de ausencia)`: 7 modelos (KNN, Ridge, Lasso, árbol, Random Forest, XGBoost, SVR)
  × 4 optimizadores = 28.

Todo con validación cruzada anidada por establecimiento (5 pliegues externos × 3 internos) y la misma prueba
de los entregables anteriores (octubre–diciembre, establecimientos no vistos).

| Qué | Dónde |
|---|---|
| Informe (Jupyter Book) | <https://besadaj.github.io/osha-ita-2023-case-outcome-classification-entregable-3/> (fuentes en `jbook/`) |
| Código | `src/osha_e3/` |
| Corridas largas | `scripts/` |
| Resultados (tablas maestras, una fila por combinación) | `results/master_clf.csv`, `results/master_reg.csv` |
| Registro de cada corrida (combinación × pliegue externo) | `results/nested/<tarea>/*.json` |
| Dashboard del entregable 2 | `dashboard/` y <https://osha-ita-2023-case-outcome.onrender.com/> |

## Estructura del código

```
src/osha_e3/
  config.py        semilla global (42), semillas derivadas, rutas, tamaños
  data.py          carga del CSV, partición del entregable 1, variables, submuestras
  models.py        modelos, balanceo, espacios de búsqueda y rejillas
  search.py        evaluador común + Grid / Random / Optuna-TPE / DEAP
  nested.py        validación anidada, registro y tabla maestra
  final.py         reajuste final y evaluación en la prueba
  evaluation.py    métricas (incl. Brier y ECE)
  stats_tests.py   Friedman, Nemenyi/CD, DeLong, DM-HLN, GW, CW, MCS, SPA, bootstrap BCa
  complexity.py    experimentos de costo computacional y perfilamiento
  interpret.py     SHAP y LIME
  analysis.py      curvas anytime, eficiencia por presupuesto, diversidad del genético
  seeds.py         sensibilidad a la semilla
nb_src/            fuentes de los capítulos-notebook (formato percent)
scripts/           run_nested.py, run_final.py, run_seeds.py, build_notebooks.py
```

## Reproducir

1. Entorno: Python 3.9 y `pip install -r requirements.txt` (versiones exactas; se usó el entorno conda `ml_venv`).
2. Datos: descargar
   [ITA_Case_Detail_Data_2023_through_12-31-2023OIICS.zip](https://www.osha.gov/sites/default/largefiles/ITA_Case_Detail_Data_2023_through_12-31-2023OIICS.zip),
   extraer el CSV (655 MB, no se versiona) y fijar `OSHA_DATA_DIR` a su carpeta.
3. Corridas (reanudables; cada celda se guarda al terminar):

```
python scripts/run_nested.py all 11     # 700 celdas de validación anidada (~1-2 h con 11 procesos)
python scripts/run_final.py all 11      # reajuste y prueba de las 140 combinaciones
python scripts/run_seeds.py 11          # sensibilidad a semillas
python scripts/build_notebooks.py       # genera y ejecuta los capítulos en jbook/
jupyter-book build jbook
ghp-import -n -p -f jbook/_build/html   # publica en gh-pages
```

Los capítulos guardan sus cálculos costosos en `results/cache/` (no versionado); la primera ejecución los calcula.
