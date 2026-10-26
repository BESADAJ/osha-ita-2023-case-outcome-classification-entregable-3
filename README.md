# Entregable 3 – OSHA ITA 2023 (Machine Learning, EST 1055-01)

Autores: Jaime Andrés Besada e Iván Berrio.

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
| Repositorio | <https://github.com/BESADAJ/osha-ita-2023-case-outcome-classification-entregable-3> |
| Informe (Jupyter Book) | <https://besadaj.github.io/osha-ita-2023-case-outcome-classification-entregable-3/> (fuentes en `jbook/`) |
| Dashboard final (Render) | <https://osha-ita-2023-case-outcome-classification.onrender.com> (fuentes en `dashboard_final/`) |
| **Notebook único con todo el entregable** | `jbook/entregable3_completo.ipynb` (también como última página del libro) |
| Código | `src/osha_e3/` |
| Corridas largas | `scripts/` |
| Resultados (tablas maestras, una fila por combinación) | `results/master_clf.csv`, `results/master_reg.csv` |
| Registro de cada corrida (combinación × pliegue externo) | `results/nested/<tarea>/*.json` |
| Predicciones de prueba de las 140 combinaciones | `results/final/<tarea>/*.npz` (sin identificadores) |
| Cálculos guardados de los capítulos (SHAP, bootstrap, tiempos…) | `results/cache/*.joblib` |
| Guía del entregable (PDF) | `docs/guia_entregable_3.pdf` |
| Dashboard del entregable 2 | `dashboard/` y <https://osha-ita-2023-case-outcome.onrender.com/> |

## Proyecto final (rama `entregable3-final`)

| Componente obligatorio | Dónde |
|---|---|
| 1. Correcciones del entregable 2 (tabla de trazabilidad) | `jbook/00_trazabilidad.md` |
| 2. Dashboard Dash multipágina (Contexto, EDA, Modelos), desplegado en Render | `dashboard_final/` (`app.py`, `pages/`, `src/`, `artifacts/`, `render.yaml`) y <https://osha-ita-2023-case-outcome-classification.onrender.com> |
| 3. Modelo original: ensamble multivista + **SECR** (selección de ensambles condicionada por región, Frank-Wolfe) | `src/osha_e3/combinador.py`, `src/osha_e3/original_final.py`, `dashboard_final/src/original_model.py`; pruebas en `tests/test_combinador.py` |
| 4. Informe en Jupyter Book | `jbook/` (capítulos 00, 13b, 16, 19, 20 y 21 nuevos; `jupyter-book build jbook`) |
| Tabla maestra de experimentos (140 + pareado + original) | `results/tabla_maestra_experimentos.csv` |
| Experimento pareado (mismas variables y particiones, 3×5) | `results/pareado/` (`resumen.json`, `pliegues.csv`, `ablacion_resumen.csv`) |

**Semilla global:** 42 (`src/osha_e3/config.py`); las derivadas están en el capítulo 15. **La prueba de octubre a diciembre
está bloqueada por código** (`OSHA_ALLOW_TEST=1` solo para reproducir cifras ya publicadas).

**Veredicto:** R1 sí, R2 sí, **R3 no**. El original tiene la mayor BA media (0,591 frente a 0,569 de KNN pareado) y el
mejor rango de Friedman, pero frente a KNN la diferencia no es significativa (capítulo 20).

### Dashboard: correr y desplegar

```
cd dashboard_final
pip install -r requirements.txt      # Python 3.11.9; versiones exactas
python app.py                        # http://127.0.0.1:8050
pytest tests                         # pruebas de humo
```

En Render: *New → Blueprint* con este repositorio y la rama `entregable3-final` (lee `dashboard_final/render.yaml`), o un
*Web Service* con `Root Directory = dashboard_final`, `Build = pip install -r requirements.txt` y
`Start = gunicorn app:server --workers 1 --threads 1 --timeout 180`.

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
python scripts/build_single_notebook.py # une todo en jbook/entregable3_completo.ipynb y lo ejecuta
jupyter-book build jbook
ghp-import -n -p -f jbook/_build/html   # publica en gh-pages
```

Los capítulos guardan sus cálculos costosos en `results/cache/` y las predicciones de prueba están en `results/final/`, ambos
versionados: el libro se puede volver a ejecutar sin repetir las corridas largas. Solo `results/cache/*.parquet` (datos
preparados con identificadores) no se versiona; se regenera desde el CSV.
