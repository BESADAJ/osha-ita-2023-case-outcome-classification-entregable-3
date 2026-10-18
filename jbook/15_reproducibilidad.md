# 15. Reproducibilidad

## Entorno

Todos los resultados del entregable 3 se calcularon en el entorno conda `ml_venv` (Python 3.9.25, Windows 11,
12 hilos lógicos, GPU NVIDIA RTX 3050 de 6 GB). Las versiones exactas están en `requirements.txt`; las
principales: numpy 1.25.2, pandas 2.1.0, scipy 1.11.2, scikit-learn 1.3.0, imbalanced-learn 0.12.4,
xgboost 2.1.3, optuna 4.1.0, deap 1.4.1, shap 0.46.0, lime 0.2.0.1, statsmodels 0.14.0, arch 7.0.0,
faiss-cpu 1.9.0 y jupyter-book 1.0.4.

Las cifras del modelo base del entregable 2 (capítulo 5) se calcularon con scikit-learn 1.2.2 en el
entorno del entregable 1 y no se recalcularon.

## Semilla global

Una sola semilla documentada, **42** (`src/osha_e3/config.py`). Las demás se derivan de ella con
`seed_for(tipo, índice) = 42 + desplazamiento × 1000 + índice`:

| Uso | Desplazamiento | Semilla |
|---|---|---|
| Partición (60 % de establecimientos para prueba), igual que en el entregable 1 | — | 42 |
| Submuestra de establecimientos (clasificación / regresión) | 1 | 1042 / 1043 |
| Pliegues externos (`StratifiedGroupKFold`) | 2 | 2042 |
| Pliegues internos del pliegue externo f | 3 | 3042 + f |
| Búsqueda (Random, TPE, genético) en el pliegue externo f | 4 | 4042 + f |
| Modelos (Random Forest, XGBoost, SMOTE, ADASYN, SAGA, LinearSVC) | — | 42 |
| Bootstrap | 100 | 100042 + índice |
| Sensibilidad a semillas (capítulo 12) | — | 7, 123, 2024 |

La semilla se pasa explícitamente a numpy, scikit-learn, imbalanced-learn, XGBoost, Optuna (`TPESampler`) y
DEAP (generador `random.Random` propio del algoritmo genético).

## Registro de experimentos

- `results/nested/<tarea>/<modelo>__<balanceo>__<optimizador>__f<pliegue>.json`: una corrida por combinación y
  pliegue externo (700 archivos), con hiperparámetros elegidos, puntaje interno, métricas externas, tiempos de
  búsqueda, reajuste y predicción, semilla y el historial de las 16 evaluaciones.
- `results/master_clf.csv` y `results/master_reg.csv`: tabla maestra, una fila por combinación (112 y 28) con
  media y desviación estándar de cada métrica externa, tiempo total, número de evaluaciones, semilla e
  hiperparámetros finales.
- `results/outer_folds_*.csv`: una fila por celda; `results/test_*.csv`: métricas en la prueba;
  `results/seeds.csv`: corridas de sensibilidad a semillas.

## Cómo correr todo

```
pip install -r requirements.txt
set OSHA_DATA_DIR=<carpeta con el CSV de OSHA>          # o export en Linux/macOS
python scripts/run_nested.py all 11      # validación anidada de las 140 combinaciones (reanudable)
python scripts/run_final.py all 11       # reajuste de cada combinación y evaluación en la prueba
python scripts/run_seeds.py 11           # sensibilidad a semillas
python scripts/build_notebooks.py        # convierte nb_src/*.py en jbook/*.ipynb y los ejecuta
jupyter-book build jbook
ghp-import -n -p -f jbook/_build/html    # publica el libro en la rama gh-pages
```

Cada celda de la validación anidada se guarda al terminar, así que una corrida interrumpida se reanuda sin
repetir trabajo. Los capítulos guardan sus cálculos costosos (SHAP, recalibración, bootstrap, experimentos de
costo) en `results/cache/`, que no se versiona; la primera ejecución los calcula.

## Dashboard del entregable 2

La carpeta `dashboard/` es la del entregable 2 sin cambios (Dash/Plotly, desplegado en Render desde un
repositorio aparte). El entregable 3 no lo modifica.
