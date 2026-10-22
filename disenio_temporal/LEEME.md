# Código del diseño temporal y del modelo original (Iván Prada)

Código con el que se calcularon las tablas de `results/disenio_temporal/` (corrida de 108 combinaciones,
capítulo 12.6) y de `results/modelo_original/` (modelo original y sus iteraciones, capítulos 16 y 17).
Es el código tal como se ejecutó; las cifras del libro salen de él.

## Contenido

| Archivo | Qué hace | Salida (relativa a esta carpeta) |
|---|---|---|
| `osha3/` | Paquete: datos y pliegues (`data.py`), métricas, registro de modelos (`models.py`), pipeline sin fugas, optimizadores (Grid, Random, Bayesiano, Genético), validación anidada (`nested.py`) y los estimadores originales (`original_models.py`) | — |
| `run_nested.py`, `summarize.py`, `pooled_analysis.py`, `count_deaths.py` | Corrida de las 108 combinaciones y sus tablas | `results/runs/`, `results/master_registry.csv`, `results/pooled_summary.csv` |
| `cribado_original.py` | Fase A: cribado de PT-bagging, MultiRandBal y la jerárquica | `results/original/cribado.csv`, `decision_fase_a.md` |
| `run_original.py` | Fase B: validación anidada de PT-bagging (4 optimizadores × 3 pliegues) | `results/original/runs/` |
| `cribado_r2.py` | Ronda 2: HAFL-XGBoost | `results/original_r2/` |
| `ronda3_2_4.py` | Ronda 3: subproblema de las clases 2–4 | `results/original_r3/` |
| `cribado_r4.py` | Ronda 4: Frank–Hall ordinal | `results/original_r4/` |
| `analisis_fuga_y_empleador.py` | Fuga de información y efecto del empleador | `results/original_justificacion/` |

En `results/modelo_original/` del repositorio solo se versionan las tablas de resumen (sin identificadores);
las predicciones `*_pred.npz` traen identificadores de establecimientos y casos y no se suben.

## Requisitos

1. **Entorno:** `environment.yml` / `requirements.txt` de esta carpeta (Python 3.9, **scikit-learn 1.2.2,
   imbalanced-learn 0.11.0, xgboost 2.1.4**, optuna 4.1.0, DEAP 1.4.1). Son versiones distintas de las del resto
   del repositorio (scikit-learn 1.3.0, imbalanced-learn 0.12.4, xgboost 2.1.3).
2. **Código del entregable 1:** `osha3/data.py` reutiliza sin modificar `osha_deliverable_utils.py` y
   `osha_group_time_study.py` (partición y preparación de variables del entregable 1). No se copian aquí. Están en
   el repositorio del entregable 1, <https://github.com/sak-buks/osha-ita-2023-case-outcome-classification>
   (raíz de la rama `entregable3-validacion-anidada`). Indicar su carpeta con la variable de entorno
   `OSHA_CODE_DIR`.
3. **Datos:** el CSV de OSHA `ITA Case Detail Data 2023 through 12-31-2023OIICS.csv` (no se versiona). Indicar su
   carpeta con `OSHA_DATA_DIR`. `load_data()` verifica que la partición coincida con la del entregable 1
   (269.516 casos y 69 muertes en desarrollo; 127.391 y 39 en prueba).

## Orden de ejecución

```
python -c "from osha3.data import materialize_folds; materialize_folds()"   # pliegues en results/folds/
python run_nested.py --workers 7 --n-evals 20                               # ~10 h
python summarize.py && python pooled_analysis.py
python cribado_original.py --workers 5                                      # ~35 min
python run_original.py --model pt_bagging --workers 5 --n-evals 12          # ~1 h 40 min
python cribado_r2.py --workers 2 --ablation && python ronda3_2_4.py && python cribado_r4.py
python analisis_fuga_y_empleador.py
```

Semilla global 42; el bucle interno usa 1042. La prueba (oct–dic) no se usó en ninguno de estos scripts.
