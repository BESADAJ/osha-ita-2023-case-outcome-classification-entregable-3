# 6. Reproducibilidad y despliegue

## Semillas y versiones

| Uso | Semilla |
|---|---|
| Sorteo del 60 % de establecimientos para la prueba, `LogisticRegression`, `DummyClassifier`, `StratifiedGroupKFold` y muestras del EDA | 42 |
| Fracciones de la curva de aprendizaje | 42 + número de pliegue |
| Bootstrap global por establecimiento | 142 |
| Bootstrap por sector | 242 |

Los resultados del modelo (archivo `evaluation.json` del entregable 1) se calcularon con scikit-learn 1.2.2 y
`newton-cg`, en el entorno `ml_env` (versiones en el `requirements.txt` del entregable 1: numpy 1.23.5, pandas
1.5.3, scikit-learn 1.2.2). El notebook del anexo se ejecutó por última vez en otro entorno (`ml_venv`, Python 3.9,
scikit-learn 1.3.0, pandas 2.1.0, numpy 1.25.2; es la tabla de versiones del final del anexo). En esa ejecución el
notebook reutilizó los resultados ya calculados en lugar de reentrenar, y sus cifras coinciden con
`evaluation.json`. Por eso las cifras de este informe corresponden a scikit-learn 1.2.2.

El dashboard se probó con Python 3.11, dash 4.4.1, plotly 7.1.0 y pandas 3.0.6, fijados en
`dashboard/requirements.txt`.

## Cómo está armado el dashboard

El dashboard **no entrena nada ni carga el CSV de 655 MB**. Un script lo prepara una sola vez:

```
dashboard/
  app.py               la aplicación (3 pestañas)
  build_data.py        exporta las tablas pequeñas a data/
  data/                tablas del EDA y del modelo (unos 170 KB)
  requirements.txt     dependencias de la aplicación
  render.yaml          configuración de Render
```

`build_data.py` toma las tablas del EDA de las salidas guardadas en el notebook del entregable 1, y las métricas, la
matriz de confusión, el bootstrap y la curva de aprendizaje de `evaluation.json`. Las curvas ROC y de
Precision-Recall se calculan con las probabilidades de `holdout_predictions.npz`.

## Cómo correrlo

```
# 1. Preparar datos (solo una vez; necesita el notebook y los resultados del entregable 1)
pip install -r requirements-build.txt
python build_data.py NOTEBOOK.ipynb evaluation.json holdout_predictions.npz daily_test_errors.csv coefficients.csv

# 2. Correr el dashboard en local
pip install -r requirements.txt
python app.py            # abre http://127.0.0.1:8050

# 3. Generar el informe
cd ../jbook
pip install "jupyter-book<2" matplotlib
python make_figures.py
jupyter-book build .
```

## Despliegue en Render

La aplicación expone `server = app.server` para que Gunicorn la sirva (`gunicorn app:server`), como indica la
guía del curso. Pasos: subir la carpeta `dashboard/` a un repositorio de GitHub; en Render crear un *Blueprint* o
*Web Service* desde ese repositorio (el `render.yaml` ya trae la configuración); esperar la construcción y copiar
la dirección pública a la introducción de este informe.
