# Entregable 2 – OSHA ITA 2023 (Machine Learning, EST 1055-01)

Predicción del resultado de los casos reportados a OSHA (4 clases) con regresión logística frente a un
`DummyClassifier`. Los datos y el modelo vienen del entregable 1.

| Qué | Dónde |
|---|---|
| Dashboard (Dash/Plotly, 3 pestañas) | https://osha-ita-2023-case-outcome.onrender.com/ |
| Informe (Jupyter Book) | https://sak-buks.github.io/osha-ita-2023-case-outcome-classification-segundo-entregable/ |
| Repo del informe (este) | https://github.com/sak-buks/osha-ita-2023-case-outcome-classification-segundo-entregable |
| Repo que lee Render | https://github.com/sak-buks/osha-ita-2023-case-outcome-classification-dashboard |

El plan gratuito de Render se duerme tras 15 minutos sin visitas; la primera carga puede tardar cerca de 1 minuto.

## Ramas de este repo

- `main`: las fuentes (aquí se edita).
- `gh-pages`: el sitio ya compilado que sirve GitHub Pages. No se edita a mano; lo genera `ghp-import`.

## Carpetas

- `dashboard/`: `app.py`, `data/` (tablas pequeñas ya calculadas), `requirements.txt`, `render.yaml` y `build_data.py`.
  El dashboard no entrena ni carga el CSV de 654 MB; solo lee `data/`.
- `jbook/`: capítulos `.md`, `_config.yml`, `_toc.yml`, `figures/`, `_generated/`, `make_figures.py` y
  `anexo_entregable1.ipynb`. `_build/` es el sitio compilado y no se versiona.

El CSV crudo de OSHA y los resultados del entregable 1 (`evaluation.json`, etc.) **no** están en el repo.

## Editar y publicar el informe

Requiere Python con `pandas`, `matplotlib`, `jupyter-book==1.0.4` y `ghp-import`.

```
cd jbook
python make_figures.py          # regenera figures/ y _generated/ desde ../dashboard/data
cd ..
jupyter-book build jbook        # genera jbook/_build/html
ghp-import -n -p -f jbook/_build/html   # publica en la rama gh-pages (reemplaza su contenido)
```

Si solo se cambian textos `.md`, basta con `jupyter-book build jbook` y `ghp-import`. El sitio se actualiza
uno o dos minutos después del push. Antes de afirmar un resultado en el texto, comprobarlo contra
`dashboard/data/`.

## Dashboard

Correr en local (Python 3.11 o más nuevo):

```
pip install -r dashboard/requirements.txt
python dashboard/app.py          # http://127.0.0.1:8050/
```

**La carpeta `dashboard/` de este repo y el repo de Render son copias separadas.** Render despliega lo que haya
en `osha-ita-2023-case-outcome-classification-dashboard` (start command: `gunicorn app:server`) y redespliega
cuando ese repo cambia. Si se modifica `app.py` o `dashboard/data/` aquí, hay que copiar el cambio también a ese
repo; ese repo contiene solo `app.py`, `data/`, `requirements.txt` y `render.yaml`.

Si se cambia `dashboard/data/`, regenerar también las figuras del informe (`make_figures.py`) para que
dashboard e informe sigan coincidiendo.
