# 21. Dashboard y despliegue

El dashboard lleva los resultados a un producto que una persona no experta puede usar: entender el problema, explorar
los datos y predecir el desenlace de un caso con una explicación. Es una aplicación **Dash multipágina** desplegada en
Render.

:::{admonition} URL pública
:class: tip
**<https://osha-ita-2023-dashboard-final.onrender.com>** (servicio `osha-ita-2023-dashboard-final`, plan gratuito).
Render asigna el subdominio al crear el servicio: si difiere, se actualiza aquí y en la introducción. El primer acceso
tras un periodo de inactividad tarda unos 30–60 s en despertar el servidor.
:::

## 21.1 Arquitectura

```text
dashboard_final/
|-- app.py              # crea la app (use_pages=True), barra de navegación fija y expone server
|-- pages/
|   |-- contexto.py     # sección 1: problema, cifra de contexto citada, pregunta, datos, diccionario, metodología, ética
|   |-- eda.py          # sección 2: filtros globales + 7 figuras con insight + explorador bivariado
|   `-- modelos.py      # sección 3: líderes, predicción en vivo + SHAP local, diagnósticos, SHAP global, CD, original
|-- src/
|   |-- data.py         # carga única y caché (lru_cache) de datos, metadatos y artefactos
|   |-- plots.py        # todas las figuras (plantilla Plotly única, paleta de 5 colores)
|   |-- explain.py      # valores de Shapley exactos (SHAP intervencional) para cualquier modelo
|   `-- original_model.py  # punto de acceso a la clase del modelo original
|-- osha_e3/            # copia de las clases de src/osha_e3 que joblib necesita para cargar los artefactos
|-- artifacts/          # 7 benchmarks + modelo original (.joblib) y sus metadata.json
|-- data/               # tablas livianas: EDA (30 030 casos sin identificadores), líderes, predicciones fuera de pliegue
|-- assets/             # estilos
|-- tests/              # pruebas de humo de páginas y callbacks (pytest)
|-- requirements.txt    # versiones exactas (las mismas de entrenamiento)
`-- render.yaml         # despliegue
```

**Decisiones de diseño.**

- **Separación interfaz / lógica / artefactos.** Las páginas solo arman componentes y callbacks. Los datos se cargan en
  `src/data.py` y las figuras se construyen en `src/plots.py`. No hay rutas absolutas: todo es relativo a la carpeta de
  la aplicación (variable de entorno `DASH_ROOT` opcional).
- **Nunca se entrena en el servidor.** Los modelos se cargan una vez (`lru_cache`) desde `artifacts/` y solo predicen.
- **Diagnósticos sin la prueba.** La matriz de confusión, las curvas ROC y PR, la calibración y el SHAP global se
  calculan con las predicciones **fuera de pliegue externo** de la validación anidada. La prueba de octubre a diciembre
  está bloqueada (capítulo 00, B1).
- **SHAP local exacto.** En lugar de la librería `shap` (pesada para el plan gratuito), `src/explain.py` enumera
  las $2^d$ coaliciones. Son 9 variables para los benchmarks; para el original, 10 grupos de variables (actividad,
  ocupación, horario, lesión, empleador…). Usa un fondo fijo de casos del desarrollo. Cumple la propiedad de eficiencia
  (la suma de aportes es igual a la predicción menos la base) y tarda alrededor de 1 s por predicción (unos 9 s con KNN).
- **Robustez.** Las entradas se validan antes de predecir: campos obligatorios, longitud y caracteres del texto libre
  y avisos de categorías no vistas. Un error de predicción se muestra en un `dbc.Alert` y la aplicación no se cae.
- **Diseño visual.** Tema FLATLY de `dash-bootstrap-components`, barra de navegación fija, rejilla responsiva
  (`dbc.Row`/`dbc.Col` con puntos de corte `xs`, `md` y `lg`), una sola plantilla Plotly y una paleta de 5 colores
  (azul marino, verde azulado, ámbar, coral y gris). Cada figura tiene título, ejes con unidades e información al pasar
  el cursor. Ninguna información depende solo del color: las barras de SHAP llevan signo y valor, y la matriz de
  confusión lleva conteos.

## 21.2 Contenido por sección

| Sección | Contenido mínimo de la guía (§7.3) | Dónde |
|---|---|---|
| Contexto | Problema y relevancia con cifra citada (BLS, 2,6 millones de casos y 946 500 con días de ausencia en 2023); pregunta y objetivos; fuente, periodo, n, variables, diccionario y tarea; diagrama del pipeline; limitaciones, ética, autores, versión y fecha | `pages/contexto.py` |
| EDA | Filtros globales (estado, sector, tipo de establecimiento, rango de fechas `DatePickerRange`, `RangeSlider` de tamaño) que actualizan todas las figuras; distribución y balance del objetivo; distribución univariada; asociación (V de Cramér); faltantes; atípicos; tendencia mensual; explorador bivariado con selectores de x, y y color; una frase de interpretación calculada por figura | `pages/eda.py` |
| Modelos | `DataTable` ordenable y filtrable con las mejores combinaciones (y las 112 a pedido), media ± d.e. e inferencia; formulario de predicción generado desde el esquema con selector de modelo, probabilidad y SHAP local; matriz de confusión, ROC, PR y calibración; SHAP global; diagrama de diferencia crítica con el original incluido; subsección del original con descripción, pseudocódigo, diferencias con la literatura, curva de convergencia, tabla de pruebas y veredicto | `pages/modelos.py` |

```{figure} figures/e3_dash_contexto.png
:name: fig-dash-contexto
:width: 95%

Sección Contexto del dashboard.
```

```{figure} figures/e3_dash_eda.png
:name: fig-dash-eda
:width: 95%

Sección EDA: filtros globales arriba y figuras con su interpretación.
```

```{figure} figures/e3_dash_modelos.png
:name: fig-dash-modelos
:width: 95%

Sección Modelos: tabla de líderes, predicción en vivo con explicación SHAP local y diagnósticos.
```

## 21.3 Guía de despliegue en Render

1. **Probar localmente.** Desde `dashboard_final/`: `pip install -r requirements.txt` y `python app.py` (abre
   <http://127.0.0.1:8050>). `pytest tests` corre las pruebas de humo. El entorno local de prueba (`dash_e3`, Python
   3.11.9) usa las mismas versiones que Render.
2. **Versiones exactas.** `requirements.txt` fija todas las librerías, incluido `gunicorn`. XGBoost queda en 2.1.3,
   la versión que entrenó los artefactos (el paquete liviano `xgboost-cpu` solo existe desde la 3.0). En Linux instala
   además NCCL, que ocupa disco pero no se carga en memoria. `render.yaml` fija `PYTHON_VERSION=3.11.9`.
3. **Subir a GitHub.** La carpeta `dashboard_final/` vive en la rama del entregable del repositorio.
4. **Crear el servicio.** En Render: *New → Blueprint*, elegir el repositorio y la rama; Render lee
   `dashboard_final/render.yaml` (`rootDir: dashboard_final`, `buildCommand: pip install -r requirements.txt`,
   `startCommand: gunicorn app:server --workers 1 --threads 1 --timeout 180`). También se puede crear un *Web Service*
   a mano con esos mismos valores.
5. **Verificar.** Revisar los *logs* de construcción y abrir las tres páginas y los callbacks (filtros del EDA,
   predicción con cada modelo) en la URL pública.

**Memoria del plan gratuito (512 MB).** Medida localmente (Python 3.11, mismas versiones) al predecir con los 8 modelos
uno tras otro:

| Momento | Memoria (RSS) |
|---|---|
| Después de iniciar la app | 100 MB |
| Con los 7 benchmarks cargados | 294 MB |
| Con el modelo original cargado | 413–424 MB |
| Pico de la sesión (SHAP del original) | 479 MB |

El SHAP del original se calcula por bloques de 512 filas (`chunk` en `src/explain.py`); sin bloques, el pico llegaba a
512 MB. Para dejar margen, `render.yaml` arranca gunicorn con 1 *worker* y 1 hilo y fija `MALLOC_ARENA_MAX=2` (menos
fragmentación de memoria en Linux). **Si Render reinicia el servicio por memoria**, se pone `DASH_SIN_ORIGINAL=1` en las
variables de entorno: la app sigue funcionando con los 7 benchmarks y la sección del original (pruebas, CD y veredicto),
pero sin predicción en vivo del original. La otra opción es un plan con más memoria.
