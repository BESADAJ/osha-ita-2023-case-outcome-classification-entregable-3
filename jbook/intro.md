# Informe del entregable 2

Este informe acompaña al dashboard del entregable 2 del curso de Machine Learning (EST 1055-01). Explica cada
resultado que aparece en el dashboard y añade los detalles que no caben en él.

**Pregunta de investigación.** ¿En qué medida las características estructuradas del lugar de trabajo, la
ocupación y el incidente permiten clasificar los cuatro resultados registrados por OSHA en incidentes del
último trimestre de 2023, en establecimientos que no se usaron para construir el modelo?

**Dashboard:** <https://osha-ita-2023-case-outcome.onrender.com/>

## Cómo está organizado

| Capítulo | Contenido | Pestaña del dashboard |
|---|---|---|
| 1. Contexto y datos | Problema, datos, partición, límites | 1. Contexto del problema |
| 2. EDA: lo que muestra el dashboard | Interpretación de las 5 gráficas | 2. EDA |
| 3. EDA: detalles adicionales | Lo que no cabía en el dashboard | (solo aquí) |
| 4. Modelos base | Regresión logística frente a la línea base | 3. Modelos base |
| 5. Conclusiones y límites | Qué se encontró y qué no | (solo aquí) |
| 6. Reproducibilidad y despliegue | Semillas, versiones, cómo correr todo | (solo aquí) |
| Anexo | Notebook completo del entregable 1 | (solo aquí) |

## Relación con el entregable 1

El dataset, la partición, el preprocesamiento y el modelo son los del entregable 1. Aquí no se reentrenó
nada: el dashboard y este informe leen los resultados que el entregable 1 guardó (métricas, probabilidades
de la prueba, coeficientes) y las tablas del EDA que el notebook ya mostraba.

:::{note}
Un modelo de regresión de soporte lineal (SVR) no aplica en este trabajo: el SVR es para objetivos numéricos
continuos y aquí el objetivo es categórico (4 clases). Por eso el modelo base del entregable 2 es la
regresión logística.
:::
