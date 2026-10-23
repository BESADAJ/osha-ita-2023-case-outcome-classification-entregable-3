# Informe del proyecto final (entregable 3)

Este Jupyter Book es el informe técnico del proyecto final de Machine Learning (EST 1055-01) sobre el archivo de casos de
lesiones y enfermedades laborales de OSHA (*Injury Tracking Application*, ITA 2023). Cubre los cuatro componentes
obligatorios de la guía:

1. **Correcciones** del entregable 2, con su tabla de trazabilidad (capítulo 00).
2. **Dashboard predictivo** en Dash Plotly, desplegado en Render, con las secciones Contexto, EDA y Modelos
   (capítulo 21): **<https://osha-ita-2023-dashboard-final.onrender.com>**.
3. **Modelo original**: un ensamble multivista con **selección de ensambles condicionada por región (SECR)**, implementado
   desde cero y comparado estadísticamente con los benchmarks (capítulos 16, 19 y 20).
4. Este **informe** reproducible.

## Resumen ejecutivo

**Problema.** Con los registros que los empleadores de EE. UU. reportan a OSHA se predice el desenlace de cada caso
registrable: fallecimiento, días de ausencia, traslado o restricción, u otro caso registrable. Es una clasificación de 4
clases con desbalance extremo (los fallecimientos son 3 de cada 10 000 casos). La métrica principal es la balanced
accuracy (BA).

**Benchmarks.** 7 modelos × 4 balanceos × 4 optimizadores = 112 combinaciones de clasificación (y 28 de regresión), con
validación cruzada anidada por establecimiento. La mejor combinación de cada modelo se elige por la BA media del bucle
externo, se reentrena y se serializa con su `metadata.json` (capítulo 13b). La mejor global es SVM lineal + SMOTE +
búsqueda aleatoria (BA 0,469 ± 0,101).

**Modelo original.** 16 modelos base, cada uno especializado en una vista del caso, se combinan con SECR. SECR aprende
pesos convexos por región, los contrae hacia los pesos globales y los resuelve con Frank-Wolfe con búsqueda lineal
exacta. Tiene propiedades demostradas: convexidad, pérdida no creciente, certificado de optimalidad y convergencia
$O(1/t)$.

**Validación.** Comparación **pareada**: mismas variables, mismas particiones, misma semilla y mismo presupuesto de
búsqueda para el original y los 7 benchmarks. Validación externa repetida 5×5 y pruebas sobre pliegues: Friedman–Nemenyi,
Nadeau–Bengio, Wilcoxon–Holm, prueba bayesiana con ROPE y δ de Cliff. La prueba de octubre a diciembre está bloqueada
por código. El veredicto sobre R1, R2 y R3 está en el capítulo 20 y en las conclusiones.

## Objetivos

1. Corregir y documentar las fallas de método del entregable 2.
2. Seleccionar con criterio estadístico la mejor combinación de cada modelo y llevarla a producción.
3. Diseñar, implementar y validar un modelo original que cumpla R1 (novedad), R2 (implementación propia) y R3
   (superioridad estadística), y reportar el resultado con honestidad.
4. Publicar un dashboard que permita a una persona no experta entender el problema, explorar los datos y predecir.

## Mapa del libro

| Parte | Capítulos |
|---|---|
| Correcciones | **00. Trazabilidad de correcciones** |
| Datos | 1. Contexto · 2–3. EDA de los entregables 1 y 2 · 4. EDA mejorado |
| Benchmarks | 5. Modelo base · 6. Diseño del pipeline · 7. Optimización de hiperparámetros · 8. Optimización computacional · 9. Clasificación · 10. Regresión · 11. Interpretabilidad · 12. Robustez · 13. Comparación estadística · **13b. Mejores combinaciones y artefactos** |
| Modelo original | **16. Literatura y novedad** · 17. PT-bagging (primer candidato, no superior) · 18. Ensamble multivista G (exploratorio) · **19. SECR: formulación, algoritmo, demostraciones y código** · **20. Validación estadística (R3)** |
| Producto y cierre | **21. Dashboard y despliegue** · 14. Conclusiones (veredicto R1–R3 y declaración de uso de IA) · 15. Reproducibilidad y bibliografía · Anexo del entregable 1 |
| Todo en uno | Notebook único con todo el entregable |

Los capítulos 6 a 13, 17, 18 y 20 son notebooks con el código que produce cada tabla y figura. Las corridas largas se
ejecutan con los scripts de `scripts/` y los capítulos leen sus resultados versionados en `results/`. El capítulo 15
explica cómo reproducir todo.

:::{note}
**Correspondencia con la estructura de la guía (Cuadro 4).** 0 Introducción → esta página · 1 Trazabilidad → cap. 00 ·
2 Datos → caps. 1–4 · 3 Benchmarks → caps. 5–13b · 4 Literatura → cap. 16 · 5 Modelo original → caps. 17–19 ·
6 Validación → cap. 20 · 7 Dashboard → cap. 21 · 8 Conclusiones → cap. 14 · 9 Reproducibilidad y bibliografía →
cap. 15.
:::
