# Informe del entregable 3

Este Jupyter Book es el entregable 3 del proyecto de Machine Learning (EST 1055-01) sobre el archivo de
casos de lesiones y enfermedades laborales de OSHA (ITA 2023). Parte del informe del entregable 2 (que
explicaba el dashboard y la regresión logística de base) y añade lo que pide la guía del tercer entregable:
**140 modelos** entrenados, optimizados y evaluados con validación cruzada anidada, cuatro métodos de
optimización de hiperparámetros, balanceo de clases, optimización computacional, calibración,
interpretabilidad y comparación estadística jerárquica.

**Preguntas de investigación.**

1. *Clasificación.* ¿En qué medida las características estructuradas del lugar de trabajo, la ocupación y
   el incidente permiten clasificar los cuatro resultados registrados por OSHA en incidentes del último
   trimestre de 2023, en establecimientos que no se usaron para construir el modelo?
2. *Regresión.* Entre los casos que terminaron en días de ausencia, ¿cuánto de la duración de la ausencia
   (en escala logarítmica) explican esas mismas características?

**Dashboard del entregable 2:** <https://osha-ita-2023-case-outcome.onrender.com/>

## Cómo está organizado

| Parte | Capítulos |
|---|---|
| Datos | 1. Contexto · 2–3. EDA de los entregables 1 y 2 · **4. EDA mejorado** (variables agrupadas, horas desde el inicio de la jornada, objetivo de regresión) |
| Modelos | 5. Modelo base del entregable 2 · **6. Diseño del pipeline combinatorio** · **7. Optimización de hiperparámetros** · **8. Optimización computacional** · **9. Clasificación** · **10. Regresión** · **11. Interpretabilidad** · **12. Robustez** · **13. Comparación estadística** |
| Cierre | 14. Conclusiones · 15. Reproducibilidad · Anexo: notebook del entregable 1 |

Los capítulos en negrita son nuevos y son notebooks con el código que produce cada tabla y figura. Las
corridas largas (validación anidada de las 140 combinaciones) se ejecutan con los scripts de `scripts/`
y los capítulos leen sus resultados de `results/`; el capítulo 15 explica cómo reproducir todo.

## Del entregable 2 al 3

| Entregable 2 | Entregable 3 |
|---|---|
| Regresión logística sin búsqueda de hiperparámetros frente a un Dummy | 7 modelos de clasificación × 4 balanceos × 4 optimizadores = 112, y 7 de regresión × 4 optimizadores = 28 |
| NAICS y SOC completos (1.433 columnas) | Subsector NAICS-3 y grupo menor SOC (~210 columnas) |
| Validación por establecimiento y tiempo, 3 pliegues | Validación anidada por establecimiento (5 externos × 3 internos); misma prueba de octubre a diciembre |
| Solo clasificación | Además, regresión de los días de ausencia |
| Intervalos bootstrap | Además, Friedman–Nemenyi, DeLong, MCS, Giacomini-White, Clark-West, Diebold-Mariano (HLN), SPA |

:::{note}
**Correcciones del entregable 1.** La guía pide incorporar las correcciones que se hicieron al primer
entregable. Se listan en la sección 3.8 a medida que se integran.
:::
