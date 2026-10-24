# Trazabilidad de correcciones

La guía del proyecto final (Bloque A, §3) pide una tabla con cada observación, la acción tomada, su ubicación en este
libro y la evidencia. Este capítulo la reúne.

:::{important}
**Estado de la retroalimentación del entregable 2.** Al cierre de este informe (23 de octubre de 2026) el docente
**no había entregado observaciones escritas** sobre el entregable 2. Por eso la tabla tiene tres partes:

- **A.** Las observaciones del entregable 1, que ya se habían atendido y se conservan con su evidencia.
- **B.** Las autocorrecciones del entregable 2 que encontró la revisión interna del equipo: fallas de método que, si
  quedaban, invalidaban la comparación del modelo original. Se tratan como observaciones y cada una tiene su fila.
- **C.** Un espacio reservado para las observaciones que el docente entregue antes del cierre.
:::

## A. Observaciones del entregable 1 (docente)

| # | Observación del docente | Acción realizada | Ubicación en el libro | Evidencia |
|---|---|---|---|---|
| A1 | La brecha entre entrenamiento y validación (0,65 frente a 0,39) no se analiza a fondo | Se descompone la brecha por clase: Fallecimiento explica ~0,25 de los ~0,28 | Sección 12.4 | Tabla de brecha por clase, `results/cache/robustez_gap.joblib` |
| A2 | La clase Death queda sin tratamiento | SMOTE, ADASYN propio y `class_weight` en los 7 modelos; recall con IC de Wilson; regla `argmax P/π^α` en el modelo original | Secciones 9.5, 9.6 y capítulo 19 | Tablas 9.5–9.6; `SafeADASYN` en `src/osha_e3/models.py` |
| A3 | Se descarta ~la mitad de los casos sin discutir su costo | Se cuantifica el descarte (55,4 %) y se mide su costo con una segunda prueba | Secciones 4.7 y 9.8 | Figura 4.7, tabla 9.8 |
| A4 | El código de entrenamiento no es visible | Capítulos como notebooks y código fuente impreso | Sección 6.8 y capítulos 6–13, 19–20 | Celdas de código de los capítulos |
| A5 | `time_unknown` supera el umbral de colinealidad | GVIF recalculado (3,81) y ablación sin la variable | Secciones 4.6 y 12.5 | `results/cache/robustez_ablation_time_unknown.joblib` |

## B. Autocorrecciones del entregable 2 (revisión interna)

| # | Observación | Acción realizada | Ubicación en el libro | Evidencia |
|---|---|---|---|---|
| B1 | **La prueba de octubre a diciembre se usó para elegir** entre 7 variantes del modelo propuesto (capítulo 18), y la sección 18.7 decía «una sola evaluación en la prueba» | Se corrige el texto y se declara el uso. Con solo los pliegues externos se llega a la misma elección (G supera a D en 5 de 5 pliegues). Desde el 23 de octubre **la prueba está bloqueada por código**: ninguna predicción, métrica ni elección nueva la usa. Las cifras de prueba del capítulo 18 quedan como exploratorias | Cap. 18 (recuadro «Declaración»), capítulos 19–20 | `LockedTest` en `src/osha_e3/data.py`; `tests/test_candado_prueba.py` (2 pruebas) |
| B2 | **Comparación no pareada**: G usaba variables (OIICS, NAICS-6, SOC-6, empleador) que los 112 benchmarks no tenían. Unos 8 de los 9 puntos de ventaja venían de las variables | Experimento pareado: los 7 benchmarks (mejor balanceo y optimizador de cada uno) se reentrenan **con las mismas variables crudas**, las mismas particiones, la misma semilla y la misma búsqueda interna (presupuesto 16) | Capítulo 20 | `src/osha_e3/paired.py`, `results/pareado/` |
| B3 | Pruebas de superioridad hechas con la prueba (McNemar, DeLong, semanas) y con 5 pliegues (Wilcoxon no puede bajar de p = 0,0625) | Validación externa **repetida 3×5** (15 pliegues; se planeó 5×5 y se recortó por tiempo) y pruebas sobre pliegues (Friedman–Nemenyi, Nadeau–Bengio, Wilcoxon–Holm, bayesiana con ROPE, δ de Cliff, IC BCa). McNemar y DeLong sobre la prueba no se hacen | Capítulo 20 | `scripts/analyze_paired.py`, `results/pareado/resumen.json` |
| B4 | Hiperparámetros del modelo propio **sin optimización anidada** | Los hiperparámetros del combinador se optimizan con TPE (Optuna) dentro de cada pliegue externo, con 16 configuraciones y 3 pliegues internos, como los benchmarks | Secciones 19.5 y 20.1 | `original_eval.tune` |
| B5 | Pesos globales de Caruana ≈ regla estándar «pesos por error global» (Cuadro 1 de la guía), lo que pone en riesgo R2 | Nueva regla de combinación propia, **SECR**, con formulación, pseudocódigo, demostraciones y ablación contra promedio simple, pesos por error global, stacking lineal y Caruana | Capítulo 19 | `src/osha_e3/combinador.py`; tabla de ablación 20.4 |
| B6 | Faltaban pruebas unitarias del aprendizaje con solución conocida, del símplex y de la convergencia | 8 pruebas de SECR, entre ellas (i) solución conocida, (ii) semilla, (iii) entradas inválidas, (iv) `clone`, pesos en el símplex, pérdida no creciente y brecha como cota | Sección 19.7 | `tests/test_combinador.py` |
| B7 | Incoherencia entre capítulos sobre los códigos OIICS: el 17 los trata como posible fuga y el 18 los usa como entrada | Postura única: la tarea es clasificar el desenlace de un caso **ya registrado**; los códigos OIICS existen cuando se clasifica el caso, así que no son fuga directa. El riesgo indirecto está en los fallecimientos (el relato menciona la muerte en el 54 % de ellos) y se declara como limitación | Sección 19.2 y capítulo 21 | Texto de ambas secciones |
| B8 | No había capítulo de literatura ni matriz de novedad para el modelo propuesto | Búsqueda en 3 bases con API pública, 8 cadenas (inglés y español) con fecha, matriz y declaración de novedad | Capítulo 16 (sección 16.5) | `scripts/literatura/buscar.py`, `results/literatura/` |
| B9 | No había artefactos serializados ni prueba de ida y vuelta | Los 7 mejores modelos y el original se reentrenan y se serializan con `metadata.json`; la ida y vuelta da probabilidades idénticas | Sección 13b | `dashboard_final/artifacts/*.metadata.json` (`roundtrip.identical = true`) |

**Resultados recalculados.** Las autocorrecciones B2 y B3 cambian la comparación del modelo propuesto: la ventaja de
~9 puntos frente a los 112 modelos del curso (capítulo 18) **no se sostiene en la comparación pareada**, porque venía de
las variables. Todas las cifras de superioridad del informe salen ahora del capítulo 20, y el ranking pareado se
reporta tal como resultó, aunque sea desfavorable (guía §3, «honestidad ante los cambios»). Las tablas de los
capítulos 6 a 13 no cambian: ninguna corrección toca sus datos, sus particiones ni sus modelos.

## C. Observaciones del docente sobre el entregable 2

| # | Observación del docente | Acción realizada | Ubicación en el libro | Evidencia |
|---|---|---|---|---|
| C1 | *Pendiente: no se recibieron observaciones escritas antes del 23 de octubre de 2026* | — | — | — |
