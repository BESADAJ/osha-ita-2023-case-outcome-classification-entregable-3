# 14. Conclusiones, limitaciones y declaración de uso de IA

## Qué se encontró

1. **Con estas 8 variables el problema tiene un techo bajo y ningún modelo lo rompe.** Las 112
   combinaciones de clasificación quedan en la prueba entre 0,38 y 0,44 de balanced accuracy y entre 0,62
   y 0,70 de AUC macro. Los mejores en ordenar las clases (Random Forest, XGBoost, la logística: AUC
   0,69–0,70, accuracy 0,51–0,52) igualan, pero no superan, a la logística sin búsqueda del entregable 2
   (balanced accuracy 0,40, AUC 0,69). En regresión las variables explican cerca del 6 % de la varianza de
   log(1 + días de ausencia) (capítulos 9 y 10).

2. **La métrica elegida decide quién «gana», y con balanced accuracy decide una clase de 39 casos.**
   Las combinaciones con mejor balanced accuracy (SVM con SMOTE, el árbol con `class_weight`) lo consiguen
   marcando miles de casos como fallecimiento para acertar unos pocos: tienen el doble de variación entre
   pliegues, no son robustas a la semilla y ordenan *peor* las clases frecuentes que Random Forest y
   XGBoost según DeLong. La corrida con diseño temporal llega a la misma conclusión por otro camino
   (capítulos 9, 12 y 13).

3. **Fallecimiento no se puede detectar de forma útil con estos datos**, ni siquiera con los tres
   tratamientos de balanceo. La mejor detección (16 de 39 con KNN y SMOTE) cuesta más de 29.000 falsas
   alarmas. Con menos de 10 casos por entrenamiento no hay patrón que aprender (capítulos 9 y 12).

4. **La optimización bayesiana es la más eficiente, pero el paisaje es plano.** Con el mismo presupuesto, el
   TPE llega con 8 evaluaciones a donde Random llega con 16 y supera a Grid con un tamaño de efecto mediano.
   Sin embargo, los cuatro métodos terminan en modelos finales equivalentes (diferencias despreciables).
   Successive Halving obtiene el mismo resultado con 9 veces menos árboles (capítulo 7).

5. **El balanceo cambia el umbral, no la capacidad de discriminar.** SMOTE, ADASYN y `class_weight`
   suben el recall de Fallecimiento y bajan el de las otras clases; en balanced accuracy las cuatro técnicas
   son estadísticamente equivalentes y en AUC `class_weight` es la peor (capítulos 9 y 13).

6. **Las probabilidades de XGBoost y de la logística están bien calibradas** (ECE 0,02); Naive Bayes, KNN,
   el árbol y Random Forest mejoran mucho con regresión isotónica (capítulo 9).

7. **Lo que pesa son el tipo de incidente, el sector y la ocupación.** SHAP y LIME coinciden en que el tipo
   de incidente domina las decisiones individuales (una enfermedad respiratoria empuja hacia días de
   ausencia), y en conjunto el subsector y el grupo ocupacional son las variables más influyentes
   (capítulo 11).

8. **En regresión, los ensambles de árboles son los únicos que sobreviven al Model Confidence Set**, son
   indistinguibles entre sí (Diebold-Mariano p = 0,60) y superan de forma real pero pequeña a una
   referencia que predice la media (SPA p ≈ 0) (capítulo 13).

9. **El modelo original (PT-bagging) no supera a los modelos del curso**: con los pliegues temporales juntos su
   balanced accuracy es 0,480 frente a 0,516 de la referencia, y cuatro iteraciones del diseño tampoco mejoran
   (capítulos 16 y 17; primer candidato, histórico. El modelo final es SECR, capítulos 19 y 20).

## Respuesta a las correcciones del entregable 1

| Comentario | Respuesta |
|---|---|
| Brecha entrenamiento–validación (0,65 frente a 0,39) | Se debe casi entera a Fallecimiento: recall 1,0 en entrenamiento y 0 en validación aporta 0,25 de los ~0,28 de brecha. En las clases frecuentes la brecha es de 0,01 a 0,05 (sección 12.4). |
| Death sin tratamiento | Tratada con SMOTE, ADASYN y `class_weight` en todos los modelos y optimizadores; se reporta con intervalos de Wilson. El tratamiento sube el recall, pero no produce un detector útil (secciones 9.5 y 9.6). |
| Casos descartados sin discutir su costo | 55,4 % descartado; el costo en desempeño de exigir establecimientos nuevos es de 0,004 a 0,05 de AUC, medido con una segunda prueba (secciones 4.7 y 9.8). |
| Código de entrenamiento no visible | Capítulos como notebooks y código fuente impreso (sección 6.8). |
| `time_unknown` sobre el umbral de colinealidad | GVIF de 3,81 confirmado; quitarla no cambia el desempeño (≤ 0,02) y en KNN y Naive Bayes mejora un poco: la variable sobra (secciones 4.6 y 12.5). |

## Límites

- **La prueba no es virgen.** El último trimestre ya se usó para reportar los entregables 1 y 2.
- **La etiqueta de días de ausencia está censurada** por el cierre del archivo anual, más en el trimestre de
  prueba: las cifras absolutas de la regresión en la prueba están sesgadas (capítulos 4 y 10).
- **Submuestra para la búsqueda.** La optimización usó ~30.000 casos (clasificación) y ~20.000 (regresión)
  de establecimientos completos; el desarrollo completo tiene 269.516 y 99.074. La curva de aprendizaje del
  entregable 2 sugiere que más datos aportarían poco, pero no se comprobó con los 140 modelos.
- **Pocos fallecimientos por pliegue** (0 a 4 en cada pliegue externo): las métricas de 4 clases son
  ruidosas y Nemenyi con 5 pliegues no puede separar combinaciones.
- **Presupuesto pequeño** (16 configuraciones por búsqueda); el algoritmo genético converge en unas 3
  generaciones.
- **SVM lineal**, no con núcleo: el núcleo RBF no escala a este tamaño (capítulo 8).
- **SMOTE y ADASYN sobre variables one-hot** generan valores fraccionarios que no corresponden a ninguna
  categoría real.
- **Retrospectivo y asociativo.** El código SOC se asigna después del reporte; SHAP describe el modelo, no
  causas; son casos reportados, no tasas de riesgo.
- **Dependencia entre empresas.** Parte de los casos de prueba pertenece a empresas (EIN) con otros
  establecimientos en el desarrollo.

## Siguientes pasos

- Modelar la duración de la ausencia con un modelo de supervivencia que trate la censura del cierre del archivo.
- Tratar Fallecimiento como un problema aparte (detección de eventos raros con todos los años disponibles),
  o reportar el clasificador de 3 clases como modelo principal.
- Quitar `time_unknown` del conjunto de variables.
- Usar la parada temprana y Successive Halving para ampliar el presupuesto de búsqueda sin más costo.
- Añadir variables con más información sobre la lesión (naturaleza, parte del cuerpo), que el archivo trae
  codificadas, verificando antes que no revelen el resultado.


## Proyecto final: veredicto sobre el modelo original (R1, R2 y R3)

| Condición | Veredicto | Evidencia |
|---|---|---|
| **R1. Novedad en la aplicación** | **Sí** | Revisión en 3 bases con API (OpenAlex, Europe PMC y arXiv), 8 cadenas en inglés y español con fecha, más WoS, Scopus y Consensus. No se encontró la regla SECR ni la selección de ensambles aplicada al desenlace de casos de lesiones laborales. Matriz de novedad y diferenciación por familia (capítulo 16.6–16.9). Riesgo residual: el profesor puede leer la novedad como de aplicación, porque la parte regional no mejoró la métrica |
| **R2. Implementación algorítmica propia** | **Sí** | Clase propia compatible con scikit-learn (`ConditionalEnsembleSelection`, `OriginalModel`) con regla de combinación nueva, Frank-Wolfe con búsqueda lineal exacta programado con numpy, 6 proposiciones demostradas, complejidad contrastada con tiempos y 8 pruebas unitarias. El codificador por objetivo y la selección de Caruana del nivel 0 también son código propio. Los modelos base son de librería, como permite el Cuadro 1 para ensambles con regla de combinación propia |
| **R3. Superioridad estadísticamente verificada** | **No** | Comparación pareada en 15 pliegues: el original tiene la mayor BA media (0,591 frente a 0,569 de KNN) y el mejor rango de Friedman (p = 6·10⁻¹¹). Pero frente al mejor benchmark ninguna prueba pareada rechaza la igualdad tras Holm: Nemenyi p = 0,81, Nadeau–Bengio p = 0,66, Wilcoxon p = 0,28; bayesiana P(mejor) = 0,947; δ = 0,19 con IC que incluye 0 (capítulo 20) |

**Tope de calificación que corresponde: 4,0** (R1 sí, R2 sí, R3 no; guía §12.1). La BA de 0,591 es optimista por
selección: el original salió de 13 candidatos (sección 19.9), y su nivel 0 se eligió mirando la repetición 0. El sesgo va
a favor de «superior», así que el veredicto R3 = no es conservador. Lo reportamos así porque es lo que
muestran los datos. La comparación del capítulo 18, con ~9 puntos de ventaja sobre los 112 modelos del curso, no era
válida: no era pareada y miraba la prueba. Corregirla cambió el resultado (capítulo 00, B1–B3).

## Hallazgos del proyecto final

1. **La información de la lesión es lo que más mejora el modelo, no el algoritmo.** Con los códigos OIICS, NAICS-6,
   SOC-6 y el empleador, casi todos los benchmarks mejoran: KNN pasa de 0,415 a 0,569 de BA (del quinto al primer lugar),
   la logística de 0,427 a 0,523 y Naive Bayes de 0,389 a 0,435. El árbol baja, de 0,461 a 0,416.
2. **El ensamble multivista con SECR es el mejor modelo en media** (BA, log-loss, AUC y F1), pero su ventaja sobre KNN
   (+0,022 de BA) es pequeña frente a la variabilidad entre pliegues.
3. **La corrección por prior ($\alpha$) aporta la mayor parte de la BA** (+0,13). Las regiones de SECR no aportan: los
   pesos óptimos son casi iguales en todas las regiones, la TPE elige compuertas distintas en cada pliegue y la
   sensibilidad muestra que con contracción fuerte (pesos globales) la BA es igual o mayor.
4. **La BA de 4 clases es muy ruidosa por los fallecimientos** (0 a 4 por pliegue): un solo caso mueve la métrica
   ±0,25. Eso limita la potencia de cualquier comparación entre modelos con esta métrica.
5. **El producto funciona sin tocar la prueba.** El dashboard predice con los 8 artefactos, explica cada predicción con
   SHAP exacto y muestra diagnósticos calculados fuera de pliegue.

## Limitaciones del proyecto final

- **Menos repeticiones de las planeadas.** Se planeó una validación repetida 5×5 y se corrió 3×5 por tiempo de cómputo
  (unas 4 h con 10 procesos; Random Forest y Naive Bayes se completaron después, con el tiempo sobrante). El recorte se decidió antes de ver
  los resultados y por el costo, no por el desempeño.
- **Hiperparámetros fijos en el nivel 0.** Los 16 modelos base del original usan hiperparámetros fijos (los del
  curso o valores por defecto razonables); solo el combinador se optimiza dentro de la validación anidada.
- **Un solo conjunto de datos.** El Cuadro 1 pide además 3 conjuntos de datos adicionales de la misma tarea (por
  ejemplo, los conjuntos armonizados de Bauder et al., 2026). No se hizo por tiempo, y la generalidad de SECR queda sin
  demostrar.
- **Fuga indirecta en fallecimientos.** Los códigos OIICS se derivan del relato del caso, que menciona la muerte en el
  54 % de los fallecimientos.
- **Las regiones no ayudaron aquí.** Es posible que SECR sí aporte en problemas donde la calidad relativa de los modelos
  cambie más entre subpoblaciones. Esa hipótesis queda sin probar.

## Trabajo futuro

- Correr las repeticiones que faltan (hasta 5×5 o 10×5) para medir con más potencia la diferencia frente a KNN, y
  reportarla como un análisis aparte, no como reemplazo de este.
- Usar una métrica principal menos dominada por una clase de 3 por 10 000 (por ejemplo, la BA de 3 clases o la
  log-loss), declarada **antes** de correr.
- Probar SECR en los conjuntos de datos adicionales y con regiones aprendidas (un árbol poco profundo como compuerta).

## Declaración de autoría y uso de herramientas (incluida IA)

Este trabajo es de **Jaime Andrés Besada e Iván Berrio**. Siguiendo la guía (§10.7 y §13), declaramos las herramientas
usadas y para qué:

| Herramienta | Para qué se usó | Qué revisó el equipo |
|---|---|---|
| Python, numpy, scipy, pandas, scikit-learn, imbalanced-learn, XGBoost, Optuna, DEAP, Dash, Plotly | Implementación, experimentos y dashboard | Todo el código |
| **Claude Code (Anthropic; modelo Claude Opus 5.5)** | Asistente de programación y redacción en el proyecto final: implementó, bajo instrucciones del equipo, el experimento pareado 5×5, el combinador SECR y sus pruebas unitarias, las pruebas de hipótesis sobre pliegues, el candado de la prueba, el dashboard multipágina y los artefactos. También redactó borradores de los capítulos 00, 13b, 16.6–16.9, 19, 20 y 21 y escribió los scripts de consulta a las bases bibliográficas | El diseño, las demostraciones, el código y las cifras se revisan contra los resultados guardados en `results/`. El equipo responde por cada línea y puede explicarla y modificarla en la sustentación |
| Consensus | Búsquedas exploratorias de literatura (capítulo 16.1) | Se verificaron los trabajos citados |
| API públicas de OpenAlex, Europe PMC y arXiv | Búsquedas reproducibles de literatura (capítulo 16.6) | Lectura de títulos y resúmenes |

Ninguna cifra del informe se escribió de memoria. Todas salen de un script versionado y de un resultado guardado en el
repositorio. Las referencias se verificaron contra su fuente; las que no se pudieron verificar se citan solo con los
datos confirmados.

Firmas:

| Integrante | Firma | Fecha |
|---|---|---|
| Jaime Andrés Besada | Firmada (ver abajo) | 2026-10-23 |
| Iván Berrio | Firmada (ver abajo) | 2026-10-23 |

**Firma de Jaime Andrés Besada:**

```{image} figures/nombre_jaime_besada.png
:alt: Firma manuscrita de Jaime Andrés Besada
:width: 220px
```

**Firma de Iván Berrio:**

```{image} figures/firma_ivan_berrio.png
:alt: Firma de Iván Berrio
:width: 260px
```
