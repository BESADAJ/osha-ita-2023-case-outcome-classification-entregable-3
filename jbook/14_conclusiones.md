# 14. Conclusiones y límites

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

## Respuesta a las correcciones del entregable 1

| Comentario | Respuesta |
|---|---|
| Brecha entrenamiento–validación (0,65 frente a 0,39) | Se debe casi entera a Fallecimiento: recall 1,0 en entrenamiento y 0 en validación aporta 0,25 de los ~0,28 de brecha. En las clases frecuentes la brecha es de 0,01 a 0,05 (sección 12.4). |
| Death sin tratamiento | Tratada con SMOTE, ADASYN y `class_weight` en todos los modelos y optimizadores; se reporta con intervalos de Wilson. El tratamiento sube el recall, pero no produce un detector útil (secciones 9.5 y 9.6). |
| Casos descartados sin discutir su costo | 55,4 % descartado; el costo en desempeño de exigir establecimientos nuevos es de 0,004 a 0,05 de AUC, medido con una segunda prueba (secciones 4.7 y 9.8). |
| Código de entrenamiento no visible | Capítulos como notebooks y código fuente impreso (sección 6.9). |
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
