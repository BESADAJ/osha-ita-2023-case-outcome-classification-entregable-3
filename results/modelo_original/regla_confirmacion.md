# Regla de la confirmación en la prueba (fijada antes de ver resultados de la Fase B)

Escrita el 2026-10-21 hacia las 01:45 (Colombia). **No se ejecuta sin el visto bueno de Ivan.**

## Datos
- Entrenamiento final: todo el desarrollo (`load_data()`: 269.516 casos, 69 muertes; fecha < 2023-09-24,
  establecimientos no reservados). Prueba: establecimientos reservados, oct–dic (127.391 casos, 39 muertes).
  La prueba no se ha usado (`osha3/data.py` solo la cuenta para verificar la partición).
- Una sola evaluación por modelo. Nada se elige con la prueba.

## Modelos y configuración (fijada en el desarrollo)
- Ganador: el optimizador de la Fase B con mayor balanced accuracy media en los 3 pliegues externos;
  se usan los hiperparámetros elegidos por ese optimizador en el **pliegue 3** (el entrenamiento más
  grande y más cercano en el tiempo a la prueba).
- Referencia: XGBoost + `class_weight` + bayesiano, con los hiperparámetros de su pliegue 3
  (`results/runs/xgboost__class_weight__bayes/fold3.json`), mismo balanceo.
- Decisión: `argmax` de las probabilidades (el umbral queda fijado en el desarrollo).

## Pruebas pareadas (todas contra la referencia, sobre los mismos casos; se reportan las tres con su p)
1. **Principal:** DeLong pareado del AUC de Fallecimiento (uno contra el resto), bilateral.
2. McNemar exacto sobre las 39 muertes (acierto sí/no de cada modelo), bilateral.
3. AUC macro: bootstrap pareado por establecimiento (2.000 réplicas, semilla 42), IC 95 % percentil de la
   diferencia y p bilateral.
No se corrigen ni se elige la mejor después; se reportan las tres.

## Decisión del 2026-10-21 (revocada)
Se había decidido no ejecutar la confirmación porque ningún candidato mejoró. Quedó revocada el mismo día con la
guía oficial del entregable (Figura 4 y sección 6.3): la confirmación se ejecutará, con esta misma regla, sobre el modelo final.
