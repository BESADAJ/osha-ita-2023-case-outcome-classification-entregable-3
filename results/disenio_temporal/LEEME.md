# Corrida con diseño temporal (Iván Berrio, 2026-10-18)

Copia de las tablas de resumen de la rama `entregable3-validacion-anidada` del repositorio
<https://github.com/sak-buks/osha-ita-2023-case-outcome-classification> (carpeta `entregable3/results/`).

Diseño: 3 pliegues externos hacia adelante en el tiempo (los del entregable 1) × 3 internos por establecimiento,
NAICS y SOC completos (con `min_frequency` como hiperparámetro), búsqueda en el 25 % del entrenamiento externo
(con todas las muertes) y reentrenamiento con el 100 %, 20 evaluaciones por optimizador, SMOTE/ADASYN solo sobre
Fallecimiento. 108 combinaciones (KNN × class_weight no se corrió) + Dummy + logística del entregable 1.

No se copian las predicciones (`*_pred.npz`) porque contienen identificadores de establecimientos y casos.
