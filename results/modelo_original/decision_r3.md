# Ronda 3 (exploratoria): subproblema de las clases 2 a 4

Generado 2026-10-21 03:11. Regla: `regla_r3.md` (escrita antes de calcular). 61637 casos (clases 2–4, 3 pliegues), 15897 establecimientos, 2000 remuestras por establecimiento.
Exploratorio, decidido después de ver que Fallecimiento no es evaluable ni mejorable con 12 muertes.

**Mejor base:** `xgboost__class_weight__random` — BA 2–4 0.527 (0.519–0.536), accuracy 0.522 (0.514–0.531), kappa 0.288 (0.275–0.301), kappa cuadrático 0.296 (0.280–0.313), AUC macro 0.716, log-loss 0.957. Sesgo por elegir el máximo de 108: pequeño con ~61.600 casos, pero existe.

| Modelo | BA 2–4 | Δ vs mejor base (IC 95 %) | IC al nivel de Holm | p bootstrap (Holm) | McNemar a favor / en contra | p McNemar (Holm) | ¿Mejora? |
|---|---|---|---|---|---|---|---|
| pt_bagging | 0.489 | -0.0382 (-0.0476 a -0.0304) | -0.0502 a -0.0284 (98.75 %) | 0.0000 (0.0000) | 6894 / 8985 | 5.9e-62 (1.2e-61) | no |
| multirandbal | 0.482 | -0.0458 (-0.0549 a -0.0378) | -0.0573 a -0.0361 (98.33 %) | 0.0000 (0.0000) | 7430 / 9902 | 7.4e-79 (3e-78) | no |
| hierarchical | 0.495 | -0.0327 (-0.0411 a -0.0250) | -0.0422 a -0.0241 (97.50 %) | 0.0000 (0.0000) | 7193 / 8763 | 1.8e-35 (1.8e-35) | no |
| hafl_xgboost | 0.502 | -0.0250 (-0.0335 a -0.0180) | -0.0335 a -0.0180 (95.00 %) | 0.0000 (0.0000) | 3705 / 5433 | 2.2e-73 (6.6e-73) | no |
| hafl_xgboost_gamma0 | 0.503 | -0.0245 (-0.0333 a -0.0174) | no cuenta (ablación) | 0.0000 (nan) | 3797 / 5462 | 2.3e-67 (nan) | no |

**Ningún candidato mejora** al mejor base en el subproblema 2–4.

## Estructura ordinal (mejor base)
Errores: 29455 de 61637; adyacentes (2↔3, 3↔4) 21083 (71.6%); no adyacentes (2↔4) 8372 (28.4%).
Entre los errores de los casos de clase 2 y 4, fracción que cae en la clase no adyacente: 0.366 (IC 95 % por establecimiento 0.345–0.389; sin orden sería ~0,5).
→ **Hay estructura ordinal** (intervalo entero por debajo de 0,5).

Matriz de confusión del mejor base (filas = verdad, columnas = predicción): `confusion_mejor_base_2_4.csv`.

Fase B de PT-bagging: no entra (ningún optimizador tenía los 3 pliegues al calcular).
