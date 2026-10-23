# Regla de la Ronda 4 (exploratoria): clasificación ordinal Frank–Hall con XGBoost

Escrita el 2026-10-21 a las 04:05 (Colombia), **antes de calcular nada**. Pedida por Ivan («probemos el ordinal, sin tope»).
**Exploratoria y posterior:** el método se eligió después de ver en la Ronda 3 que los errores de 2–4 tienen estructura
ordinal. Es el quinto candidato; el efecto de selección se declara. No toca `results/runs/` ni la prueba (oct–dic).

## Método (Frank y Hall, 2001, «A simple approach to ordinal classification», ECML)
- Orden de gravedad de OSHA (regla de registro 29 CFR 1904.7: se anota el desenlace más grave): Otro (4) < Traslado o
  restricción (3) < Días de ausencia (2) < Fallecimiento (1). Gravedad s = 0, 1, 2, 3.
- Tres modelos binarios: P(s > 0), P(s > 1), P(s > 2). Probabilidades: P(4) = 1 − P(s>0); P(3) = P(s>0) − P(s>1);
  P(2) = P(s>1) − P(s>2); P(1) = P(s>2). Si alguna sale negativa (los binarios no son monótonos), se recorta a 0 y se
  renormaliza (desviación declarada; el paper no lo trata).
- Clasificador base: XGBoost binario. Cribado: hiperparámetros de la **mejor base de la Ronda 3**
  (`results/runs/xgboost__class_weight__random/fold<k>.json`), mismo preprocesamiento y **mismos pesos de muestra**
  (`compute_sample_weight('balanced')` sobre las 4 clases). Lo único que cambia es la descomposición ordinal.
- Control de reproducción: el mismo pipeline con XGBoost multiclase debe reproducir las predicciones guardadas de esa
  base (diferencia máxima de probabilidades < 10⁻⁴); si no, el cribado se detiene y se reporta.

## Criterio (exploratorio, igual que la Ronda 3)
- Subproblema 2–4 con los 3 pliegues juntos (61.637 casos): P(Fallecimiento) descartada, renormalización, argmax.
- Comparación con la mejor base (`xgboost__class_weight__random`): ΔBA 2–4 con bootstrap pareado por establecimiento
  (las mismas 2.000 remuestras, semilla 42) y McNemar exacto.
- **Cribado (m = 1):** «pasa» si ΔBA 2–4 > 0 y el IC 95 % del bootstrap pareado excluye 0.
- Se reportan siempre, sin elegir después: BA de 4 clases, muertes acertadas y casos marcados, recall 2–4, AUC macro,
  kappa, kappa cuadrático 2–4, log-loss y la fracción de errores no adyacentes.

## Etapa 2 (solo si el cribado pasa)
Validación anidada como la corrida base: FH-XGBoost, balanceo `class_weight` (pesos de muestra), 4 optimizadores ×
3 pliegues = 12 celdas, 20 evaluaciones, el espacio y la rejilla de XGBoost del registro. Cada optimizador se compara con
la mejor base igual que arriba, con **Holm sobre los 4 optimizadores**; «mejora» solo si el IC al nivel de Holm excluye 0.
Si el cribado no pasa: se detiene y se reporta.
