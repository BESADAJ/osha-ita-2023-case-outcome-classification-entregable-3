# Ronda 4 (exploratoria): Frank–Hall ordinal con XGBoost

Generado 2026-10-21 04:05. Regla: `regla_r4.md` (escrita antes de calcular).
Control de reproducción (XGBoost multiclase reajustado vs predicciones guardadas de `xgboost__class_weight__random`): diferencia máxima de probabilidades 0.00e+00 → OK.

- **base**: BA 2–4 0.5274 (0.5188–0.5365); accuracy 2–4 0.522; kappa 0.288; kappa cuadrático 0.296; fracción de errores no adyacentes 0.366. 4 clases: BA 0.395, muertes 0/12 marcando 57, AUC macro 0.745, log-loss 0.964.
- **frank_hall**: BA 2–4 0.5249 (0.5162–0.5336); accuracy 2–4 0.519; kappa 0.284; kappa cuadrático 0.292; fracción de errores no adyacentes 0.350. 4 clases: BA 0.394, muertes 0/12 marcando 37, AUC macro 0.739, log-loss 0.987.

ΔBA 2–4 (Frank–Hall − base) = -0.0025, IC 95 % bootstrap pareado -0.0055 a +0.0004, p = 0.1000. McNemar: 2587 a favor / 2786 en contra, p = 0.0069.

**NO PASA** el cribado: se detiene. Cinco candidatos probados; ninguno mejoró.
Exploratorio y posterior (método elegido tras la Ronda 3); efecto de selección declarado.
