# Decisión de la Ronda 2 (HAFL-XGBoost)

Generado 2026-10-21 02:56. Regla: `regla_r2.md` (fijada antes de ver resultados).
Referencia `xgboost__class_weight__bayes`: H = 8/12, casos marcados por pliegue [1117, 7, 4213], AUC-PR = 0.0018, R234 = 0.466, AUC = 0.704.

| Criterio | Candidato | Referencia | Cumple |
|---|---|---|---|
| 1. Muertes acertadas (mismos casos marcados) | 8/12 (Wilson 0.39–0.86) | 8/12 | no |
| 2. AUC-PR de Fallecimiento | 0.0019 | 0.0018 | sí |
| 3a. Recall medio 2–4 | 0.466 | 0.466 | sí |
| 3b. AUC macro (tolerancia 0,005) | 0.703 | 0.704 | sí |

**NO PASA** la regla: se detiene aquí. Se probaron cuatro candidatos y ninguno mejoró.
Efecto de selección: cuatro candidatos probados; con 12 muertes, una diferencia puede ser azar.
