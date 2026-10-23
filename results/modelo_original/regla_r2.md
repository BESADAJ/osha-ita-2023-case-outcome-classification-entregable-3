# Regla de paso de la Ronda 2 (HAFL-XGBoost), fijada antes de ver resultados

Escrita el 2026-10-21 hacia las 03:00 (Colombia), antes de ajustar el candidato.
Fuente: `modelo_original/ronda2_focal_xgboost.md`. Aquí solo se vuelve operativa.

- Referencia: XGBoost + `class_weight` + bayesiano (`results/runs/xgboost__class_weight__bayes/fold<k>_pred.npz`).
- k_f = casos que la referencia marca como Fallecimiento en el pliegue f (argmax): 1.117, 7 y 4.213.
- Decisión del candidato a casos marcados iguales: en cada pliegue se marcan como Fallecimiento los k_f casos
  con mayor P(Fallecimiento); el resto recibe el argmax entre las clases 2, 3 y 4.
- Todo se mide con los 3 pliegues juntos (12 muertes):
  1. muertes acertadas del candidato (suma de los 3 pliegues) **> 8** (las de la referencia);
  2. AUC-PR de Fallecimiento (pliegues juntos) **mayor** que la de la referencia;
  3. recall medio de las clases 2–4 (con la decisión de arriba) **≥** el de la referencia, **y**
     AUC macro ≥ AUC macro de la referencia − 0,005.
- Pasa solo si cumple 1, 2 y 3. Si no pasa: se detiene (sin Fase B de esta ronda) y el informe dice que se probaron
  cuatro candidatos y ninguno mejoró.
- Efecto de selección declarado: cuatro candidatos probados; con 12 muertes, pasar puede ser azar. Se reporta Wilson.
- Ablación informativa (no entra en la regla): el mismo objetivo con γ_start = 0 (entropía cruzada con el hessiano
  exacto), para separar el efecto de la pérdida focal del efecto del objetivo propio.
