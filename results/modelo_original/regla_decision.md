# Regla de decisión de la Fase A (fijada antes de ver resultados de los candidatos)

Escrita el 2026-10-21 hacia las 01:40 (Colombia), antes de ajustar cualquier candidato.
Fuente: `modelo_original/plan_cribado_candidatos.md` (criterio de mejora) y la respuesta de Ivan del 2026-10-21.
Aquí solo se vuelven operativas las palabras «mismas», «no baja» y «de forma clara».

## Referencia
La mejor de las 8 combinaciones de la corrida base con `class_weight` (Random Forest y XGBoost × 4 optimizadores):
la de mayor balanced accuracy media en los 3 pliegues externos según `results/master_registry.csv`
(XGBoost + `class_weight` + bayesiano; ya conocida antes de este trabajo). Se reportan también las otras 7.

## Cantidades (los 3 pliegues temporales juntos: 12 muertes, 61.649 casos)
- H = muertes acertadas (de 12); F = casos marcados como Fallecimiento.
- R234 = recall medio de las clases 2, 3 y 4. AUC = AUC macro uno contra el resto. LL = log-loss.
- La decisión de cada modelo es `argmax` de `predict_proba`, igual que en `osha3/metrics.py`.

## Un candidato «mejora» si cumple (a) y (b)
(a) Fallecimiento, una de estas dos:
  - a1: más muertes con las mismas falsas alarmas: H_c > H_r y F_c ≤ 1,10 · F_r;
  - a2: las mismas (o más) muertes con menos casos marcados: H_c ≥ H_r y F_c < F_r.
(b) Resto: R234_c ≥ R234_r − 0,005 **y** AUC_c ≥ AUC_r − 0,01 **y** LL_c ≤ 1,10 · LL_r.

## Elección
1. Si cumplen uno o más: el que más muertes acierte; si empatan, el que marque menos casos;
   si siguen empatados, el orden fijo PT-bagging, MultiRandBal, jerárquica.
2. Si no cumple ninguno: pasa a la Fase B el que más muertes acierte con menos casos marcados
   (mismo orden), y el informe dice que **no mejoró**.
3. No se corre nada más para desempatar ni se cambia la regla después de ver los números.
Sesgo declarado: elegir el mejor de 3 candidatos sobrestima su ventaja.
