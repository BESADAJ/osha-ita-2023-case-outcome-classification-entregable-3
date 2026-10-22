# Regla de la Ronda 3 (exploratoria): subproblema de las clases 2 a 4

Escrita el 2026-10-21 a las 03:10 (Colombia), **antes de calcular nada**. Fuente: `modelo_original/ronda3_subproblema_2_4.md`.
**Exploratorio, decidido después de ver que Fallecimiento no es evaluable ni mejorable con 12 muertes.** No reajusta ni
reentrena nada, no toca `results/runs/` ni la prueba (oct–dic) y no cambia el criterio de las Rondas 1 y 2.

## Datos
- Predicciones guardadas (`fold<k>_pred.npz`) de los 3 pliegues temporales, juntos; solo los casos con clase verdadera
  2, 3 o 4. Se verifica que todos los modelos tengan los mismos casos en el mismo orden (`case_id`).
- En cada modelo se descarta P(Fallecimiento), se renormalizan P(2), P(3), P(4) (si suman 0: 1/3 cada una) y la
  predicción es el argmax entre 2, 3 y 4.
- Modelos: las 110 celdas de `results/runs/` (108 combinaciones del curso + Dummy + logística del entregable 1); los
  tres candidatos de la Ronda 1 (`results/original/cribado/`); HAFL-XGBoost y su ablación γ = 0
  (`results/original_r2/cribado/`); y PT-bagging de la Fase B **solo por optimizador con los 3 pliegues terminados al
  momento del cálculo**. A las 03:10 ninguno los tiene (2 de 12 celdas), así que la Fase B **no entra**.

## Métricas por modelo
Balanced accuracy de 2 a 4 (BA234 = recall medio), accuracy, kappa, kappa cuadrático ponderado (orden 2 > 3 > 4),
AUC macro uno contra el resto sobre las tres clases y log-loss (con las probabilidades renormalizadas).
Intervalos: bootstrap por establecimiento (2.000 remuestras, semilla 42, percentil 95 %) para BA234, accuracy, kappa
y kappa cuadrático; AUC y log-loss solo como punto (su bootstrap no aporta a la decisión).

## Comparaciones
1. **Mejor base:** entre las 108 combinaciones del curso (Dummy y la logística del entregable 1 no son elegibles), la de
   mayor BA234. Sesgo por elegir el máximo: declarado (pequeño con ~61.600 casos).
2. **Candidatos** (m = 4): PT-bagging, MultiRandBal y jerárquica (Ronda 1) y HAFL-XGBoost (Ronda 2). La ablación γ = 0 se
   reporta pero no cuenta en m. Para cada uno contra el mejor base:
   - diferencia de BA234 con bootstrap **pareado** por establecimiento (mismas 2.000 remuestras para todos);
     p bilateral = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0));
   - McNemar exacto (binomial sobre los pares discordantes de acierto/error en 2–4);
   - Holm sobre los 4 candidatos, por separado para cada prueba; se reporta además el intervalo del bootstrap al
     nivel de Holm que le toca (1 − 0,05/(m − i + 1) para el i-ésimo p más chico).
   - Se llama **«mejora»** solo si Δ > 0 y el intervalo bootstrap al nivel de Holm excluye 0. Aun así es exploratoria.
3. **Estructura ordinal** (para orientar la búsqueda de métodos, no para elegir ganador): matriz de confusión del mejor
   base; fracción de errores entre clases no adyacentes (2↔4) y adyacentes (2↔3, 3↔4). Indicador: entre los errores de
   los casos de clase 2 y 4 (que tienen un vecino adyacente y uno no adyacente), la fracción que cae en la clase no
   adyacente; sin estructura ordinal sería ~0,5. Se dice «hay estructura ordinal» solo si el intervalo bootstrap por
   establecimiento de esa fracción queda entero por debajo de 0,5.

## Reglas
Todo se reporta, con sus intervalos. No se descarta ninguna comparación ni se añade otra después de ver los números.
