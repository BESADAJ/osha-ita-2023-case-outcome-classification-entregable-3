# Proyección de tiempo de la Fase B (PT-bagging) y recorte aplicado

Escrito 2026-10-21 ~02:25 (Colombia), antes de lanzar la Fase B.

## Mediciones reales [obs]
- Un árbol sin podar (`entropy`, `max_features=None`, 1.283 columnas), solo: 0,22 s (5.000 filas), 1,32 s (20.000),
  4,1 s (50.000), 15,4 s (138.669).
- Fase A (5 procesos a la vez), ajuste + predicción de PT-bagging con 100 árboles: 469 s (pliegue 1),
  1.047 s (pliegue 2), 1.822 s (pliegue 3). Fase A completa: 32 min de pared (01:42 a 02:14).

## Proyección [inf]
Supuestos: entrenamiento interno = 2/3 de la muestra de búsqueda (9.300 / 15.800 / 23.000 filas); ~190 árboles
de media por evaluación; factor 0,75 por las configuraciones podadas; ×1,2 por carga de 5 procesos.

| Celda (por optimizador) | 20 evaluaciones | 12 evaluaciones |
|---|---|---|
| Pliegue 1 | ~1,6 h | ~1,0 h |
| Pliegue 2 | ~3,0 h | ~2,0 h |
| Pliegue 3 | ~5,0 h | ~3,3 h |
| 4 optimizadores (CPU) | ~38 h | ~25 h |
| Pared con 5 procesos | **~9,5 h** | **~6 h** |

«Horas de cómputo» se interpreta como horas de pared de este equipo (5 procesos). A + B con 20 evaluaciones:
0,5 + ~9,5 ≈ 10 h o más, en el límite y con incertidumbre alta → se aplica el **recorte (1)**:
**12 evaluaciones por optimizador** (A + B ≈ 6,5 h; termina hacia las 09:00 si la proyección acierta).
Consecuencia: la rejilla debe tener ≤ 12 puntos; pasa de 18 a 12:
`n_estimators [100, 200] × max_depth [None, 10] × min_samples_leaf [1, 10, 50]` (se quita `max_depth=30`).
No se recorta la muestra de búsqueda ni los optimizadores.
