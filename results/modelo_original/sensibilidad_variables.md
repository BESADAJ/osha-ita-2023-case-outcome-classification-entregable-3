# Sensibilidad a las variables (prueba rápida, 2026-10-21)

**Qué es:** una prueba exploratoria mía, **no** parte del pipeline del grupo. Script: `sensibilidad_variables_feat_test.py` (se corre con la ruta del CSV como argumento).
**Diseño:** solo clases 2, 3 y 4 (Fallecimiento fuera); 890.934 casos; partición 70/30 por establecimiento (aleatoria, no temporal, semilla 42); `HistGradientBoostingClassifier` (150 iteraciones, `class_weight='balanced'`), entrenado con 250.000 filas; categorías de alta cardinalidad limitadas a las 250 más frecuentes; NAICS a 3 dígitos y SOC a grupo menor. Una sola partición, sin intervalos de confianza.

| Conjunto de variables | BA (clases 2–4) | AUC macro (clases 2–4) |
|---|---|---|
| A. Las 8 variables del grupo | 0,533 | 0,722 |
| B. A + tamaño del establecimiento (size, empleados, horas) | 0,541 | 0,729 |
| C. A + naturaleza, parte del cuerpo, evento y fuente (`*_pred`) | 0,598 | 0,787 |
| D. B + C | 0,605 | 0,793 |

**Lectura [inf]:** A reproduce el techo (~0,53) de la corrida del grupo. El tamaño del establecimiento aporta poco. Los campos `*_pred` suben ~0,065 de BA y ~0,065 de AUC, pero la guía de sustentación del entregable 1 los excluye porque derivan de la narrativa y no se sabe si existen antes del desenlace; tampoco se sabe si la narrativa menciona el resultado. Documentar como limitación y como sensibilidad, no como cambio de diseño.
