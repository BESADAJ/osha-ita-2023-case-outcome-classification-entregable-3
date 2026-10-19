# 3. EDA: detalles que no están en el dashboard

Todo con los datos de desarrollo. El análisis completo, con el código, está en el anexo (notebook del
entregable 1). Aquí van los resultados que sostienen las decisiones del modelo.

## 3.1 Descripción de cada variable

- **Categóricas.** NAICS tiene 715 códigos distintos y 700 de ellos pesan menos del 1 % cada uno (en total, el
  39 % de los casos). SOC tiene 598 códigos más los faltantes (582 categorías por debajo del 1 %, 36,7 % de los casos). Estado tiene 55, con California
  como el más frecuente (13,3 %). El 93,7 % de los casos viene de establecimientos privados (código 1) y el 89,7 % son
  lesiones.
- **Categorías raras.** 354 códigos NAICS y 309 SOC tienen menos de 30 casos (1,2 % y 0,9 % de los casos).
  Se conservan: la regularización L2 de la regresión logística contrae los coeficientes con poco soporte y
  `handle_unknown='ignore'` evita errores si aparece un código nuevo en la prueba.
- **Horas.** La hora de inicio es asimétrica a la derecha (asimetría 0,77; mediana 7 h, turnos de mañana). La
  hora del incidente es casi simétrica (asimetría −0,01; mediana 12 h). Las pruebas de normalidad rechazan en las
  dos (p ≈ 0), algo esperable con tantos datos y con una variable discreta y acotada; no se transforman. Como
  la hora es cíclica (la hora 23 está pegada a la 0), se calculó también la media circular (7,3 h y 11,7 h).
- **Valores atípicos.** La regla del IQR no marca ninguno: 0–23 es el rango válido del reloj.

## 3.2 Relaciones entre predictores

- **Entre las dos horas:** correlación moderada (Pearson 0,25; Spearman 0,34). La mediana entre el inicio de
  la jornada y el incidente es 4 horas.
- **Horas frente al resultado:** 4 de 18 comparaciones salen significativas tras corregir por comparaciones
  múltiples (Holm), pero el mayor tamaño de efecto es |r| = 0,154: pequeño o despreciable. Las horas no
  ordenan el resultado de forma monótona, por eso se codifican como categorías y no con una pendiente lineal.
- **Entre categóricas:** la asociación más fuerte es NAICS con tipo de establecimiento (V = 0,513). Industria,
  ocupación y establecimiento describen el mismo lugar desde ángulos distintos.
- **Multicolinealidad (GVIF):** solo `time_unknown` pasa el umbral (3,80), por construcción: casi siempre
  coincide con «hora desconocida». El resto queda por debajo de 1,2 una vez ajustado por grados de libertad.

## 3.3 Estructura multivariada

- **Dimensionalidad alta:** 10 componentes explican solo el 56,8 % de la varianza y hacen falta 49 para el
  80 %. Esto se debe a que NAICS, SOC y estado aportan cientos de indicadores casi independientes.
- **Atípicos multivariados:** Isolation Forest marca el 16,8 % de los casos y Mahalanobis el 1,2 %, y los dos
  criterios coinciden poco. Sobrerrepresentan enfermedades no traumáticas: son subpoblaciones legítimas, no
  errores, por eso no se eliminan.
- **Grupos:** k-means elige 3 grupos (silueta 0,31) que se distinguen sobre todo por el sector. Su asociación con
  el resultado es baja (V = 0,022): reflejan el tipo de lugar de trabajo, no clases del objetivo.

## 3.4 Auditoría de fuga de información

Cada predictor se evaluó solo, con una regresión logística de una variable, validada hacia adelante en
agosto–septiembre. Un AUC cercano a 1 sería una alarma.

| Predictor | Macro AUC solo |
|---|---|
| `naics_code` | 0,629 |
| `soc_code` | 0,618 |
| `type_of_incident` | 0,572 |
| `hora de inicio` | 0,517 |
| `hora del incidente` | 0,517 |
| `establishment_type` | 0,515 |
| `state` | 0,512 |
| `time_unknown` | 0,499 |

Ninguno se acerca a 1; el más alto es 0,629. Aun así, el código SOC se asigna retrospectivamente y no se puede
asegurar que existiera al momento del incidente. Por eso el problema se plantea como clasificación
retrospectiva.

## 3.5 Establecimientos, repeticiones y calidad de datos

- **Concentración:** la mediana es de 3 casos por establecimiento y el máximo de 2.249. Los 100 mayores
  concentran el 12,2 % de los casos. Los casos de un mismo establecimiento no son independientes, y por eso la
  partición separa establecimientos completos y el bootstrap remuestrea establecimientos.
- **Duplicados:** no hay filas exactamente duplicadas. Hay 64.748 repeticiones de la pareja
  establecimiento–número de caso, pero repetir el número no demuestra duplicación (cambian fecha o resultado),
  así que no se eliminan.
- **Códigos no documentados:** el tipo de establecimiento 0 (243 casos) no está en el diccionario oficial. Se
  conserva como categoría propia sin suponer que signifique algo.
- **Fechas:** hay 562 casos con fecha de incidente fuera de 2023 (otros años de reporte); se excluyen.

## 3.6 Tiempo y geografía

- **Serie diaria completa y regular:** no hay días sin casos entre el 1 de enero y el corte.
- **Estacionariedad:** ADF (p = 0,109) y KPSS (p > 0,10) no rechazan su hipótesis nula. Los dos resultados no
  son contradictorios pero tampoco concluyentes con un solo año y un ciclo semanal fuerte.
- **Feriados:** reducen el volumen (razón media feriado/día de referencia 0,70; el menor es Memorial Day,
  0,46). Es un efecto de cantidad, no necesariamente de tipo de resultado.
- **Deriva:** la composición del resultado cambia poco entre trimestres (diferencia total máxima 0,036). No hay
  evidencia de un cambio fuerte dentro de 2023, pero sí cambios pequeños que justifican validar hacia adelante.
- **Posible maduración de la etiqueta:** los días de ausencia pueden seguir acumulándose después del reporte,
  así que los casos más recientes podrían figurar como menos graves. No se puede verificar sin versiones
  históricas del archivo.
- **Geografía:** no hay coordenadas, solo estado. Entre los 41 estados con al menos 1.000 casos, la proporción
  de días de ausencia va del 23,5 % al 69,2 %. Puede reflejar composición industrial y prácticas de reporte, no
  riesgo geográfico.

## 3.7 Del EDA al modelo

| Hallazgo | Decisión |
|---|---|
| Fallecimiento es el 0,026 % | Conservar 4 clases; métricas macro y por clase; intervalos |
| Días de ausencia/restricción y fecha de muerte revelan el resultado | Excluirlos del modelo |
| Faltantes de SOC y horas no son MCAR | Categoría «Unknown» dentro del Pipeline |
| Códigos nominales de alta cardinalidad | One-hot + regularización L2 |
| Horas sin relación monótona con el resultado | Horas como categorías |
| Atípicos son subpoblaciones legítimas | No eliminar filas |
| Casos agrupados por establecimiento | Separar establecimientos; bootstrap por establecimiento |
| Patrón semanal y feriados | Partición hacia adelante con purga de 7 días |

## 3.8 Correcciones solicitadas en el entregable 1

El entregable 1 obtuvo 4,5/5,0 (base de datos 4,6; EDA 4,6; modelo base 4,2). Los comentarios del profesor y
dónde se atiende cada uno en este informe:

| Comentario del profesor | Qué se hizo | Dónde |
|---|---|---|
| La brecha entre entrenamiento y validación (0,65 frente a 0,39) no se analiza a fondo | Se descompone la brecha por clase. En la logística del entregable 1, Fallecimiento tiene recall 1,0 en entrenamiento y 0 en validación: esa clase sola explica unos 0,25 de los ~0,28 de brecha (el modelo memoriza los pocos fallecimientos con 1.433 columnas). Se repite el análisis con los modelos de este entregable | Sección 12.4 |
| La clase Death queda sin tratamiento | Se trata con SMOTE, ADASYN y `class_weight` en los 7 modelos y con los 4 optimizadores; se reporta su recall y su precisión con intervalos de Wilson y el balanced accuracy con y sin esa clase | Secciones 9.5 y 9.6 |
| El diseño descarta cerca de la mitad de los casos sin discutir su costo | Se cuantifica el descarte (55,4 %), se explica por qué cada bloque queda fuera y se discute el costo; el bloque de octubre a diciembre de los establecimientos de desarrollo se usa como segunda prueba (solo separa por tiempo) para medir cuánto cuesta exigir establecimientos nuevos | Secciones 4.7 y 9.8 |
| El código de entrenamiento no es visible en el notebook | Los capítulos nuevos son notebooks con el código que produce cada resultado, y una sección imprime el código fuente real de entrenamiento y evaluación | Sección 6.9 y capítulos 6 a 13 |
| `time_unknown` supera el umbral de colinealidad que el propio trabajo fija | Se recalcula el GVIF (sigue siendo 3,81: es redundante con la hora «Unknown») y se reajustan los 7 mejores modelos sin ella para medir su efecto | Secciones 4.6 y 12.5 |

Mejoras adicionales al EDA por hallazgos del desarrollo del proyecto (capítulo 4): agrupación jerárquica de
NAICS y SOC, horas desde el inicio de la jornada, el objetivo de regresión (días de ausencia) y la censura
de ese objetivo al final del año.
