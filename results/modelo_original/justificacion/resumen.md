# Fuga de información y efecto del empleador (datos de desarrollo: 269.516 casos, 69 muertes)

## 1. Fuga directa (fracción de casos de cada clase con la variable presente)

|                      |   dafw>0 |   djtr>0 |   date_of_death |
|:---------------------|---------:|---------:|----------------:|
| Fallecimiento        |    0.145 |    0     |               1 |
| Días de ausencia     |    0.994 |    0.293 |               0 |
| Traslado/restricción |    0.004 |    0.995 |               0 |
| Otro                 |    0.002 |    0.003 |               0 |

## 2. Narrativas que nombran el desenlace (fracción de casos de cada clase)

|                      |   muerte |   dias_de_ausencia |   restriccion_o_traslado |   regreso_al_trabajo |   cualquiera |
|:---------------------|---------:|-------------------:|-------------------------:|---------------------:|-------------:|
| Fallecimiento        |    0.536 |              0     |                    0     |                0     |        0.536 |
| Días de ausencia     |    0.001 |              0.004 |                    0.003 |                0.002 |        0.008 |
| Traslado/restricción |    0     |              0.001 |                    0.004 |                0.001 |        0.007 |
| Otro                 |    0.001 |              0.001 |                    0.002 |                0.002 |        0.005 |

Los ejemplos de fragmentos de narrativa no se versionan (texto libre de los casos).

## 3. Información mutua con el desenlace (nats; mayor = más informativa)

|                    |   informacion_mutua_nats |
|:-------------------|-------------------------:|
| nature_title_pred  |                   0.1323 |
| source_title_pred  |                   0.1238 |
| event_title_pred   |                   0.1134 |
| naics_code         |                   0.0869 |
| naics_3            |                   0.0687 |
| soc_code           |                   0.0642 |
| part_title_pred    |                   0.0556 |
| type_of_incident   |                   0.035  |
| time_started_work  |                   0.0241 |
| state              |                   0.0231 |
| time_of_incident   |                   0.0134 |
| establishment_type |                   0.0048 |
| time_unknown       |                   0.0031 |

Nota: la información mutua de variables con muchas categorías (naics_code, soc_code) está inflada por su cardinalidad; se añade naics_3 como referencia.

## 4. Efecto del empleador: traslado/restricción (3) frente a días de ausencia (2)

Casos de clase 2 o 3: 179545; establecimientos con >= 20 de esos casos: 1416 (72094 casos).
Para cada establecimiento se compara cuántos casos de traslado tuvo con los esperados si se comportara como el promedio de su estrato (industria NAICS de 3 dígitos x naturaleza x parte del cuerpo). z = (observado - esperado) / desviación esperada; sin efecto del empleador, var(z) ~ 1.
- var(z) observada: 12.93; bajo la nula (200 permutaciones dentro de estrato): 0.98 (máx. 1.08).
- Establecimientos con |z| > 3: 41.0% (esperado sin efecto: ~0,3 %).
- Proporción de traslado por establecimiento: mediana 0.47, percentil 10 0.03, percentil 90 0.87.

Misma industria (NAICS 3 dígitos), distintos establecimientos (>= 20 casos 2/3 cada uno):

|   n3 |   establecimientos |   prop_traslado_min |   prop_traslado_max |
|-----:|-------------------:|--------------------:|--------------------:|
|  622 |                313 |                0    |                0.95 |
|  492 |                190 |                0    |                1    |
|  493 |                170 |                0    |                1    |
|  424 |                104 |                0    |                1    |
|  623 |                 98 |                0    |                0.75 |
|  311 |                 78 |                0.04 |                0.98 |
|  336 |                 64 |                0.05 |                1    |
|  721 |                 45 |                0    |                1    |

Proporción de traslado según tamaño del establecimiento (`size`):

|   size |   casos |   prop_traslado |
|-------:|--------:|----------------:|
|      1 |    1055 |           0.429 |
|      2 |    2306 |           0.318 |
|     21 |   18913 |           0.388 |
|     22 |   47804 |           0.422 |
|      3 |  109467 |           0.467 |

## 5. ¿Cuándo se carga el registro?

created_timestamp: mínimo 2024-01-02, mediana 2024-02-26, máximo 2024-12-31; días desde el incidente: mediana 283, percentil 10 176.

