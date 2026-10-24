# 1. Contexto y datos

## El problema

Cada caso del archivo ITA de OSHA es una lesión o enfermedad laboral que un establecimiento registró. Cada caso
termina en uno de cuatro resultados:

1. Fallecimiento (Death)
2. Días de ausencia
3. Traslado o restricción laboral
4. Otro caso registrable

La tarea es **clasificar** el resultado de un caso a partir de datos que se conocen del establecimiento y del
incidente: 8 predictores en los entregables 1 y 2, y 9 en el entregable 3 (ver abajo). Es una clasificación retrospectiva sobre casos ya reportados. No predice quién se lesionará y no
atribuye causas.

**Tarea de regresión (nueva en el entregable 3).** La guía del tercer entregable exige también modelos de
regresión. El objetivo es el **número de días de ausencia** de los casos que terminaron en «días de ausencia»,
en escala `log(1 + días)`, con las mismas variables. En clasificación esa columna se excluye porque revela
el resultado; en regresión es el objetivo y solo se usa dentro de esa clase (capítulo 4).

## Los datos

- **Fuente:** [OSHA ITA 2023, detalle de casos](https://www.osha.gov/itadata). Datos abiertos del Departamento
  de Trabajo de EE. UU.; no tienen derechos de autor en EE. UU. El CSV original (655 MB) no se redistribuye.
- **Tamaño:** 890.934 casos y 91.334 establecimientos, 49 columnas.
- **Unidad de observación:** un caso reportado, agrupado por establecimiento.
- **Objetivo:** `incident_outcome`, categórica nominal (códigos 1 a 4, sin orden de gravedad).

En el entregable 3, NAICS y SOC entran agrupados (subsector de 3 dígitos y grupo menor SOC) y se añade
una variable de horas desde el inicio de la jornada (capítulo 4).

| Predictor | Tipo | Qué es |
|---|---|---|
| `state` | categórica (55) | Estado o territorio |
| `naics_code` | categórica (715) | Sector económico del establecimiento |
| `establishment_type` | categórica | Privado, gobierno estatal o local |
| `soc_code` | categórica (598) | Ocupación del trabajador |
| `type_of_incident` | categórica (6) | Lesión, enfermedad respiratoria, pérdida auditiva, etc. |
| `time_unknown` | binaria | La hora del incidente se reportó como desconocida |
| `time_started_work_hour` | hora 0–23 | Hora de inicio de la jornada |
| `time_of_incident_hour` | hora 0–23 | Hora del incidente |

**Variables excluidas por fuga de información:** los días de ausencia, los días de restricción y la fecha de
fallecimiento revelan el resultado. Se usan solo para revisar la coherencia de los datos. Tampoco se usan los
identificadores, nombres, direcciones ni los textos libres.

## La partición

Antes de mirar cualquier resultado se separaron los datos de dos formas a la vez:

- **Por establecimiento:** se sorteó (semilla 42) el 60 % de los establecimientos para la prueba. Ningún
  establecimiento aparece en desarrollo y en prueba.
- **Por tiempo:** el desarrollo usa enero a septiembre (hasta el 23); la prueba usa octubre a diciembre. Entre
  los dos hay una purga de 7 días.

| Resultado | Desarrollo | Prueba |
|---|---|---|
| Fallecimiento | 69 | 39 |
| Días de ausencia | 99.697 | 46.169 |
| Traslado / restricción | 79.848 | 37.231 |
| Otro caso registrable | 89.902 | 43.952 |
| **Total** | **269.516** (34.029 establecimientos) | **127.391** (34.182 establecimientos) |

**Por qué así.** Si se mezclaran casos al azar, el modelo vería en el entrenamiento casos del mismo
establecimiento y de los mismos días que luego evalúa, y parecería mejor de lo que es. Separar por empresa
y por tiempo imita el uso real: predecir en lugares y fechas que el modelo no conoce.

## Límites y ética

- El código de ocupación (SOC) lo asigna OSHA después del reporte. No hay prueba de que existiera al momento
  del incidente.
- Son casos reportados, no tasas: los conteos dependen del tamaño del establecimiento y de sus prácticas de
  reporte. El archivo no es una muestra representativa de todos los trabajadores.
- Las combinaciones de fecha, ocupación y lugar pueden permitir reidentificar casos raros. No se publican
  narrativas ni ejemplos individuales.
- El 52 % de los casos de prueba pertenece a empresas (EIN) que tienen otros establecimientos en el
  desarrollo. Ningún establecimiento se repite, pero las prácticas de registro de una empresa pueden hacer la
  prueba algo optimista.
