# 16. Revisión de literatura y novedad del modelo original

:::{admonition} Dos modelos en este capítulo
:class: note
Las secciones 16.1 a 16.5 documentan la revisión del **primer candidato** (PT-bagging, capítulo 17), que no superó a los
benchmarks y quedó como modelo explorado. Las secciones 16.6 a 16.9 documentan la revisión del **modelo original final**
(SECR, capítulos 19 y 20) y su declaración de novedad. Las cifras salen de las exportaciones de las búsquedas o de las
respuestas guardadas de cada consulta; no se completó nada de memoria.
:::

:::{admonition} Secciones 16.1 a 16.5: histórico
:class: warning
Estas secciones documentan la revisión del **primer candidato** (PT-bagging). Se conservan como registro, pero ya no
describen el modelo original final. **Las secciones vigentes son 16.6 a 16.9** (SECR).
:::

El primer candidato a modelo original fue **PT-bagging** (*probability thresholding bagging*). Fue propuesto por Collell, Prelec y Patil
(*Neurocomputing*, 2018; versión previa en arXiv, 2016). El método combina tres pasos: bagging de árboles sin podar
entrenados con la distribución natural de las clases, promedio de sus probabilidades y decisión
`argmax P(k|x)/λ_k`, donde `λ_k` es la proporción de la clase *k*. El modelo, sus fórmulas y sus resultados están en
el capítulo 17. Este capítulo documenta cómo se buscó si el método ya se había aplicado al problema: clasificar el
desenlace de un caso registrable de OSHA (Fallecimiento, Días de ausencia, Traslado o restricción, Otro).

La novedad que se reclama es de **aplicación**, no de algoritmo: PT-bagging ya existe. Lo que se buscó es si alguien
lo había usado para clasificar la gravedad de lesiones laborales.

## 16.1 Búsquedas realizadas

| # | Fecha | Base | Cadena | Resultados | Revisados | Relevantes | Observación |
|---|---|---|---|---|---|---|---|
| 1 | 2026-10-20 | Web of Science | `("occupational injury" OR "workplace injury" OR "work-related injury" OR OSHA) AND (severity OR outcome OR hospitali* OR fatal*) AND ("machine learning" OR classification OR prediction)`, 2005–2026 | 140 | 140 (búsqueda de texto en título y resumen exportados) | pendiente de registrar | Aplicación. Ningún registro menciona *bagging*, *threshold-moving*, prior de clase, Random Balance, metaclasificación ni modelos ordinales. Uno usa pérdida focal (aprendizaje profundo sobre narrativas de construcción, 2024). 34 mencionan OSHA y ninguno el *Injury Tracking Application* (ITA). |
| 2 | 2026-10-20 | Web of Science | **pendiente: la cadena no quedó guardada** | 374 | 374 (búsqueda de texto) | pendiente de registrar | Modelos para datos desbalanceados. De aquí salen PT-bagging (doi 10.1016/j.neucom.2017.08.035) y MultiRandBal (doi 10.1016/j.knosys.2019.105434). Solo 1 registro menciona OSHA. |
| 3 | 2026-10-21 | Scopus | `TITLE-ABS-KEY(("occupational injury" OR OSHA) AND ("voting" OR "soft voting" OR "stacking") AND (severity OR outcome))` | 2 | 2 | 1 | Mermer, Zhang y Demir (2026) usan stacking y voting en accidentes agrícolas. El otro resultado no es de aprendizaje automático. La cadena revisa una familia descartada (ensambles por votación), no PT-bagging. |
| 4 | 2026-10-21 | Consensus (Semantic Scholar, OpenAlex, PubMed) | `bagging threshold moving class prior imbalanced classification occupational injury severity` | 10 (máximo que muestra el plan) | 10 | 3 | Aparecen el artículo del método (Collell et al., 2018 y 2016). En lesiones, Sarkar et al. (2020, acería) y Luo et al. (2023, construcción) usan sobremuestreo y Random Forest; Jeong et al. (2018, tránsito) usan bagging con sobremuestreo. Ninguno usa umbral = prior. |
| 5 | 2026-10-21 | Consensus | `clasificación de la gravedad de lesiones laborales con aprendizaje automático y clases desbalanceadas` (en español) | 10 | 10 | 7 | Todos los resultados están en inglés. Sarkar 2020, Obasi et al. 2025 (OSHA, industria extractiva, SMOTETomek), Luo 2023, Davoudi Kakhki et al. 2019, Wang 2026 (OSHA, construcción, ordinal y pérdida focal), Recal et al. 2021 (balanced accuracy, multiclase) y Salarian et al. 2023 (agrupamiento y SMOTE). Ninguno usa PT-bagging. |
| 6 | 2026-10-21 | Consensus | `focal loss XGBoost occupational injury severity classification` | no registrado | título y resumen | 1 | Wang (2026) ya combina un objetivo focal propio con *gradient boosting* en datos de OSHA. Afecta la novedad de la iteración HAFL-XGBoost (capítulo 17, Ronda 2). |
| 7 | 2026-10-21 | Búsqueda web | `"focal loss" XGBoost OSHA OR "occupational injury" severity` | no registrado | primeros resultados | 1 | Confirma el antecedente de Wang (2026). |
| 8 | 2026-10-21 | Consensus | `OSHA Injury Tracking Application ITA case detail data machine learning predict` | no registrado | título y resumen | 0 | No se halló ningún estudio de modelado con los registros ITA. |
| 9 | 2026-10-21 | Consensus | búsqueda de Wang 2026 con «ordinal» | no registrado | título y resumen | 2 | La clasificación ordinal ya se usa en gravedad de accidentes: tránsito (Zhu, 2021) y construcción en Turquía (logística ordinal). Afecta la novedad de la iteración Frank–Hall (Ronda 4). |

**Cómo leer la tabla.**
- «Revisados (búsqueda de texto)» quiere decir que se buscaron términos del método (*bagging*, *threshold-moving*,
  prior, Random Balance, focal, ordinal, stacking, OSHA, ITA) en el título y el resumen de cada registro exportado.
  No se leyeron los 514 resúmenes uno por uno.
- El texto completo se leyó solo en los métodos candidatos: PT-bagging, MultiRandBal, metaclasificación jerárquica y
  HAFL-XGBoost.
- «Relevantes» cuenta los estudios de gravedad de lesiones o accidentes con aprendizaje automático cuyo método se
  parece a alguno de los candidatos.

## 16.2 Matriz de novedad: trabajos más cercanos

| Trabajo | Datos | Desenlace | Método frente al desbalance | ¿Usa PT-bagging? | Diferencia con la propuesta |
|---|---|---|---|---|---|
| Wang (2026), *Scientific Reports* | 22.217 accidentes de construcción de OSHA (2015–2023) | Gravedad ordinal | Pesos por frecuencia inversa y un objetivo focal propio en tres *gradient boosting*; optimización bayesiana | No | Usa informes de accidentes de construcción, no los registros ITA de todas las industrias. Corrige el desbalance en el entrenamiento (pesos y pérdida). PT-bagging no repondera: corrige solo al decidir. |
| Mermer, Zhang y Demir (2026), *Journal of Agromedicine* | 2.421 incidentes agrícolas (AgInjuryNews) | Fatal / no fatal | Ensambles (bagging, boosting, stacking, voting) y SHAP | No | Es binario y con otra fuente de datos. Su bagging usa la regla de mayoría, sin umbral = prior. |
| Obasi et al. (2025), *Results in Engineering* | 2.645 investigaciones de OSHA en industria extractiva | Gravedad | SMOTETomek, optimización bayesiana, RF, GBM y XGBoost | No | Rebalancea con datos sintéticos. Usa las narrativas como variables; aquí se excluyen por posible fuga (capítulo 17.8). |
| Sarkar et al. (2020), *Safety Science* | Una acería (informes de investigación e inspección) | Fatal / caso médico / primeros auxilios | Cuatro variantes de SMOTE más SVM, RF, CART y otros | No | Sobremuestreo antes de entrenar, en una sola empresa. |
| Recal et al. (2021), *J. Intelligent & Fuzzy Systems* | Accidentes laborales | Binario y 3 clases | Comparación de MLR, SVM, C5.0, SGB y NN con balanced accuracy | No | Usa la misma métrica (balanced accuracy), pero sin un umbral derivado de esa métrica. |
| Davoudi Kakhki et al. (2019), *Safety Science* | 33.000 reclamaciones de compensación en agroindustria | Clases de costo | SVM, *boosted trees* y Naive Bayes | No | Otro desenlace (costo) y otra fuente de datos. |
| Jeong et al. (2018), *Accident Analysis & Prevention* | 297.113 choques de tránsito (Michigan) | 5 niveles de gravedad | Bagging con sobre- o submuestreo | No | Es la familia más cercana (bagging), pero rebalancea las bolsas. El problema es de tránsito. |

## 16.3 Métodos afines por familia algorítmica

La guía pide comparar con 5 a 10 métodos afines **por familia**, no solo por aplicación. La columna «Limitación»
indica qué problema deja abierto cada familia y que PT-bagging intenta resolver: decidir con la regla que maximiza la
balanced accuracy **sin deformar la distribución de entrenamiento** y sin fijar un costo de antemano.

| Familia | Ejemplo | Qué hace con el desbalance | Qué conserva PT-bagging | Qué cambia PT-bagging | Limitación de la familia |
|---|---|---|---|---|---|
| Bagging con remuestreo | UnderBagging y SMOTEBagging; Jeong et al. (2018); iBRF (Mohammadpour et al., 2026) | Rebalancea cada bolsa | Bagging de árboles | Bolsas con la distribución natural y umbral al decidir | El remuestreo descalibra las probabilidades y hay que elegir la proporción antes de entrenar |
| Sensible al costo | `class_weight` en RF y XGBoost (capítulos 6 a 9); pesos inversos en Wang (2026) | Cambia la pérdida o la impureza | Árboles | Los árboles se construyen sin pesos | El costo se fija antes de entrenar y cambia la forma de los árboles |
| Umbral sobre un modelo único (*threshold-moving*) | RF o XGBoost + `P/prior` (ablación D, capítulo 17.6) | Mueve el umbral después de entrenar | La regla de decisión | Añade el bagging de árboles sin podar | Depende de la calibración de un solo modelo |
| Random Balance | MultiRandBal (Rodríguez et al., *KBS*, 2020); iteración 1 | Proporciones de clase al azar por árbol, con SMOTE | Bagging de árboles | Proporciones fijas (naturales) y umbral | Deforma la distribución de cada árbol |
| Metaclasificación jerárquica | Bhattacharya, Jurkovitz y Shatkay (*BMC Med. Inform. Decis. Mak.*, 2018); iteración 2 | Separa la clase rara en un primer nivel | — | Un solo nivel | El metaclasificador aprende el prior real y casi nunca marca la clase rara (capítulo 17.5) |
| Boosting con pérdida focal | HAFL-XGBoost (Jiang et al., *Neural Comput. & Applic.*, 2023); Wang (2026); iteración 3 | Repondera los casos difíciles durante el entrenamiento | Árboles | Sin pérdida propia | Hay que elegir γ; ya se aplicó con datos de OSHA (Wang, 2026) |
| Clasificación ordinal | Frank y Hall (ECML, 2001); Wang (2026); iteración 4 | Usa el orden de gravedad | — | Trata las clases como nominales | Ya se aplicó en gravedad de accidentes |
| Ensambles por votación o stacking | Mermer et al. (2026); RXK-VEM (descartado) | Combina modelos | Promedio de probabilidades | Umbral = prior | Ya se aplicó en lesiones agrícolas |
| Self-Paced Ensemble | Asadi et al. (2023, choques en zonas de obra) | Submuestreo guiado por la dificultad | — | — | Es binario; se descartó para 4 clases |
| Sobremuestreo + modelo clásico | Sarkar et al. (2020), Obasi et al. (2025), Salarian et al. (2023) | SMOTE y variantes | — | Sin datos sintéticos | Datos sintéticos en variables categóricas one-hot |

**Qué no resuelve PT-bagging (adelanto del capítulo 17).** La regla `P/λ` es óptima si las probabilidades están bien
estimadas. Con 16 a 37 muertes de entrenamiento por pliegue, eso no se cumple. Los árboles sin podar dan
`P(Fallecimiento) = 0` exacto en más del 99 % de los casos, y los árboles podados marcan como Fallecimiento a uno de
cada tres o cuatro casos. Por eso el método no supera a los modelos del curso (R3 = no).

## 16.4 Declaración de novedad

> Al mejor de nuestro conocimiento y según la búsqueda descrita (Web of Science, Scopus y Consensus, 2026-10-20 y
> 2026-10-21), no se encontraron estudios que apliquen PT-bagging (bagging de árboles con la distribución natural de
> las clases y decisión `argmax P(k|x)/λ_k`, Collell, Prelec y Patil, 2018) a la clasificación del desenlace de casos
> registrables de lesiones y enfermedades laborales. Tampoco se encontró ningún estudio de modelado con los registros
> ITA de OSHA.
>
> Los trabajos más cercanos se diferencian así:
> - **Wang (2026):** datos de OSHA de construcción, gravedad ordinal, pesos inversos y pérdida focal en el
>   entrenamiento.
> - **Mermer et al. (2026):** bagging y voting por mayoría, desenlace binario, accidentes agrícolas.
> - **Obasi et al. (2025):** OSHA en industria extractiva, SMOTETomek y variables tomadas de las narrativas.
> - **Jeong et al. (2018):** bagging con remuestreo en choques de tránsito.
>
> Ninguno conserva la distribución natural de entrenamiento y corrige el desbalance solo al decidir, con la regla que
> maximiza la balanced accuracy.

**Alcance.** La novedad es de aplicación. Además, la declaración es **provisional** mientras falten los puntos de
16.5. Las iteraciones del diseño no reclaman novedad propia: la pérdida focal en boosting y la clasificación ordinal
ya se aplicaron con datos de OSHA (Wang, 2026).

## 16.5 Pendientes para cumplir la sección 5.2

- [ ] **Tercera base formal.** Consensus agrega Semantic Scholar, OpenAlex y PubMed, pero no es una de las bases que
  nombra la guía. Falta Google Scholar o arXiv con fecha y número de resultados. Cadenas propuestas, sin ejecutar:
  - `"threshold-moving" bagging "injury severity"`
  - `"PT-bagging" OR "probability thresholding bagging" occupational`
- [ ] **Cadena de la búsqueda 2 de Web of Science.** Hay que recuperarla del historial de WoS. Si no se puede, hay que
  repetir la búsqueda y anotar la cadena.
- [ ] **Scopus con la familia de PT-bagging.** Cadena propuesta, sin ejecutar:
  `TITLE-ABS-KEY((bagging AND ("threshold moving" OR "threshold-moving" OR "class prior")) AND ("occupational injury" OR "workplace injury" OR OSHA OR "accident severity"))`
- [ ] **Una cadena en español en una base formal**, por ejemplo Google Scholar:
  `"gravedad de lesiones laborales" "aprendizaje automático"`.
- [ ] **Contar los relevantes** de las búsquedas 1 y 2 de WoS, revisando título y resumen de los registros que
  coincidan.
- [ ] **Pasar las referencias a `references.bib`** y citarlas con `{cite}`.


---

## 16.6 Revisión de literatura del modelo original final (SECR) — vigente

PT-bagging (secciones 16.1 a 16.5) fue el primer candidato a modelo original y **no superó** a los modelos del curso
(capítulo 17). El modelo original final es el **ensamble multivista con selección condicionada por región (SECR)**, del
capítulo 19. Esta sección documenta su revisión de literatura según la guía §5.2: al menos tres bases, al menos cinco
cadenas que combinen la familia del modelo, el problema y sinónimos en inglés y en español, la fecha de cada consulta,
una matriz de novedad y una declaración de novedad.

**Bases y procedimiento.** Las consultas son reproducibles: `scripts/literatura/buscar.py` las lanza contra las API
públicas de **OpenAlex** (índice bibliográfico abierto que reemplazó a Microsoft Academic), **Europe PMC** (incluye
PubMed/MEDLINE) y **arXiv**. Guarda la URL exacta, el total de resultados y los 25 primeros títulos por relevancia en
`results/literatura/`. Las tres admiten operadores booleanos. Semantic Scholar respondió «429 Too Many Requests» sin
clave y Crossref no admite operadores booleanos, por eso no se usaron. A estas tres se suman las búsquedas en Web of
Science, Scopus y Consensus de la sección 16.1, que cubren la aplicación y las familias de ensambles (stacking y
votación). Fecha de las consultas: **2026-10-23** (hora de Colombia).

**Revisión.** De cada consulta se leyeron los títulos de los 25 primeros resultados por relevancia. Se abrió el resumen
de los que trataban ensambles o combinación de modelos para gravedad de lesiones o accidentes, o la combinación
ponderada local de clasificadores. «Relevantes» cuenta esos trabajos.

| # | Cadena (booleana) | OpenAlex | Europe PMC | arXiv | Revisados | Relevantes | Observación |
|---|---|---|---|---|---|---|---|
| 1 | `"ensemble selection" AND (injury OR injuries)` | 371 | 74 | 0 | 25 + 25 | 6 | Selección **dinámica** de ensambles para gravedad de choques de tránsito (Aziz et al. 2025 y 2026) y para recuperación postraumática; un método de pesos con competencia local. Ninguno es laboral ni usa pesos convexos por región |
| 2 | `"ensemble selection" AND (occupational OR workplace OR "work-related")` | 161 | 10 | 0 | 25 + 10 | 2 | Selección dinámica de clasificadores con clases desbalanceadas; ensamble adaptativo con regularización. Ninguno es de lesiones laborales |
| 3 | `(ensemble OR stacking) AND "occupational injury" AND (severity OR outcome)` | 1 532 | 43 | 0 | 25 + 25 | 8 | Aplicación: Khairuddin et al. (2022), J. Wang et al. (2025, informes de OSHA con ensambles y LLM), gravedad desde informes con aprendizaje profundo, días de ausencia en minería. Todos usan stacking, votación o modelos únicos |
| 4 | `"Frank-Wolfe" AND (ensemble OR "classifier combination" OR "model averaging")` | 599 | 16 | 2 | 25 + 16 | 5 | Familia del algoritmo: combinación de clasificadores de una clase con restricción ℓp, fusión local con ℓp dinámica, *boosting* con Frank-Wolfe. Ninguna aplicación a lesiones |
| 5 | `("dynamic ensemble selection" OR "local weighting" OR "region-specific weights") AND (injury OR accident)` | 364 | 59 | 1 | 25 + 25 | 7 | Selección dinámica para gravedad de tránsito (Aziz 2025, 2026; una versión con optimización bayesiana guiada por la dificultad de las instancias); DES con kernels |
| 6 | `"mixture of experts" AND ("occupational injury" OR "workplace injury" OR "injury severity")` | 41 | 6 | 0 | 25 + 6 | 0 | Mezclas de expertos neuronales para trayectorias de vehículos y visión; ninguna para lesiones laborales |
| 7 | `("selección de ensambles" OR "ensamble de modelos" OR "combinación de clasificadores") AND ("lesiones laborales" OR "accidentes de trabajo" OR "accidente laboral")` (español) | 0 | 0 | 0 | — | 0 | Sin resultados en español |
| 8 | `OSHA AND ("machine learning" OR classification) AND (severity OR outcome OR "days away")` | 9 611 | 1 025 | 1 | 25 + 25 | 5 | Ensambles interpretables para gravedad en construcción, agroindustria, emparejamiento de registros de lesiones y una revisión sistemática de aprendizaje automático en seguridad y salud en el trabajo. Ninguno usa los registros ITA *Case Detail* |

**Nota sobre arXiv.** La API de arXiv respondió «Rate exceeded» en la primera pasada. `scripts/literatura/buscar_arxiv.py`
repite las 8 cadenas con 30 segundos entre consultas y guarda los totales en `results/literatura/arXiv_<n>.json`. La
tabla 16.6b los resume.

| # | arXiv: resultados | Títulos | Relevantes |
|---|---|---|---|
| 1 | 0 | — | 0 |
| 2 | 0 | — | 0 |
| 3 | 0 | — | 0 |
| 4 | 2 | *Lp-Norm Constrained One-Class Classifier Combination*; *Locally Adaptive One-Class Classifier Fusion with Dynamic ℓp-Norm Constraints…* | 2 (familia: combinación ponderada, detección de anomalías; ninguna con lesiones) |
| 5 | 1 | *Distribution-free inference for LightGBM and GLM with Tweedie loss* | 0 |
| 6 | 0 | — | 0 |
| 7 | 0 | — | 0 |
| 8 | 1 | *Abex-rat: … Classification of Occupational Accident Reports* | 1 (aplicación: clasifica informes de accidentes laborales con aumento de datos y entrenamiento adversario; no combina modelos con pesos) |

Tabla 16.6b. Resultados de arXiv (consultas del 2026-10-23, segunda pasada con 30 s entre consultas).

## 16.7 Matriz de novedad: trabajos más cercanos a SECR

| Trabajo | Problema | Mecanismo de combinación | ¿Pesos convexos por región? | ¿Contracción hacia lo global? | ¿Garantía de optimización? |
|---|---|---|---|---|---|
| {cite:t}`caruana2004ensemble` | General | Selección codiciosa con reemplazo, pesos globales | No | — | No (paso fijo) |
| {cite:t}`hassan2025workplace` | Gravedad de incidentes laborales (OSHA) | Stacking con meta-modelo | No | No | No aplica |
| {cite:t}`wang2025classifying` | Tipo de accidente en informes de OSHA (construcción) | Votación suave y stacking | No | No | No aplica |
| {cite:t}`mermer2026explainable` | Accidentes agrícolas fatales | Bagging, boosting, stacking y votación | No | No | No aplica |
| {cite:t}`aziz2025interpretable` | Gravedad de choques de tránsito | Selección **dinámica** por vecindario (DES-MI, KNORA) | Pesos locales por k-NN al predecir | No | No |
| {cite:t}`garcia2018dynamic` | Multiclase desbalanceado (general) | DES-MI: competencia local ponderada por el desbalance | Local por vecinos | No | No |
| {cite:t}`wolpert1992stacked` | General | Stacking: meta-modelo sobre predicciones fuera de pliegue | No | No | Depende del meta-modelo |
| Combinación con restricción ℓp (consulta 4) | Detección de anomalías | Pesos con norma ℓp, a veces locales | Parcial (fusión local) | No | Sí (optimización convexa) |
| **SECR (este trabajo)** | **Desenlace de casos ITA de OSHA (4 clases)** | **Combinación convexa por región con Frank-Wolfe exacto** | **Sí** | **Sí (λ pseudo-observaciones)** | **Sí: convexidad, monotonía, brecha y O(1/t)** |

## 16.8 Diferenciación por familia algorítmica

| Familia | Qué conserva SECR | Qué cambia SECR | Por qué la familia no resuelve la limitación abordada |
|---|---|---|---|
| Promedio simple y votación suave | Mezcla de probabilidades | Pesos aprendidos y por región | Da el mismo peso a vistas buenas y malas en todas partes |
| Pesos por error global ($w_m\propto1/\text{log-loss}_m$) | Pesos ≥ 0 que suman 1 | Minimiza la pérdida **de la mezcla**, no la de cada modelo, y por región | No tiene en cuenta la redundancia entre modelos ni la heterogeneidad del espacio de entrada |
| Stacking lineal {cite}`wolpert1992stacked` | Aprende la combinación con predicciones fuera de pliegue | Pesos convexos e interpretables por vista, con estructura por región | El meta-modelo multinomial no se lee como pesos de vistas y adaptarlo por región multiplica los parámetros |
| Selección de Caruana {cite}`caruana2004ensemble` | Biblioteca de modelos y log-loss fuera de pliegue | Búsqueda lineal exacta, certificado de optimalidad y pesos por región | Pesos globales y paso fijo que puede empeorar la pérdida (proposición 6 del capítulo 19) |
| Selección dinámica (KNORA, DES-MI) {cite}`garcia2018dynamic` | Pesos que dependen de la entrada | Regiones interpretables y fijas en lugar de vecinos; pérdida convexa; contracción | Cada predicción busca k vecinos (inferencia costosa) y en regiones escasas no hay contracción |
| Mezcla de expertos {cite}`jacobs1991adaptive` | La compuerta decide qué experto pesa más | Expertos fijos ya entrenados; compuerta como partición con contracción; sin entrenamiento conjunto | El entrenamiento conjunto es no convexo y suele ser neuronal (excluido por la guía) |
| Aprendizaje multivista por partición de variables {cite}`kumar2016multiview` | Un modelo por vista | La combinación de vistas cambia por región | Combina las vistas con votación o pesos globales |
| Contracción empírica-bayesiana {cite}`efron1973stein` | Mezcla local–global ponderada por $n_g$ | Se aplica a los **pesos de combinación** de un ensamble | En la literatura revisada se aplica a medias o coeficientes, no a pesos de un ensamble de clasificadores |

## 16.9 Declaración de novedad del modelo original (SECR)

> Al mejor de nuestro conocimiento y según la búsqueda descrita (OpenAlex, Europe PMC y arXiv, 8 cadenas en inglés y
> español, 2026-10-23; más Web of Science, Scopus y Consensus, 2026-10-20 a 2026-10-23), **no se encontraron estudios
> que apliquen una combinación convexa de modelos base con pesos condicionados por región y contraídos hacia los pesos
> globales, resuelta con Frank-Wolfe (SECR), a la clasificación del desenlace de casos registrables de lesiones y
> enfermedades laborales.** Tampoco se encontró la regla SECR aplicada a otros problemas en los resultados revisados.
>
> Los trabajos más cercanos se diferencian así:
> - **Hassan y Khalifa (2025)** y **J. Wang et al. (2025):** stacking o votación con datos de OSHA, con un meta-modelo o
>   pesos globales.
> - **Aziz et al. (2025)** y **García et al. (2018):** pesos locales por vecindario al predecir, sin pérdida convexa, sin
>   contracción y en otro fenómeno (tránsito).
> - **Caruana et al. (2004):** pesos globales con paso fijo; SECR lo generaliza (proposición 6 del capítulo 19).

**Alcance de la novedad.** La novedad es **de mecanismo y de aplicación**. Es de mecanismo porque la regla de combinación
(ec. 19.2), su algoritmo y sus propiedades demostradas son nuevos frente a los métodos afines de la tabla 16.8. Es de
aplicación porque no se halló ningún uso de esa regla, ni de la selección de ensambles en general, para el desenlace de
casos ITA. Frank-Wolfe, la selección de Caruana y la contracción empírica-bayesiana existen por separado y se citan como
fuentes. La contribución es su combinación en una regla de ponderación por región con garantías, que es lo que pide el
Cuadro 1 de la guía («pesos según la región del espacio de entrada… con regla de actualización derivada y contrastada
contra reglas estándar»). La búsqueda tiene límites: se revisaron títulos y resúmenes, no textos completos, y solo los
25 primeros resultados de cada consulta.
