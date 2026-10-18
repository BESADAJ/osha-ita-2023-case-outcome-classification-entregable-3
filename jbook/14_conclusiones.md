# 14. Conclusiones y límites

## Qué se encontró

1. **El problema es difícil con estos datos.** Con 8 predictores estructurados, la regresión logística supera con
   claridad a la línea base en balanced accuracy (0,40 frente a 0,25), macro F1 score (0,40 frente a 0,13) y macro
   AUC (0,69 frente a 0,50), pero su desempeño es moderado: acierta el 52 % de los casos.
2. **La señal viene del lugar de trabajo y de la ocupación.** Sector (V de Cramér 0,24) y ocupación (0,21) son los
   predictores más asociados con el resultado, seguidos por tipo de incidente (0,15). Las horas casi no aportan.
3. **Fallecimiento no se puede detectar de forma fiable.** Hay 39 casos en la prueba y el modelo identifica 1.
   Con tan pocos casos no se puede sostener ninguna afirmación sobre esa clase.
4. **No hay indicios de fuga de información.** El accuracy está lejos del rango de alerta (80–90 %) y ningún
   predictor, solo, supera un AUC de 0,63.
5. **Más datos, con este modelo, parecen no ayudar.** La validación casi no mejora al triplicar el entrenamiento.

## Límites

- **No es una prueba intacta.** Los resultados del último trimestre y agregados de él se examinaron en análisis
  anteriores del proyecto, así que la prueba no es completamente virgen.
- **Retrospectivo.** El código de ocupación lo asigna OSHA después del reporte; el modelo no sirve como alerta
  temprana.
- **Dependencia entre empresas.** El 52 % de los casos de prueba pertenece a empresas con otros establecimientos
  en el desarrollo. No es fuga directa, pero puede hacer la prueba algo optimista.
- **Intervalos condicionados.** Los intervalos de confianza cubren la variación entre establecimientos de un
  trimestre, no la del entrenamiento ni la del calendario.
- **Casos reportados.** No hay horas trabajadas ni trabajadores como denominador: nada de esto son tasas de riesgo
  ni se debe usar para clasificar empleadores.
- **Asociaciones, no causas.** Los coeficientes comparan categorías dentro del modelo y pueden reflejar confusión
  y prácticas de reporte.

## Siguientes pasos

El entregable 1 ya los dejaba anotados: agrupar los códigos raros de NAICS y SOC por jerarquía, incluir las
horas transcurridas desde el inicio de la jornada, probar una partición que separe también por empresa (EIN) y
evaluar modelos no lineales.
