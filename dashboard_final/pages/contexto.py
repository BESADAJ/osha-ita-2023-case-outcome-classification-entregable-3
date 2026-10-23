"""Sección 1: Contexto del problema, datos, metodología, limitaciones y autores."""
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from src import data

dash.register_page(__name__, path='/', name='Contexto', title='Contexto · OSHA ITA 2023')

d = data.eda()
counts = d.outcome.value_counts().reindex(data.CLASSES)


def kpi(value, label, icon):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div([html.I(className=f'bi {icon} me-2 text-primary', **{'aria-hidden': 'true'}),
                  html.Span(value, className='h3 mb-0')]),
        html.Div(label, className='text-muted small')]), className='h-100 shadow-sm'), xs=12, sm=6, lg=3, className='mb-3')


steps = [('ETL', 'CSV de OSHA (890 mil casos) → 9 variables + códigos; partición por establecimiento y fecha'),
         ('EDA', 'Desbalance (fallecimiento 0,03 %), asociaciones, faltantes'),
         ('140 modelos', '7 modelos × 4 balanceos × 4 optimizadores (y 28 de regresión)'),
         ('Validación anidada', '5 pliegues externos × 3 internos por establecimiento; presupuesto 16'),
         ('Comparación estadística', 'Friedman–Nemenyi, Wilcoxon–Holm, Nadeau–Bengio, bayesiana'),
         ('Modelo original', 'Ensamble multivista + SECR (Frank-Wolfe por región)')]

pipeline = dbc.Row([dbc.Col(dbc.Card(dbc.CardBody([html.Div(f'{i + 1}. {t}', className='fw-bold text-primary'),
                                                   html.Small(s)]), className='h-100'), xs=12, sm=6, lg=2, className='mb-2')
                    for i, (t, s) in enumerate(steps)], className='g-2')

dictionary = dbc.Table.from_dataframe(
    __import__('pandas').DataFrame([
        ('state', 'Estado de EE. UU. del establecimiento', 'categórica', 'curso y original'),
        ('naics3 / naics6', 'Actividad económica (NAICS 3 y 6 dígitos)', 'categórica', 'curso (3) / original (6)'),
        ('establishment_type', 'Tipo de establecimiento (privado, gobierno estatal/local)', 'categórica', 'ambos'),
        ('soc_minor / soc6', 'Ocupación del trabajador (SOC)', 'categórica', 'curso / original'),
        ('type_of_incident', 'Lesión, enfermedad de la piel, respiratoria, envenenamiento, auditiva, otra', 'categórica', 'ambos'),
        ('start_hour, incident_hour, hours_since_start, time_unknown', 'Horario del incidente', 'categórica', 'ambos'),
        ('nature, part, event, source, sec_source', 'Códigos OIICS de la lesión (asignados por OSHA al relato)', 'categórica', 'original'),
        ('ein, company', 'Empleador (codificación por objetivo fuera de pliegue)', 'categórica', 'original'),
        ('incident_outcome (objetivo)', 'Fallecimiento / Días de ausencia / Traslado o restricción / Otro registrable',
         '4 clases', 'clasificación'),
    ], columns=['Variable', 'Descripción', 'Tipo', 'Modelos que la usan']), striped=True, bordered=False, hover=True,
    size='sm', responsive=True)

layout = html.Div([
    dbc.Row(dbc.Col(html.Div([
        html.H1('¿Cómo termina un caso de lesión laboral?', className='display-6 fw-bold'),
        html.P(['Con los registros que los empleadores de EE. UU. reportan a OSHA (formulario 300/301, ',
                html.Em('Injury Tracking Application'), ' 2023), predecimos el desenlace de cada caso: ',
                html.B('fallecimiento, días de ausencia, traslado/restricción u otro caso registrable'), '.'],
               className='lead'),
        dbc.Alert([html.I(className='bi bi-info-circle me-2', **{'aria-hidden': 'true'}),
                   'En 2023 se registraron 2,6 millones de lesiones y enfermedades no fatales en el sector privado de '
                   'EE. UU.; 946 500 causaron días de ausencia (',
                   html.A('Bureau of Labor Statistics, Employer-Reported Workplace Injuries and Illnesses – 2023',
                          href='https://www.bls.gov/news.release/archives/osh_11082024.htm', target='_blank'),
                   ', 8 de noviembre de 2024). ',
                   'Saber temprano qué casos terminarán en ausencia o restricción ayuda a priorizar la atención y el '
                   'retorno al trabajo.'], color='light', className='border'),
    ], className='p-4 bg-white rounded shadow-sm'))),
    dbc.Row([kpi(f'{len(d):,}'.replace(',', '.'), 'casos de desarrollo usados para entrenar (ene–sep 2023)', 'bi-people'),
              kpi(f"{counts['Días de ausencia'] / len(d) * 100:.1f} %".replace('.', ','), 'terminan con días de ausencia', 'bi-calendar-x'),
              kpi(f"{int(counts['Fallecimiento'])}", 'fallecimientos en la submuestra (clase muy rara)', 'bi-exclamation-octagon'),
              kpi('8 modelos', '7 benchmarks del curso + 1 modelo original', 'bi-diagram-3')], className='mt-3'),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader('Por qué importa'), dbc.CardBody([
            html.P('Los casos con días de ausencia y con restricción cuestan más a trabajadores y empresas, y su '
                   'identificación temprana orienta la prevención y la gestión del retorno al trabajo. OSHA publica los '
                   'registros de forma abierta, pero no clasifica automáticamente el desenlace probable de un caso.'),
            html.P('El problema es de clasificación multiclase con desbalance extremo: los fallecimientos son 3 de cada '
                   '10 000 casos, así que la métrica principal es la exactitud balanceada (balanced accuracy).')])],
            className='h-100'), lg=6, className='mb-3'),
        dbc.Col(dbc.Card([dbc.CardHeader('Pregunta de investigación y objetivos'), dbc.CardBody([
            html.P(html.B('¿Con qué exactitud balanceada se puede predecir el desenlace de un caso registrable de OSHA a '
                          'partir del lugar de trabajo, la ocupación, el horario y la lesión, y supera un modelo original '
                          'a los modelos del curso?')),
            html.Ul([html.Li('Comparar 7 modelos × 4 balanceos × 4 optimizadores con validación anidada.'),
                     html.Li('Diseñar e implementar un modelo original (SECR) y validarlo con 3+ pruebas de hipótesis.'),
                     html.Li('Llevar los mejores modelos a un producto interactivo y reproducible.')])])],
            className='h-100'), lg=6, className='mb-3'),
    ], className='mt-2'),
    dbc.Card([dbc.CardHeader('Datos'), dbc.CardBody([
        html.P(['Fuente: ', html.A('OSHA, ITA Case Detail Data 2023 (CSV con códigos OIICS)',
                                   href='https://www.osha.gov/Establishment-Specific-Injury-and-Illness-Data', target='_blank'),
                '. 890 372 casos de 2023; partición del entregable 1: 60 % de los establecimientos reservados para la prueba '
                '(octubre–diciembre) y el resto, hasta el 23 de septiembre, para el desarrollo. Los modelos se entrenan con '
                'una submuestra de 30 030 casos por establecimientos completos. Tarea: clasificación de 4 clases.']),
        dictionary])], className='mb-3'),
    dbc.Card([dbc.CardHeader('Metodología'), dbc.CardBody(pipeline)], className='mb-3'),
    dbc.Card([dbc.CardHeader('Limitaciones, ética y autores'), dbc.CardBody([
        html.Ul([
            html.Li('Los datos los reportan los empleadores: hay subregistro y diferencias entre sectores y estados.'),
            html.Li('Los códigos OIICS se asignan a partir del relato del caso; están disponibles cuando el caso se '
                    'clasifica, pero el relato de los fallecimientos suele mencionar la muerte (riesgo de fuga indirecta, '
                    'discutido en el libro).'),
            html.Li('Los fallecimientos son tan pocos que su recall por pliegue es muy variable; las cifras se reportan '
                    'con media ± desviación estándar.'),
            html.Li('El modelo no debe usarse para decisiones sobre personas concretas (seguros, despidos): es una '
                    'herramienta de análisis agregado y de priorización.'),
            html.Li('El conjunto de prueba (octubre–diciembre) está bloqueado: todos los diagnósticos de esta app salen '
                    'de los pliegues externos de la validación anidada.')]),
        html.P(['Autores: Jaime Andrés Besada e Iván Berrio. Informe completo: ',
                html.A('Jupyter Book', href='https://besadaj.github.io/osha-ita-2023-case-outcome-classification-entregable-3/',
                       target='_blank'), '. Versión 3.0 (23 de octubre de 2026).'], className='mb-0')])], className='mb-3'),
])
