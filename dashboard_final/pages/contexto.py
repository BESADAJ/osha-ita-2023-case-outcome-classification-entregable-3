"""Sección 1: Contexto del problema, datos, metodología, limitaciones y autores."""
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from src import data

dash.register_page(__name__, path='/', name='Contexto', title='Contexto · OSHA ITA 2023')

d = data.eda()
counts = d.outcome.value_counts().reindex(data.CLASSES)
RES = data.original_results()
SUMMARY = {r['Modelo']: r for r in RES['summary']} if RES else {}


def num(x, nd=1):
    return f'{x:.{nd}f}'.replace('.', ',')


def stat(value, label, accent=False, unit=None):
    return html.Div([html.Div([value, html.Small(unit) if unit else None], className='v'), html.Div(label, className='l')],
                    className='stat accent' if accent else 'stat')


def section(eyebrow, title, desc, body):
    return html.Section([html.H2(title, className='section-title'),
                         html.P(desc, className='section-desc') if desc else None, body], className='section')


STEPS = [('ETL', 'CSV de OSHA (890 372 casos) → 9 variables del lugar de trabajo + códigos; partición por establecimiento y fecha'),
         ('EDA', 'Desbalance extremo (fallecimiento 0,03 %), asociaciones con V de Cramér, faltantes y atípicos'),
         ('140 modelos', '7 modelos × 4 balanceos × 4 optimizadores para clasificación, y 28 de regresión'),
         ('Validación anidada', '5 pliegues externos × 3 internos por establecimiento; presupuesto de 16 configuraciones'),
         ('Comparación estadística', 'Friedman–Nemenyi, Wilcoxon–Holm, Nadeau–Bengio y prueba bayesiana con ROPE'),
         ('Modelo original', 'Ensamble multivista + SECR, comparado en 3 × 5 pliegues pareados con los 7 benchmarks')]

DICT = pd.DataFrame([
    ('state', 'Estado de EE. UU. del establecimiento', 'Categórica', 'Benchmarks y original'),
    ('naics3 / naics6', 'Actividad económica (NAICS de 3 y 6 dígitos)', 'Categórica', 'Benchmarks (3) · original (6)'),
    ('establishment_type', 'Tipo de establecimiento (privado, gobierno estatal o local)', 'Categórica', 'Ambos'),
    ('soc_minor / soc6', 'Ocupación del trabajador (SOC)', 'Categórica', 'Benchmarks (menor) · original (6)'),
    ('type_of_incident', 'Lesión, piel, respiratoria, envenenamiento, auditiva u otra enfermedad', 'Categórica', 'Ambos'),
    ('start_hour, incident_hour, hours_since_start, time_unknown', 'Horario del incidente', 'Categórica', 'Ambos'),
    ('nature, part, event, source, sec_source', 'Códigos OIICS de la lesión (asignados por OSHA al relato)', 'Categórica',
     'Original'),
    ('ein, company', 'Empleador (codificación por objetivo fuera de pliegue)', 'Categórica', 'Original'),
    ('incident_outcome', 'Objetivo: Fallecimiento · Días de ausencia · Traslado o restricción · Otro registrable',
     '4 clases', '—'),
], columns=['Variable', 'Descripción', 'Tipo', 'Modelos que la usan'])
DICT_NOTE = ('«Benchmarks» se refiere a la búsqueda de las 112 combinaciones, con 9 variables. En la comparación final '
             'con el original, los 7 benchmarks se reentrenan con las mismas 18 variables que usa el original.')


def dict_table():
    head = html.Thead(html.Tr([html.Th(c) for c in DICT.columns]))
    body = html.Tbody([html.Tr([html.Td(html.Span(r.Variable, className='mono')), html.Td(r.Descripción), html.Td(r.Tipo),
                                html.Td(r._4)]) for r in DICT.itertuples()])
    return html.Div([html.Div(html.Table([head, body], className='table-clean'), style={'overflowX': 'auto'}),
                     html.P(DICT_NOTE, className='note mt-2 mb-0')])


orig, knn = SUMMARY.get('Original (SECR)'), SUMMARY.get('KNN')
headline = None
if orig and knn:
    headline = html.Div([
        html.B('Resultado. '),
        f'El modelo original (SECR) tiene la mayor balanced accuracy media en la comparación pareada '
        f'({orig["BA (media ± d.e.)"].split(" ")[0].replace(".", ",")} frente a '
        f'{knn["BA (media ± d.e.)"].split(" ")[0].replace(".", ",")} de KNN, 15 pliegues), pero la ventaja '
        'no es estadísticamente significativa. ',
        dcc.Link('Ver la evaluación →', href='/modelos#original')], className='callout mt-4')

layout = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div('Contexto', className='eyebrow'),
                html.H1('¿Cómo termina un caso de lesión laboral?', className='page-title'),
                html.P(['Con los registros que los empleadores de EE. UU. reportan a OSHA (formularios 300/301, ',
                        html.Em('Injury Tracking Application'), ' 2023) clasificamos el desenlace de cada caso en cuatro '
                        'clases: fallecimiento, días de ausencia, traslado o restricción, u otro caso registrable.'],
                       className='page-lead'),
            ], lg=6, className='align-self-center'),
            dbc.Col(html.Img(src=dash.get_asset_url('hero.jpg'), className='hero-img',
                             alt='Soldador trabajando sobre una viga de acero en un taller, con chispas y careta protectora'),
                    lg=6, className='mt-4 mt-lg-0'),
        ], className='g-4'),
        html.Div([
            stat(f'{len(d):,}'.replace(',', '.'), 'casos de desarrollo con los que se entrena (enero–septiembre de 2023)'),
            stat(num(counts['Días de ausencia'] / len(d) * 100), 'de los casos terminan con días de ausencia', unit='%'),
            stat(f"{int(counts['Fallecimiento'])}", 'fallecimientos en la submuestra: la clase más rara (0,03 %)'),
            stat('7 + 1', 'modelos comparados: 7 benchmarks y 1 modelo original', accent=True),
        ], className='stats'),
        headline,
    ], className='page-head'),

    section('Problema', 'Por qué importa',
            None,
            dbc.Row([
                dbc.Col([
                    html.P(['En 2023 se registraron ', html.B('2,6 millones'), ' de lesiones y enfermedades no fatales en el '
                            'sector privado de EE. UU., y ', html.B('946 500'), ' causaron días de ausencia (',
                            html.A('Bureau of Labor Statistics, Employer-Reported Workplace Injuries and Illnesses – 2023',
                                   href='https://www.bls.gov/news.release/archives/osh_11082024.htm', target='_blank'),
                            ', 8 de noviembre de 2024).']),
                    html.P('Los casos con días de ausencia o con restricción cuestan más a trabajadores y empresas. Saber qué '
                           'casos terminarán así orienta la prevención y la gestión del retorno al trabajo. OSHA publica los '
                           'registros de forma abierta, pero no clasifica el desenlace probable de un caso.'),
                    html.P(['Es una clasificación multiclase con desbalance extremo: los fallecimientos son 3 de cada 10 000 '
                            'casos. Por eso la métrica principal es la ', html.B('balanced accuracy'),
                            ' (BA, el promedio del recall de las cuatro clases).'], className='mb-0'),
                ], lg=6, className='mb-4 mb-lg-0'),
                dbc.Col(html.Div([
                    html.Div('Pregunta de investigación', className='block-title'),
                    html.P('¿Con qué balanced accuracy se puede predecir el desenlace de un caso registrable de OSHA a partir '
                           'del lugar de trabajo, la ocupación, el horario y la lesión, y supera un modelo original a los '
                           'modelos de referencia?', className='mb-3', style={'fontSize': '16px'}),
                    html.Div('Objetivos', className='block-title'),
                    html.Ol([html.Li('Comparar 7 modelos × 4 balanceos × 4 optimizadores con validación anidada.'),
                             html.Li('Diseñar e implementar un modelo original (SECR) y validarlo con al menos 3 pruebas '
                                     'de hipótesis.'),
                             html.Li('Llevar los mejores modelos a un producto interactivo y reproducible.')],
                            className='mb-0 ps-3')], className='panel panel-soft'), lg=6),
            ])),

    section('Datos', 'Fuente, partición y variables',
            ['Fuente: ', html.A('OSHA, ITA Case Detail Data 2023 (CSV con códigos OIICS)',
                                href='https://www.osha.gov/Establishment-Specific-Injury-and-Illness-Data', target='_blank'),
             '. 890 372 casos de 2023. Partición: el 60 % de los establecimientos se reserva para la prueba '
             '(octubre–diciembre, bloqueada) y el resto, hasta el 23 de septiembre, es el desarrollo. Los modelos se '
             'entrenan con una submuestra de 30 030 casos tomada por establecimientos completos.'],
            dict_table()),

    section('Método', 'Del dato crudo al veredicto', None,
            html.Div([html.Div([html.Div(f'0{i + 1}', className='n'), html.Div(t, className='t'),
                                html.Div(s, className='s')], className='st') for i, (t, s) in enumerate(STEPS)],
                     className='steps')),

    section('Alcance', 'Limitaciones, ética y autoría', None, dbc.Row([
        dbc.Col(html.Ul([
            html.Li('Los datos los reportan los empleadores: hay subregistro y diferencias entre sectores y estados.'),
            html.Li('Los códigos OIICS se asignan a partir del relato del caso. Existen cuando el caso se clasifica, pero el '
                    'relato de los fallecimientos suele mencionar la muerte: hay riesgo de fuga indirecta en esa clase.'),
            html.Li('Los fallecimientos son tan pocos (0 a 4 por pliegue) que un solo caso mueve la BA de un pliegue en '
                    '±0,25. Por eso las cifras se reportan con media ± desviación estándar.'),
            html.Li('El modelo no debe usarse para decisiones sobre personas concretas (seguros, despidos). Es una '
                    'herramienta de análisis agregado y de priorización.'),
            html.Li('La prueba de octubre a diciembre está bloqueada: todos los diagnósticos de esta aplicación salen de '
                    'los pliegues externos de la validación anidada.')], className='ps-3 mb-0'), lg=8),
        dbc.Col(html.Div([
            html.Div('Autores', className='block-title'),
            html.P('Iván Berrio y Jaime Andrés Besada', className='mb-2'),
            html.Div('Versión', className='block-title'),
            html.P('3.1 · 23 de octubre de 2026', className='mb-2'),
            html.Div('Informe completo', className='block-title'),
            html.A('Jupyter Book del proyecto',
                   href='https://besadaj.github.io/osha-ita-2023-case-outcome-classification-entregable-3/', target='_blank'),
        ], className='panel panel-soft'), lg=4, className='mt-3 mt-lg-0')])),
])
