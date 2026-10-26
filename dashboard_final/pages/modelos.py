"""Sección 3: tabla de líderes, predicción en vivo con SHAP local, diagnósticos, interpretabilidad global,
comparación estadística y subsección del modelo original."""
import re

import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme

from src import data, plots
from src.explain import exact_shapley

dash.register_page(__name__, path='/modelos', name='Modelos', title='Modelos · OSHA ITA 2023')

MODELS = data.available_models()
RES = data.original_results()
PF = data.proyecto_final()

# Estilo único de las tablas (cabecera clara, sin bordes verticales, números alineados).
TABLE = dict(style_table={'overflowX': 'auto', 'border': '1px solid #e2e8f0', 'borderRadius': '6px'},
             style_cell={'fontFamily': 'Inter, Segoe UI, sans-serif', 'fontSize': 13, 'padding': '8px 10px',
                         'textAlign': 'left', 'border': 'none', 'borderBottom': '1px solid #f1f5f9', 'color': '#0f172a',
                         'whiteSpace': 'normal', 'height': 'auto', 'minWidth': '84px'},
             style_header={'backgroundColor': '#f8fafc', 'color': '#475569', 'fontWeight': 600, 'fontSize': 12,
                           'borderBottom': '1px solid #e2e8f0'},
             style_filter={'backgroundColor': '#ffffff', 'borderBottom': '1px solid #e2e8f0'})

# ------------------------------------------------------------------------------------- tabla de líderes
lb = data.leaderboard()
lb_tab = pd.DataFrame({
    'Modelo': lb.model.map(data.PRETTY), 'Balanceo': lb.balance.map(data.PRETTY), 'Optimizador': lb.optimizer.map(data.PRETTY),
    'Mejor del modelo': np.where(lb.best_of_model, 'sí', ''),
    'BA media': lb.outer_recall_mean.round(4), 'BA d.e.': lb.outer_recall_std.round(4),
    'BA (media ± d.e.)': [f'{m:.3f} ± {s:.3f}' for m, s in zip(lb.outer_recall_mean, lb.outer_recall_std)],
    'F1 macro': lb.outer_f1_mean.round(4), 'AUC macro': lb.outer_auc_mean.round(4),
    'Log-loss': lb.outer_log_loss_mean.round(4), 'Inferencia (ms/caso)': lb.inference_ms_per_case.round(4),
}).sort_values('BA media', ascending=False)

FMT3 = Format(precision=3, scheme=Scheme.fixed, decimal_delimiter=',')
LB_COLS = [('Modelo', 'text', None), ('Balanceo', 'text', None), ('Optimizador', 'text', None),
           ('BA media', 'numeric', FMT3), ('BA d.e.', 'numeric', FMT3), ('F1 macro', 'numeric', FMT3),
           ('AUC macro', 'numeric', FMT3), ('Log-loss', 'numeric', FMT3), ('Inferencia (ms/caso)', 'numeric', FMT3)]
leaderboard = dash_table.DataTable(
    id='leaderboard', data=lb_tab[lb_tab['Mejor del modelo'] == 'sí'].to_dict('records'),
    columns=[{'name': c, 'id': c, 'type': k, **({'format': f} if f else {})} for c, k, f in LB_COLS],
    sort_action='native', filter_action='native', filter_options={'placeholder_text': 'Filtrar…'}, page_size=10,
    **TABLE, style_cell_conditional=[{'if': {'column_type': 'numeric'}, 'textAlign': 'right'}],
    style_data_conditional=[{'if': {'row_index': 0}, 'backgroundColor': '#f8fafc', 'fontWeight': 600}])

# ------------------------------------------------------------------------------- formulario de predicción
d0 = data.eda()
TOP = {'state': None, 'naics6': 300, 'establishment_type': None, 'soc6': 300, 'type_of_incident': None,
       'nature': 150, 'part': 120, 'event': 150, 'source': 200}
HOURS = ['Unknown'] + [str(h) for h in range(24)]
FORM_FIELDS = ['state', 'naics6', 'establishment_type', 'soc6', 'type_of_incident', 'start_hour', 'incident_hour',
               'nature', 'part', 'event', 'source']
DEFAULTS = {c: d0[c].mode().iloc[0] for c in FORM_FIELDS if c in d0}
DEFAULTS.update({'start_hour': '7', 'incident_hour': '10'})
TITLE_COL = {'naics6': 'industry_description', 'soc6': 'soc_description', 'nature': 'nature_title_pred',
             'part': 'part_title_pred', 'event': 'event_title_pred', 'source': 'source_title_pred'}


def field_options(c):
    if c in ('start_hour', 'incident_hour'):
        return [{'label': h if h == 'Unknown' else f'{h}:00', 'value': h} for h in HOURS]
    vals = data.options(c, TOP.get(c))
    if c in TITLE_COL:
        title = d0.groupby(c)[TITLE_COL[c]].agg(lambda s: s.mode().iloc[0])
        return [{'label': f'{v} · {title.get(v, "")}'[:70], 'value': v} for v in vals]
    if c == 'type_of_incident':
        return [{'label': f'{v} · {data.TYPE_OF_INCIDENT.get(v, "")}', 'value': v} for v in vals]
    return [{'label': v, 'value': v} for v in vals]


form = dbc.Row([dbc.Col([dbc.Label(data.LABELS[c], html_for=f'in-{c}'),
                         dcc.Dropdown(field_options(c), DEFAULTS.get(c), id=f'in-{c}', clearable=False)],
                        xs=12, sm=6, lg=4, className='mb-2') for c in FORM_FIELDS] +
               [dbc.Col([dbc.Label('Empresa (opcional; solo modelo original)', html_for='in-company'),
                         dbc.Input(id='in-company', placeholder='Unknown', type='text', maxLength=80)], xs=12, sm=6, lg=4)])


def model_label(m):
    return data.PRETTY.get(m, m)


def build_row(vals):
    """Fila de entrada con las 18 variables, derivando las del curso igual que ``data.prepare_features``."""
    v = dict(zip(FORM_FIELDS, vals[:-1]))
    company = (vals[-1] or '').strip().upper() or 'Unknown'
    start, inc = v['start_hour'], v['incident_hour']
    if start != 'Unknown' and inc != 'Unknown':
        el = (int(inc) - int(start)) % 24
        edges = [(1, '0-1'), (3, '2-3'), (5, '4-5'), (7, '6-7'), (9, '8-9'), (12, '10-12'), (24, '13+')]
        hss = next(lbl for hi, lbl in edges if el <= hi)
    else:
        hss = 'Unknown'
    row = {'state': v['state'], 'naics3': str(v['naics6'])[:3], 'establishment_type': v['establishment_type'],
           'soc_minor': str(v['soc6'])[:4] if v['soc6'] != 'Unknown' else 'Unknown',
           'type_of_incident': v['type_of_incident'], 'time_unknown': '1' if inc == 'Unknown' else '0',
           'start_hour': start, 'incident_hour': inc, 'hours_since_start': hss,
           'nature': v['nature'], 'part': v['part'], 'event': v['event'], 'source': v['source'],
           'sec_source': 'Unknown', 'naics6': v['naics6'], 'soc6': v['soc6'], 'ein': 'Unknown', 'company': company}
    return pd.DataFrame([row])


GROUPS_ORIGINAL = {'state': ['state'], 'naics': ['naics3', 'naics6'], 'establishment_type': ['establishment_type'],
                   'soc': ['soc_minor', 'soc6'], 'type_of_incident': ['type_of_incident'],
                   'horario': ['time_unknown', 'start_hour', 'incident_hour', 'hours_since_start'],
                   'nature': ['nature'], 'part': ['part'], 'event': ['event', 'source', 'sec_source'],
                   'empleador': ['ein', 'company']}
data.LABELS.update({'naics': 'Actividad (NAICS 3 y 6)', 'soc': 'Ocupación (SOC)', 'horario': 'Horario',
                    'empleador': 'Empleador'})


def background_rows():
    bg = data.background().copy()
    bg['sec_source'], bg['ein'], bg['company'] = 'Unknown', 'Unknown', 'Unknown'
    return bg[data.FEATURES + data.EXTRA]


# ------------------------------------------------------------------------------- subsección del original
def num(x, nd=3, sign=False):
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    return s.replace('.', ',')


def tests_table():
    if not RES:
        return html.Div('Los resultados del experimento pareado no están disponibles.', className='callout warn')
    t = pd.DataFrame(RES['tests']).map(es)
    return dash_table.DataTable(data=t.to_dict('records'), columns=[{'name': c, 'id': c} for c in t.columns],
                                sort_action='native', **{**TABLE, 'style_cell': {**TABLE['style_cell'], 'fontSize': 12.5,
                                                                                 'maxWidth': '260px'}},
                                style_data_conditional=[{'if': {'filter_query': '{Conclusión} contains "Se rechaza"'},
                                                         'backgroundColor': '#ecfdf5'}])


def es(s):
    """Coma decimal en un texto (los separadores de listas pasan a punto y coma)."""
    s = str(s).replace(', ', '; ')
    return re.sub(r'(\d)\.(\d)', lambda m: m.group(1) + ',' + m.group(2), s)


def paired_table():
    if not RES:
        return html.Div()
    rows = []
    for r in RES['summary']:
        rows.append({'Modelo': 'Original (SECR)' if 'original' in r['Modelo'].lower() else r['Modelo'],
                     'Balanceo': data.PRETTY.get(r['Balanceo'], r['Balanceo']) if 'Original' not in r['Modelo'] else 'Corrección por prior (α)',
                     'Optimizador': data.PRETTY.get(r['Optimizador'], r['Optimizador']) if 'Original' not in r['Modelo'] else 'Bayesiana (TPE) del combinador',
                     'BA (media ± d.e.)': es(r['BA (media ± d.e.)']), 'IC BCa 95 %': es(r['IC BCa 95 % de la media']),
                     'F1 macro': es(f"{r['F1 macro']:.3f}"), 'AUC macro': es(f"{r['AUC']:.3f}"),
                     'Entrenamiento (min/pliegue)': es(f"{r['Entrenamiento (s/pliegue)'] / 60:.1f}"),
                     'Inferencia (ms/caso)': es(f"{r['Inferencia (ms/caso)']:.3f}")})
    num_cols = ['BA (media ± d.e.)', 'IC BCa 95 %', 'F1 macro', 'AUC macro', 'Entrenamiento (min/pliegue)', 'Inferencia (ms/caso)']
    return dash_table.DataTable(data=rows, columns=[{'name': c, 'id': c} for c in rows[0]], **TABLE,
                                style_cell_conditional=[{'if': {'column_id': c}, 'textAlign': 'right'} for c in num_cols],
                                style_data_conditional=[{'if': {'filter_query': '{Modelo} contains "Original"'},
                                                         'backgroundColor': '#eff6ff', 'fontWeight': 600}])


PSEUDO = r'''Entrada: probabilidades fuera de pliegue P_m(x_i) de M modelos base (vistas), etiquetas y_i,
         región r(x_i) dada por la variable de compuerta, λ ≥ 0, T pasos, tolerancia ε, n_min
Salida:  pesos globales w₀ ∈ Δ_M y pesos por región w_g ∈ Δ_M

1. T[i,m] ← max(P_m(x_i)[y_i], 10⁻⁶)                     # probabilidad de la clase verdadera
2. w₀ ← FW(T, a = 1/n)                                    # pesos globales
3. para cada región g con n_g ≥ n_min:
       a_i ← 1[i∈g]/(n_g+λ) + λ/((n_g+λ)·n)               # contracción hacia lo global
       w_g ← FW(T, a, inicio = w₀)
FW(T, a): w ← mejor vértice; repetir t = 0..T−1:
       q ← T w ;  ∇ ← −Tᵀ(a/q) ;  s ← argmin ∇ ;  G ← ∇ᵀ(w − e_s)
       si G ≤ ε: parar                                    # brecha = cota de F(w) − F*
       γ ← argmin_{γ∈[0,1]} F(w + γ(e_s − w))              # búsqueda lineal exacta (bisección)
       w ← w + γ(e_s − w)
Predicción: q(x) = Σ_m w_{r(x),m} P_m(x);  ŷ = argmax_k q_k(x) / π_k^α'''

DIFFS = [('Stacking lineal (Wolpert, 1992; Hassan y Khalifa, 2025 con datos OSHA)',
          'Meta-modelo entrenado sin restricciones; SECR usa combinación convexa (pesos ≥ 0 que suman 1) y por región.'),
         ('Selección de Caruana (Caruana et al., 2004)', 'Pesos globales con paso fijo; SECR es el caso general con búsqueda '
          'lineal exacta, certificado de optimalidad y pesos por región con contracción.'),
         ('Selección dinámica (DES/KNORA; García et al., 2018; Aziz et al., 2025)', 'Elige modelos por vecindario en tiempo '
          'de predicción (k-NN en el espacio de entrada); SECR aprende pesos por región una vez, sin vecinos, con pérdida convexa.'),
         ('Mezcla de expertos', 'La compuerta es un modelo entrenado conjuntamente con expertos; en SECR los expertos están '
          'fijos y la compuerta es una partición interpretable con contracción.')]


def section(id_, eyebrow, title, desc, *children):
    return html.Section([html.H2(title, className='section-title'),
                         html.P(desc, className='section-desc') if desc else None, *children], id=id_, className='section')


def simple_table(head, rows, num_cols=()):
    return html.Div(html.Table([html.Thead(html.Tr([html.Th(h) for h in head])),
                                html.Tbody(rows)], className='table-clean'), style={'overflowX': 'auto'})


def candidates_table():
    if not PF:
        return html.Div()
    groups = {'17': 'Primer candidato (PT-bagging) y sus iteraciones · diseño temporal de 3 pliegues',
              '18': 'Variantes del stacking multivista · 5 pliegues externos y prueba (exploratorio)',
              '19–20': 'Modelo final'}
    rows, cur = [], None
    for c in PF['candidates']:
        if c['cap'] != cur:
            cur = c['cap']
            rows.append(html.Tr(html.Td(groups[cur], colSpan=3), className='group'))
        final = c['cap'] == '19–20'
        rows.append(html.Tr([html.Td(html.B(c['diseno']) if final else c['diseno']), html.Td(c['datos']),
                             html.Td(c['resultado'])], className='hl' if final else None))
    return simple_table(['Diseño', 'Datos usados para elegir', 'Resultado'], rows)


# Cifras de cabecera, todas leídas de los resultados versionados.
SUM = {r['Modelo']: r for r in RES['summary']} if RES else {}
BA = {m: float(SUM[m]['BA (media ± d.e.)'].split(' ')[0]) for m in SUM}
DELTA = BA.get('Original (SECR)', np.nan) - BA.get('KNN', np.nan)
FRIEDMAN_P = RES['tests'][0]['p crudo'] if RES else '—'
BYREP = {int(k): v for k, v in PF['diff_by_rep'].items()} if PF else {}
ABL = {r['variante']: r for r in RES['ablation']} if RES else {}


def stat(value, label, accent=False):
    return html.Div([html.Div(value, className='v'), html.Div(label, className='l')],
                    className='stat accent' if accent else 'stat')


def more(summary, *children):
    """Detalle plegable: lo esencial queda a la vista y el resto a un clic."""
    return html.Details([html.Summary(summary), html.Div(children, className='pt-3')], className='more mt-3')


subnav = html.Nav([html.A(t, href=f'#{i}') for i, t in [
    ('benchmarks', 'Benchmarks'), ('prediccion', 'Predicción'), ('diagnostico', 'Diagnóstico'),
    ('original', 'Modelo original'), ('comparacion', 'Comparación'), ('ablacion', 'Ablación')]],
    className='subnav', **{'aria-label': 'Secciones de la página'})

# --------------------------------------------------------------------------------------------- layout
layout = html.Div([
    html.Div([
        html.Div('Modelos', className='eyebrow'),
        html.H1('El modelo original gana en media, pero no por un margen claro', className='page-title'),
        html.P('Con las mismas variables y los mismos 15 pliegues, el ensamble multivista con SECR obtiene la mayor balanced '
               'accuracy de los 8 modelos. Su ventaja sobre KNN, el mejor benchmark, es pequeña frente a la variación entre '
               'pliegues y ninguna prueba pareada la confirma.', className='page-lead'),
        html.Div([stat(num(BA.get('Original (SECR)', np.nan)), 'BA media del original (15 pliegues)', accent=True),
                  stat(num(BA.get('KNN', np.nan)), 'BA media de KNN, el mejor benchmark'),
                  stat(num(DELTA, sign=True), 'diferencia; IC BCa 95 % [−0,032; 0,056]'),
                  stat(num(ABL['SECR sin corrección por prior (α = 0)']['ba_mean']) if ABL else '—',
                       'BA del original sin la corrección por prior: el componente que más aporta')],
                 className='stats'),
    ], className='page-head'),
    subnav,

    section('benchmarks', 'Benchmarks', 'Mejor combinación de cada modelo',
            'Validación anidada de 7 modelos × 4 balanceos × 4 optimizadores con las 9 variables del lugar de trabajo. '
            'Cada fila es la combinación con mayor BA media en los 5 pliegues externos. Estos mismos 7 modelos se reentrenan con las variables ampliadas '
            'para compararlos con el original.',
            dbc.Switch(id='show-all', label='Mostrar las 112 combinaciones', value=False, className='mb-2'),
            leaderboard),

    section('prediccion', 'Predicción', 'Un caso, paso a paso',
            'Los benchmarks usan las 9 variables del lugar de trabajo; el original añade la lesión y el empleador. El '
            'modelo elegido aquí es también el del diagnóstico.',
            dbc.Row([dbc.Col([dbc.Label('Modelo', html_for='model-sel'),
                              dcc.Dropdown([{'label': model_label(m), 'value': m} for m in MODELS],
                                           'original' if 'original' in MODELS else MODELS[0], id='model-sel',
                                           clearable=False)], xs=12, md=5),
                     dbc.Col(html.Div(id='model-info', className='note pt-md-4'), xs=12, md=7)], className='mb-3'),
            html.Div([form,
                      dbc.Button('Predecir', id='btn-pred', color='primary', className='mt-2')],
                     className='panel panel-soft'),
            html.Div(id='pred-alert', className='mt-3'),
            dbc.Row([dbc.Col(html.Div(dcc.Loading(dcc.Graph(id='g-proba', config={'displaylogo': False},
                                                            figure=plots.empty('Pulse «Predecir».', height=150)),
                                                  type='dot', color='#94a3b8'), className='panel'), lg=5),
                     dbc.Col(html.Div(dcc.Loading(dcc.Graph(id='g-shap', config={'displaylogo': False},
                                                            figure=plots.empty('Aquí aparecerá el aporte de cada variable.', height=150)),
                                                  type='dot', color='#94a3b8'), className='panel'), lg=7)],
                    className='grid-gap mt-1')),

    section('diagnostico', 'Diagnóstico', 'Dónde acierta y dónde se equivoca',
            'Predicciones fuera de pliegue externo de los 30 030 casos de desarrollo.',
            html.P(id='i-diag', className='finding mb-3'),
            dbc.Row([dbc.Col(html.Div(dcc.Graph(id='g-cm'), className='panel'), lg=6),
                     dbc.Col(html.Div(dcc.Graph(id='g-cal'), className='panel'), lg=6),
                     dbc.Col(html.Div(dcc.Graph(id='g-roc'), className='panel'), lg=6),
                     dbc.Col(html.Div(dcc.Graph(id='g-pr'), className='panel'), lg=6),
                     dbc.Col(html.Div([dcc.Graph(id='g-shapglob'), html.P(id='i-shapglob', className='finding')],
                                      className='panel'), lg=12)], className='grid-gap')),

    section('original', 'Modelo original', 'Ensamble multivista + SECR',
            '16 modelos base, cada uno especializado en una vista del caso (lugar de trabajo, lesión, códigos finos, '
            'empleador), se combinan con pesos aprendidos por región del espacio de entrada y contraídos hacia unos pesos '
            'globales cuando la región tiene pocos datos. Los pesos se optimizan con Frank-Wolfe, que garantiza que la '
            'pérdida no sube y certifica qué tan cerca está del óptimo. La decisión final divide cada probabilidad por la '
            'frecuencia de su clase.',
            more('Algoritmo, diferencias con la literatura y convergencia',
                 dbc.Row([dbc.Col([html.Div('Pseudocódigo', className='block-title mb-2'),
                                   html.Pre(PSEUDO, className='pseudo')], lg=6),
                          dbc.Col([html.Div('Diferencias con métodos afines', className='block-title mb-2'),
                                   simple_table(['Método afín', 'Qué cambia SECR'],
                                                [html.Tr([html.Td(a), html.Td(b)]) for a, b in DIFFS])], lg=6)],
                         className='grid-gap'),
                 html.Div(dcc.Graph(id='g-conv', config={'displaylogo': False}), className='panel mt-3'),
                 html.P('La log-loss no sube nunca y la brecha de Frank-Wolfe, que acota la distancia al óptimo, cae varios '
                        'órdenes de magnitud.', className='finding')),
            more('Los 13 diseños probados antes de llegar a SECR',
                 html.P('Elegir el mejor de varios intentos sobrestima su ventaja, así que se listan todos con los datos que '
                        'se miraron para elegirlos. Las cifras de bloques distintos no son comparables entre sí.',
                        className='note'),
                 candidates_table())),

    section('comparacion', 'Comparación pareada', 'Mismas variables, mismos 15 pliegues',
            'Los 7 benchmarks se reentrenan con las mismas columnas que el original y se evalúan en los mismos 3 × 5 '
            'pliegues externos, con el mismo presupuesto de búsqueda.',
            paired_table(),
            dbc.Row([dbc.Col(html.Div([dcc.Graph(id='g-cd', figure=(plots.cd_diagram(
                RES['cd']['names'], np.array(RES['cd']['ranks']), RES['cd']['cd'], highlight='Original (SECR)')
                if RES else plots.empty('Resultados pendientes'))),
                html.P(es(RES['cd']['insight']) if RES else '', className='finding')], className='panel'), lg=5),
                     dbc.Col(html.Div([dcc.Graph(id='g-folds', figure=plots.paired_folds(PF['per_fold']) if PF
                                                 else plots.empty()),
                                       html.P((f'La ventaja media sobre KNN es {num(BYREP[0], sign=True)} en la repetición 0 '
                                               f'(pliegues ya vistos al elegir el nivel 0) y {num(BYREP[1], sign=True)} y '
                                               f'{num(BYREP[2], sign=True)} en las otras dos. La variación entre pliegues '
                                               '(d.e. ≈ 0,08) es unas cuatro veces la ventaja media.')
                                              if BYREP else '', className='finding')], className='panel'), lg=7)],
                    className='grid-gap mt-3'),
            more('Pruebas de hipótesis (Friedman, Nemenyi, Nadeau–Bengio, Wilcoxon y bayesiana)', tests_table())),

    section('ablacion', 'Ablación', 'Qué aporta cada componente',
            'Mismos modelos base y mismos pliegues; se quita un componente cada vez o se cambia la regla de combinación.',
            html.Div([dcc.Graph(figure=plots.ablation_bar(RES['ablation']) if RES else plots.empty()),
                      html.P(('La corrección por prior es lo que más aporta: sin ella la BA cae a '
                              f"{num(ABL['SECR sin corrección por prior (α = 0)']['ba_mean'])} (Holm p = "
                              f"{num(ABL['SECR sin corrección por prior (α = 0)']['p_holm'])}). Las regiones no aportan: "
                              f"sin ellas la BA es {num(ABL['SECR sin regiones (solo pesos globales)']['ba_mean'])}, sin "
                              'diferencia significativa. Las variantes con * difieren de SECR tras Holm.')
                             if ABL else '', className='finding')], className='panel'),
            more('Sensibilidad a los hiperparámetros',
                 dbc.Row([dbc.Col(html.Div(dcc.Graph(figure=plots.sensitivity(PF['sensitivity'], prm) if PF
                                                     else plots.empty()), className='panel'), md=6, xl=3)
                          for prm in ['alpha', 'lam', 'gate', 'frac_datos']], className='grid-gap'),
                 html.P('α decide la BA (≈ 0,46 con α ≤ 0,5 y ≈ 0,59 con α = 1). Con contracción fuerte o sin compuerta la BA '
                        'sube a ≈ 0,62: la búsqueda interna eligió compuertas que no generalizan. No se cambió el modelo por '
                        'esto, porque sería elegir con los pliegues de la evaluación.', className='finding mt-3'))),
])


@callback(Output('leaderboard', 'data'), Input('show-all', 'value'))
def toggle_all(show_all):
    return (lb_tab if show_all else lb_tab[lb_tab['Mejor del modelo'] == 'sí']).to_dict('records')


@callback(Output('model-info', 'children'), Output('g-cm', 'figure'), Output('g-cal', 'figure'), Output('g-roc', 'figure'),
          Output('g-pr', 'figure'), Output('i-diag', 'children'), Output('g-shapglob', 'figure'),
          Output('i-shapglob', 'children'), Input('model-sel', 'value'))
def update_model(m):
    meta = data.metadata(m)
    if m == 'original':
        hp = meta['hyperparameters']
        info = (f"Hiperparámetros (TPE sobre todo el desarrollo): compuerta = {hp['gate']}, λ = {hp['lam']:.1f}, "
                f"pasos = {hp['n_steps']}, región mínima = {hp['min_region']}, α = {hp['alpha']:.2f}. "
                f"Tamaño del artefacto: {meta['size_mb']} MB.")
    else:
        info = (f"{data.PRETTY[meta['balance']]} + {data.PRETTY[meta['optimizer']]}; hiperparámetros: "
                + ', '.join(f'{k} = {v:.4g}' if isinstance(v, float) else f'{k} = {v}' for k, v in meta['hyperparameters'].items())
                + f". BA externa {data.fmt(meta['outer_metrics']['recall']['mean'])} ± {data.fmt(meta['outer_metrics']['recall']['sd'])}.")
    o = data.oof(m)
    if o is None:
        e = plots.empty('Diagnóstico pendiente.')
        return es(info), e, e, e, e, '', e, ''
    y, P = o.y.to_numpy(), o[[f'p{k}' for k in range(4)]].to_numpy()
    prior = np.bincount(y, minlength=4) / len(y)
    pred = (P / prior ** (meta['hyperparameters']['alpha'] if m == 'original' else 0)).argmax(1)
    rec = [((pred == k) & (y == k)).sum() / max((y == k).sum(), 1) for k in range(4)]
    txt = ('Recall por clase: ' + ', '.join(f'{c}: {data.fmt_pct(r * 100, nd=0)}' for c, r in zip(data.CLASSES, rec)) +
           f'. BA con los 30 030 casos juntos = {num(np.mean(rec))}' +
           (' (con el α del modelo final; la cifra de la comparación, 0,591, es la media de 15 pliegues). '
            if m == 'original' else ' (la tabla de líderes da la media por pliegue). ') +
           '«Traslado/restricción» es la clase que más se confunde con «Otro registrable»; las curvas de calibración '
           'muestran si las probabilidades pueden leerse como riesgos.')
    sg = data.shap_global()
    og = data.DATA / 'shap_original_groups.csv'
    if m == 'original' and og.exists():
        s = pd.read_csv(og, index_col=0)['Original (SECR)']
        g_fig = plots.shap_global_bar(s, m)
        g_txt = (f'Por grupos de variables, lo que más mueve la probabilidad del original es «{s.idxmax()}»; el '
                 'mejor benchmark con 9 variables (SVM) no ve los códigos de la lesión.')
    elif m in getattr(sg, 'index', []):
        s = sg.loc[m]
        g_fig = plots.shap_global_bar(s, m)
        g_txt = f'La variable que más mueve la probabilidad de la clase predicha es «{data.LABELS[s.idxmax()]}».'
    else:
        g_fig, g_txt = plots.empty('SHAP global no disponible para este modelo.'), ''
    return es(info), plots.confusion(y, pred), plots.calibration(y, P), plots.roc_pr(y, P, 'roc'), plots.roc_pr(y, P, 'pr'), \
        txt, g_fig, g_txt


@callback(Output('g-proba', 'figure'), Output('g-shap', 'figure'), Output('pred-alert', 'children'),
          Input('btn-pred', 'n_clicks'), State('model-sel', 'value'),
          *[State(f'in-{c}', 'value') for c in FORM_FIELDS], State('in-company', 'value'), prevent_initial_call=True)
def predict(n, m, *vals):
    missing = [data.LABELS[c] for c, v in zip(FORM_FIELDS, vals) if v in (None, '')]
    if missing:
        return no_update, no_update, dbc.Alert('Complete: ' + ', '.join(missing), color='danger')
    company = vals[-1] or ''
    if len(company) > 80 or any(ch in company for ch in '<>{}'):
        return no_update, no_update, dbc.Alert('El nombre de la empresa no es válido.', color='danger')
    try:
        row = build_row(vals)
        est = data.model(m)
        cols = data.FEATURES if m != 'original' else data.FEATURES + data.EXTRA
        p = est.predict_proba(row[cols])[0]
        if m == 'original':
            pred = int(est.combiner_.decide(p[None])[0])
            groups = GROUPS_ORIGINAL
            bg = background_rows().head(4)
        else:
            pred = int(p.argmax())
            groups = {c: [c] for c in data.FEATURES}
            bg = background_rows()[data.FEATURES]
        phi = exact_shapley(lambda X: est.predict_proba(X[cols]), row[cols], bg[cols] if m == 'original' else bg,
                            groups, pred)
        unseen = [data.LABELS[c] for c in ['naics6', 'soc6', 'nature'] if row[c].iloc[0] not in set(d0[c])]
        msg = dbc.Alert([html.B(f'Predicción: {data.CLASSES[pred]}'),
                         f' con {model_label(m)} (probabilidad {data.fmt(p[pred] * 100, 1)} %). ',
                         f'Suma de aportes SHAP = {data.fmt(phi.sum() * 100, 1)} pp sobre la base '
                         f'{data.fmt(phi.attrs["base"] * 100, 1)} %.'
                         + (f' Aviso: {", ".join(unseen)} no se vio en el entrenamiento; se trata como categoría infrecuente.'
                            if unseen else '')], color='info')
        return plots.proba_bar(p, pred), plots.shap_bar(phi, data.CLASSES[pred]), msg
    except Exception as exc:  # noqa: BLE001 - la app nunca se cae por una entrada
        return no_update, no_update, dbc.Alert(f'No se pudo predecir: {type(exc).__name__}. Revise las entradas.', color='danger')


@callback(Output('g-conv', 'figure'), Input('model-sel', 'value'))
def conv(_):
    if not RES:
        return plots.empty('Curva pendiente')
    return plots.convergence(RES['convergence']['loss'], RES['convergence']['gap'])
