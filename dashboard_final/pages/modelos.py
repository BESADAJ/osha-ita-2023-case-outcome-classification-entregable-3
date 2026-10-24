"""Sección 3: tabla de líderes, predicción en vivo con SHAP local, diagnósticos, interpretabilidad global,
comparación estadística y subsección del modelo original."""
import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update

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
                         'whiteSpace': 'normal', 'height': 'auto'},
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

leaderboard = dash_table.DataTable(
    id='leaderboard', data=lb_tab[lb_tab['Mejor del modelo'] == 'sí'].to_dict('records'),
    columns=[{'name': c, 'id': c, 'type': 'numeric' if lb_tab[c].dtype.kind == 'f' else 'text'}
             for c in lb_tab.columns if c != 'Mejor del modelo'],
    sort_action='native', filter_action='native', page_size=10, **TABLE,
    style_cell_conditional=[{'if': {'column_type': 'numeric'}, 'textAlign': 'right'}],
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
    t = pd.DataFrame(RES['tests'])
    return dash_table.DataTable(data=t.to_dict('records'), columns=[{'name': c, 'id': c} for c in t.columns],
                                sort_action='native', **{**TABLE, 'style_cell': {**TABLE['style_cell'], 'fontSize': 12.5,
                                                                                 'maxWidth': '260px'}},
                                style_data_conditional=[{'if': {'filter_query': '{Conclusión} contains "Se rechaza"'},
                                                         'backgroundColor': '#ecfdf5'}])


def paired_table():
    if not RES:
        return html.Div()
    t = pd.DataFrame(RES['summary'])
    return dash_table.DataTable(data=t.to_dict('records'), columns=[{'name': c, 'id': c} for c in t.columns],
                                sort_action='native', **TABLE,
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
    return html.Section([html.Div(eyebrow, className='eyebrow'), html.H2(title, className='section-title'),
                         html.P(desc, className='section-desc') if desc else None, *children], id=id_, className='section')


def simple_table(head, rows, num_cols=()):
    return html.Div(html.Table([html.Thead(html.Tr([html.Th(h) for h in head])),
                                html.Tbody(rows)], className='table-clean'), style={'overflowX': 'auto'})


def candidates_table():
    if not PF:
        return html.Div()
    groups = {'17': 'Capítulo 17 · primer candidato (PT-bagging) y sus iteraciones · diseño temporal de 3 pliegues',
              '18': 'Capítulo 18 · variantes del stacking multivista · 5 pliegues externos y prueba (exploratorio)',
              '19–20': 'Capítulos 19 y 20 · modelo original final'}
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


def verdict_strip():
    req = [('R1 · Novedad', 'Sí', 'ok', 'Revisión en 3 bases con 8 cadenas; no se halló SECR aplicado a este problema '
            '(cap. 16.6–16.9).'),
           ('R2 · Implementación propia', 'Sí', 'ok', 'Clase compatible con scikit-learn, Frank-Wolfe en numpy, '
            '6 proposiciones y 8 pruebas unitarias (cap. 19).'),
           ('R3 · Superioridad', 'No', 'bad', 'Mayor BA media, pero ninguna prueba pareada contra KNN rechaza la '
            'igualdad tras Holm (cap. 20).')]
    return html.Div([html.Div([html.Div(k, className='k'), html.Div(html.Span(s, className=f'pill {c}'), className='s'),
                               html.Div(d, className='d')], className='req') for k, s, c, d in req], className='verdict')


subnav = html.Nav([html.A(t, href=f'#{i}') for i, t in [
    ('benchmarks', 'Benchmarks'), ('prediccion', 'Predicción'), ('diagnostico', 'Diagnóstico'), ('original', 'Modelo original'),
    ('candidatos', 'Candidatos'), ('comparacion', 'Comparación pareada'), ('ablacion', 'Ablación'), ('veredicto', 'Veredicto')]],
    className='subnav', **{'aria-label': 'Secciones de la página'})

# --------------------------------------------------------------------------------------------- layout
layout = html.Div([
    html.Div([
        html.Div('Sección 3 · Modelos', className='eyebrow'),
        html.H1('Siete benchmarks del curso y un modelo original', className='page-title'),
        html.P('El original (ensamble multivista + SECR) obtiene la mayor balanced accuracy media cuando se compara con los '
               '7 benchmarks con las mismas variables y los mismos pliegues, pero su ventaja sobre KNN no es '
               'estadísticamente significativa.', className='page-lead'),
        html.Div([stat(num(BA.get('Original (SECR)', np.nan)), 'BA media del original (15 pliegues pareados)', accent=True),
                  stat(num(BA.get('KNN', np.nan)), 'BA media de KNN, el mejor benchmark'),
                  stat(num(DELTA, sign=True), 'diferencia; IC BCa 95 % [−0,032; 0,056] incluye el 0'),
                  stat(FRIEDMAN_P.replace('.', ','), 'p de Friedman (global): los 8 modelos no son iguales')],
                 className='stats'),
        html.Div(verdict_strip(), className='mt-3'),
    ], className='page-head'),
    subnav,

    section('benchmarks', '3.1 · Benchmarks', 'Tabla de líderes del Entregable 2',
            'Mejores combinaciones del Entregable 2 (9 variables): una fila por modelo, la combinación de balanceo y '
            'optimizador con mayor BA media en los 5 pliegues externos de la validación anidada. La comparación con el '
            'modelo original usa estos mismos 7 modelos reentrenados con las variables ampliadas (sección 3.6).',
            dbc.Switch(id='show-all', label='Mostrar las 112 combinaciones', value=False, className='mb-2'),
            leaderboard),

    section('prediccion', '3.2 · Predicción', 'Predicción en vivo con explicación',
            'Describa un caso y elija el modelo. Los benchmarks usan las 9 variables del lugar de trabajo; el original usa '
            'además los códigos de la lesión y el empleador. NAICS de 3 dígitos, ocupación SOC menor y horas desde el '
            'inicio de la jornada se derivan automáticamente. El modelo elegido también controla el diagnóstico (3.3).',
            dbc.Row([dbc.Col([dbc.Label('Modelo', html_for='model-sel'),
                              dcc.Dropdown([{'label': model_label(m), 'value': m} for m in MODELS],
                                           'original' if 'original' in MODELS else MODELS[0], id='model-sel',
                                           clearable=False)], xs=12, md=5),
                     dbc.Col(html.Div(id='model-info', className='note pt-md-4'), xs=12, md=7)], className='mb-3'),
            html.Div([form,
                      dbc.Button('Predecir y explicar', id='btn-pred', color='primary', className='mt-2')],
                     className='panel panel-soft'),
            html.Div(id='pred-alert', className='mt-3'),
            dbc.Row([dbc.Col(html.Div(dcc.Loading(dcc.Graph(id='g-proba', config={'displaylogo': False},
                                                            figure=plots.empty('Pulse «Predecir y explicar».')),
                                                  type='dot', color='#94a3b8'), className='panel'), lg=5),
                     dbc.Col(html.Div(dcc.Loading(dcc.Graph(id='g-shap', config={'displaylogo': False},
                                                            figure=plots.empty('Aquí aparecerá la explicación SHAP local.')),
                                                  type='dot', color='#94a3b8'), className='panel'), lg=7)],
                    className='grid-gap mt-1')),

    section('diagnostico', '3.3 · Diagnóstico', 'Cómo se equivoca el modelo seleccionado',
            'Calculado con las predicciones fuera de pliegue externo (30 030 casos); la prueba de octubre a diciembre no se '
            'usa.',
            html.P(id='i-diag', className='finding mb-3'),
            dbc.Row([dbc.Col(html.Div(dcc.Graph(id='g-cm'), className='panel'), lg=6),
                     dbc.Col(html.Div(dcc.Graph(id='g-cal'), className='panel'), lg=6),
                     dbc.Col(html.Div(dcc.Graph(id='g-roc'), className='panel'), lg=6),
                     dbc.Col(html.Div(dcc.Graph(id='g-pr'), className='panel'), lg=6),
                     dbc.Col(html.Div([dcc.Graph(id='g-shapglob'), html.P(id='i-shapglob', className='finding')],
                                      className='panel'), lg=12)], className='grid-gap')),

    section('original', '3.4 · Modelo original', 'Ensamble multivista + SECR',
            '16 modelos base, cada uno especializado en una vista de los datos (lugar de trabajo, lesión OIICS, códigos '
            'finos con codificación por objetivo, empleador), se combinan con SECR: selección de ensambles condicionada '
            'por región con contracción. Los pesos de cada región minimizan la log-loss fuera de pliegue sobre el símplex '
            'con Frank-Wolfe y se contraen hacia los pesos globales con λ pseudo-observaciones. La compuerta, λ, los pasos, '
            'el tamaño mínimo de región y el α de la decisión se eligen con TPE dentro de cada pliegue externo, con el '
            'mismo presupuesto (16) que los benchmarks.',
            dbc.Row([dbc.Col([html.Div('Pseudocódigo', className='block-title mb-2'), html.Pre(PSEUDO, className='pseudo')],
                             lg=6),
                     dbc.Col([html.Div('Diferencias frente a la literatura', className='block-title mb-2'),
                              simple_table(['Método afín', 'Qué cambia SECR'],
                                           [html.Tr([html.Td(a), html.Td(b)]) for a, b in DIFFS])], lg=6)],
                    className='grid-gap'),
            html.Div(dcc.Graph(id='g-conv', config={'displaylogo': False}), className='panel mt-3'),
            html.P('La log-loss no sube nunca (proposición 2) y la brecha de Frank-Wolfe, que acota la distancia al óptimo '
                   '(proposición 3), cae varios órdenes de magnitud.', className='finding')),

    section('candidatos', '3.5 · Candidatos', 'Trece diseños probados antes del modelo final',
            'La guía (§13) pide declarar todos los diseños probados y qué datos se miraron para elegir, porque elegir el '
            'mejor de varios candidatos sobrestima su ventaja. Las cifras de bloques distintos no se comparan entre sí.',
            candidates_table(),
            html.P('El diseño de SECR (regla, hiperparámetros y rangos) se fijó antes de correr el experimento pareado, y '
                   'sus hiperparámetros los eligió TPE con predicciones internas de cada pliegue externo (cap. 19.9).',
                   className='finding mt-3')),

    section('comparacion', '3.6 · Comparación pareada', 'Mismas variables, mismas particiones, 3 × 5 pliegues',
            'Los 7 benchmarks se reentrenan con las mismas columnas que el original y se evalúan en los mismos 15 pliegues '
            'externos, con el mismo presupuesto de búsqueda.',
            paired_table(),
            dbc.Row([dbc.Col(html.Div([dcc.Graph(id='g-cd', figure=(plots.cd_diagram(
                RES['cd']['names'], np.array(RES['cd']['ranks']), RES['cd']['cd'], highlight='Original (SECR)')
                if RES else plots.empty('Resultados pendientes'))),
                html.P(RES['cd']['insight'] if RES else '', className='finding')], className='panel'), lg=5),
                     dbc.Col(html.Div([dcc.Graph(id='g-folds', figure=plots.paired_folds(PF['per_fold']) if PF
                                                 else plots.empty()),
                                       html.P((f'La ventaja media sobre KNN es {num(BYREP[0], sign=True)} en la repetición 0 '
                                               f'(pliegues ya vistos al elegir el nivel 0) y {num(BYREP[1], sign=True)} y '
                                               f'{num(BYREP[2], sign=True)} en las repeticiones 1 y 2. En 15 pliegues la '
                                               'variación entre pliegues (d.e. ≈ 0,08) es unas cuatro veces la ventaja media.')
                                              if BYREP else '', className='finding')], className='panel'), lg=7)],
                    className='grid-gap mt-3'),
            html.Div('Pruebas de hipótesis (familia de 4, ajuste de Holm)', className='block-title mt-4 mb-2'),
            tests_table()),

    section('ablacion', '3.7 · Ablación y sensibilidad', 'Qué aporta cada componente',
            'Mismos modelos base, mismas predicciones y mismos 15 pliegues; se retira un componente cada vez o la regla '
            'de combinación se reemplaza por una regla estándar. La sensibilidad es descriptiva: no se usó para elegir '
            'nada, porque eso sería ajustar con los pliegues del veredicto.',
            dbc.Row([dbc.Col(html.Div([dcc.Graph(figure=plots.ablation_bar(RES['ablation']) if RES else plots.empty()),
                                       html.P(('La corrección por prior (α) es lo que más aporta: sin ella la BA cae a '
                                               f"{num(ABL['SECR sin corrección por prior (α = 0)']['ba_mean'])} (Holm p = "
                                               f"{num(ABL['SECR sin corrección por prior (α = 0)']['p_holm'])}). Las regiones "
                                               f"no aportan: sin ellas la BA es {num(ABL['SECR sin regiones (solo pesos globales)']['ba_mean'])}, "
                                               'sin diferencia significativa. Las variantes marcadas con * difieren de SECR '
                                               'de forma significativa tras Holm.')
                                              if 'SECR sin corrección por prior (α = 0)' in ABL else '',
                                              className='finding')], className='panel'), lg=12)],
                    className='grid-gap'),
            html.Div('Sensibilidad de SECR (descriptiva; media ± d.e. en 15 pliegues)', className='block-title mt-4 mb-2'),
            dbc.Row([dbc.Col(html.Div(dcc.Graph(figure=plots.sensitivity(PF['sensitivity'], prm) if PF else plots.empty()),
                                      className='panel'), md=6, xl=3)
                     for prm in ['alpha', 'lam', 'gate', 'frac_datos']], className='grid-gap'),
            html.P('α decide la BA (≈ 0,46 con α ≤ 0,5 y ≈ 0,59 con α = 1). Con contracción fuerte (λ ≥ 1000) o sin compuerta '
                   'la BA sube a ≈ 0,62, más que el 0,591 del modelo con TPE: la búsqueda interna eligió compuertas que no '
                   'generalizan. No se cambió el modelo por esto, porque sería elegir con los pliegues del veredicto. Con el '
                   '10 % de los datos internos el método se vuelve inestable.', className='finding mt-3')),

    section('veredicto', '3.8 · Veredicto', 'R1 sí, R2 sí, R3 no: tope de calificación 4,0',
            None,
            verdict_strip(),
            html.Div([html.B('Resultado del experimento. '), RES['verdict'] if RES else 'Veredicto pendiente.'],
                     className='callout warn mt-3', id='verdict'),
            html.Div([html.B('Por qué «R3 = no» es conservador. '),
                      'La BA de 0,591 es optimista por selección: el original salió de 13 diseños, y su nivel 0 se eligió '
                      'mirando los 5 pliegues de la repetición 0. Ese sesgo favorece declarar «superior»; corregirlo solo '
                      'podría reducir la ventaja. No se repitió la corrida con más repeticiones, otra semilla u otra '
                      'métrica después de ver los resultados, porque eso sería buscar la significancia (guía §13).'],
                     className='callout mt-3')),
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
        return info, e, e, e, e, '', e, ''
    y, P = o.y.to_numpy(), o[[f'p{k}' for k in range(4)]].to_numpy()
    prior = np.bincount(y, minlength=4) / len(y)
    pred = (P / prior ** (meta['hyperparameters']['alpha'] if m == 'original' else 0)).argmax(1)
    rec = [((pred == k) & (y == k)).sum() / max((y == k).sum(), 1) for k in range(4)]
    txt = ('Recall por clase: ' + ', '.join(f'{c}: {r * 100:.0f} %' for c, r in zip(data.CLASSES, rec)) +
           f'. BA = {np.mean(rec):.3f}. ' + ('«Traslado/restricción» es la clase que más se confunde con «Otro registrable»; '
                                             'las curvas de calibración muestran si las probabilidades pueden leerse como riesgos.'))
    sg = data.shap_global()
    og = data.DATA / 'shap_original_groups.csv'
    if m == 'original' and og.exists():
        s = pd.read_csv(og, index_col=0)['Original (SECR)']
        g_fig = plots.shap_global_bar(s, m)
        g_txt = (f'Por grupos de variables, lo que más mueve la probabilidad del original es «{s.idxmax()}»; el '
                 'mejor benchmark del curso (SVM) no ve los códigos de la lesión (capítulo 20.8 del informe).')
    elif m in getattr(sg, 'index', []):
        s = sg.loc[m]
        g_fig = plots.shap_global_bar(s, m)
        g_txt = f'La variable que más mueve la probabilidad de la clase predicha es «{data.LABELS[s.idxmax()]}».'
    else:
        g_fig, g_txt = plots.empty('SHAP global no disponible para este modelo.'), ''
    return info, plots.confusion(y, pred), plots.calibration(y, P), plots.roc_pr(y, P, 'roc'), plots.roc_pr(y, P, 'pr'), \
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
