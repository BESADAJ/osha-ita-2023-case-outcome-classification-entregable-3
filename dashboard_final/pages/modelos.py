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
    sort_action='native', filter_action='native', page_size=10, style_table={'overflowX': 'auto'},
    style_cell={'fontFamily': 'Inter, Segoe UI, sans-serif', 'fontSize': 13, 'padding': '6px', 'textAlign': 'left'},
    style_header={'backgroundColor': '#1f3b73', 'color': 'white', 'fontWeight': 'bold'},
    style_data_conditional=[{'if': {'row_index': 0}, 'backgroundColor': '#e8f4f2', 'fontWeight': 'bold'}])

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


form = dbc.Row([dbc.Col([dbc.Label(data.LABELS[c], html_for=f'in-{c}', className='small mb-0'),
                         dcc.Dropdown(field_options(c), DEFAULTS.get(c), id=f'in-{c}', clearable=False)],
                        xs=12, sm=6, lg=4, className='mb-2') for c in FORM_FIELDS] +
               [dbc.Col([dbc.Label('Empresa (opcional; solo modelo original)', html_for='in-company', className='small mb-0'),
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
def tests_table():
    if not RES:
        return dbc.Alert('Los resultados del experimento pareado se están calculando.', color='warning')
    t = pd.DataFrame(RES['tests'])
    return dash_table.DataTable(data=t.to_dict('records'), columns=[{'name': c, 'id': c} for c in t.columns],
                                style_table={'overflowX': 'auto'}, style_cell={'fontSize': 12, 'textAlign': 'left',
                                                                               'whiteSpace': 'normal', 'height': 'auto',
                                                                               'maxWidth': '260px'},
                                style_header={'backgroundColor': '#1f3b73', 'color': 'white', 'fontWeight': 'bold'},
                                sort_action='native')


def paired_table():
    if not RES:
        return html.Div()
    t = pd.DataFrame(RES['summary'])
    return dash_table.DataTable(data=t.to_dict('records'), columns=[{'name': c, 'id': c} for c in t.columns],
                                sort_action='native', style_table={'overflowX': 'auto'},
                                style_cell={'fontSize': 12, 'textAlign': 'left'},
                                style_header={'backgroundColor': '#1f3b73', 'color': 'white', 'fontWeight': 'bold'},
                                style_data_conditional=[{'if': {'filter_query': '{Modelo} contains "original"'},
                                                         'backgroundColor': '#fdecee', 'fontWeight': 'bold'}])


PSEUDO = r'''
Entrada: probabilidades fuera de pliegue P_m(x_i) de M modelos base (vistas), etiquetas y_i,
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
Predicción: q(x) = Σ_m w_{r(x),m} P_m(x);  ŷ = argmax_k q_k(x) / π_k^α
'''

DIFFS = [('Stacking lineal (Wolpert, 1992; Hassan y Khalifa, 2025 con datos OSHA)',
          'Meta-modelo entrenado sin restricciones; SECR usa combinación convexa (pesos ≥ 0 que suman 1) y por región.'),
         ('Selección de Caruana (Caruana et al., 2004)', 'Pesos globales con paso fijo; SECR es el caso general con búsqueda '
          'lineal exacta, certificado de optimalidad y pesos por región con contracción.'),
         ('Selección dinámica (DES/KNORA; García et al., 2018; Aziz et al., 2025)', 'Elige modelos por vecindario en tiempo '
          'de predicción (k-NN en el espacio de entrada); SECR aprende pesos por región una vez, sin vecinos, con pérdida convexa.'),
         ('Mezcla de expertos', 'La compuerta es un modelo entrenado conjuntamente con expertos; en SECR los expertos están '
          'fijos y la compuerta es una partición interpretable con contracción.')]

original_section = dbc.Card([dbc.CardHeader(html.H3('Modelo original: ensamble multivista + SECR', className='h5 mb-0')),
                             dbc.CardBody([
    html.P(['16 modelos base, cada uno especializado en una vista de los datos (lugar de trabajo, lesión OIICS, códigos '
            'finos con codificación por objetivo, empleador), se combinan con ', html.B('SECR'), ': selección de ensambles '
            'condicionada por región con contracción. Los pesos de cada región minimizan la log-loss fuera de pliegue sobre '
            'el símplex con Frank-Wolfe, y se contraen hacia los pesos globales con λ pseudo-observaciones. Los '
            'hiperparámetros (variable de compuerta, λ, pasos, tamaño mínimo de región y α de la decisión) se eligen con '
            'búsqueda bayesiana TPE dentro de cada pliegue externo, con el mismo presupuesto (16) que los benchmarks.']),
    dbc.Row([dbc.Col([html.H4('Pseudocódigo', className='h6'), html.Pre(PSEUDO, className='small bg-light p-2 border rounded',
                                                                       style={'whiteSpace': 'pre-wrap'})], lg=6),
             dbc.Col([html.H4('Diferencias frente a la literatura', className='h6'),
                      dbc.Table([html.Thead(html.Tr([html.Th('Método afín'), html.Th('Qué cambia SECR')]))] +
                                [html.Tbody([html.Tr([html.Td(a), html.Td(b)]) for a, b in DIFFS])],
                                size='sm', bordered=False, striped=True, className='small'),
                      dcc.Graph(id='g-conv', config={'displaylogo': False})], lg=6)]),
    html.H4('Comparación pareada (mismas variables, mismas particiones, 3 × 5 pliegues)', className='h6 mt-3'),
    paired_table(),
    html.H4('Pruebas de hipótesis', className='h6 mt-3'), tests_table(),
    dbc.Alert(RES['verdict'] if RES else 'Veredicto pendiente.', color=('success' if RES and RES.get('superior') else 'warning'),
              className='mt-3 mb-0', id='verdict')])], className='mb-3 shadow-sm')

# --------------------------------------------------------------------------------------------- layout
layout = html.Div([
    html.H2('Modelos', className='h3'),
    dbc.Card(dbc.CardBody([
        html.H3('Tabla de líderes: mejores combinaciones del entregable 2', className='h5'),
        html.P('Una fila por modelo: la combinación (balanceo, optimizador) con mayor balanced accuracy (BA) media en los 5 '
               'pliegues externos de la validación anidada. La tabla se puede ordenar y filtrar.', className='small text-muted'),
        dbc.Switch(id='show-all', label='Mostrar las 112 combinaciones', value=False, className='mb-2'),
        leaderboard,
        html.P('Mejores combinaciones del Entregable 2 (9 variables). La comparación con el modelo original usa estos '
               'mismos 7 modelos reentrenados con las variables ampliadas (capítulo 20 del informe; tabla «Comparación '
               'pareada» al final de esta página).', className='small text-muted mt-2 mb-0')]),
        className='mb-3 shadow-sm'),
    dbc.Card(dbc.CardBody([
        dbc.Row([dbc.Col([dbc.Label('Modelo', html_for='model-sel', className='fw-bold'),
                          dcc.Dropdown([{'label': model_label(m), 'value': m} for m in MODELS],
                                       'original' if 'original' in MODELS else MODELS[0], id='model-sel', clearable=False)],
                         md=6),
                 dbc.Col(html.Div(id='model-info', className='small text-muted pt-md-4'), md=6)]),
    ]), className='mb-3 shadow-sm'),
    dbc.Card(dbc.CardBody([
        html.H3('Predicción en vivo', className='h5'),
        html.P('Describa el caso. Los modelos del curso usan las 9 variables del lugar de trabajo; el modelo original usa '
               'además los códigos de la lesión y el empleador. NAICS de 3 dígitos, ocupación SOC menor y horas desde el inicio '
               'se derivan automáticamente.', className='small text-muted'),
        form,
        dbc.Button([html.I(className='bi bi-play-fill me-1', **{'aria-hidden': 'true'}), 'Predecir y explicar'],
                   id='btn-pred', color='primary', className='mt-2'),
        html.Div(id='pred-alert', className='mt-2'),
        dbc.Row([dbc.Col(dcc.Loading(dcc.Graph(id='g-proba', config={'displaylogo': False},
                                               figure=plots.empty('Pulse «Predecir y explicar» para ver la probabilidad.'))), lg=5),
                 dbc.Col(dcc.Loading(dcc.Graph(id='g-shap', config={'displaylogo': False},
                                               figure=plots.empty('Aquí aparecerá la explicación SHAP local.'))), lg=7)])]),
        className='mb-3 shadow-sm'),
    dbc.Card(dbc.CardBody([
        html.H3('Diagnóstico del modelo seleccionado', className='h5'),
        html.P('Calculado con las predicciones fuera de pliegue externo de la validación anidada (30 030 casos); la prueba '
               'de octubre a diciembre no se usa.', className='small text-muted'),
        dbc.Row([dbc.Col(dcc.Graph(id='g-cm'), lg=6), dbc.Col(dcc.Graph(id='g-cal'), lg=6)]),
        dbc.Row([dbc.Col(dcc.Graph(id='g-roc'), lg=6), dbc.Col(dcc.Graph(id='g-pr'), lg=6)]),
        html.P(id='i-diag', className='small text-muted'),
        dcc.Graph(id='g-shapglob'), html.P(id='i-shapglob', className='small text-muted')]), className='mb-3 shadow-sm'),
    dbc.Card(dbc.CardBody([
        html.H3('Comparación estadística', className='h5'),
        html.P('Friedman con post hoc de Nemenyi sobre los 15 pliegues del experimento pareado, con el modelo original '
               'incluido en el universo (métrica: balanced accuracy).', className='small text-muted'),
        dcc.Graph(id='g-cd', figure=(plots.cd_diagram(RES['cd']['names'], np.array(RES['cd']['ranks']), RES['cd']['cd'],
                                                       highlight='Original (SECR)')
                                     if RES else plots.empty('Resultados pendientes'))),
        html.P(RES['cd']['insight'] if RES else '', className='small text-muted')]), className='mb-3 shadow-sm'),
    original_section,
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
    txt = ('Insight: recall por clase ' + ', '.join(f'{c}: {r * 100:.0f} %' for c, r in zip(data.CLASSES, rec)) +
           f'. BA = {np.mean(rec):.3f}. ' + ('«Traslado/restricción» es la clase que más se confunde con «Otro registrable»; '
                                             'las curvas de calibración muestran si las probabilidades pueden leerse como riesgos.'))
    sg = data.shap_global()
    og = data.DATA / 'shap_original_groups.csv'
    if m == 'original' and og.exists():
        s = pd.read_csv(og, index_col=0)['Original (SECR)']
        g_fig = plots.shap_global_bar(s, m)
        g_txt = (f'Insight: por grupos de variables, lo que más mueve la probabilidad del original es «{s.idxmax()}»; el '
                 'mejor benchmark del curso (SVM) no ve los códigos de la lesión (capítulo 20.8 del informe).')
    elif m in getattr(sg, 'index', []):
        s = sg.loc[m]
        g_fig = plots.shap_global_bar(s, m)
        g_txt = f'Insight: la variable que más mueve la probabilidad de la clase predicha es «{data.LABELS[s.idxmax()]}».'
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
