"""Sección 2: análisis exploratorio con filtros globales que actualizan todas las figuras."""
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dcc, html

from src import data, plots

dash.register_page(__name__, path='/eda', name='EDA', title='EDA · OSHA ITA 2023')

d0 = data.eda()
STATES = data.options('state')
SECTORS = data.options('naics2')
EST = data.options('establishment_type')
CAT_VARS = ['state', 'naics2', 'naics3', 'establishment_type', 'soc_minor', 'type_of_incident', 'incident_hour',
            'hours_since_start', 'nature', 'part', 'event', 'source', 'size']
NUM_VARS = ['days_away', 'days_restricted', 'employees']
CORR_VARS = ['state', 'naics2', 'establishment_type', 'soc_minor', 'type_of_incident', 'hours_since_start',
             'nature', 'part', 'event', 'outcome']
MIN_DATE, MAX_DATE = d0.date.min().date(), d0.date.max().date()


def insight(id_):
    return html.P(id=id_, className='finding')


def figure(gid, iid, lg=6, h=360):
    return dbc.Col(html.Div([dcc.Loading(dcc.Graph(id=gid, config={'displaylogo': False}, style={'minHeight': f'{h}px'}),
                                         type='dot', color='#94a3b8'), insight(iid)], className='panel'),
                   xs=12, lg=lg)


def group(eyebrow, title, desc, *rows):
    return html.Section([html.H2(title, className='section-title'),
                         html.P(desc, className='section-desc'), *rows], className='section')


filters = html.Div(dbc.Row([
    dbc.Col([dbc.Label('Estado', html_for='f-state'),
             dcc.Dropdown(STATES, multi=True, id='f-state', placeholder='Todos')], xs=12, md=6, lg=3),
    dbc.Col([dbc.Label('Sector NAICS (2 díg.)', html_for='f-sector'),
             dcc.Dropdown(SECTORS, multi=True, id='f-sector', placeholder='Todos')], xs=12, md=6, lg=2),
    dbc.Col([dbc.Label('Tipo de establecimiento', html_for='f-est'),
             dcc.Dropdown(EST, multi=True, id='f-est', placeholder='Todos')], xs=12, md=6, lg=2),
    dbc.Col([dbc.Label('Fecha del incidente', html_for='f-date'),
             dcc.DatePickerRange(id='f-date', min_date_allowed=MIN_DATE, max_date_allowed=MAX_DATE, start_date=MIN_DATE,
                                 end_date=MAX_DATE, display_format='DD/MM/YYYY')], xs=12, md=6, lg=3),
    dbc.Col([dbc.Label('Empleados (percentil)', html_for='f-emp'),
             dcc.RangeSlider(0, 100, 10, value=[0, 100], id='f-emp', marks={0: '0', 50: '50', 100: '100'})],
            xs=12, md=6, lg=2),
], className='g-2 align-items-end'), className='filterbar')

layout = html.Div([
    html.Div([
        html.Div('Análisis exploratorio', className='eyebrow'),
        html.H1('Qué muestran los datos antes de modelar', className='page-title'),
        html.P('30 030 casos de desarrollo, de enero a septiembre de 2023. Los filtros actualizan todas las figuras.',
               className='page-lead'),
    ], className='page-head', style={'borderBottom': 'none', 'paddingBottom': '16px'}),
    filters,
    html.Div(id='eda-n', className='count-badge mt-2'),
    group('Desenlace', 'El desenlace y su evolución',
          'El desbalance entre clases define la métrica del proyecto; la estabilidad mensual respalda el corte temporal '
          'de la prueba.',
          dbc.Row([figure('g-target', 'i-target'), figure('g-month', 'i-month')], className='grid-gap')),
    group('Variables', 'Distribución univariada', 'Composición del desenlace en las categorías más frecuentes de la '
          'variable elegida; las numéricas se muestran en escala logarítmica.',
          dbc.Row(dbc.Col([dbc.Label('Variable', html_for='uni-var'),
                           dcc.Dropdown(CAT_VARS + NUM_VARS, 'type_of_incident', id='uni-var', clearable=False,
                                        optionHeight=30)], xs=12, md=4), className='mb-3'),
          html.Div([dcc.Loading(dcc.Graph(id='g-uni', config={'displaylogo': False}), type='dot', color='#94a3b8'),
                    insight('i-uni')], className='panel')),
    group('Asociaciones y calidad', 'Qué variables se relacionan con el desenlace y cuánto falta',
          'V de Cramér entre pares de categóricas (0 = independencia, 1 = asociación perfecta) y proporción de casos sin '
          'dato por variable.',
          dbc.Row([figure('g-corr', 'i-corr', 7, 520), figure('g-miss', 'i-miss', 5, 520)], className='grid-gap'),
          dbc.Row([figure('g-out', 'i-out', 12)], className='grid-gap mt-3')),
    group('Explorador', 'Explorador bivariado', 'Elija una variable para el eje x, la respuesta y el color.',
          dbc.Row([dbc.Col([dbc.Label('Eje x (categórica)', html_for='bx'),
                            dcc.Dropdown(CAT_VARS, 'naics2', id='bx', clearable=False)], xs=12, md=4),
                   dbc.Col([dbc.Label('Eje y / respuesta', html_for='by'),
                            dcc.Dropdown([{'label': 'Desenlace (composición %)', 'value': 'outcome'}] +
                                         [{'label': data.LABELS[c], 'value': c} for c in NUM_VARS], 'outcome', id='by',
                                         clearable=False)], xs=12, md=4),
                   dbc.Col([dbc.Label('Color', html_for='bc'),
                            dcc.Dropdown(['outcome'] + CAT_VARS, 'outcome', id='bc', clearable=False)], xs=12, md=4)],
                  className='g-2 mb-3'),
          html.Div([dcc.Loading(dcc.Graph(id='g-bi', config={'displaylogo': False}), type='dot', color='#94a3b8'),
                    insight('i-bi')], className='panel')),
])


def filtered(states, sectors, est, start, end, emp):
    d = d0
    if states:
        d = d[d.state.isin(states)]
    if sectors:
        d = d[d.naics2.isin(sectors)]
    if est:
        d = d[d.establishment_type.isin(est)]
    if start and end:
        d = d[(d.date >= pd.Timestamp(start)) & (d.date <= pd.Timestamp(end))]
    if emp and tuple(emp) != (0, 100):
        lo, hi = d0.employees.quantile([emp[0] / 100, emp[1] / 100])
        d = d[d.employees.between(lo, hi)]
    return d


FILTERS = [Input('f-state', 'value'), Input('f-sector', 'value'), Input('f-est', 'value'),
           Input('f-date', 'start_date'), Input('f-date', 'end_date'), Input('f-emp', 'value')]


def pct(x):
    nd = 2 if 0 < x < 0.001 else 1
    return f'{x * 100:.{nd}f} %'.replace('.', ',')


@callback(Output('eda-n', 'children'), Output('g-target', 'figure'), Output('i-target', 'children'),
          Output('g-month', 'figure'), Output('i-month', 'children'), Output('g-corr', 'figure'),
          Output('i-corr', 'children'), Output('g-miss', 'figure'), Output('i-miss', 'children'),
          Output('g-out', 'figure'), Output('i-out', 'children'), *FILTERS)
def update_all(states, sectors, est, start, end, emp):
    d = filtered(states, sectors, est, start, end, emp)
    n = f'{len(d):,}'.replace(',', '.')
    if len(d) < 30:
        e = plots.empty('Muy pocos casos con estos filtros (menos de 30).')
        return [f'{n} casos seleccionados.'] + [e, ''] * 5
    share = d.outcome.value_counts(normalize=True).reindex(data.CLASSES).fillna(0)
    maj = share.idxmax()
    i_target = (f'«{maj}» es la clase mayoritaria ({pct(share.max())}); los fallecimientos son '
                f'{int((d.outcome == "Fallecimiento").sum())} casos ({pct(share["Fallecimiento"])}). El desbalance extremo '
                'justifica la balanced accuracy como métrica principal y las técnicas de balanceo.')
    m = pd.crosstab(d.date.dt.month, d.outcome, normalize='index').reindex(columns=data.CLASSES).fillna(0)
    i_month = (f'La participación de «Días de ausencia» varía entre {pct(m["Días de ausencia"].min())} y '
               f'{pct(m["Días de ausencia"].max())} según el mes: la composición es estable en el tiempo, por eso un '
               'corte temporal de prueba (oct–dic) es razonable.')
    cols = CORR_VARS
    fig_corr = plots.cramer_heatmap(d, cols)
    v = {c: plots.cramers_v(d[c], d.outcome) for c in cols if c != 'outcome'}
    top = sorted(v, key=v.get, reverse=True)[:3]
    i_corr = ('Las variables más asociadas al desenlace son ' +
              ', '.join(f'{data.LABELS.get(c, c)} (V = {v[c]:.2f})'.replace('.', ',') for c in top) +
              '. Los códigos de la lesión (OIICS) aportan información que las 9 variables del lugar de trabajo no tienen.')
    miss_cols = ['soc_minor', 'start_hour', 'incident_hour', 'hours_since_start', 'nature', 'part', 'event', 'source',
                 'naics3', 'state']
    mrate = {c: float(d[c].astype(str).eq('Unknown').mean()) for c in miss_cols}
    worst = max(mrate, key=mrate.get)
    i_miss = (f'«{data.LABELS[worst]}» es la variable con más faltantes ({pct(mrate[worst])}). Los faltantes '
              'se modelan como la categoría «Unknown», porque no faltan al azar (dependen del empleador).')
    da = d.days_away[d.days_away > 0]
    i_out = (f'La mediana de días de ausencia es {da.median():.0f} y el percentil 99 es {da.quantile(.99):.0f}; '
             'la cola larga (topada en 180 días por OSHA) motiva el logaritmo en la tarea de regresión.') if len(da) else ''
    return ([f'{n} casos con los filtros actuales', plots.target_bar(d), i_target, plots.monthly(d), i_month,
             fig_corr, i_corr, plots.missing_bar(d, miss_cols), i_miss, plots.outlier_box(d), i_out])


@callback(Output('g-uni', 'figure'), Output('i-uni', 'children'), Input('uni-var', 'value'), *FILTERS)
def update_uni(var, *f):
    d = filtered(*f)
    if len(d) < 30:
        return plots.empty(), ''
    fig = plots.hist_feature(d, var)
    if var in NUM_VARS:
        v = d[var].dropna()
        return fig, (f'{pct((v > 0).mean())} de los casos tiene valor positivo; la distribución es muy '
                     'asimétrica, por eso se muestra en escala logarítmica.')
    top = d[var].value_counts()
    g = d[d[var] == top.index[0]].outcome.eq('Días de ausencia').mean()
    return fig, (f'La categoría más frecuente es «{top.index[0]}» ({pct(top.iloc[0] / len(d))} de los casos), '
                 f'con {pct(g)} de casos con días de ausencia frente a {pct(d.outcome.eq("Días de ausencia").mean())} en total.')


@callback(Output('g-bi', 'figure'), Output('i-bi', 'children'), Input('bx', 'value'), Input('by', 'value'),
          Input('bc', 'value'), *FILTERS)
def update_bi(x, y, color, *f):
    d = filtered(*f)
    if len(d) < 30:
        return plots.empty(), ''
    fig = plots.bivariate(d, x, y, color)
    if y == 'outcome':
        r = d.groupby(x).outcome.apply(lambda s: s.eq('Días de ausencia').mean())
        r = r[d[x].value_counts().reindex(r.index) >= 50]
        txt = (f'Entre categorías con al menos 50 casos, la proporción de días de ausencia va de '
               f'{pct(r.min())} («{r.idxmin()}») a {pct(r.max())} («{r.idxmax()}»).') if len(r) else ''
    else:
        med = d.groupby(x)[y].median().dropna()
        txt = (f'La mediana de {data.LABELS[y].lower()} cambia de {med.min():.0f} a {med.max():.0f} según '
               f'{data.LABELS.get(x, x).lower()}.') if len(med) else ''
    return fig, txt
