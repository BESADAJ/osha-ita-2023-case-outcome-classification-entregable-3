"""Punto de entrada del dashboard (Dash multipágina). Expone ``server`` para gunicorn en Render.

Páginas: Contexto (/), EDA (/eda) y Modelos (/modelos). Los datos y modelos se cargan una sola vez y el
servidor nunca entrena ni optimiza: solo lee artefactos de ``artifacts/``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # paquete osha_e3 (clases necesarias para cargar los artefactos) y src/

import dash  # noqa: E402
import dash_bootstrap_components as dbc  # noqa: E402
from dash import html  # noqa: E402

from src import data  # noqa: E402

FONTS = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap'
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP, FONTS],
                title='OSHA ITA 2023 · Desenlace de casos', suppress_callback_exceptions=True)
server = app.server

data.eda()  # carga y deja en caché los datos al iniciar

topbar = html.Header(dbc.Container(html.Div([
    html.A([html.Span(className='mark', **{'aria-hidden': 'true'}), 'OSHA ITA 2023',
            html.Span('Desenlace de casos de lesión laboral', className='sub')], href='/', className='brand'),
    dbc.Nav([dbc.NavLink('Contexto', href='/', active='exact'),
             dbc.NavLink('EDA', href='/eda', active='exact'),
             dbc.NavLink('Modelos', href='/modelos', active='exact')], className='mainnav'),
], className='inner'), fluid='xxl', className='px-3 px-md-4'), className='topbar')

footer = html.Footer(dbc.Container(html.Div([
    html.Span('Iván Berrio y Jaime Andrés Besada · Machine Learning (EST 1055-01), Universidad del Norte · '),
    html.Span('Datos: OSHA Injury Tracking Application 2023 · Versión 3.1, octubre de 2026 · '),
    html.A('Informe (Jupyter Book)', href='https://besadaj.github.io/osha-ita-2023-case-outcome-classification-entregable-3/',
           target='_blank')]), fluid='xxl', className='px-3 px-md-4'), className='footer')

app.layout = html.Div([topbar, dbc.Container(dash.page_container, fluid='xxl', className='px-3 px-md-4'), footer])

if __name__ == '__main__':
    app.run(debug=False, port=8050)
