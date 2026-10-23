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

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
                title='OSHA ITA 2023 · Desenlace de casos', suppress_callback_exceptions=True)
server = app.server

data.eda()  # carga y deja en caché los datos al iniciar

navbar = dbc.Navbar(dbc.Container([
    dbc.NavbarBrand([html.I(className='bi bi-shield-plus me-2', **{'aria-hidden': 'true'}),
                     'OSHA ITA 2023 · ¿Cómo termina un caso de lesión laboral?'], href='/'),
    dbc.Nav([dbc.NavItem(dbc.NavLink('Contexto', href='/', active='exact')),
             dbc.NavItem(dbc.NavLink('EDA', href='/eda', active='exact')),
             dbc.NavItem(dbc.NavLink('Modelos', href='/modelos', active='exact'))], navbar=True, className='ms-auto'),
], fluid=True), color='primary', dark=True, sticky='top', className='mb-3')

footer = html.Footer(dbc.Container(html.Small(
    'Jaime Andrés Besada e Iván Berrio · Machine Learning (EST 1055-01) · Datos: OSHA Injury Tracking Application 2023 '
    '· Versión 3.0, octubre de 2026', className='text-muted')), className='py-3 mt-4 border-top')

app.layout = html.Div([navbar, dbc.Container(dash.page_container, fluid=True, className='px-3 px-md-4'), footer])

if __name__ == '__main__':
    app.run(debug=False, port=8050)
