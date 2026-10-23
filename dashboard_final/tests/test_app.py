"""Pruebas de humo del dashboard: las páginas responden, los callbacks no fallan y la predicción valida entradas."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402
from pages import eda, modelos  # noqa: E402
from src import data  # noqa: E402
from src.explain import exact_shapley  # noqa: E402

VALS = [modelos.DEFAULTS.get(c) for c in modelos.FORM_FIELDS] + ['']


def test_paginas_responden():
    c = app.server.test_client()
    for path in ['/', '/eda', '/modelos', '/_dash-layout', '/_dash-dependencies']:
        assert c.get(path).status_code == 200


def test_filtros_eda_actualizan_todas_las_figuras():
    out = eda.update_all(['CA'], None, None, str(eda.MIN_DATE), str(eda.MAX_DATE), [10, 90])
    assert len(out) == 11 and 'casos' in out[0]
    vacio = eda.update_all(['ZZ'], None, None, None, None, [0, 100])
    assert 'Muy pocos' in vacio[1].layout.annotations[0].text


def test_prediccion_y_shap_eficiencia():
    m = data.available_models()[0]
    fig_p, fig_s, alert = modelos.predict(1, m, *VALS)
    assert 'Predicción' in str(alert)
    est = data.model('tree')
    row = modelos.build_row(VALS)[data.FEATURES]
    bg = modelos.background_rows()[data.FEATURES]
    cls = int(est.predict_proba(row).argmax())
    phi = exact_shapley(est.predict_proba, row, bg, {c: [c] for c in data.FEATURES}, cls)
    assert np.isclose(phi.sum(), phi.attrs['full'] - phi.attrs['base'])


def test_entradas_invalidas_no_rompen_la_app():
    vals = list(VALS)
    vals[0] = None
    assert 'Complete' in str(modelos.predict(1, 'tree', *vals)[2])
    vals = list(VALS)
    vals[-1] = '<script>'
    assert 'no es válido' in str(modelos.predict(1, 'tree', *vals)[2])
