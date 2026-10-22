"""Pruebas del ensamble multivista (``src/osha_e3/ensemble.py``).

Se ejecutan con ``pytest tests/test_ensemble.py`` o, sin pytest, con ``python tests/test_ensemble.py``.
Usan una muestra pequeña del desarrollo para que corran en segundos.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sklearn.base import clone  # noqa: E402

from osha_e3 import data as D, ensemble as ENS, nested as N  # noqa: E402


def _sample(n=3000, seed=0):
    sub = D.classification_sets()[0]
    rng = np.random.default_rng(seed)
    groups = rng.permutation(sub.establishment_id.unique())
    keep = sub.establishment_id.isin(set(groups[: max(1, int(len(groups) * n / len(sub)))]))
    X = ENS.add_full_codes(sub.loc[keep].reset_index(drop=True))
    return X, X.y_clf.to_numpy() - 1, X.establishment_id.to_numpy()


def _small_views():
    return [('curso_logreg', ENS.ViewModel('course:logreg', params={'C': 0.1, 'penalty': 'l2'})),
            ('finos', ENS.ViewModel('te_lr', ['naics6', 'soc6', 'nature'], {'C': 0.3}))]


def test_i_codificacion_fuera_de_pliegue_no_usa_su_etiqueta():
    """Cada fila se codifica solo con filas de *otros* establecimientos (pliegues por grupo).

    Se reproducen los mismos pliegues del codificador y se verifica que la codificación de cada fila
    coincide con la calculada usando solo las filas fuera de su pliegue, y que su establecimiento nunca
    está entre ellas.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    X, y, g = _sample()
    enc = ENS.MulticlassTargetEncoder(['naics6', 'soc6'], n_splits=5, seed=1)
    Z = enc.fit_transform(X, y, g)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=1)
    for a, b in cv.split(X, y, g):
        assert not set(g[a]) & set(g[b]), 'un establecimiento aparece en las dos partes del pliegue'
        prior, tables = enc._stats(X.iloc[a], y[a])
        assert np.allclose(Z[b], enc._apply(X.iloc[b], prior, tables))


def test_ii_reproducible_con_semilla():
    X, y, g = _sample()
    p1 = ENS.StackedEnsemble(_small_views(), n_splits=3, seed=7).fit(X, y, g).predict_proba(X.head(200))
    p2 = ENS.StackedEnsemble(_small_views(), n_splits=3, seed=7).fit(X, y, g).predict_proba(X.head(200))
    assert np.allclose(p1, p2)
    assert np.allclose(p1.sum(axis=1), 1)


def test_iii_compatible_con_clone():
    est = ENS.StackedEnsemble(_small_views(), meta_C=0.5, n_splits=3, seed=3)
    c = clone(est)
    assert c.get_params()['meta_C'] == 0.5 and not hasattr(c, 'meta_')


def test_iv_meta_modelo_entrenado_con_predicciones_fuera_de_pliegue():
    """Las predicciones fuera de pliegue del nivel 0 difieren de las del modelo ajustado con todo (no se
    reutilizan predicciones de entrenamiento) y cubren todas las filas."""
    X, y, g = _sample()
    ens = ENS.StackedEnsemble(_small_views(), n_splits=3, seed=5).fit(X, y, g)
    for n in ens.names_:
        oof = ens.oof_[n]
        assert oof.shape == (len(y), 4) and (oof.sum(axis=1) > 0).all()
        full = ens.models_[n].predict_proba(X)
        assert not np.allclose(oof, full)
    n_classes = len(np.unique(y))
    assert ens.meta_.coef_.shape == (n_classes if n_classes > 2 else 1, 4 * len(ens.names_))
    assert ens.predict_proba(X.head(50)).shape == (50, 4)  # siempre 4 columnas


def test_v_entradas_invalidas():
    X, y, g = _sample()
    ens = ENS.StackedEnsemble(_small_views(), n_splits=3).fit(X, y, g)
    with pytest.raises(KeyError):
        ens.predict_proba(X.drop(columns=['naics6']))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    ok = 0
    for t in tests:
        try:
            t()
            ok += 1
            print('OK  ', t.__name__)
        except Exception as exc:  # noqa: BLE001
            print('FALLA', t.__name__, type(exc).__name__, exc)
    print(f'{ok} de {len(tests)} pruebas pasan')
