"""El conjunto de prueba (octubre-diciembre) está bloqueado: ningún código nuevo puede usarlo."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from osha_e3 import data as D  # noqa: E402


def test_prueba_bloqueada_por_defecto(monkeypatch):
    monkeypatch.delenv(D.TEST_ENV, raising=False)
    sub, dev, test = D.classification_sets()
    assert isinstance(test, D.LockedTest)
    with pytest.raises(PermissionError):
        len(test)
    with pytest.raises(PermissionError):
        test.y_clf
    with pytest.raises(PermissionError):
        test['y_clf']
    assert not sub.split.eq('test').any() and not dev.split.eq('test').any()


def test_regresion_tambien_bloqueada(monkeypatch):
    monkeypatch.delenv(D.TEST_ENV, raising=False)
    assert isinstance(D.regression_sets()[2], D.LockedTest)
