"""Caché en disco para cálculos costosos de los capítulos del libro."""
import joblib

from . import config as C


def cached(key, fn, refresh=False):
    """Devuelve ``fn()`` guardado en ``results/cache/<key>.joblib`` (lo calcula la primera vez).

    Parameters
    ----------
    key : str
        Nombre del archivo de caché.
    fn : callable
        Función sin argumentos que produce el resultado.
    refresh : bool
        Si es ``True`` recalcula aunque exista la caché.
    """
    path = C.RESULTS / 'cache' / f'{key}.joblib'
    if path.exists() and not refresh:
        return joblib.load(path)
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, path, compress=3)
    return out
