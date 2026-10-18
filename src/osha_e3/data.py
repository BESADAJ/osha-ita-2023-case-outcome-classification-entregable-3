"""Carga, partición y preparación de variables.

La partición es exactamente la del entregable 1: 60 % de los establecimientos (semilla 42) y
octubre-diciembre para la prueba; el resto, hasta el 23 de septiembre (purga de 7 días), para el
desarrollo. La preparación de variables es determinista (no aprende estadísticas), por eso puede
aplicarse antes de la validación cruzada sin fuga de información; todo lo que aprende de los datos
(agrupar categorías raras, one-hot, balanceo, modelo) vive dentro del ``Pipeline``.
"""
import numpy as np
import pandas as pd

from . import config as C

RAW_FEATURES = ['state', 'naics_code', 'establishment_type', 'soc_code', 'type_of_incident',
                'time_unknown', 'time_started_work', 'time_of_incident']
EXTRA = ['id', 'establishment_id', 'ein', 'incident_outcome', 'date_of_incident',
         'dafw_num_away', 'djtr_num_tr']

# Variables que entran al modelo después de ``prepare_features``.
FEATURES = ['state', 'naics3', 'establishment_type', 'soc_minor', 'type_of_incident',
            'time_unknown', 'start_hour', 'incident_hour', 'hours_since_start']


def _cache_file():
    return C.RESULTS / 'cache' / 'osha_2023_prepared.parquet'


def load_raw(path=None):
    """Lee las columnas necesarias del CSV crudo y conserva solo incidentes de 2023.

    Returns
    -------
    pandas.DataFrame
        Una fila por caso, columnas de texto más ``date`` (datetime).
    """
    path = path or C.data_path()
    d = pd.read_csv(path, usecols=RAW_FEATURES + EXTRA, dtype=str, encoding='latin1')
    d['date'] = pd.to_datetime(d.date_of_incident, format='%m/%d/%Y', errors='coerce')
    d = d.loc[d.date.dt.year.eq(2023)].copy()
    assert d.establishment_id.notna().all() and d.id.is_unique
    assert d.incident_outcome.isin(['1', '2', '3', '4']).all()
    return d


def assign_split(d):
    """Etiqueta cada caso como ``development``, ``test`` o ``excluded`` (diseño del entregable 1)."""
    ordered = np.asarray(sorted(pd.unique(d.establishment_id)), dtype=object)
    selected = np.random.default_rng(C.SEED).permutation(ordered)
    held = set(selected[:int(np.floor(len(ordered) * C.HOLDOUT_GROUP_FRACTION))])
    is_held = d.establishment_id.isin(held)
    cutoff = pd.Timestamp(C.CUTOFF)
    dev = ~is_held & d.date.lt(cutoff - pd.Timedelta(days=C.GAP_DAYS))
    test = is_held & d.date.ge(cutoff)
    split = pd.Series('excluded', index=d.index, dtype=object)
    split[dev] = 'development'
    split[test] = 'test'
    return split


def _hour(series):
    h = pd.to_numeric(series.str.extract(r'^(\d{1,2}):', expand=False), errors='coerce')
    return h.where(h.between(0, 23))


def prepare_features(raw):
    """Construye las 9 variables categóricas del modelo a partir de las 8 originales.

    Cambios frente a los entregables 1 y 2 (siguientes pasos que ellos dejaron anotados):

    * ``naics3``: subsector NAICS de 3 dígitos (≈ 90 categorías) en lugar de 715 códigos.
    * ``soc_minor``: grupo menor SOC (``'51-2'`` de ``'51-2092'``, ≈ 95 categorías) en lugar de 598.
    * ``hours_since_start``: horas entre el inicio de la jornada y el incidente (mod 24), en tramos.

    Las horas siguen como categorías (el EDA mostró que no ordenan el resultado de forma monótona).
    Los faltantes quedan como la categoría ``'Unknown'``.
    """
    d = raw[RAW_FEATURES].replace(r'^\s*$', np.nan, regex=True)
    out = pd.DataFrame(index=raw.index)
    out['state'] = d.state
    out['naics3'] = d.naics_code.str[:3]
    out['establishment_type'] = d.establishment_type
    soc = d.soc_code.where(~d.soc_code.isin(['9999', '0000']))
    out['soc_minor'] = soc.str[:4]
    out['type_of_incident'] = d.type_of_incident
    out['time_unknown'] = d.time_unknown
    start = _hour(d.time_started_work)
    inc = _hour(d.time_of_incident).where(~d.time_unknown.eq('1'))
    out['start_hour'] = start
    out['incident_hour'] = inc
    elapsed = (inc - start) % 24
    out['hours_since_start'] = pd.cut(elapsed, [-0.5, 1.5, 3.5, 5.5, 7.5, 9.5, 12.5, 24],
                                      labels=['0-1', '2-3', '4-5', '6-7', '8-9', '10-12', '13+'])
    for c in ['start_hour', 'incident_hour']:
        out[c] = out[c].map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    out = out.astype(object).where(out.notna(), 'Unknown').astype(str)
    return out[FEATURES]


def load_prepared(refresh=False):
    """Datos preparados con partición, objetivo de clasificación y de regresión (con caché parquet).

    Columnas: ``FEATURES`` + ``id, establishment_id, ein, date, split, y_clf`` (1-4) y ``y_reg``
    (``log1p`` de los días de ausencia; solo definida en casos con resultado «días de ausencia»).
    """
    path = _cache_file()
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    raw = load_raw()
    df = prepare_features(raw)
    df['id'] = raw.id
    df['establishment_id'] = raw.establishment_id
    df['ein'] = raw.ein.fillna('')
    df['date'] = raw.date
    df['split'] = assign_split(raw)
    df['y_clf'] = raw.incident_outcome.astype(int)
    days = pd.to_numeric(raw.dafw_num_away, errors='coerce')
    df['days_away'] = days
    df['y_reg'] = np.where(df.y_clf.eq(2) & days.gt(0), np.log1p(days), np.nan)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False)
    return df.reset_index(drop=True)


def subsample_groups(df, n_rows, seed):
    """Submuestra de establecimientos completos hasta acumular ~``n_rows`` casos.

    Muestrear establecimientos (y no casos) conserva la estructura agrupada que la validación por
    grupos necesita y no deja casos de un mismo establecimiento a ambos lados de un pliegue.
    """
    groups = np.sort(df.establishment_id.unique())
    order = np.random.default_rng(seed).permutation(groups)
    sizes = df.groupby('establishment_id').size().reindex(order).to_numpy()
    keep = order[:int(np.searchsorted(np.cumsum(sizes), n_rows)) + 1]
    return df.loc[df.establishment_id.isin(keep)].reset_index(drop=True)


def classification_sets(df=None):
    """(desarrollo submuestreado, desarrollo completo, prueba) para clasificación."""
    df = load_prepared() if df is None else df
    dev = df.loc[df.split.eq('development')].reset_index(drop=True)
    test = df.loc[df.split.eq('test')].sort_values('date', kind='stable').reset_index(drop=True)
    sub = subsample_groups(dev, C.N_SUB_CLF, C.seed_for('subsample'))
    return sub, dev, test


def regression_sets(df=None):
    """(desarrollo submuestreado, desarrollo completo, prueba) para regresión (días de ausencia)."""
    df = load_prepared() if df is None else df
    df = df.loc[df.y_reg.notna()]
    dev = df.loc[df.split.eq('development')].reset_index(drop=True)
    test = df.loc[df.split.eq('test')].sort_values('date', kind='stable').reset_index(drop=True)
    sub = subsample_groups(dev, C.N_SUB_REG, C.seed_for('subsample', 1))
    return sub, dev, test
