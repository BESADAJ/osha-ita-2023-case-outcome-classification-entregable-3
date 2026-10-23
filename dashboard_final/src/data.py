"""Carga y caché de datos y modelos del dashboard. Todo se lee una sola vez al iniciar el servidor.

Las rutas salen de la carpeta del proyecto (``DASH_ROOT`` puede cambiarla); no hay rutas absolutas.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('DASH_ROOT', Path(__file__).resolve().parents[1]))
DATA, ART = ROOT / 'data', ROOT / 'artifacts'

CLASSES = ['Fallecimiento', 'Días de ausencia', 'Traslado/restricción', 'Otro registrable']
MODELS = ['svm', 'tree', 'logreg', 'xgboost', 'knn', 'random_forest', 'naive_bayes']
PRETTY = {'knn': 'KNN', 'naive_bayes': 'Naive Bayes', 'logreg': 'Regresión logística', 'tree': 'Árbol de decisión',
          'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'svm': 'SVM lineal',
          'original': 'Modelo original (multivista + SECR)', 'none': 'Sin balanceo', 'smote': 'SMOTE',
          'adasyn': 'ADASYN', 'class_weight': 'class_weight', 'grid': 'Grid', 'random': 'Random',
          'bayes': 'Bayesiana (TPE)', 'genetic': 'Genética'}
FEATURES = ['state', 'naics3', 'establishment_type', 'soc_minor', 'type_of_incident', 'time_unknown', 'start_hour',
            'incident_hour', 'hours_since_start']
EXTRA = ['nature', 'part', 'event', 'source', 'sec_source', 'naics6', 'soc6', 'ein', 'company']
LABELS = {'state': 'Estado', 'naics3': 'Subsector NAICS (3 díg.)', 'establishment_type': 'Tipo de establecimiento',
          'soc_minor': 'Ocupación SOC (grupo menor)', 'type_of_incident': 'Tipo de incidente',
          'time_unknown': 'Hora desconocida', 'start_hour': 'Hora de inicio de jornada',
          'incident_hour': 'Hora del incidente', 'hours_since_start': 'Horas desde el inicio',
          'nature': 'Naturaleza de la lesión (OIICS)', 'part': 'Parte del cuerpo (OIICS)',
          'event': 'Evento (OIICS)', 'source': 'Fuente (OIICS)', 'sec_source': 'Fuente secundaria (OIICS)',
          'naics6': 'NAICS (6 díg.)', 'soc6': 'Ocupación SOC (6 díg.)', 'ein': 'EIN del empleador',
          'company': 'Empresa', 'naics2': 'Sector NAICS (2 díg.)', 'outcome': 'Desenlace del caso',
          'days_away': 'Días de ausencia', 'days_restricted': 'Días de traslado/restricción',
          'employees': 'Empleados promedio', 'size': 'Tamaño del establecimiento'}
TYPE_OF_INCIDENT = {'1': 'Lesión', '2': 'Trastorno de la piel', '3': 'Condición respiratoria', '4': 'Envenenamiento',
                    '5': 'Pérdida auditiva', '6': 'Otra enfermedad'}


@lru_cache(maxsize=1)
def eda():
    d = pd.read_csv(DATA / 'eda.csv.gz', dtype=str)
    for c in ['days_away', 'days_restricted', 'employees']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['date'] = pd.to_datetime(d['date'])
    d['month'] = d.date.dt.month
    return d


@lru_cache(maxsize=1)
def leaderboard():
    return pd.read_csv(DATA / 'leaderboard.csv')


@lru_cache(maxsize=None)
def metadata(name):
    return json.loads((ART / f'{name}.metadata.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=None)
def model(name):
    """Artefacto ya entrenado (el dashboard nunca entrena ni optimiza)."""
    return joblib.load(ART / f'{name}.joblib')


def available_models():
    return [m for m in MODELS + ['original'] if (ART / f'{m}.joblib').exists()]


@lru_cache(maxsize=None)
def oof(name):
    p = DATA / f'oof_{name}.csv.gz'
    return pd.read_csv(p) if p.exists() else None


@lru_cache(maxsize=1)
def shap_global():
    p = DATA / 'shap_global.csv'
    return pd.read_csv(p, index_col=0) if p.exists() else None


@lru_cache(maxsize=1)
def original_results():
    p = DATA / 'original_results.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


@lru_cache(maxsize=1)
def background(n=12, seed=42):
    """Fondo fijo para SHAP (casos del desarrollo)."""
    d = eda()
    return d.sample(n, random_state=seed).reset_index(drop=True)


def options(col, top=None):
    """Valores observados de ``col`` en el desarrollo (los ``top`` más frecuentes, o todos ordenados)."""
    vc = eda()[col].value_counts()
    return vc.index.tolist()[:top] if top else sorted(vc.index.tolist())


def fmt(x, nd=3):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{nd}f}'.replace('.', ',')
