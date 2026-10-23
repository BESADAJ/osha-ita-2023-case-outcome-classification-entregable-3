"""Clase del modelo original (guía §7.2). La implementación vive en ``osha_e3`` (copia de ``src/osha_e3`` del
repositorio) para que ``joblib`` encuentre las clases al cargar ``artifacts/original.joblib``:

* ``osha_e3.combinador.ConditionalEnsembleSelection``: combinador SECR (Frank-Wolfe por región con contracción).
* ``osha_e3.original_final.OriginalModel``: ensamble multivista de 16 modelos base + SECR.
"""
from osha_e3.combinador import ConditionalEnsembleSelection, frank_wolfe  # noqa: F401
from osha_e3.original_final import OriginalModel  # noqa: F401
