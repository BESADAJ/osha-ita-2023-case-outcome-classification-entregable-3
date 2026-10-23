"""Completa ``original.joblib`` ya entrenado: fuerza predicción determinista (``make_deterministic``), repite la prueba de
ida y vuelta exacta y escribe ``original.metadata.json``. El ajuste (scripts/build_artifacts.py original) tardó ~2 h con la
CPU compartida con el experimento pareado (23-oct-2026, 00:12–02:10)."""
import time

import _bootstrap  # noqa: F401
import joblib
from threadpoolctl import threadpool_limits

from build_artifacts import ART, roundtrip_rows, save, versions
from osha_e3 import config as C, paired as PR

threadpool_limits(1)
est = joblib.load(ART / 'original.joblib')
est.make_deterministic()
X_rt = roundtrip_rows(PR.VPLUS)
t0 = time.perf_counter()
est.predict_proba(X_rt)
pred_ms = (time.perf_counter() - t0) / len(X_rt) * 1000
meta = {'model': 'original (ensamble multivista + SECR)', 'seed': C.SEED, 'hyperparameters': est.params_,
        'inner_score_ba': est.inner_score_, 'base_models': est.stack_.names_,
        'weights_global': dict(zip(est.stack_.names_, est.combiner_.weights_global_.round(4).tolist())),
        'n_regions_with_own_weights': len(est.combiner_.weights_region_),
        'trained_on': '30030 casos de la submuestra de desarrollo (ene-sep 2023)',
        'fit_seconds': 'aprox. 7100 (CPU compartida; ver docstring)', 'inference_ms_per_case': pred_ms,
        'versions': versions(), 'input_schema': None, 'classes': C.CLASS_NAMES,
        'outer_metrics': 'ver results/pareado/resumen.json (validación anidada repetida)'}
X, y, g = PR.paired_data()
from build_artifacts import schema  # noqa: E402
meta['input_schema'] = schema(X, PR.VPLUS)
save('original', est, meta, X_rt, PR.VPLUS)
