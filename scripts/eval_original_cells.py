"""Evalúa (TPE anidado + SECR + ablación) las celdas del original que ya existan; reanudable. Uso: python scripts/eval_original_cells.py"""
import json
import _bootstrap  # noqa: F401
from threadpoolctl import threadpool_limits
from analyze_paired import _eval_cell, OE, PR

threadpool_limits(1)
for c in PR.load('original'):
    if not (OE.OUT / f"r{c['rep']}_f{c['fold']}.json").exists():
        _eval_cell(c['rep'], c['fold'])
        r = json.loads((OE.OUT / f"r{c['rep']}_f{c['fold']}.json").read_text(encoding='utf-8'))
        print(c['rep'], c['fold'], r['params'], {k: round(v['recall'], 4) for k, v in r['metrics'].items()}, flush=True)
