"""Construye ``results/cache/osha_2023_fullcodes.parquet``: códigos completos, OIICS y empleador por caso.

``ensemble.add_full_codes`` lo lee (capítulos 18 a 20, experimento pareado, pruebas de ``tests/test_ensemble.py``). No se
versiona porque conserva el EIN y el nombre de la empresa; se regenera desde el CSV crudo de OSHA (``OSHA_DATA_DIR``).

Reglas (las mismas del archivo usado en las corridas; ``--check`` verifica igualdad exacta con un archivo existente):

* casos de 2023 (igual que ``data.load_raw``);
* ``naics6`` = ``naics_code``; ``soc6`` = ``soc_code`` con 9999/0000 como «Unknown»;
* ``nature, part, event, source, sec_source`` = ``*_code_pred`` (códigos OIICS asignados por OSHA al relato);
* ``ein`` = EIN; ``company`` = nombre de la empresa en mayúsculas y sin espacios sobrantes;
* faltantes y cadenas vacías como «Unknown».

Uso: python scripts/build_fullcodes.py [--check]
"""
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from osha_e3 import config as C

OUT = C.RESULTS / 'cache' / 'osha_2023_fullcodes.parquet'
SRC = {'naics6': 'naics_code', 'soc6': 'soc_code', 'nature': 'nature_code_pred', 'part': 'part_code_pred',
       'event': 'event_code_pred', 'source': 'source_code_pred', 'sec_source': 'sec_source_code_pred',
       'ein': 'ein', 'company': 'company_name'}


def build():
    raw = pd.read_csv(C.data_path(), usecols=['id', 'date_of_incident'] + list(SRC.values()), dtype=str,
                      encoding='latin1')
    raw = raw.loc[pd.to_datetime(raw.date_of_incident, format='%m/%d/%Y', errors='coerce').dt.year.eq(2023)]
    out = pd.DataFrame({'id': raw['id']})
    for new, old in SRC.items():
        col = raw[old].str.strip().replace('', np.nan)
        if new == 'soc6':
            col = col.where(~col.isin(['9999', '0000']))
        if new == 'company':
            col = col.str.upper()
        out[new] = col.fillna('Unknown')
    return out.reset_index(drop=True)


if __name__ == '__main__':
    new = build()
    if '--check' in sys.argv and OUT.exists():
        old = pd.read_parquet(OUT)
        m = old.merge(new, on='id', suffixes=('_old', ''), how='outer', indicator=True)
        print('filas:', len(old), len(new), '| solo en uno:', int((m._merge != 'both').sum()))
        for c in SRC:
            print(c, 'diferencias:', int((m[c + '_old'] != m[c]).sum()))
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        new.to_parquet(OUT, index=False)
        print('escrito', OUT, new.shape)
