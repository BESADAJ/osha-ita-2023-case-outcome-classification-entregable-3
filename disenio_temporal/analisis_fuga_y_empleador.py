"""Análisis descriptivo para la sustentación (solo datos de desarrollo; la prueba oct–dic no se toca).

1. Fuga directa: dafw_num_away, djtr_num_tr y date_of_death frente a incident_outcome.
2. Fuga en las narrativas: frecuencia, por clase, de expresiones que nombran el desenlace
   (muerte, días de ausencia, trabajo restringido o traslado, regreso al trabajo).
3. Cuánto informan las variables *_pred (codificadas a partir de la narrativa) frente a las 8 del diseño.
4. Efecto del empleador: entre casos de clase 2 (días de ausencia) y 3 (traslado o restricción), ¿varía
   la proporción de clase 3 entre establecimientos más de lo que explican la industria y la lesión?
5. Momento de carga: created_timestamp frente a la fecha del incidente.

python analisis_fuga_y_empleador.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from osha3.data import DATA_FILE, RAW_FEATURES, assign_test_groups, CUTOFF, GAP_DAYS

OUT = Path(__file__).resolve().parent / 'results' / 'original_justificacion'
NARR = ['NEW_NAR_WHAT_HAPPENED', 'NEW_NAR_BEFORE_INCIDENT', 'NEW_INCIDENT_LOCATION', 'NEW_NAR_INJURY_ILLNESS',
        'NEW_NAR_OBJECT_SUBSTANCE', 'NEW_INCIDENT_DESCRIPTION']
PRED = ['nature_title_pred', 'part_title_pred', 'event_title_pred', 'source_title_pred']
PATTERNS = {
    'muerte': r'\b(died|dies|death|deceased|fatal(?:ity|ly)?|passed away|pronounced dead|expired)\b',
    'dias_de_ausencia': r'\b(days? away|lost time|lost work|off work|out of work|missed (?:work|\d+ days?)|time off'
                        r'|days? off|unable to (?:work|return)|away from work)\b',
    'restriccion_o_traslado': r'\b(light duty|restricted (?:duty|work|activity)|restrictions?|modified (?:duty|work)'
                              r'|transitional duty|limited duty|alternate (?:duty|work)|job transfer|transferred to)\b',
    'regreso_al_trabajo': r'\b(returned to work|return(?:ed)? to (?:full|regular) duty|back to work'
                          r'|released to (?:full|regular)|full duty)\b',
}
CLASSES = {'1': 'Fallecimiento', '2': 'Días de ausencia', '3': 'Traslado/restricción', '4': 'Otro'}


def load_dev():
    cols = (['id', 'establishment_id', 'incident_outcome', 'date_of_incident', 'dafw_num_away', 'djtr_num_tr',
             'date_of_death', 'created_timestamp', 'size'] + RAW_FEATURES + NARR + PRED)
    d = pd.read_csv(DATA_FILE, usecols=sorted(set(cols)), dtype=str, encoding='latin1')
    d['date'] = pd.to_datetime(d.date_of_incident, format='%m/%d/%Y', errors='coerce')
    d = d.loc[d.date.dt.year.eq(2023)].copy()
    held, _ = assign_test_groups(d.establishment_id)
    dev = d.loc[~d.establishment_id.isin(held) & d.date.lt(CUTOFF - pd.Timedelta(days=GAP_DAYS))].copy()
    assert len(dev) == 269516 and int(dev.incident_outcome.eq('1').sum()) == 69, 'partición distinta'
    return dev


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load_dev()
    lines = ['# Fuga de información y efecto del empleador (datos de desarrollo: 269.516 casos, 69 muertes)', '']

    # 1. Fuga directa.
    dafw = pd.to_numeric(dev.dafw_num_away, errors='coerce').fillna(0) > 0
    djtr = pd.to_numeric(dev.djtr_num_tr, errors='coerce').fillna(0) > 0
    death = dev.date_of_death.notna()
    t1 = pd.DataFrame({'dafw>0': dafw, 'djtr>0': djtr, 'date_of_death': death}).groupby(dev.incident_outcome).mean()
    t1.index = [CLASSES[i] for i in t1.index]
    t1.to_csv(OUT / 'fuga_directa.csv')
    lines += ['## 1. Fuga directa (fracción de casos de cada clase con la variable presente)', '',
              t1.round(3).to_markdown(), '']

    # 2. Narrativas.
    text = dev[NARR].fillna('').agg(' '.join, axis=1).str.lower()
    hits = pd.DataFrame({k: text.str.contains(p, regex=True) for k, p in PATTERNS.items()})
    t2 = hits.groupby(dev.incident_outcome).mean()
    t2.index = [CLASSES[i] for i in t2.index]
    t2['cualquiera'] = hits.any(axis=1).groupby(dev.incident_outcome).mean().values
    t2.to_csv(OUT / 'fuga_narrativas.csv')
    examples = []
    for k, p in PATTERNS.items():
        sel = dev.loc[hits[k]].sample(min(3, int(hits[k].sum())), random_state=42)
        for _, r in sel.iterrows():
            frag = re.search(r'.{0,60}' + p + r'.{0,60}', ' '.join(r[NARR].fillna('')).lower())
            examples.append({'patron': k, 'clase': CLASSES[r.incident_outcome],
                             'fragmento': frag.group(0) if frag else ''})
    pd.DataFrame(examples).to_csv(OUT / 'fuga_narrativas_ejemplos.csv', index=False)
    lines += ['## 2. Narrativas que nombran el desenlace (fracción de casos de cada clase)', '',
              t2.round(3).to_markdown(), '', 'Ejemplos: `fuga_narrativas_ejemplos.csv`.', '']

    # 3. Información mutua con el desenlace.
    y = dev.incident_outcome
    mi = {c: mutual_info_score(y, dev[c].fillna('NA')) for c in RAW_FEATURES + PRED}
    mi['naics_3'] = mutual_info_score(y, dev.naics_code.fillna('NA').str[:3])
    t3 = pd.Series(mi, name='informacion_mutua_nats').sort_values(ascending=False)
    t3.to_csv(OUT / 'informacion_mutua.csv')
    lines += ['## 3. Información mutua con el desenlace (nats; mayor = más informativa)', '',
              t3.round(4).to_frame().to_markdown(), '',
              'Nota: la información mutua de variables con muchas categorías (naics_code, soc_code) está inflada '
              'por su cardinalidad; se añade naics_3 como referencia.', '']

    # 4. Efecto del empleador sobre traslado (3) frente a ausencia (2).
    s = dev.loc[y.isin(['2', '3'])].copy()
    s['y3'] = s.incident_outcome.eq('3').astype(float)
    s['estrato'] = s.naics_code.str[:3].fillna('NA') + '|' + s.nature_title_pred.fillna('NA') + '|' + \
        s.part_title_pred.fillna('NA')
    s['p_estrato'] = s.groupby('estrato').y3.transform('mean')
    g = s.groupby('establishment_id').agg(n=('y3', 'size'), O=('y3', 'sum'), E=('p_estrato', 'sum'),
                                          V=('p_estrato', lambda p: float(np.sum(p * (1 - p)))))
    g = g.loc[(g.n >= 20) & (g.V > 0)]
    g['z'] = (g.O - g.E) / np.sqrt(g.V)
    g['prop_traslado'] = g.O / g.n
    g.to_csv(OUT / 'empleador_por_establecimiento.csv')
    rng = np.random.default_rng(42)
    perm_var = []
    strat = pd.factorize(s.estrato)[0]
    est = pd.factorize(s.establishment_id)[0]
    yv, pv = s.y3.to_numpy(), s.p_estrato.to_numpy()
    base_order = np.argsort(strat, kind='stable')
    n_e = np.bincount(est)
    E_e = np.bincount(est, weights=pv)
    V_e = np.bincount(est, weights=pv * (1 - pv))
    keep = (n_e >= 20) & (V_e > 0)
    for _ in range(200):   # nula: se baraja y3 dentro de cada estrato (sin efecto del establecimiento)
        shuffled = np.lexsort((rng.random(len(yv)), strat))   # orden aleatorio dentro de estrato
        yp = np.empty_like(yv)
        yp[base_order] = yv[shuffled]
        O_e = np.bincount(est, weights=yp)
        perm_var.append(float(np.var(((O_e - E_e) / np.sqrt(np.where(V_e > 0, V_e, 1)))[keep])))
    same_ind = s.assign(n3=s.naics_code.str[:3]).groupby(['n3', 'establishment_id']).y3.agg(['size', 'mean'])
    same_ind = same_ind.loc[same_ind['size'] >= 20].groupby(level=0)['mean'].agg(['count', 'min', 'max'])
    same_ind = same_ind.loc[same_ind['count'] >= 10].sort_values('count', ascending=False).head(8)
    size_tab = s.groupby('size').y3.agg(['size', 'mean'])
    lines += ['## 4. Efecto del empleador: traslado/restricción (3) frente a días de ausencia (2)', '',
              f'Casos de clase 2 o 3: {len(s)}; establecimientos con >= 20 de esos casos: {len(g)} '
              f'({int(g.n.sum())} casos).',
              'Para cada establecimiento se compara cuántos casos de traslado tuvo con los esperados si se '
              'comportara como el promedio de su estrato (industria NAICS de 3 dígitos x naturaleza x parte del '
              'cuerpo). z = (observado - esperado) / desviación esperada; sin efecto del empleador, var(z) ~ 1.',
              f'- var(z) observada: {np.var(g.z):.2f}; bajo la nula (200 permutaciones dentro de estrato): '
              f'{np.mean(perm_var):.2f} (máx. {np.max(perm_var):.2f}).',
              f'- Establecimientos con |z| > 3: {np.mean(np.abs(g.z) > 3):.1%} (esperado sin efecto: ~0,3 %).',
              f'- Proporción de traslado por establecimiento: mediana {g.prop_traslado.median():.2f}, '
              f'percentil 10 {g.prop_traslado.quantile(.1):.2f}, percentil 90 {g.prop_traslado.quantile(.9):.2f}.',
              '', 'Misma industria (NAICS 3 dígitos), distintos establecimientos (>= 20 casos 2/3 cada uno):', '',
              same_ind.round(2).rename(columns={'count': 'establecimientos', 'min': 'prop_traslado_min',
                                                'max': 'prop_traslado_max'}).to_markdown(), '',
              'Proporción de traslado según tamaño del establecimiento (`size`):', '',
              size_tab.round(3).rename(columns={'size': 'casos', 'mean': 'prop_traslado'}).to_markdown(), '']

    # 5. Momento de carga del registro.
    created = pd.to_datetime(dev.created_timestamp, format='%d%b%y:%H:%M:%S', errors='coerce')
    lag = (created - dev.date).dt.days
    lines += ['## 5. ¿Cuándo se carga el registro?', '',
              f'created_timestamp: mínimo {created.min():%Y-%m-%d}, mediana {created.median():%Y-%m-%d}, '
              f'máximo {created.max():%Y-%m-%d}; días desde el incidente: mediana {lag.median():.0f}, '
              f'percentil 10 {lag.quantile(.1):.0f}.', '']
    (OUT / 'resumen.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
