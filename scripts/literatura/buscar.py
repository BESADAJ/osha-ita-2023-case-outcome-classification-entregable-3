"""Búsqueda sistemática de literatura para el modelo original (capítulo 16, sección del modelo SECR).

Consulta tres bases con API pública y búsqueda booleana (OpenAlex, Europe PMC y arXiv) con las mismas
cadenas, y guarda la respuesta (total y títulos de los primeros 25 resultados por relevancia) con la fecha y la
URL exacta en ``results/literatura/``. Semantic Scholar respondió 429 («Too Many Requests») sin clave y Crossref no
admite operadores booleanos, por eso no se usan.

Uso: python scripts/literatura/buscar.py
"""
import csv
import datetime
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'results' / 'literatura'
OUT.mkdir(parents=True, exist_ok=True)

# Cada cadena: lista de términos unidos con AND; un término puede ser una alternativa (A OR B).
QUERIES = [
    ['"ensemble selection"', '(injury OR injuries)'],
    ['"ensemble selection"', '(occupational OR workplace OR "work-related")'],
    ['(ensemble OR stacking)', '"occupational injury"', '(severity OR outcome)'],
    ['"Frank-Wolfe"', '(ensemble OR "classifier combination" OR "model averaging")'],
    ['("dynamic ensemble selection" OR "local weighting" OR "region-specific weights")', '(injury OR accident)'],
    ['"mixture of experts"', '("occupational injury" OR "workplace injury" OR "injury severity")'],
    ['("selección de ensambles" OR "ensamble de modelos" OR "combinación de clasificadores")',
     '("lesiones laborales" OR "accidentes de trabajo" OR "accidente laboral")'],
    ['OSHA', '("machine learning" OR classification)', '(severity OR outcome OR "days away")'],
]


def as_text(terms):
    return ' AND '.join(terms)


def get(url, tries=3, wait=4):
    req = urllib.request.Request(url, headers={'User-Agent': 'osha-ita-2023-entregable3 (uso académico)'})
    for i in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120).read().decode('utf-8')
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(wait * (i + 1))
    return json.dumps({'error': str(err)})


def openalex(terms):
    url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'search': as_text(terms), 'per-page': 25})
    d = json.loads(get(url))
    return url, d.get('meta', {}).get('count'), [w.get('title') for w in d.get('results', [])]


def europepmc(terms):
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(
        {'query': as_text(terms), 'format': 'json', 'pageSize': 25})
    d = json.loads(get(url))
    return url, d.get('hitCount'), [r.get('title') for r in d.get('resultList', {}).get('result', [])]


def arxiv(terms):
    def one(t):
        t = t.strip('()')
        alts = [a.strip() for a in t.split(' OR ')]
        q = ' OR '.join(f'all:{a}' for a in alts)
        return f'({q})' if len(alts) > 1 else q
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode(
        {'search_query': ' AND '.join(one(t) for t in terms), 'max_results': 25})
    x = get(url, tries=2, wait=10)
    time.sleep(4)
    total = re.search(r'<opensearch:totalResults[^>]*>(\d+)<', x)
    titles = [re.sub(r'\s+', ' ', t).strip() for t in re.findall(r'<entry>.*?<title>(.*?)</title>', x, re.S)]
    return url, int(total.group(1)) if total else None, titles


if __name__ == '__main__':
    stamp = datetime.datetime.now().astimezone().isoformat(timespec='minutes')
    rows = []
    for i, terms in enumerate(QUERIES, 1):
        for base, fn in [('OpenAlex', openalex), ('Europe PMC', europepmc), ('arXiv', arxiv)]:
            url, total, titles = fn(terms)
            (OUT / f'{base.replace(" ", "")}_{i}.json').write_text(json.dumps(
                {'fecha': stamp, 'cadena': as_text(terms), 'url': url, 'total': total, 'titulos': titles},
                ensure_ascii=False, indent=1), encoding='utf-8')
            rows.append({'n': i, 'fecha': stamp, 'base': base, 'cadena': as_text(terms), 'resultados': total,
                         'titulos_guardados': len(titles)})
            print(i, base, total, flush=True)
    with open(OUT / 'resumen.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
