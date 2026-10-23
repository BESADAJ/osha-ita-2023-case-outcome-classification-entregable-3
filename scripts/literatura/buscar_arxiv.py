"""Reintento de las 8 cadenas en arXiv (la API respondió «Rate exceeded» en la primera pasada): una consulta cada 30 s."""
import json, time, datetime
import buscar as B

stamp = datetime.datetime.now().astimezone().isoformat(timespec='minutes')
for i, terms in enumerate(B.QUERIES, 1):
    for attempt in range(3):
        url, total, titles = B.arxiv(terms)
        if total is not None:
            break
        time.sleep(60)
    (B.OUT / f'arXiv_{i}.json').write_text(json.dumps({'fecha': stamp, 'cadena': B.as_text(terms), 'url': url, 'total': total,
                                                        'titulos': titles}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(i, total, titles[:5], flush=True)
    time.sleep(30)
