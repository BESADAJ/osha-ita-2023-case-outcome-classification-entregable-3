"""Sustituye, en orden, los marcadores «**Interpretación.** TODO» de una fuente nb_src por textos dados.

Uso: python scripts/fill_todos.py nb_src/X.py textos.md   (textos separados por una línea '---')
"""
import sys
from pathlib import Path

src, texts = Path(sys.argv[1]), Path(sys.argv[2]).read_text(encoding='utf-8').split('\n---\n')
s = src.read_text(encoding='utf-8')
marker = '# **Interpretación.** TODO'
assert s.count(marker) == len(texts), (s.count(marker), len(texts))
for t in texts:
    body = '\n'.join('# ' + l if l.strip() else '#' for l in t.strip().splitlines())
    s = s.replace(marker, '# **Interpretación.** ' + body[2:], 1)
src.write_text(s, encoding='utf-8')
print('ok', len(texts))
