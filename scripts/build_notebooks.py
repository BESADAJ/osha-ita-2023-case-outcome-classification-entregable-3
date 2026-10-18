"""Convierte las fuentes ``nb_src/*.py`` (formato percent) en notebooks del libro y los ejecuta.

Formato: ``# %%`` abre una celda de código y ``# %% [markdown]`` una de texto (cada línea de
texto empieza con ``# ``).

Uso: python scripts/build_notebooks.py [nombre ...] [--no-exec]
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def parse(path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = '\n'.join(buf).strip('\n')
        if kind == 'md':
            text = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        elif text.strip():
            cells.append(nbformat.v4.new_code_cell(text))

    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('# %%'):
            flush()
            kind, buf = ('md' if '[markdown]' in line else 'code'), []
        else:
            buf.append(line)
    flush()
    return cells


def build(name, execute=True):
    src = ROOT / 'nb_src' / f'{name}.py'
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.cells = parse(src)
    out = ROOT / 'jbook' / f'{name}.ipynb'
    if execute:
        t0 = time.time()
        NotebookClient(nb, timeout=None, kernel_name='python3',
                       resources={'metadata': {'path': str(ROOT / 'jbook')}}).execute()
        print(f'{name}: ejecutado en {time.time() - t0:.0f} s', flush=True)
    nbformat.write(nb, out)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    names = args or sorted(p.stem for p in (ROOT / 'nb_src').glob('*.py'))
    for n in names:
        build(n, execute='--no-exec' not in sys.argv)
