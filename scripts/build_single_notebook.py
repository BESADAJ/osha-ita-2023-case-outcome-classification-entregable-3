"""Construye un único notebook con todo el entregable 3 (``jbook/entregable3_completo.ipynb``).

Une, en el orden del libro, los capítulos en Markdown (convertidos a celdas; las directivas MyST se
traducen a Markdown estándar para que se vean también en Jupyter) y los capítulos-notebook (con su
código), lo ejecuta de principio a fin y añade al final el anexo del entregable 1 con sus salidas
guardadas (ese notebook depende del código del entregable 1 y no se reejecuta).

Uso: python scripts/build_single_notebook.py [--no-exec | --from-built]
"""
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / 'jbook'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import parse  # noqa: E402

ORDER = ['intro.md', '00_trazabilidad.md', '01_contexto.md', '02_eda_dashboard.md', '03_eda_adicional.md',
         '04_eda_mejorado', '05_modelo_base_e2.md', '06_diseno_pipeline', '07_optimizacion_hiperparametros',
         '08_optimizacion_computacional', '09_clasificacion', '10_regresion', '11_interpretabilidad',
         '12_robustez', '13_comparacion_estadistica', '13b_mejores_combinaciones.md', '16_literatura_novedad.md',
         '17_modelo_original', '18_modelo_propuesto', '19_modelo_original_secr.md', '20_validacion_estadistica',
         '21_dashboard.md', '14_conclusiones.md', '15_reproducibilidad.md']


def myst_to_markdown(text):
    """Traduce las directivas MyST usadas en el libro a Markdown estándar."""
    def include(m):
        return (BOOK / m.group(1).strip()).read_text(encoding='utf-8')
    text = re.sub(r'```\{include\}\s*(\S+)\s*\n```', include, text)

    def figure(m):
        path, body = m.group(1).strip(), m.group(2)
        caption = ' '.join(l for l in body.splitlines() if l.strip() and not l.strip().startswith(':'))
        return f'![{caption}]({path})\n\n*{caption}*'
    text = re.sub(r'```\{figure\}\s*(\S+)\n(.*?)```', figure, text, flags=re.S)

    def admonition(m):
        kind, arg, body = m.group(1), m.group(2).strip(), m.group(3)
        title = 'Nota' if kind == 'note' else (arg or kind)
        body = '\n'.join(l for l in body.splitlines() if not l.startswith(':class:'))
        return '\n'.join(f'> {l}' if l.strip() else '>' for l in (f'**{title}.**\n' + body.strip()).splitlines())
    text = re.sub(r':::\{(\w+)\}([^\n]*)\n(.*?)\n:::', admonition, text, flags=re.S)

    def literalinclude(m):
        path, opts = m.group(1).strip(), m.group(2)
        lines = (BOOK / path).read_text(encoding='utf-8').splitlines()
        rng = re.search(r':lines:\s*(\d+)-(\d+)', opts)
        if rng:
            lines = lines[int(rng.group(1)) - 1:int(rng.group(2))]
        return '```python\n' + '\n'.join(lines) + '\n```'
    text = re.sub(r'```\{literalinclude\}\s*(\S+)\n(.*?)```', literalinclude, text, flags=re.S)
    text = re.sub(r'```\{bibliography\}.*?```', '*(Lista de referencias: `jbook/references.bib`.)*', text, flags=re.S)
    text = re.sub(r'\{cite(?::t)?\}`([^`]+)`', r'[\1]', text)  # citas: la clave BibTeX entre corchetes
    text = re.sub(r'\{eq\}`([^`]+)`', r'(\1)', text)
    text = re.sub(r'\{numref\}`([^`]+)`', r'la figura «\1»', text)
    return re.sub(r'\$\$ \((eq-[\w-]+)\)', '$$', text)  # etiquetas de ecuaciones MyST


def build(execute=True, from_built=False):
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    cells = [nbformat.v4.new_markdown_cell(
        '# Proyecto final (entregable 3): notebook único\n\n'
        'Este notebook reúne **todo el proyecto final en un solo archivo**, en el mismo orden que el Jupyter Book: '
        'trazabilidad de correcciones (00), contexto y EDA (1 a 4), los 140 modelos y su evaluación (5 a 13), mejores '
        'combinaciones y artefactos (13b), literatura y novedad (16), modelos originales explorados (17 y 18), el modelo '
        'original final SECR (19), su validación estadística (20), el dashboard (21), conclusiones con el veredicto '
        'R1–R3 y la declaración de uso de IA (14), reproducibilidad y bibliografía (15) y, como anexo, el notebook del '
        'entregable 1.\n\n'
        'Con `--from-built` (como se generó esta versión) **las salidas son las de cada capítulo ya ejecutado**: no se '
        'reejecuta, porque la prueba de octubre a diciembre está bloqueada por código y los capítulos 9 a 13 y 18 la '
        'usaron antes del bloqueo.\n\n'
        'Cada capítulo con código empieza con su propia celda de preparación, así que puede ejecutarse desde ese punto. '
        'Las corridas largas (validación anidada de las 140 combinaciones y su evaluación en la prueba) se hacen con '
        '`scripts/run_nested.py` y `scripts/run_final.py`; aquí se leen sus resultados de `results/`. Ejecutar desde la '
        'carpeta `jbook/`.')]
    for item in ORDER:
        if item.endswith('.md'):
            cells.append(nbformat.v4.new_markdown_cell(myst_to_markdown((BOOK / item).read_text(encoding='utf-8'))))
        elif from_built:  # el capítulo ya ejecutado del libro, con sus salidas (no se reejecuta)
            cells.extend(nbformat.read(BOOK / f'{item}.ipynb', as_version=4).cells)
        else:
            cells.extend(parse(ROOT / 'nb_src' / f'{item}.py'))
    nb.cells = cells
    if execute and not from_built:
        t0 = time.time()
        NotebookClient(nb, timeout=None, kernel_name='python3',
                       resources={'metadata': {'path': str(BOOK)}}).execute()
        print(f'notebook único ejecutado en {time.time() - t0:.0f} s', flush=True)
    annex = nbformat.read(BOOK / 'anexo_entregable1.ipynb', as_version=4)
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '# Anexo: notebook del entregable 1\n\nSe incluye con sus salidas guardadas; depende del código del '
        'entregable 1 (`osha_deliverable_utils.py` y relacionados) y no se reejecuta aquí.'))
    for c in annex.cells:
        if c.cell_type == 'markdown':
            c.source = re.sub(r'^(#+)', r'#\1', c.source, flags=re.M)  # bajar un nivel los títulos del anexo
        nb.cells.append(c)
    for c in nb.cells:  # las etiquetas y enlaces internos ya existen en los capítulos: aquí serían duplicados
        if c.cell_type == 'markdown':
            c.source = re.sub(r'\$\$ \(([\w-]+)\)', '$$', c.source)
            c.source = re.sub(r'\{eq\}`([^`]+)`', r'(\1)', c.source)
            c.source = re.sub(r'\[([^\]]+)\]\((?!https?:)[^)]+\.md\)', r'\1', c.source)
    out = BOOK / 'entregable3_completo.ipynb'
    nbformat.write(nb, out)
    print('escrito:', out, '| celdas:', len(nb.cells))


if __name__ == '__main__':
    build(execute='--no-exec' not in sys.argv, from_built='--from-built' in sys.argv)
