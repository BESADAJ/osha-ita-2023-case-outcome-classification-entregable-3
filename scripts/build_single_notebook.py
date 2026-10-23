"""Construye un único notebook con todo el entregable 3 (``jbook/entregable3_completo.ipynb``).

Une, en el orden del libro, los capítulos en Markdown (convertidos a celdas; las directivas MyST se
traducen a Markdown estándar para que se vean también en Jupyter) y los capítulos-notebook (con su
código), lo ejecuta de principio a fin y añade al final el anexo del entregable 1 con sus salidas
guardadas (ese notebook depende del código del entregable 1 y no se reejecuta).

Uso: python scripts/build_single_notebook.py [--no-exec]
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

ORDER = ['intro.md', '01_contexto.md', '02_eda_dashboard.md', '03_eda_adicional.md', '04_eda_mejorado',
         '05_modelo_base_e2.md', '06_diseno_pipeline', '07_optimizacion_hiperparametros',
         '08_optimizacion_computacional', '09_clasificacion', '10_regresion', '11_interpretabilidad',
         '12_robustez', '13_comparacion_estadistica', '16_literatura_novedad.md', '17_modelo_original',
         '18_modelo_propuesto', '14_conclusiones.md', '15_reproducibilidad.md']


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
    return re.sub(r':::\{(\w+)\}([^\n]*)\n(.*?)\n:::', admonition, text, flags=re.S)


def build(execute=True):
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    cells = [nbformat.v4.new_markdown_cell(
        '# Entregable 3 — notebook único\n\n'
        'Este notebook reúne **todo el entregable 3 en un solo archivo**, en el mismo orden que el Jupyter Book: '
        'contexto y EDA (capítulos 1 a 4), modelo base del entregable 2 (5), diseño del pipeline y los 140 modelos (6), '
        'optimización de hiperparámetros (7) y computacional (8), evaluación de clasificación (9) y regresión (10), '
        'interpretabilidad (11), robustez (12), comparación estadística (13), el modelo propuesto (18: stacking '
        'multivista), conclusiones (14), reproducibilidad (15) '
        'y, como anexo, el notebook del entregable 1.\n\n'
        'Cada capítulo con código empieza con su propia celda de preparación, así que puede ejecutarse desde ese punto. '
        'Las corridas largas (validación anidada de las 140 combinaciones y su evaluación en la prueba) se hacen con '
        '`scripts/run_nested.py` y `scripts/run_final.py`; aquí se leen sus resultados de `results/`. Ejecutar desde la '
        'carpeta `jbook/`.')]
    for item in ORDER:
        if item.endswith('.md'):
            cells.append(nbformat.v4.new_markdown_cell(myst_to_markdown((BOOK / item).read_text(encoding='utf-8'))))
        else:
            cells.extend(parse(ROOT / 'nb_src' / f'{item}.py'))
    nb.cells = cells
    if execute:
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
    out = BOOK / 'entregable3_completo.ipynb'
    nbformat.write(nb, out)
    print('escrito:', out, '| celdas:', len(nb.cells))


if __name__ == '__main__':
    build(execute='--no-exec' not in sys.argv)
