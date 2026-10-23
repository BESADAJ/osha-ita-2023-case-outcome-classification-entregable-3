"""Arma ``jbook/entregable3_completo.ipynb`` con las salidas ya guardadas de cada capítulo, sin reejecutar nada.

Es la alternativa a ``build_single_notebook.py --no-exec``, que deja el notebook único sin salidas. Usa el mismo
orden (``ORDER``) y la misma traducción de MyST de ``build_single_notebook.py``, pero toma cada capítulo-notebook de
``jbook/<capítulo>.ipynb``, ya ejecutado con ``build_notebooks.py``. Así se puede añadir un capítulo sin volver a
correr los demás (varios dependen de cachés y entornos distintos).

Uso: python scripts/assemble_single_notebook.py
"""
import re
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_single_notebook import BOOK, ORDER, myst_to_markdown  # noqa: E402

INTRO = (
    '# Entregable 3 — notebook único\n\n'
    'Este notebook reúne **todo el entregable 3 en un solo archivo**, en el mismo orden que el Jupyter Book: '
    'contexto y EDA (capítulos 1 a 4), modelo base del entregable 2 (5), diseño del pipeline y los 140 modelos (6), '
    'optimización de hiperparámetros (7) y computacional (8), evaluación de clasificación (9) y regresión (10), '
    'interpretabilidad (11), robustez (12), comparación estadística (13), revisión de literatura y modelo original '
    '(16 y 17, versión preliminar), conclusiones (14), reproducibilidad (15) y, como anexo, el notebook del '
    'entregable 1.\n\n'
    'Las salidas son las de cada capítulo del libro tal como se ejecutó por separado (este archivo se arma con '
    '`scripts/assemble_single_notebook.py`, sin reejecutar). Cada capítulo con código empieza con su propia celda de '
    'preparación, así que puede ejecutarse desde ese punto. Las corridas largas se hacen con `scripts/run_nested.py`, '
    '`scripts/run_final.py` y `disenio_temporal/`; aquí se leen sus resultados de `results/`. Ejecutar desde la '
    'carpeta `jbook/`.')


def main():
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.cells = [nbformat.v4.new_markdown_cell(INTRO)]
    for item in ORDER:
        if item.endswith('.md'):
            nb.cells.append(nbformat.v4.new_markdown_cell(myst_to_markdown((BOOK / item).read_text(encoding='utf-8'))))
            continue
        chapter = nbformat.read(BOOK / f'{item}.ipynb', as_version=4)
        for c in chapter.cells:
            if c.cell_type == 'markdown':
                c.source = myst_to_markdown(c.source)
            nb.cells.append(c)
    annex = nbformat.read(BOOK / 'anexo_entregable1.ipynb', as_version=4)
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '# Anexo: notebook del entregable 1\n\nSe incluye con sus salidas guardadas; depende del código del '
        'entregable 1 (`osha_deliverable_utils.py` y relacionados) y no se reejecuta aquí.'))
    for c in annex.cells:
        if c.cell_type == 'markdown':
            c.source = re.sub(r'^(#+)', r'#\1', c.source, flags=re.M)  # bajar un nivel los títulos del anexo
        nb.cells.append(c)
    for i, c in enumerate(nb.cells):        # ids únicos: los capítulos se generaron por separado
        c['id'] = f'celda-{i:04d}'
    nbformat.validate(nb)
    out = BOOK / 'entregable3_completo.ipynb'
    nbformat.write(nb, out)
    print('escrito:', out, '| celdas:', len(nb.cells))


if __name__ == '__main__':
    main()
