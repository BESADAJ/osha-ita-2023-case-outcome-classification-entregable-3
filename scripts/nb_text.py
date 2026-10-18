"""Imprime las salidas de texto de un notebook ejecutado (para revisar cifras)."""
import sys
import nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    for o in c.outputs:
        if o.output_type == 'stream':
            print(f'[{i}]', o.text[:3000])
        elif o.output_type == 'error':
            print(f'[{i}] ERROR', o.ename, o.evalue)
        elif 'data' in o and 'text/plain' in o.data and 'image/png' not in o.data:
            print(f'[{i}]', o.data['text/plain'][:3000])
