"""Explicaciones locales con valores de Shapley exactos (SHAP intervencional; Lundberg y Lee, 2017).

Con pocas variables (o grupos de variables) se pueden enumerar *todas* las coaliciones. Para la coalición S,
las variables de S toman el valor del caso explicado y las demás el de cada fila del fondo; v(S) es la
probabilidad media de la clase explicada. Entonces

    φ_j = Σ_{S ⊆ F∖{j}} |S|! (d − |S| − 1)! / d! · [v(S ∪ {j}) − v(S)],

que cumple Σ_j φ_j = v(F) − v(∅) (eficiencia). Costo: 2^d · B filas predichas en una sola llamada.
Solo usa numpy y pandas; sirve para cualquier modelo con ``predict_proba``.
"""
from math import factorial

import numpy as np
import pandas as pd


def exact_shapley(predict_proba, x, background, groups, cls):
    """Valores de Shapley exactos de ``x`` (DataFrame de una fila) para la clase ``cls``.

    Parameters
    ----------
    predict_proba : callable(DataFrame) -> ndarray (n, K)
    x : DataFrame de una fila
    background : DataFrame (B filas, mismas columnas)
    groups : dict {nombre: [columnas]}; cada grupo es un jugador
    cls : int

    Returns
    -------
    pandas.Series con φ por grupo; ``attrs['base']`` = v(∅) y ``attrs['full']`` = v(F).
    """
    names = list(groups)
    d, B = len(names), len(background)
    bg = background.reset_index(drop=True)
    xv = {c: x.iloc[0][c] for g in names for c in groups[g]}
    rows = []
    for m in range(2 ** d):
        z = bg.copy()
        for j in range(d):
            if (m >> j) & 1:
                for c in groups[names[j]]:
                    z[c] = xv[c]
        rows.append(z)
    p = predict_proba(pd.concat(rows, ignore_index=True))[:, cls].reshape(2 ** d, B).mean(axis=1)
    w = [factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)]
    size = np.array([bin(m).count('1') for m in range(2 ** d)])
    phi = np.zeros(d)
    for j in range(d):
        without = np.array([m for m in range(2 ** d) if not (m >> j) & 1])
        phi[j] = np.sum([w[s] for s in size[without]] * (p[without | (1 << j)] - p[without]))
    out = pd.Series(phi, index=names)
    out.attrs['base'], out.attrs['full'] = float(p[0]), float(p[-1])
    return out
