"""Estilo común de las figuras del libro.

Paleta categórica en orden fijo (validada para daltonismo en pares adyacentes); el color sigue a la
entidad (p. ej. cada optimizador siempre tiene el mismo color), nunca a su posición en un ranking.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as C

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

OPT_COLORS = dict(zip(['grid', 'random', 'bayes', 'genetic'], SERIES[:4]))
BAL_COLORS = dict(zip(['none', 'smote', 'adasyn', 'class_weight'], SERIES[:4]))
CLASS_COLORS = dict(zip(range(4), SERIES[:4]))
MODEL_ORDER_CLF = ['knn', 'naive_bayes', 'logreg', 'tree', 'random_forest', 'xgboost', 'svm']
MODEL_ORDER_REG = ['knn', 'ridge', 'lasso', 'tree', 'random_forest', 'xgboost', 'svr']
MODEL_COLORS = dict(zip(MODEL_ORDER_CLF, SERIES[:7])) | {'ridge': SERIES[1], 'lasso': SERIES[2],
                                                         'svr': SERIES[6]}


def setup():
    """Aplica el estilo: fondo claro, rejilla tenue, líneas de 2 px, sin bordes superiores."""
    mpl.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
        'axes.prop_cycle': mpl.cycler(color=SERIES), 'font.size': 10, 'axes.titlesize': 11,
        'legend.frameon': False, 'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    })


def save(fig, name):
    """Guarda la figura en ``jbook/figures/<name>.png``."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f'{name}.png')
    return fig


def show():
    plt.show()
