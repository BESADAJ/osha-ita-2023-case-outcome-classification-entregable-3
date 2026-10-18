"""Los cuatro métodos de optimización de hiperparámetros sobre un evaluador común.

Todos los métodos llaman al mismo ``Evaluator``: cada evaluación ajusta el *pipeline completo*
(one-hot → balanceo → modelo) en cada pliegue interno y lo puntúa en la parte de validación de ese
pliegue. Así no hay fuga de información en ningún método (tampoco en Optuna ni en DEAP, que no
integran ``Pipeline`` de forma nativa) y el presupuesto (número de configuraciones) es idéntico.

Codificación común del espacio: cada hiperparámetro es un gen en [0, 1] que ``decode`` traduce a
su rango (lineal o logarítmico) o a una categoría. Random Search muestrea genes uniformes; el
algoritmo genético evoluciona genes; Optuna usa ``suggest_*`` sobre los mismos rangos; Grid Search
usa rejillas explícitas dentro de esos rangos.
"""
import itertools
import time
import warnings

import numpy as np
import optuna
from deap import base, creator, tools
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import ParameterGrid

from . import models as M

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Hiperparámetros del algoritmo genético (justificados en el capítulo de optimización).
GA = dict(pop_size=6, tournament=3, cx_prob=0.8, blend_alpha=0.5, mut_prob=0.3, mut_sigma=0.15,
          mut_indpb=0.5, elite=1)


def decode(space, genes):
    """Traduce un vector de genes en [0, 1] a un diccionario de hiperparámetros.

    Parameters
    ----------
    space : dict
        ``{nombre: ('int'|'float', lo, hi, log) | ('cat', opciones)}``.
    genes : array-like
        Un valor en [0, 1] por hiperparámetro, en el orden de ``space``.
    """
    out = {}
    for g, (k, spec) in zip(np.clip(genes, 0, 1), space.items()):
        if spec[0] == 'cat':
            out[k] = spec[1][min(int(g * len(spec[1])), len(spec[1]) - 1)]
            continue
        _, lo, hi, log = spec
        v = float(np.exp(np.log(lo) + g * (np.log(hi) - np.log(lo)))) if log else lo + g * (hi - lo)
        out[k] = int(round(v)) if spec[0] == 'int' else v
    return out


def _key(params):
    return tuple(sorted((k, round(v, 6) if isinstance(v, float) else v) for k, v in params.items()))


def score_predictions(task, y_true, est, X):
    """Métrica de validación: balanced accuracy (clasificación) o −RMSE (regresión); mayor es mejor."""
    if task == 'clf':
        return balanced_accuracy_score(y_true, est.predict_proba(X).argmax(axis=1))
    return -float(np.sqrt(mean_squared_error(y_true, est.predict(X))))


class Evaluator:
    """Evalúa configuraciones con validación cruzada interna y registra cada evaluación.

    Parameters
    ----------
    task, name, balance : str
        Tarea, modelo y técnica de balanceo.
    X : pandas.DataFrame
        Variables preparadas del entrenamiento externo.
    y : numpy.ndarray
    folds : list of (train_idx, valid_idx)
        Pliegues internos (por establecimiento).
    seed : int
    budget : int
        Máximo de configuraciones distintas a evaluar.
    """

    def __init__(self, task, name, balance, X, y, folds, seed, budget):
        self.task, self.name, self.balance = task, name, balance
        self.X, self.y, self.folds, self.seed, self.budget = X, y, folds, seed, budget
        self.cache, self.history, self.t0 = {}, [], time.perf_counter()
        self.prepared = None

    def _prepare(self):
        """Codificación y remuestreo de cada pliegue interno, calculados una sola vez.

        El one-hot y SMOTE/ADASYN no dependen de los hiperparámetros del modelo, así que se ajustan
        una vez por pliegue (solo con su parte de entrenamiento) y se reutilizan en las ``budget``
        evaluaciones. Es exactamente el mismo cálculo que el ``Pipeline`` completo, sin repetirlo.
        """
        self.prepared = []
        for tr, va in self.folds:
            enc = M.make_encoder().fit(self.X.iloc[tr])
            A, ya = enc.transform(self.X.iloc[tr]), self.y[tr]
            sampler = M.make_sampler(self.balance, self.seed) if self.task == 'clf' else None
            if sampler is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    A, ya = sampler.fit_resample(A, ya)
            B = enc.transform(self.X.iloc[va])
            if self.name == 'random_forest':
                A, B = A.toarray(), B.toarray()
            self.prepared.append((A, ya, B, self.y[va]))
        self.prepare_seconds = time.perf_counter() - self.t0

    @property
    def exhausted(self):
        return len(self.history) >= self.budget

    def __call__(self, params):
        k = _key(params)
        if k in self.cache:  # las repeticiones no consumen presupuesto (relevante en el genético)
            return self.cache[k]
        if self.prepared is None:
            self._prepare()
        start = time.perf_counter()
        scores = []
        for A, ya, B, yb in self.prepared:
            est = M.make_estimator(self.task, self.name, params, self.balance, self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                est.fit(A, ya)
            scores.append(score_predictions(self.task, yb, est, B))
        value = float(np.mean(scores))
        self.cache[k] = value
        self.history.append({'eval': len(self.history) + 1, 'params': dict(params), 'score': value,
                             'score_sd': float(np.std(scores)), 'seconds': time.perf_counter() - start,
                             'elapsed': time.perf_counter() - self.t0})
        return value

    def best(self):
        h = max(self.history, key=lambda r: r['score'])
        return h['params'], h['score']


def grid_search(ev, space, grid, rng):
    """Grid Search: recorre la rejilla explícita (≤ presupuesto) en orden."""
    for params in list(ParameterGrid(grid))[:ev.budget]:
        full = {k: params.get(k, decode({k: s}, [0.5])[k]) for k, s in space.items()}
        ev(full)
    return {}


def random_search(ev, space, grid, rng):
    """Random Search: genes uniformes en [0, 1] decodificados al espacio (escala log incluida)."""
    tries = 0
    while not ev.exhausted and tries < 20 * ev.budget:
        ev(decode(space, rng.random(len(space))))
        tries += 1
    return {}


def bayes_search(ev, space, grid, rng):
    """Optimización bayesiana con Optuna.

    Sustituto: *Tree-structured Parzen Estimator* (TPE), que modela p(x | y bueno) = l(x) y
    p(x | y malo) = g(x) con estimadores de Parzen. Adquisición: Mejora Esperada (EI); en TPE
    maximizar EI equivale a maximizar l(x)/g(x). Las primeras 5 pruebas son aleatorias
    (exploración) y luego TPE explota las regiones con mayor razón l/g.
    """
    sampler = optuna.samplers.TPESampler(seed=int(rng.integers(1 << 31)), n_startup_trials=5,
                                         multivariate=True)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    def objective(trial):
        params = {}
        for k, spec in space.items():
            if spec[0] == 'cat':
                params[k] = trial.suggest_categorical(k, spec[1])
            elif spec[0] == 'int':
                params[k] = trial.suggest_int(k, spec[1], spec[2], log=spec[3])
            else:
                params[k] = trial.suggest_float(k, spec[1], spec[2], log=spec[3])
        return ev(params)

    tries = 0
    while not ev.exhausted and tries < 3 * ev.budget:
        study.optimize(objective, n_trials=1)
        tries += 1
    return {}


def _ensure_creator():
    if not hasattr(creator, 'FitnessMax'):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create('Individual', list, fitness=creator.FitnessMax)


def genetic_search(ev, space, grid, rng):
    """Algoritmo genético (DEAP) con elitismo.

    * Representación: un gen real en [0, 1] por hiperparámetro.
    * Selección: torneo de tamaño 3 (presión moderada, robusta a escalas de la aptitud).
    * Cruce: mezcla BLX-α (α = 0,5) con probabilidad 0,8.
    * Mutación: gaussiana (σ = 0,15, probabilidad por gen 0,5) con probabilidad 0,3.
    * Elitismo: el mejor individuo pasa intacto a la siguiente generación.
    * Población: 6 individuos; se itera hasta agotar el presupuesto de evaluaciones.

    Devuelve la diversidad genética (distancia media entre pares, normalizada) por generación.
    """
    _ensure_creator()
    tb = base.Toolbox()
    d = len(space)
    pyrng = __import__('random').Random(int(rng.integers(1 << 31)))
    tb.register('select', tools.selTournament, tournsize=GA['tournament'])

    def evaluate(ind):
        return (ev(decode(space, ind)),)

    def diversity(pop):
        g = np.asarray(pop)
        pairs = list(itertools.combinations(range(len(g)), 2))
        return float(np.mean([np.linalg.norm(g[i] - g[j]) for i, j in pairs]) / np.sqrt(d))

    pop = [creator.Individual(rng.random(d).tolist()) for _ in range(GA['pop_size'])]
    for ind in pop:
        ind.fitness.values = evaluate(ind)
    log = [{'generation': 0, 'diversity': diversity(pop), 'best': max(i.fitness.values[0] for i in pop),
            'evals': len(ev.history)}]
    gen = 0
    while not ev.exhausted and gen < 50:
        gen += 1
        elite = tools.selBest(pop, GA['elite'])
        offspring = [creator.Individual(list(x)) for x in tb.select(pop, len(pop) - GA['elite'])]
        for a, b in zip(offspring[::2], offspring[1::2]):
            if pyrng.random() < GA['cx_prob']:
                tools.cxBlend(a, b, GA['blend_alpha'])
        for m in offspring:
            if pyrng.random() < GA['mut_prob']:
                for i in range(d):
                    if pyrng.random() < GA['mut_indpb']:
                        m[i] += pyrng.gauss(0, GA['mut_sigma'])
            m[:] = list(np.clip(m, 0, 1))
        for m in offspring:
            if ev.exhausted and _key(decode(space, m)) not in ev.cache:
                m.fitness.values = (-np.inf,)
            else:
                m.fitness.values = evaluate(m)
        pop = [creator.Individual(list(e)) for e in elite] + offspring
        for e, src in zip(pop[:GA['elite']], elite):
            e.fitness.values = src.fitness.values
        log.append({'generation': gen, 'diversity': diversity(pop),
                    'best': max(i.fitness.values[0] for i in pop), 'evals': len(ev.history)})
    return {'ga_log': log}


SEARCHERS = {'grid': grid_search, 'random': random_search, 'bayes': bayes_search, 'genetic': genetic_search}


def run_search(method, ev, task, name, seed):
    """Ejecuta un método de búsqueda y devuelve (mejores params, mejor puntaje interno, extras)."""
    rng = np.random.default_rng(seed)
    extra = SEARCHERS[method](ev, M.SPACES[task][name], M.GRIDS[task][name], rng)
    params, score = ev.best()
    return params, score, extra


def refit(task, name, params, balance, X, y, seed):
    """Ajusta el pipeline final con los hiperparámetros elegidos sobre todo el conjunto dado."""
    pipe = M.make_pipeline(task, name, params, balance, seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipe.fit(X, y)
    return pipe


def clone_pipeline(pipe):
    return clone(pipe)
