"""Experimentos de optimización computacional (sección 4 de la guía).

Cada función devuelve un ``DataFrame`` con tiempos promediados sobre ``repeats`` repeticiones
(``time.perf_counter``), memoria pico (``tracemalloc``) y desempeño, para contrastar la
complejidad teórica con la medición empírica.
"""
import cProfile
import io
import pstats
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso, Ridge, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from . import models as M

warnings.filterwarnings('ignore')

THEORY = pd.DataFrame([
    ('KNN (fuerza bruta)', 'O(1) (guarda los datos)', 'O(n·p) por consulta'),
    ('KNN (KD-Tree / Ball Tree)', 'O(p·n log n)', 'O(log n) por consulta en p bajo; → O(n·p) en p alto'),
    ('KNN aproximado (FAISS IVF)', 'O(n·p·iter) (k-means de centroides)', 'O((n/nlist)·nprobe·p) por consulta'),
    ('Naive Bayes (fit / partial_fit)', 'O(n·p)', 'O(K·p) por consulta'),
    ('Regresión logística (SAGA)', 'O(n·p) por época', 'O(K·p)'),
    ('Ridge / Lasso (SAGA, CD)', 'O(n·p) por época / iteración', 'O(p)'),
    ('Árbol de decisión', 'O(p·n log n)', 'O(profundidad)'),
    ('Random Forest', 'O(T·m·n log n), m = max_features·p', 'O(T·profundidad)'),
    ('XGBoost exacto', 'O(T·p·n log n)', 'O(T·profundidad)'),
    ('XGBoost hist', 'O(T·p·n) + O(T·p·bins)', 'O(T·profundidad)'),
    ('SVM kernel (SVC)', 'O(n²·p) a O(n³)', 'O(n_SV·p)'),
    ('LinearSVC / SGD', 'O(n·p) por época', 'O(K·p)'),
], columns=['Modelo', 'Entrenamiento', 'Inferencia'])


def timed(fn, repeats=3):
    """Tiempo medio y desviación de ``fn()`` en ``repeats`` repeticiones y memoria pico (MB)."""
    times = []
    for i in range(repeats):
        if i == 0:
            tracemalloc.start()
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
        if i == 0:
            peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
            tracemalloc.stop()
    return float(np.mean(times)), float(np.std(times)), float(peak), out


def scaling_exponent(ns, ts):
    """Pendiente de log(t) contra log(n): exponente empírico de la curva de escalamiento."""
    return float(np.polyfit(np.log(ns), np.log(ts), 1)[0])


def scaling_curves(X, y, ns, specs, repeats=2):
    """Tiempo de entrenamiento al variar n para varios estimadores (``specs``: nombre → fábrica)."""
    rows = []
    for name, make in specs.items():
        for n in ns:
            mean, sd, peak, _ = timed(lambda: make().fit(X[:n], y[:n]), repeats)
            rows.append({'modelo': name, 'n': n, 'fit_s': mean, 'fit_sd': sd, 'mem_mb': peak})
    return pd.DataFrame(rows)


def knn_benchmark(Xtr, ytr, Xte, yte, k=31, dims=10, seed=42):
    """KNN exacto (fuerza bruta, KD-Tree, Ball Tree) frente a FAISS (IVF, aproximado).

    Los árboles solo ayudan en dimensión baja, así que se comparan sobre una proyección SVD a
    ``dims`` componentes; la fuerza bruta se mide además sobre el one-hot completo.
    """
    import faiss
    svd = TruncatedSVD(dims, random_state=seed).fit(Xtr)
    Ztr = np.ascontiguousarray(svd.transform(Xtr), dtype=np.float32)
    Zte = np.ascontiguousarray(svd.transform(Xte), dtype=np.float32)
    rows = []
    for algo, A, B, label in [('brute', Xtr, Xte, f'fuerza bruta (one-hot, p={Xtr.shape[1]})'),
                              ('brute', Ztr, Zte, f'fuerza bruta (SVD, p={dims})'),
                              ('kd_tree', Ztr, Zte, f'KD-Tree (SVD, p={dims})'),
                              ('ball_tree', Ztr, Zte, f'Ball Tree (SVD, p={dims})')]:
        knn = KNeighborsClassifier(k, algorithm=algo, n_jobs=1)
        fit_s, _, mem, _ = timed(lambda: knn.fit(A, ytr), 1)
        pred_s, _, _, pred = timed(lambda: knn.predict(B), 1)
        rows.append({'método': label, 'fit_s': fit_s, 'predict_s': pred_s, 'mem_mb': mem,
                     'balanced_accuracy': balanced_accuracy_score(yte, pred)})
    exact = KNeighborsClassifier(k, algorithm='brute').fit(Ztr, ytr).kneighbors(Zte, return_distance=False)
    for nlist, nprobe in [(64, 1), (64, 8)]:
        def build():
            q = faiss.IndexFlatL2(dims)
            index = faiss.IndexIVFFlat(q, dims, nlist)
            index.train(Ztr)
            index.add(Ztr)
            index.nprobe = nprobe
            return index, q
        fit_s, _, mem, (index, _q) = timed(build, 1)
        faiss.omp_set_num_threads(1)
        pred_s, _, _, (_, I) = timed(lambda: index.search(Zte, k), 1)
        votes = np.asarray(ytr)[np.clip(I, 0, None)]
        pred = np.array([np.bincount(v, minlength=4).argmax() for v in votes])
        recall = np.mean([len(set(a) & set(b)) / k for a, b in zip(I, exact)])
        rows.append({'método': f'FAISS IVF (nlist={nlist}, nprobe={nprobe})', 'fit_s': fit_s, 'predict_s': pred_s,
                     'mem_mb': mem, 'balanced_accuracy': balanced_accuracy_score(yte, pred),
                     'recall_vecinos': recall})
    return pd.DataFrame(rows)


def linear_solvers(Xtr, ytr, Xte, yte, alpha_ridge=10.0, alpha_lasso=1e-3):
    """Ridge y Lasso: solver por defecto frente a SAGA (lineal en n y p por época)."""
    rows = []
    for name, est in [('Ridge (cholesky, denso)', Ridge(alpha_ridge, solver='cholesky')),
                      ('Ridge (sparse_cg)', Ridge(alpha_ridge, solver='sparse_cg')),
                      ('Ridge (SAGA)', Ridge(alpha_ridge, solver='saga', max_iter=200, tol=1e-3, random_state=0)),
                      ('Lasso (coordinate descent)', Lasso(alpha_lasso, max_iter=3000, tol=1e-3)),
                      ('Lasso (SAGA vía SGD L1)', None)]:
        if est is None:
            from sklearn.linear_model import SGDRegressor
            est = SGDRegressor(penalty='l1', alpha=alpha_lasso, max_iter=50, tol=1e-3, random_state=0)
        A = Xtr.toarray() if 'cholesky' in name else Xtr
        fit_s, sd, mem, _ = timed(lambda: est.fit(A, ytr), 3)
        B = Xte.toarray() if 'cholesky' in name else Xte
        pred_s, _, _, pred = timed(lambda: est.predict(B), 3)
        rows.append({'método': name, 'fit_s': fit_s, 'fit_sd': sd, 'predict_s': pred_s, 'mem_mb': mem,
                     'rmse': float(np.sqrt(mean_squared_error(yte, pred)))})
    return pd.DataFrame(rows)


def naive_bayes_online(X, y, Xte, yte, chunk=20_000):
    """``fit`` en memoria frente a ``partial_fit`` por bloques (aprendizaje incremental)."""
    rows = []
    fit_s, _, mem, nb = timed(lambda: BernoulliNB(alpha=1.0).fit(X, y), 3)
    rows.append({'método': 'BernoulliNB.fit (todo en memoria)', 'fit_s': fit_s, 'mem_mb': mem,
                 'balanced_accuracy': balanced_accuracy_score(yte, nb.predict(Xte))})

    def online():
        m = BernoulliNB(alpha=1.0)
        for i in range(0, X.shape[0], chunk):
            m.partial_fit(X[i:i + chunk], y[i:i + chunk], classes=np.arange(4))
        return m
    fit_s, _, mem, nb2 = timed(online, 3)
    rows.append({'método': f'BernoulliNB.partial_fit (bloques de {chunk:,})', 'fit_s': fit_s, 'mem_mb': mem,
                 'balanced_accuracy': balanced_accuracy_score(yte, nb2.predict(Xte))})
    return pd.DataFrame(rows)


def xgboost_methods(Xtr, ytr, Xva, yva, Xte, yte, gpu=True):
    """XGBoost: método exacto frente a ``hist``, early stopping y GPU (``device='cuda'``)."""
    base = dict(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                random_state=0, verbosity=0)
    configs = [('exact (CPU, 12 hilos)', dict(tree_method='exact', n_jobs=12)),
               ('hist (CPU, 12 hilos)', dict(tree_method='hist', n_jobs=12)),
               ('hist + early stopping (CPU)', dict(tree_method='hist', n_jobs=12, early_stopping_rounds=20))]
    if gpu:
        configs.append(('hist (GPU, cuda)', dict(tree_method='hist', device='cuda')))
    rows = []
    for name, cfg in configs:
        est = XGBClassifier(**base, **cfg)
        kw = {'eval_set': [(Xva, yva)], 'verbose': False} if 'early_stopping_rounds' in cfg else {}
        try:
            fit_s, sd, mem, _ = timed(lambda: est.fit(Xtr, ytr, **kw), 2)
        except Exception as exc:  # sin GPU disponible
            rows.append({'método': name, 'error': str(exc)[:80]})
            continue
        pred_s, _, _, p = timed(lambda: est.predict_proba(Xte), 2)
        rows.append({'método': name, 'fit_s': fit_s, 'fit_sd': sd, 'predict_s': pred_s, 'mem_mb': mem,
                     'arboles': int(getattr(est, 'best_iteration', base['n_estimators'] - 1) + 1)
                     if 'early_stopping_rounds' in cfg else base['n_estimators'],
                     'balanced_accuracy': balanced_accuracy_score(yte, p.argmax(axis=1))})
    return pd.DataFrame(rows)


def svm_methods(X, y, Xte, yte, ns):
    """SVC con núcleo RBF frente a LinearSVC y SGD (bisagra) al crecer n."""
    rows = []
    for n in ns:
        for name, make in [('SVC (RBF)', lambda: SVC(C=1.0, kernel='rbf', cache_size=500)),
                           ('LinearSVC', lambda: LinearSVC(C=0.1, dual=True, max_iter=3000, tol=1e-3)),
                           ('SGDClassifier (hinge)', lambda: SGDClassifier(loss='hinge', alpha=1e-4, max_iter=50,
                                                                           tol=1e-3, random_state=0))]:
            if name.startswith('SVC') and n > 12_000:
                continue  # cuadrático/cúbico: se omite por costo; se extrapola con el exponente
            est = make()
            fit_s, _, mem, _ = timed(lambda: est.fit(X[:n], y[:n]), 1)
            pred_s, _, _, pred = timed(lambda: est.predict(Xte), 1)
            rows.append({'método': name, 'n': n, 'fit_s': fit_s, 'predict_s': pred_s, 'mem_mb': mem,
                         'balanced_accuracy': balanced_accuracy_score(yte, pred)})
    return pd.DataFrame(rows)


def joblib_backends(work, n_tasks=12, n_jobs=6):
    """Mismo lote de tareas en serie, con ``loky`` (procesos) y con ``threading`` (hilos)."""
    rows = []
    t0 = time.perf_counter()
    [work(i) for i in range(n_tasks)]
    serial = time.perf_counter() - t0
    rows.append({'backend': 'serie (1 proceso)', 'n_jobs': 1, 'segundos': serial, 'aceleración': 1.0})
    for backend in ['loky', 'threading']:
        with parallel_backend(backend, n_jobs=n_jobs):
            t0 = time.perf_counter()
            Parallel()(delayed(work)(i) for i in range(n_tasks))
            s = time.perf_counter() - t0
        rows.append({'backend': backend, 'n_jobs': n_jobs, 'segundos': s, 'aceleración': serial / s})
    return pd.DataFrame(rows)


def profile(fn, top=15):
    """Perfil de CPU con cProfile; devuelve las ``top`` funciones con más tiempo acumulado."""
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    rows = []
    for (file, line, func), (cc, nc, tt, ct, _) in st.stats.items():
        rows.append({'función': f'{func} ({file.split(chr(92))[-1].split("/")[-1]}:{line})', 'llamadas': nc,
                     'tiempo_propio_s': tt, 'tiempo_acumulado_s': ct})
    return pd.DataFrame(rows).sort_values('tiempo_acumulado_s', ascending=False).head(top)


def encode(Xtr, *others):
    """One-hot ajustado con ``Xtr`` y aplicado a los demás conjuntos."""
    enc = M.make_encoder().fit(Xtr)
    return [enc.transform(Xtr)] + [enc.transform(o) for o in others]
