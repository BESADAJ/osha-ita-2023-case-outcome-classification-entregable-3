"""Evalúa una variante del ensamble en los 5 pliegues externos y en la prueba (oct-dic).

Uso: python run_eval.py <nombre_variante>
Guarda las probabilidades (fuera de pliegue externo y de prueba) en results/ensemble/<variante>/.
"""
import sys, time, pickle, json
from joblib import Parallel, delayed
from common import *

VARIANTS = {
    'stack_full': dict(),
    'stack_curso': dict(injury=False, fine=False, employer=False),
    'stack_sin_empleador': dict(employer=False),
    'stack_sin_finos': dict(fine=False),
    'unico_todo': None,
    'stack_v2': dict(full=True),
    'stack_v3': dict(full='diverso'),
}
OUT = C.RESULTS / 'ensemble'


ALL_TE = list(dict.fromkeys(ENS.FINE_TE + ENS.EMPLOYER_TE + D.FEATURES + ENS.INJURY))


def views_for(name):
    if name == 'unico_todo':  # un solo XGBoost con todas las variables (sin stacking)
        return [('xgb_todo', ENS.ViewModel('te_xgb', ALL_TE))]
    return ENS.default_views(**VARIANTS[name])


def run_fold(name, f):
    from threadpoolctl import threadpool_limits
    X, y, g = outer_data()
    path = OUT / name / f'fold{f}.npz'
    if path.exists():
        return
    with threadpool_limits(2):
        if f == 'test':
            sub, dev, test = D.classification_sets()
            Xt = ENS.add_full_codes(test)
            yt = Xt.y_clf.to_numpy() - 1
            t0 = time.time()
            ens = ENS.StackedEnsemble(views_for(name)).fit(X, y, g)
            fit_s = time.time() - t0
            P = ens.base_proba(Xt); p = ens.predict_proba(Xt, P)
            pc = ens.predict_proba(Xt, P, combiner='caruana')
            np.savez_compressed(path, p=p.astype(np.float32), p_caruana=pc.astype(np.float32), y=yt, fit_s=fit_s,
                                weights=json.dumps(ens.weights_),
                                **{f'base_{k}': v.astype(np.float32) for k, v in P.items()})
            np.savez_compressed(OUT / name / 'oof_train.npz', y=y, **{k: v.astype(np.float32) for k, v in ens.oof_.items()})
            pickle.dump(ens.meta_, open(OUT / name / 'meta_test.pkl', 'wb'))
        else:
            tr, te = N.outer_folds('clf')[f]
            t0 = time.time()
            ens = ENS.StackedEnsemble(views_for(name)).fit(X.iloc[tr].reset_index(drop=True), y[tr], g[tr])
            fit_s = time.time() - t0
            P = ens.base_proba(X.iloc[te]); p = ens.predict_proba(X.iloc[te], P)
            pc = ens.predict_proba(X.iloc[te], P, combiner='caruana')
            np.savez_compressed(path, p=p.astype(np.float32), p_caruana=pc.astype(np.float32), y=y[te], fit_s=fit_s,
                                weights=json.dumps(ens.weights_),
                                **{f'base_{k}': v.astype(np.float32) for k, v in P.items()})


if __name__ == '__main__':
    names = sys.argv[1:] or ['stack_full']
    for n in names:
        (OUT / n).mkdir(parents=True, exist_ok=True)
    jobs = [(n, f) for n in names for f in ['test', 0, 1, 2, 3, 4]]
    Parallel(n_jobs=min(6, len(jobs)), backend='loky', verbose=5)(delayed(run_fold)(n, f) for n, f in jobs)
    print('listo', names)
