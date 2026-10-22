import time, pickle
from common import *
X, y, g = outer_data()
tr, te = N.outer_folds('clf')[0]
t = time.time()
ens = ENS.StackedEnsemble().fit(X.iloc[tr].reset_index(drop=True), y[tr], g[tr])
print('tiempo', round(time.time() - t), 's')
P = ens.base_proba(X.iloc[te]); p = ens.predict_proba(X.iloc[te], P)
print('STACK', per_class(y[te], p), {k: round(v, 4) for k, v in report(y[te], p).items() if k in ('accuracy','f1','log_loss','ba3','ece')})
for n, pb in P.items():
    print(n.ljust(16), per_class(y[te], pb), round(report(y[te], pb)['accuracy'], 4))
pickle.dump(ens, open(C.RESULTS / 'cache' / 'exp_ens_f0.pkl', 'wb'))
