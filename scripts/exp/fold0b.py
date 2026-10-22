from common import *
import pickle
X, y, g = outer_data()
tr, te = N.outer_folds('clf')[0]
ens = ENS.StackedEnsemble().fit(X.iloc[tr].reset_index(drop=True), y[tr], g[tr])
pickle.dump(ens, open(C.RESULTS / 'cache' / 'exp_ens_f0.pkl', 'wb'))
print('stack', per_class(y[te], ens.predict_proba(X.iloc[te])))
for n, pb in ens.base_proba(X.iloc[te]).items():
    print(n.ljust(14), per_class(y[te], pb))
P = ens.base_proba(X.iloc[te]); avg = np.mean([P[n] for n in ens.names_ if n not in ('knn','naive_bayes')], axis=0)
print('promedio simple', per_class(y[te], avg))
