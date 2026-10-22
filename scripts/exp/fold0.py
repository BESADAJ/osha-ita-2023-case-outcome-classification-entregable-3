import time
from common import *
X, y, g = outer_data()
tr, te = N.outer_folds('clf')[0]
t = time.time()
ens = ENS.StackedEnsemble().fit(X.iloc[tr].reset_index(drop=True), y[tr], g[tr])
p = ens.predict_proba(X.iloc[te])
print('tiempo', round(time.time() - t), 's')
for rule in ['argmax', 'prior']:
    print('stack', rule, {k: round(v, 4) for k, v in report(y[te], p, ens.decide(p, rule)).items()})
for n, pb in ens.base_proba(X.iloc[te]).items():
    print('base', n, {k: round(v, 4) for k, v in report(y[te], pb).items() if k in ('auc', 'log_loss', 'ba3', 'ba_decision', 'accuracy')})
f = pd.read_csv(C.RESULTS / 'outer_folds_clf.csv'); f0 = f[f.fold == 0]
print('mejores del curso en el pliegue 1: BA', f0.outer_recall.max().round(4), '| AUC', f0.outer_auc.max().round(4), '| logloss', f0.outer_log_loss.min().round(4), '| acc', f0.outer_accuracy.max().round(4))
print('meta coef abs medio por modelo:', {n: round(float(abs(ens.meta_.coef_[:, i*4:(i+1)*4]).mean()), 3) for i, n in enumerate(ens.names_)})
