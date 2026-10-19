"""Pruebas estadísticas para comparar modelos.

Clasificación: Friedman → Nemenyi (diagrama CD) → DeLong con Holm → delta de Cliff.
Regresión: MCS → Giacomini-White / Clark-West → Diebold-Mariano (HLN) con bootstrap estacionario
→ d de Cohen. Además: Reality Check / SPA, bootstrap BCa por establecimiento.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------------------------- Friedman/Nemenyi
def friedman(scores):
    """Test de Friedman sobre una matriz bloques × modelos (mayor = mejor).

    Returns
    -------
    dict con estadístico chi², g.l., p-valor, rangos promedio (1 = mejor) y N, k.
    """
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, scores)
    chi2, p = stats.friedmanchisquare(*scores.T)
    # Corrección de Iman-Davenport (distribución F), menos conservadora.
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else np.inf
    p_ff = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (n - 1))
    return {'chi2': float(chi2), 'df': k - 1, 'p_value': float(p), 'iman_davenport_F': float(ff),
            'p_value_F': float(p_ff), 'avg_ranks': ranks.mean(axis=0), 'n_blocks': n, 'k': k}


def nemenyi_cd(k, n, alpha=0.05):
    """Diferencia crítica de Nemenyi: q_alpha * sqrt(k(k+1)/(6N)).

    q_alpha es el cuantil del rango studentizado con k grupos y g.l. infinitos dividido por √2.
    """
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def cd_diagram(names, avg_ranks, cd, ax, title=''):
    """Diagrama de Diferencia Crítica (Demšar, 2006).

    Ubica cada modelo según su rango promedio y une con una barra a los grupos cuya diferencia de
    rangos es menor que la CD (estadísticamente equivalentes según Nemenyi).
    """
    order = np.argsort(avg_ranks)
    r = np.asarray(avg_ranks)[order]
    nm = [names[i] for i in order]
    k = len(r)
    lo, hi = 1, max(k, int(np.ceil(r.max())))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.hlines(0.85, lo, hi, color='k', lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0.85, 0.88, color='k', lw=1)
        ax.text(t, 0.9, str(t), ha='center', va='bottom', fontsize=8)
    ax.hlines(0.97, lo, lo + cd, color='C3', lw=2)
    ax.text(lo + cd / 2, 0.985, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=8, color='C3')
    half = (k + 1) // 2
    for i, (name, rank) in enumerate(zip(nm, r)):
        left = i < half
        y = 0.75 - 0.6 * ((i if left else k - 1 - i) + 0.5) / max(half, 1)
        x_text = lo - 0.4 if left else hi + 0.4
        ax.plot([rank, rank, x_text], [0.85, y, y], color='0.3', lw=0.8)
        ax.text(x_text, y, f'{name} ({rank:.2f})', ha='right' if left else 'left', va='center', fontsize=7.5)
    # Barras de grupos equivalentes (cliques maximales de rangos contiguos).
    groups, level = [], 0
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    for a, b in groups:
        y = 0.82 - 0.025 * level
        ax.hlines(y, r[a] - 0.03, r[b] + 0.03, color='C0', lw=3)
        level += 1
    if title:
        ax.set_title(title, fontsize=10, pad=26)
    return groups


# ----------------------------------------------------------------------------------- DeLong
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        a = i
        while a < n and z[a] == z[i]:
            a += 1
        t[i:a] = 0.5 * (i + a - 1) + 1
        i = a
    out = np.empty(n)
    out[j] = t
    return out


def delong_test(y_true, s1, s2):
    """Test de DeLong para dos AUC correlacionadas (mismas observaciones).

    Implementación rápida de Sun y Xu (2014). ``y_true`` binario.

    Returns
    -------
    dict con auc1, auc2, diferencia, error estándar, z y p-valor bilateral.
    """
    y = np.asarray(y_true).astype(bool)
    m, n = y.sum(), (~y).sum()
    preds = np.vstack([s1, s2])
    pos, neg = preds[:, y], preds[:, ~y]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-300)))
    z = diff / se
    return {'auc1': float(aucs[0]), 'auc2': float(aucs[1]), 'diff': float(diff), 'se': se, 'z': float(z),
            'p_value': float(2 * stats.norm.sf(abs(z)))}


def adjust(pvalues, method='holm'):
    """p-valores ajustados por Holm-Bonferroni (``'holm'``) o Benjamini-Hochberg (``'fdr_bh'``)."""
    return multipletests(pvalues, method=method)[1]


# ----------------------------------------------------------------------------- Tamaño del efecto
def cliffs_delta(a, b):
    """Delta de Cliff: P(a > b) − P(a < b) ∈ [−1, 1]; |δ| < 0,147 despreciable, < 0,33 pequeño,
    < 0,474 mediano, si no grande (Romano et al., 2006)."""
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).mean()
    lt = (a[:, None] < b[None, :]).mean()
    return float(gt - lt)


def cliffs_label(d):
    d = abs(d)
    return 'despreciable' if d < 0.147 else 'pequeño' if d < 0.33 else 'mediano' if d < 0.474 else 'grande'


def cohens_d_paired(diff):
    """d de Cohen para diferencias pareadas: media / desviación estándar."""
    diff = np.asarray(diff, dtype=float)
    return float(diff.mean() / diff.std(ddof=1))


# ------------------------------------------------------------------- Diebold-Mariano y familia
def _hac_var(d, h):
    """Varianza de largo plazo (Newey-West con h-1 rezagos) de la media de ``d``."""
    d = np.asarray(d, dtype=float) - np.mean(d)
    T = len(d)
    gamma0 = d @ d / T
    v = gamma0
    for lag in range(1, h):
        v += 2 * (1 - lag / h) * (d[lag:] @ d[:-lag]) / T
    return v / T


def diebold_mariano(loss1, loss2, h=1):
    """Test de Diebold-Mariano con la corrección de Harvey-Leybourne-Newbold (1997).

    d_t = loss1_t − loss2_t. H0: E[d_t] = 0. Un estadístico positivo indica que el modelo 2 es mejor.
    La corrección HLN multiplica el estadístico por sqrt((T+1−2h+h(h−1)/T)/T) y usa una t con T−1 g.l.
    """
    d = np.asarray(loss1, dtype=float) - np.asarray(loss2, dtype=float)
    T = len(d)
    dm = d.mean() / np.sqrt(_hac_var(d, h))
    k = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * k
    return {'mean_diff': float(d.mean()), 'dm': float(dm), 'dm_hln': float(stat),
            'p_value': float(2 * stats.t.sf(abs(stat), df=T - 1)), 'T': T}


def clark_west(y, pred_small, pred_large):
    """Test de Clark-West (2007) para modelos anidados (H1: el grande predice mejor).

    f_t = e1_t² − (e2_t² − (ŷ1_t − ŷ2_t)²); se contrasta la media de f_t con una t unilateral (HAC).
    """
    e1, e2 = y - pred_small, y - pred_large
    f = e1 ** 2 - (e2 ** 2 - (pred_small - pred_large) ** 2)
    t = f.mean() / np.sqrt(_hac_var(f, 1))
    return {'mean_adj_diff': float(f.mean()), 't': float(t), 'p_value': float(stats.norm.sf(t))}


def giacomini_white(loss1, loss2, instruments=None):
    """Test de habilidad predictiva condicional de Giacomini-White (2006), horizonte 1.

    H0: E[h_{t-1} d_t] = 0 con instrumentos h_{t-1} = (1, d_{t-1}). Estadístico de Wald
    T * Zbar' Ω⁻¹ Zbar ~ χ²(q). Si se rechaza, el signo de la media de d indica el mejor modelo.
    """
    d = np.asarray(loss1, dtype=float) - np.asarray(loss2, dtype=float)
    if instruments is None:
        instruments = np.column_stack([np.ones(len(d) - 1), d[:-1]])
        d = d[1:]
    Z = instruments * d[:, None]
    T, q = Z.shape
    zbar = Z.mean(axis=0)
    omega = (Z - zbar).T @ (Z - zbar) / T
    stat = float(T * zbar @ np.linalg.solve(omega, zbar))
    return {'stat': stat, 'df': q, 'p_value': float(stats.chi2.sf(stat, q)), 'mean_diff': float(d.mean())}


def stationary_bootstrap_ci(d, reps=2000, block=7, seed=0, alpha=0.05):
    """IC del promedio de ``d`` con el bootstrap estacionario de Politis y Romano (1994)."""
    from arch.bootstrap import StationaryBootstrap
    bs = StationaryBootstrap(block, np.asarray(d, dtype=float), seed=seed)
    ci = bs.conf_int(np.mean, reps=reps, method='percentile', size=1 - alpha)
    return float(ci[0, 0]), float(ci[1, 0])


def model_confidence_set(losses, alpha=0.10, reps=2000, block=7, seed=0):
    """Model Confidence Set de Hansen, Lunde y Nason (2011) con ``arch``.

    Parameters
    ----------
    losses : pandas.DataFrame
        Pérdidas T × modelos (por ejemplo, error cuadrático diario medio).
    """
    from arch.bootstrap import MCS
    mcs = MCS(losses, size=alpha, reps=reps, block_size=block, method='R', bootstrap='stationary', seed=seed)
    mcs.compute()
    pv = mcs.pvalues.rename(columns={'Pvalue': 'p_value_mcs'})
    pv['in_mcs'] = pv.index.isin(mcs.included)
    pv['avg_loss'] = losses.mean().reindex(pv.index)
    return pv.sort_values('avg_loss')


def spa_test(benchmark_loss, model_losses, reps=2000, block=7, seed=0):
    """Superior Predictive Ability de Hansen (2005) (y Reality Check de White como cota superior)."""
    from arch.bootstrap import SPA
    spa = SPA(benchmark_loss, model_losses, reps=reps, block_size=block, bootstrap='stationary', seed=seed)
    spa.compute()
    return {k: float(v) for k, v in spa.pvalues.items()}


# ------------------------------------------------------------------------------- Bootstrap BCa
def group_bca_ci(values, weights, groups, reps=2000, seed=0, alpha=0.05):
    """IC BCa de una media ponderada remuestreando *establecimientos* completos.

    Se agregan numerador y denominador por establecimiento y se aplica ``scipy.stats.bootstrap``
    (BCa) sobre los establecimientos; el estadístico es sum(num)/sum(den).
    """
    df = pd.DataFrame({'g': groups, 'num': np.asarray(values) * np.asarray(weights), 'den': weights})
    agg = df.groupby('g')[['num', 'den']].sum()
    num, den = agg.num.to_numpy(float), agg.den.to_numpy(float)
    G, S_num, S_den = len(num), num.sum(), den.sum()
    theta = S_num / S_den
    # Bootstrap: cada réplica remuestrea G establecimientos con reemplazo (conteos multinomiales).
    rng = np.random.default_rng(seed)
    boot = np.empty(reps)
    for i in range(0, reps, 200):
        k = min(200, reps - i)
        w = rng.multinomial(G, np.full(G, 1.0 / G), size=k)
        boot[i:i + k] = (w @ num) / (w @ den)
    # Corrección de sesgo z0 y aceleración a (jackknife dejando fuera un establecimiento, O(G)).
    z0 = stats.norm.ppf(np.clip((boot < theta).mean() + 0.5 * (boot == theta).mean(), 1e-6, 1 - 1e-6))
    jack = (S_num - num) / (S_den - den)
    d = jack.mean() - jack
    a = (d ** 3).sum() / (6.0 * ((d ** 2).sum()) ** 1.5)
    q = []
    for z in stats.norm.ppf([alpha / 2, 1 - alpha / 2]):
        q.append(stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
    lo, hi = np.quantile(boot, q)
    return float(lo), float(hi)
