"""Funciones que construyen las figuras Plotly (plantilla y paleta únicas para todo el dashboard)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .data import CLASSES, LABELS, PRETTY, TYPE_OF_INCIDENT

# Paleta limitada (guía §7.4): cuatro colores semánticos, uno por clase del desenlace, más tinta, gris y un acento
# reservado al modelo original. «Otro registrable», la clase residual, va en gris.
INK, TEAL, AMBER, CORAL, GRAY = '#0f172a', '#0e7490', '#d97706', '#b91c1c', '#94a3b8'
ACCENT, SLATE = '#1d4ed8', '#64748b'
NAVY = INK  # alias conservado por compatibilidad
CLASS_COLORS = dict(zip(CLASSES, [CORAL, TEAL, AMBER, SLATE]))
SEQ = [[0, '#ffffff'], [0.5, '#93c5fd'], [1, '#1e3a8a']]

_axis = dict(gridcolor='#eef2f6', zeroline=False, linecolor='#cbd5e1', ticks='outside', tickcolor='#cbd5e1', ticklen=4,
             title=dict(font=dict(size=12, color='#475569')), tickfont=dict(size=11.5, color='#475569'), automargin=True)
pio.templates['osha'] = go.layout.Template(layout=go.Layout(
    separators=',.',  # coma decimal y punto de miles, como el resto del dashboard
    font=dict(family='Inter, Segoe UI, system-ui, sans-serif', size=12.5, color='#0f172a'),
    colorway=[INK, TEAL, AMBER, CORAL, GRAY], paper_bgcolor='white', plot_bgcolor='white',
    margin=dict(l=56, r=16, t=56, b=48), title=dict(x=0, xref='paper', y=0.98, yanchor='top',
                                                    font=dict(size=14, color='#0f172a')),
    xaxis=_axis, yaxis=_axis, hoverlabel=dict(bgcolor='white', bordercolor='#cbd5e1', font=dict(size=12)),
    legend=dict(orientation='h', y=-0.18, x=0, font=dict(size=11.5, color='#475569'))))
pio.templates.default = 'osha'


def empty(msg='Sin datos para los filtros elegidos.', height=280):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=GRAY), x=0.5, y=0.5, xref='paper', yref='paper')
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height)
    return fig


def target_bar(d):
    vc = d.outcome.value_counts().reindex(CLASSES).fillna(0)
    pct = vc / vc.sum() * 100
    fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=[CLASS_COLORS[c] for c in vc.index],
                           text=[f'{p:.1f} %'.replace('.', ',') for p in pct], textposition='outside',
                           hovertemplate='%{x}<br>%{y:,} casos<extra></extra>'))
    fig.update_layout(title='Distribución del desenlace (balance de clases)', yaxis_title='Casos (n)', xaxis_title='Desenlace')
    return fig


def hist_feature(d, col, top=15):
    if col in ('days_away', 'days_restricted', 'employees'):
        v = d[col].dropna()
        v = v[v > 0]
        fig = go.Figure(go.Histogram(x=np.log10(v), nbinsx=40, marker_color=NAVY,
                                     hovertemplate='log10 = %{x}<br>%{y} casos<extra></extra>'))
        fig.update_layout(title=f'{LABELS[col]} (escala log10, solo > 0)', xaxis_title=f'log10({LABELS[col]})',
                          yaxis_title='Casos (n)')
        return fig
    ct = pd.crosstab(d[col], d.outcome).reindex(columns=CLASSES).fillna(0)
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index[:top]]
    if col == 'type_of_incident':
        ct.index = [f'{TYPE_OF_INCIDENT.get(str(i), i)} ({i})' for i in ct.index]
    fig = go.Figure([go.Bar(y=ct.index.astype(str), x=ct[c], name=c, orientation='h', marker_color=CLASS_COLORS[c],
                            hovertemplate='%{y}: %{x:,} casos<extra>' + c + '</extra>') for c in CLASSES])
    fig.update_layout(barmode='stack', title=f'{LABELS.get(col, col)}: {top} categorías más frecuentes por desenlace',
                      xaxis_title='Casos (n)', yaxis=dict(autorange='reversed', title=''), height=460)
    return fig


def cramers_v(a, b):
    ct = pd.crosstab(a, b).to_numpy()
    n = ct.sum()
    if n == 0 or min(ct.shape) < 2:
        return np.nan
    e = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
    chi2 = ((ct - e) ** 2 / np.where(e > 0, e, 1)).sum()
    return float(np.sqrt(chi2 / n / (min(ct.shape) - 1)))


def cramer_heatmap(d, cols):
    M = np.array([[1.0 if a == b else cramers_v(d[a], d[b]) for b in cols] for a in cols])
    names = [LABELS.get(c, c) for c in cols]
    fig = go.Figure(go.Heatmap(z=M, x=names, y=names, colorscale=SEQ, zmin=0, zmax=1,
                               text=np.round(M, 2), texttemplate='%{text}', colorbar=dict(title='V'),
                               hovertemplate='%{y} × %{x}<br>V de Cramér = %{z:.2f}<extra></extra>'))
    fig.update_layout(title='Asociación entre variables (V de Cramér; categóricas)', height=560,
                      xaxis=dict(tickangle=-40))
    return fig


def missing_bar(d, cols):
    miss = pd.Series({LABELS.get(c, c): float((d[c].isna() | d[c].astype(str).eq('Unknown')).mean() * 100) for c in cols})
    miss = miss.sort_values()
    fig = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation='h', marker_color=SLATE,
                           text=[f'{v:.1f} %'.replace('.', ',') for v in miss.values], textposition='outside', cliponaxis=False,
                           hovertemplate='%{y}: %{x:.1f} % faltante<extra></extra>'))
    fig.update_layout(title='Valores faltantes o «Unknown» por variable', xaxis_title='Casos sin dato (%)', height=460,
                      margin=dict(r=48))
    return fig


def outlier_box(d):
    fig = go.Figure()
    for col, color in [('days_away', TEAL), ('days_restricted', AMBER)]:
        v = d[col].dropna()
        v = v[v > 0]
        fig.add_trace(go.Box(y=v, name=LABELS[col], marker_color=color, boxpoints='outliers',
                             hovertemplate='%{y} días<extra></extra>'))
    fig.update_layout(title='Duración de ausencia y restricción: atípicos (escala log)', yaxis_type='log',
                      yaxis_title='Días (escala log)', showlegend=False)
    return fig


def monthly(d):
    ct = pd.crosstab(d.date.dt.to_period('M').astype(str), d.outcome).reindex(columns=CLASSES).fillna(0)
    share = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure([go.Scatter(x=share.index, y=share[c], name=c, mode='lines+markers', line_color=CLASS_COLORS[c],
                                hovertemplate='%{x}: %{y:.1f} %<extra>' + c + '</extra>') for c in CLASSES])
    fig.update_layout(title='Participación mensual de cada desenlace', yaxis_title='Casos del mes (%)', xaxis_title='Mes')
    return fig


def bivariate(d, x, y, color):
    if y in ('days_away', 'days_restricted', 'employees'):
        dd = d[[x, y, color]].dropna()
        top = dd[x].value_counts().index[:12]
        dd = dd[dd[x].isin(top)]
        fig = go.Figure()
        groups = CLASSES if color == 'outcome' else dd[color].value_counts().index[:5]
        palette = [CLASS_COLORS.get(g, c) for g, c in zip(groups, [NAVY, TEAL, AMBER, CORAL, GRAY])]
        for g, c in zip(groups, palette):
            s = dd[dd[color] == g]
            fig.add_trace(go.Box(x=s[x].astype(str), y=s[y].clip(lower=0.5), name=str(g), marker_color=c, boxpoints=False))
        fig.update_layout(boxmode='group', yaxis_type='log', title=f'{LABELS.get(y, y)} según {LABELS.get(x, x)}',
                          yaxis_title=f'{LABELS.get(y, y)} (log)', xaxis_title=LABELS.get(x, x))
        return fig
    top_x = d[x].value_counts().index[:12]
    dd = d[d[x].isin(top_x)]
    ct = pd.crosstab(dd[x].astype(str), dd[color].astype(str), normalize='index') * 100
    cols = [c for c in (CLASSES if color == 'outcome' else ct.sum().sort_values(ascending=False).index[:5]) if c in ct]
    fig = go.Figure([go.Bar(x=ct.index, y=ct[c], name=str(c),
                            marker_color=CLASS_COLORS.get(c, [NAVY, TEAL, AMBER, CORAL, GRAY][i % 5]),
                            hovertemplate='%{x}: %{y:.1f} %<extra>' + str(c) + '</extra>') for i, c in enumerate(cols)])
    fig.update_layout(barmode='stack', title=f'Composición de {LABELS.get(color, color)} según {LABELS.get(x, x)}',
                      yaxis_title='Casos de la categoría (%)', xaxis_title=LABELS.get(x, x))
    return fig


def proba_bar(p, pred):
    fig = go.Figure(go.Bar(x=CLASSES, y=p * 100, marker_color=[CLASS_COLORS[c] for c in CLASSES],
                           marker_line_width=[3 if i == pred else 0 for i in range(4)], marker_line_color='#111',
                           text=[f'{v * 100:.1f} %'.replace('.', ',') for v in p], textposition='outside'))
    fig.update_layout(title='Probabilidad estimada por desenlace', yaxis_title='Probabilidad (%)',
                      yaxis_range=[0, max(100, p.max() * 115)], height=330)
    return fig


def shap_bar(phi, cls_name, title='Explicación local (SHAP exacto)'):
    s = phi.sort_values()
    fig = go.Figure(go.Bar(x=s.values * 100, y=[LABELS.get(i, i) for i in s.index], orientation='h',
                           marker_color=[ACCENT if v > 0 else '#e11d48' for v in s.values],
                           hovertemplate='%{y}: %{x:+.2f} puntos porcentuales<extra></extra>'))
    fig.update_layout(title=f'{title}: aporte a P({cls_name})', xaxis_title='Aporte a la probabilidad (puntos porcentuales)',
                      height=380, margin=dict(l=200))
    return fig


def confusion(y, pred):
    cm = pd.crosstab(pd.Series(y, name='real'), pd.Series(pred, name='pred')).reindex(index=range(4), columns=range(4)).fillna(0)
    rec = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig = go.Figure(go.Heatmap(z=rec.values, x=CLASSES, y=CLASSES, colorscale=SEQ,
                               zmin=0, zmax=100, text=cm.values.astype(int), texttemplate='%{text}',
                               hovertemplate='Real %{y}<br>Predicho %{x}<br>%{z:.1f} % de la fila<extra></extra>',
                               colorbar=dict(title='% fila')))
    fig.update_layout(title='Matriz de confusión (conteos; color = % de la fila)', xaxis_title='Predicho',
                      yaxis=dict(title='Real', autorange='reversed'), height=420, margin=dict(l=150))
    return fig


def roc_pr(y, P, kind='roc'):
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    fig = go.Figure()
    for k, c in enumerate(CLASSES):
        yk = (y == k).astype(int)
        if yk.sum() == 0:
            continue
        if kind == 'roc':
            a, b, _ = roc_curve(yk, P[:, k])
            name = f'{c} (AUC {auc(a, b):.3f})'.replace('.', ',')
        else:
            b, a, _ = precision_recall_curve(yk, P[:, k])
            name = f'{c} (prevalencia {yk.mean() * 100:.1f} %)'.replace('.', ',')
        step = max(1, len(a) // 400)
        fig.add_trace(go.Scatter(x=a[::step], y=b[::step], mode='lines', name=name, line_color=CLASS_COLORS[c]))
    if kind == 'roc':
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dot', color=GRAY), name='Azar'))
        fig.update_layout(title='Curvas ROC uno contra el resto', xaxis_title='Tasa de falsos positivos',
                          yaxis_title='Tasa de verdaderos positivos')
    else:
        fig.update_layout(title='Curvas precisión-exhaustividad', xaxis_title='Exhaustividad (recall)',
                          yaxis_title='Precisión')
    fig.update_layout(height=420)
    return fig


def calibration(y, P, bins=10):
    fig = go.Figure()
    for k, c in enumerate(CLASSES[1:], start=1):
        p, yk = P[:, k], (y == k)
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, bins - 1)
        mp = [p[idx == b].mean() for b in range(bins) if (idx == b).any()]
        fr = [yk[idx == b].mean() for b in range(bins) if (idx == b).any()]
        fig.add_trace(go.Scatter(x=mp, y=fr, mode='lines+markers', name=c, line_color=CLASS_COLORS[c]))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dot', color=GRAY), name='Calibración perfecta'))
    fig.update_layout(title='Diagrama de calibración (deciles de probabilidad)', xaxis_title='Probabilidad predicha media',
                      yaxis_title='Frecuencia observada', height=420)
    return fig


def shap_global_bar(s, model):
    s = s.sort_values()
    fig = go.Figure(go.Bar(x=s.values * 100, y=[LABELS.get(i, i) for i in s.index], orientation='h', marker_color=NAVY,
                           hovertemplate='%{y}: %{x:.2f} pp<extra></extra>'))
    fig.update_layout(title=f'Importancia global SHAP: {PRETTY.get(model, model)}',
                      xaxis_title='|SHAP| medio (puntos porcentuales de la clase predicha)', height=380, margin=dict(l=200))
    return fig


def cd_diagram(names, ranks, cd, highlight=None):
    """Diagrama de diferencia crítica (Demšar, 2006) en Plotly."""
    order = np.argsort(ranks)
    fig = go.Figure()
    k = len(ranks)
    for pos, i in enumerate(order):
        y = -(pos + 1)
        color = ACCENT if names[i] == highlight else INK
        fig.add_trace(go.Scatter(x=[ranks[i]], y=[y], mode='markers+text', marker=dict(size=12, color=color),
                                 text=[f'  {names[i]} ({ranks[i]:.2f})'.replace('.', ',')], textposition='middle right',
                                 hovertemplate=f'{names[i]}<br>rango medio %{{x:.2f}}<extra></extra>', showlegend=False))
    best = ranks[order[0]]
    fig.add_shape(type='rect', x0=best, x1=best + cd, y0=-k - 0.6, y1=-0.4, fillcolor='rgba(29,78,216,0.08)', line_width=0)
    fig.add_annotation(x=best + cd / 2, y=-0.2, text=f'CD de Nemenyi = {cd:.2f}'.replace('.', ','), showarrow=False,
                       font=dict(color=ACCENT))
    fig.update_layout(title='Diferencia crítica de Nemenyi (rango 1 = mejor)',
                      xaxis=dict(title='Rango medio en los pliegues (menor es mejor)', range=[0.5, k + 2.5]),
                      yaxis=dict(visible=False), height=60 + 45 * k)
    return fig


def convergence(loss, gap):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss, mode='lines+markers', name='Log-loss F(wᵗ)', line_color=NAVY))
    fig.add_trace(go.Scatter(y=gap, mode='lines+markers', name='Brecha de Frank-Wolfe Gᵗ', line_color=ACCENT, yaxis='y2'))
    fig.update_layout(title='Convergencia de Frank-Wolfe (pesos globales, un pliegue)', xaxis_title='Iteración t',
                      yaxis_title='Log-loss fuera de pliegue', yaxis2=dict(title='Brecha (escala log)', overlaying='y',
                                                                            side='right', type='log'), height=380)
    return fig


def paired_folds(rows):
    """BA por pliegue del original y de KNN (15 pliegues; la repetición 0 son los pliegues ya vistos al elegir G)."""
    fig = go.Figure()
    x = [f'R{r["rep"]}·P{r["fold"]}' for r in rows]
    for i, r in enumerate(rows):
        fig.add_trace(go.Scatter(x=[x[i], x[i]], y=[r['knn'], r['original']], mode='lines', showlegend=False,
                                 line=dict(color='#cbd5e1', width=2), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=[r['knn'] for r in rows], mode='markers', name='KNN (mejor benchmark)',
                             marker=dict(size=9, color=INK, symbol='circle-open', line=dict(width=2)),
                             hovertemplate='%{x}<br>KNN: BA %{y:.3f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=x, y=[r['original'] for r in rows], mode='markers', name='Original (SECR)',
                             marker=dict(size=9, color=ACCENT), hovertemplate='%{x}<br>Original: BA %{y:.3f}<extra></extra>'))
    fig.add_vrect(x0=-0.5, x1=4.5, fillcolor='rgba(217,119,6,0.08)', line_width=0)
    fig.add_annotation(x=2, y=1.06, yref='paper', showarrow=False, text='Repetición 0: pliegues vistos al elegir G',
                       font=dict(size=11, color=AMBER))
    fig.update_layout(title='Balanced accuracy por pliegue: original frente a KNN (3 repeticiones × 5 pliegues)',
                      yaxis_title='Balanced accuracy', xaxis_title='Repetición · pliegue', height=380,
                      margin=dict(t=72), legend=dict(y=-0.28))
    return fig


def ablation_bar(rows):
    """BA media ± d.e. de SECR completo y de cada variante (mismos modelos base, mismos 15 pliegues)."""
    rows = sorted(rows, key=lambda r: r['ba_mean'])
    names = [r['variante'] for r in rows]
    colors = [ACCENT if 'original' in r['variante'] else ('#cbd5e1' if (r.get('p_holm') is None or r['p_holm'] != r['p_holm']
              or r['p_holm'] >= 0.05) else GRAY) for r in rows]
    txt = []
    for r in rows:
        p = r.get('p_holm')
        txt.append('' if p is None or p != p else (f'p Holm = {p:.3f}'.replace('.', ',') + (' *' if p < 0.05 else '')))
    fig = go.Figure(go.Bar(x=[r['ba_mean'] for r in rows], y=names, orientation='h', marker_color=colors,
                           error_x=dict(type='data', array=[r['ba_sd'] for r in rows], color='#94a3b8', thickness=1),
                           hovertemplate='%{y}<br>BA media %{x:.3f}<extra></extra>'))
    for n, s in zip(names, txt):  # p de Holm en una columna fija a la derecha, sin tapar las barras de error
        fig.add_annotation(x=1.0, xref='paper', xanchor='left', y=n, text=s or 'referencia', showarrow=False,
                           font=dict(size=11.5, color='#475569'))
    fig.update_layout(title='Ablación: BA media (± d.e.) al quitar o reemplazar cada componente',
                      xaxis=dict(title='Balanced accuracy (15 pliegues)', range=[0, 0.75]), height=380,
                      margin=dict(l=10, r=130))
    return fig


SENS_LABELS = {'alpha': 'α (corrección por prior)', 'lam': 'λ (contracción)', 'n_steps': 'Pasos de Frank-Wolfe',
               'gate': 'Variable de compuerta', 'frac_datos': 'Fracción de datos internos'}


def sensitivity(rows, param):
    r = [x for x in rows if x['param'] == param]
    fig = go.Figure(go.Scatter(x=[x['value'] for x in r], y=[x['ba_mean'] for x in r], mode='lines+markers',
                               line=dict(color=ACCENT), marker=dict(size=8),
                               error_y=dict(type='data', array=[x['ba_sd'] for x in r], color='#cbd5e1', thickness=1),
                               hovertemplate='%{x}: BA %{y:.3f}<extra></extra>'))
    fig.update_layout(title=SENS_LABELS.get(param, param),
                      xaxis=dict(title='', type='category', tickangle=-40 if param == 'gate' else 0),
                      yaxis=dict(title='BA media', range=[0, 0.8]), height=320, margin=dict(l=48, r=8))
    return fig
