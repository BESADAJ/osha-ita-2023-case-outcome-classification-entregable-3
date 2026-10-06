"""Genera las figuras y tablas del JBook a partir de ../dashboard/data (las mismas que usa el dashboard).

Uso (desde la carpeta jbook/):  python make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent / "dashboard" / "data"
FIG = HERE / "figures"
GEN = HERE / "_generated"
FIG.mkdir(exist_ok=True)
GEN.mkdir(exist_ok=True)

CLASSES = ["Fallecimiento (Death)", "Días de ausencia", "Traslado / restricción", "Otro caso registrable"]
COL = dict(zip(CLASSES, ["#b2182b", "#377eb8", "#4daf4a", "#984ea3"]))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def es(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png")
    plt.close()


# ---------- EDA
d = pd.read_csv(DATA / "eda_resultado.csv")
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(d["resultado"], d["casos"], color=[COL[c] for c in d["resultado"]])
for i, (c, p) in enumerate(zip(d["casos"], d["porcentaje"])):
    ax.text(c + 1500, i, f"{c:,} ({es(p)} %)".replace(",", "."), va="center")
ax.invert_yaxis(); ax.set_xlim(0, d["casos"].max() * 1.45); ax.set_xlabel("Casos (desarrollo)")
save("eda_resultado")

d = pd.read_csv(DATA / "eda_faltantes.csv").sort_values("faltantes_pct")
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(d["predictor"], d["faltantes_pct"], color="#377eb8")
for i, v in enumerate(d["faltantes_pct"]):
    ax.text(v + 0.3, i, f"{es(v, 1)} %", va="center")
ax.set_xlim(0, 24); ax.set_xlabel("% faltante (desarrollo)")
save("eda_faltantes")

d = pd.read_csv(DATA / "eda_sector_resultado.csv").set_index("sector")[CLASSES[1:]]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
im = ax.imshow(d.values, cmap="Blues", vmin=0, vmax=60, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES[1:]); ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index)
for i in range(d.shape[0]):
    for j in range(d.shape[1]):
        ax.text(j, i, es(d.values[i, j], 1), ha="center", va="center",
                color="white" if d.values[i, j] > 35 else "black")
plt.colorbar(im, label="% del sector"); ax.spines[:].set_visible(False)
save("eda_sector")

d = pd.read_csv(DATA / "eda_cramer.csv").sort_values("cramer_v")
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(d["predictor"], d["cramer_v"], color="#377eb8")
for i, v in enumerate(d["cramer_v"]):
    ax.text(v + 0.004, i, es(v), va="center")
ax.set_xlim(0, 0.3); ax.set_xlabel("V de Cramér con el resultado")
save("eda_cramer")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
m = pd.read_csv(DATA / "eda_mes.csv")
x = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep*"]
w = pd.read_csv(DATA / "eda_dia_semana.csv")
for ax, df, xs, t in [(axes[0], m, x, "Por mes (*septiembre hasta el 23)"), (axes[1], w, w["dia"], "Por día de la semana")]:
    bottom = np.zeros(len(df))
    for c in CLASSES[1:]:
        ax.bar(xs, df[c], bottom=bottom, color=COL[c], label=c); bottom += df[c].to_numpy()
    ax.set_title(t); ax.set_ylabel("Casos")
axes[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(1.1, -0.35), ncol=3, frameon=False)
save("eda_tiempo")

# ---------- Modelos
cm = pd.read_csv(DATA / "modelo_confusion_LogisticRegression.csv", index_col=0)
pct = cm.div(cm.sum(axis=1), axis=0) * 100
short = [c.split(" (")[0] for c in CLASSES]
fig, ax = plt.subplots(figsize=(6.5, 4.6))
ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(4)); ax.set_xticklabels(short, rotation=25, ha="right"); ax.set_yticks(range(4)); ax.set_yticklabels(short)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{es(pct.values[i, j], 1)} %", ha="center", va="center", color="white" if pct.values[i, j] > 45 else "black")
ax.set_xlabel("Predicho"); ax.set_ylabel("Observado"); ax.spines[:].set_visible(False)
save("modelo_confusion")

roc = pd.read_csv(DATA / "modelo_roc.csv"); auc = pd.read_csv(DATA / "modelo_auc_por_clase.csv").set_index("clase")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot([0, 1], [0, 1], "--", color="gray")
for c in CLASSES:
    r = roc[roc["clase"] == c]
    axes[0].plot(r["fpr"], r["tpr"], color=COL[c], label=f"{c.split(' (')[0]} (AUC {es(auc.loc[c, 'auc'])})")
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("Recall (true positive rate)"); axes[0].legend(fontsize=8)
pr = pd.read_csv(DATA / "modelo_pr.csv")
for c in CLASSES[1:]:
    r = pr[pr["clase"] == c]
    axes[1].plot(r["recall"], r["precision"], color=COL[c], label=f"{c} (AP {es(auc.loc[c, 'ap'])}; base {es(auc.loc[c, 'prevalencia'])})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_ylim(0, 1); axes[1].legend(fontsize=7)
save("modelo_roc_pr")

lc = pd.read_csv(DATA / "modelo_curva_aprendizaje.csv")
g = lc.groupby("fraccion").agg(casos=("casos", "mean"), ent=("entrenamiento", "mean"), val=("validacion", "mean"),
                               ent_s=("entrenamiento", "std"), val_s=("validacion", "std")).reset_index()
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.errorbar(g["casos"], g["ent"], yerr=g["ent_s"], fmt="o-", color="#377eb8", label="Entrenamiento", capsize=3)
ax.errorbar(g["casos"], g["val"], yerr=g["val_s"], fmt="o-", color="#e66101", label="Validación cronológica", capsize=3)
ax.set_xlabel("Casos de entrenamiento (promedio de 3 pliegues)"); ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0.3, 0.75); ax.legend()
save("modelo_aprendizaje")

# ---------- Tablas en markdown
boot = pd.read_csv(DATA / "modelo_bootstrap.csv"); met = pd.read_csv(DATA / "modelo_metricas.csv").pivot(index="metrica", columns="modelo", values="valor")
names = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1 score", "roc_auc_macro_ovr": "Macro AUC (one-vs-rest)"}
rows = ["| Métrica | Regresión logística [IC 95 %] | Dummy [IC 95 %] |", "|---|---|---|"]
for k, n in names.items():
    b = boot[boot["metric"] == k].set_index("model")
    cell = lambda m: f"{es(met.loc[k, m])} [{es(b.loc[m, 'lower'])} – {es(b.loc[m, 'upper'])}]"
    rows.append(f"| {n} | {cell('LogisticRegression')} | {cell('DummyClassifier')} |")
(GEN / "tabla_metricas.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

pc = pd.read_csv(DATA / "modelo_por_clase.csv"); pc = pc[pc["modelo"] == "LogisticRegression"]
rows = ["| Resultado | Precision | Recall | F1 score | Casos en prueba |", "|---|---|---|---|---|"]
for r in pc.itertuples():
    rows.append(f"| {r.clase} | {es(r.precision)} | {es(r.sensibilidad)} | {es(r.f1)} | {f'{r.casos:,}'.replace(',', '.')} |")
(GEN / "tabla_por_clase.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

cv = pd.read_csv(DATA / "modelo_cv.csv")
rows = ["| Modelo | Pliegue | Accuracy | Balanced accuracy | Macro F1 score | Macro AUC |", "|---|---|---|---|---|---|"]
for r in cv.itertuples():
    rows.append(f"| {r.modelo} | {r.pliegue} | {es(r.accuracy)} | {es(r.balanced_accuracy)} | {es(r.macro_f1)} | {es(r.roc_auc_macro_ovr)} |")
(GEN / "tabla_cv.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("ok")
