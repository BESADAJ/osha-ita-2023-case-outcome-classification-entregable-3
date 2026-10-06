"""Construye data/ para el dashboard a partir de los resultados del entregable 1.

Se corre UNA vez (en el PC, no en Render). El dashboard solo lee los archivos pequeños
que este script escribe, así no necesita el CSV de 654 MB ni entrenar nada.

Uso:
    python build_data.py NOTEBOOK.ipynb EVALUATION.json HOLDOUT_PREDICTIONS.npz DAILY_ERRORS.csv [COEFFICIENTS.csv]

Fuentes:
  - Tablas de EDA: salidas HTML ya guardadas en el notebook (celdas indicadas abajo).
  - Métricas, matriz de confusión, bootstrap, curva de aprendizaje: evaluation.json.
  - Curvas ROC y PR: se calculan con las probabilidades de holdout_predictions.npz.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

CLASSES = ["Fallecimiento (Death)", "Días de ausencia", "Traslado / restricción", "Otro caso registrable"]


def nb_table(nb, cell, out):
    html = "".join(nb["cells"][cell]["outputs"][out]["data"]["text/html"])
    return pd.read_html(io.StringIO(html))[0]


def flat(df):
    """Aplana los encabezados de varios niveles que genera pandas al guardar HTML."""
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for c in df.columns:
            parts = [p for p in c if not str(p).startswith("Unnamed")]
            cols.append(parts[-1] if parts else "indice")
        df.columns = cols
    else:
        df.columns = ["indice" if str(c).startswith("Unnamed") else c for c in df.columns]
    return df


def build_eda(nb):
    # 1. Distribución del resultado (celda 22)
    t = flat(nb_table(nb, 22, 0))
    t.columns = ["resultado", "casos", "porcentaje"]
    t.to_csv(OUT / "eda_resultado.csv", index=False)

    # 2. Faltantes por predictor (celda 11) y por resultado (celda 11)
    m = flat(nb_table(nb, 11, 6))
    m.columns = ["predictor", "faltantes_n", "faltantes_pct"]
    m.to_csv(OUT / "eda_faltantes.csv", index=False)
    mo = flat(nb_table(nb, 11, 9))
    mo.columns = ["resultado"] + list(mo.columns[1:])
    mo.to_csv(OUT / "eda_faltantes_por_resultado.csv", index=False)

    # 3. Resultado por sector industrial, en % (celda 28)
    s = flat(nb_table(nb, 28, 1))
    s.columns = ["sector"] + CLASSES
    s.to_csv(OUT / "eda_sector_resultado.csv", index=False)
    # tamaño de cada sector (celda 25)
    sz = flat(nb_table(nb, 25, 0))
    sz.columns = ["sector", "casos", "porcentaje"]
    sz.to_csv(OUT / "eda_sector_tamano.csv", index=False)

    # 4. V de Cramér predictor vs resultado (celda 31)
    v = nb_table(nb, 31, 0)
    v = v[["Predictor", "V de Cramér", "NMI"]]
    v.columns = ["predictor", "cramer_v", "nmi"]
    v.to_csv(OUT / "eda_cramer.csv", index=False)

    # 5. Casos por mes (celda 43) y por día de la semana (celda 44)
    mes = flat(nb_table(nb, 43, 0))
    mes.columns = ["mes"] + CLASSES + ["total"]
    mes.to_csv(OUT / "eda_mes.csv", index=False)
    dia = nb_table(nb, 44, 2)
    dia.columns = ["dia"] + CLASSES + ["total"]
    dia.to_csv(OUT / "eda_dia_semana.csv", index=False)


def curve_points(x, y, n=200):
    """Reduce una curva a ~n puntos para que el dashboard cargue rápido."""
    if len(x) <= n:
        return x, y
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return x[idx], y[idx]


def build_model(ev, npz_path, daily_path, coef_path):
    z = np.load(npz_path, allow_pickle=False)
    y, lr, dummy = z["y"], z["lr"], z["dummy"]

    # Métricas globales + bootstrap
    rows = []
    for model, d in ev["holdout_metrics"].items():
        for k, val in d.items():
            rows.append({"modelo": model, "metrica": k, "valor": val})
    pd.DataFrame(rows).to_csv(OUT / "modelo_metricas.csv", index=False)
    pd.DataFrame(ev["bootstrap"]["intervals"]).to_csv(OUT / "modelo_bootstrap.csv", index=False)
    pd.DataFrame(ev["bootstrap"]["paired_differences"]).to_csv(OUT / "modelo_bootstrap_diferencias.csv", index=False)

    # Reporte por clase
    rep = []
    for model, r in ev["classification_reports"].items():
        for k, name in zip("1234", CLASSES):
            rep.append({"modelo": model, "clase": name, "precision": r[k]["precision"],
                        "sensibilidad": r[k]["recall"], "f1": r[k]["f1-score"], "casos": r[k]["support"]})
    pd.DataFrame(rep).to_csv(OUT / "modelo_por_clase.csv", index=False)

    # Matrices de confusión (filas = real, columnas = predicho)
    for model, cm in ev["confusion_matrices"].items():
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(OUT / f"modelo_confusion_{model}.csv")

    # ROC y PR uno-contra-el-resto, solo para la logística (el Dummy es la diagonal / la prevalencia)
    roc_rows, pr_rows, auc_rows = [], [], []
    for k, name in enumerate(CLASSES):
        b = (y == k + 1)
        fpr, tpr, _ = roc_curve(b, lr[:, k])
        fx, fy = curve_points(fpr, tpr)
        roc_rows += [{"clase": name, "fpr": a, "tpr": c} for a, c in zip(fx, fy)]
        p, r, _ = precision_recall_curve(b, lr[:, k])
        px, py = curve_points(r[::-1], p[::-1])
        pr_rows += [{"clase": name, "recall": a, "precision": c} for a, c in zip(px, py)]
        auc_rows.append({"clase": name, "auc": roc_auc_score(b, lr[:, k]),
                         "ap": average_precision_score(b, lr[:, k]), "prevalencia": b.mean()})
    pd.DataFrame(roc_rows).to_csv(OUT / "modelo_roc.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(OUT / "modelo_pr.csv", index=False)
    pd.DataFrame(auc_rows).to_csv(OUT / "modelo_auc_por_clase.csv", index=False)

    # Curva de aprendizaje: media de los 3 pliegues por fracción
    lc = pd.DataFrame([{
        "fraccion": r["fraction"], "pliegue": r["fold"], "casos": r["train_rows"],
        "entrenamiento": r["train_metrics"]["balanced_accuracy"],
        "validacion": r["validation_metrics"]["balanced_accuracy"],
        "f1_entrenamiento": r["train_metrics"]["macro_f1"],
        "f1_validacion": r["validation_metrics"]["macro_f1"]} for r in ev["learning_curve"]])
    lc.to_csv(OUT / "modelo_curva_aprendizaje.csv", index=False)

    # Validación cruzada cronológica (usar "metrics": en las filas del Dummy,
    # "validation_metrics" repite las de la logística)
    cv = pd.DataFrame([{"modelo": r["model"], "pliegue": r["fold"], **r["metrics"]} for r in ev["cv"]])
    cv.to_csv(OUT / "modelo_cv.csv", index=False)

    pd.read_csv(daily_path).to_csv(OUT / "modelo_error_diario.csv", index=False)

    info = {k: ev[k] for k in ["train_rows", "test_rows", "train_groups", "test_groups", "encoded_features",
                               "class_counts", "death_sensitivity_wilson_ci95", "death_correctly_identified"]}
    info["sklearn"] = ev["signature"]["sklearn"]
    info["parametros"] = ev["signature"]["parameters"]
    (OUT / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")

    if coef_path:
        pd.read_csv(coef_path, index_col=0).to_csv(OUT / "modelo_coeficientes.csv")


if __name__ == "__main__":
    nb_p, ev_p, npz_p, daily_p = sys.argv[1:5]
    coef_p = sys.argv[5] if len(sys.argv) > 5 else None
    nb = json.load(open(nb_p, encoding="utf-8"))
    ev = json.load(open(ev_p, encoding="utf-8"))
    build_eda(nb)
    build_model(ev, npz_p, daily_p, coef_p)
    print("Listo:", sorted(p.name for p in OUT.iterdir()))
