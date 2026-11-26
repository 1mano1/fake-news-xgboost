"""
04_eval_shap_errors.py — Integrante 4: Evaluación comparativa, explicabilidad y errores

Entrada : data/processed/X_test.parquet, y_test.parquet
          data/interim/welfake_clean.parquet (texto original, vía índice id)
          models/xgb_iter{1,2,3}_*.json  y  models/{tfidf,svd}_{title,text}.joblib
Salida  : reports/tables/tabla07_metricas_test.{csv,md}
          reports/tables/tabla08_errores.{csv,md}
          reports/tables/04_errores_texto.md      (texto de las 6 noticias, para análisis)
          reports/tables/04_terminos_svd.csv      (términos de los componentes SVD influyentes)
          reports/tables/04_eval_log.json
          reports/figures/fig09 a fig12 (300 dpi)

Los modelos se cargan tal como los entregó la etapa 3; aquí no se reentrena
ni se ajusta nada (tampoco el umbral de decisión, fijo en 0.5).

Convención de etiquetas: label = 0 -> Real  |  label = 1 -> Fake (clase positiva)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
THRESHOLD = 0.5
N_ERRORS_PER_TYPE = 3
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MOD = ROOT / "models"
TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TITLE_COLS = [f"title_svd_{i:02d}" for i in range(50)]
MODELS = {
    "Iteración 1 — Baseline": ("xgb_iter1_baseline", TITLE_COLS),
    "Iteración 2 — Enriquecida": ("xgb_iter2_features", None),
    "Iteración 3 — Optimizada": ("xgb_iter3_tuned", None),
}
FINAL = "Iteración 3 — Optimizada"
NAMES = {0: "Real", 1: "Fake"}


def save_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(TAB / f"{stem}.csv", index=False, encoding="utf-8")
    (TAB / f"{stem}.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Datos y modelos
# ---------------------------------------------------------------------------
X = pd.read_parquet(PROC / "X_test.parquet")
y = pd.read_parquet(PROC / "y_test.parquet").iloc[:, 0].astype(int)
assert X.shape == (12527, 160), X.shape
assert X.index.equals(y.index)
assert np.isfinite(X.to_numpy()).all()

models, probs = {}, {}
for name, (stem, cols) in MODELS.items():
    model = XGBClassifier()
    model.load_model(MOD / f"{stem}.json")
    cols = cols or list(X.columns)
    models[name] = (model, cols)
    probs[name] = model.predict_proba(X[cols])[:, 1]

# ---------------------------------------------------------------------------
# 2. Tabla 7: métricas en Test
# ---------------------------------------------------------------------------
rows = []
for name, p in probs.items():
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rows.append({
        "Modelo": name,
        "Exactitud": accuracy_score(y, pred),
        "Precisión": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp),
    })
table7 = pd.DataFrame(rows).round(4)
save_table(table7, "tabla07_metricas_test")
print(table7.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Figura 9: curvas ROC, PR y matriz de confusión del modelo final
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for (name, p), color, style in zip(probs.items(), ["#999999", "#E69F00", "#0072B2"], [":", "--", "-"]):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    short = name.split(" — ")[0]
    axes[0].plot(fpr, tpr, color=color, ls=style, label=f"{short} ({roc_auc_score(y, p):.4f})")
    axes[1].plot(rec, prec, color=color, ls=style, label=f"{short} ({average_precision_score(y, p):.4f})")
axes[0].plot([0, 1], [0, 1], color="black", lw=0.6, alpha=0.4)
axes[0].set(title="(a) Curva ROC", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
axes[1].set(title="(b) Curva precisión-recall", xlabel="Recall", ylabel="Precisión")
axes[0].legend(title="ROC-AUC", frameon=False, loc="lower right")
axes[1].legend(title="PR-AUC", frameon=False, loc="lower left")

cm = confusion_matrix(y, (probs[FINAL] >= THRESHOLD).astype(int))
axes[2].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})",
                     ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
axes[2].set(title="(c) Matriz de confusión, iteración 3", xlabel="Clase predicha", ylabel="Clase verdadera",
            xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
fig.tight_layout()
fig.savefig(FIG / "fig09_test_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. SHAP (iteración 3). Los valores están en log-odds; positivo empuja a Fake.
# ---------------------------------------------------------------------------
final_model = models[FINAL][0]
explainer = shap.TreeExplainer(final_model)
explanation = explainer(X)
sv = explanation.values

# Comprobación de aditividad: base + suma(SHAP) debe igualar el margen del modelo
margin = final_model.predict(X, output_margin=True)
additivity_error = float(np.abs(explanation.base_values + sv.sum(axis=1) - margin).max())
assert additivity_error < 1e-2, additivity_error

plt.figure()
shap.summary_plot(explanation, X, max_display=20, show=False, plot_size=(9, 8))
plt.xlabel("Valor SHAP (log-odds; positivo empuja hacia Fake)")
cbar = plt.gcf().axes[-1]
cbar.set_ylabel("Valor de la variable")
cbar.set_yticklabels(["Bajo", "Alto"])
plt.savefig(FIG / "fig10_shap_summary.png")
plt.close("all")

mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
groups = {
    "SVD título (50)": [c for c in X.columns if c.startswith("title_svd_")],
    "SVD cuerpo (100)": [c for c in X.columns if c.startswith("text_svd_")],
}
groups["Estructurales y sentimiento (10)"] = [
    c for c in X.columns if c not in groups["SVD título (50)"] + groups["SVD cuerpo (100)"]
]
group_share = {g: float(mean_abs[cols].sum() / mean_abs.sum()) for g, cols in groups.items()}

# Términos con mayor carga en los componentes SVD más influyentes
svd_rows = []
vocab, comps = {}, {}
for field in ("title", "text"):
    vocab[field] = joblib.load(MOD / f"tfidf_{field}.joblib").get_feature_names_out()
    comps[field] = joblib.load(MOD / f"svd_{field}.joblib").components_
for col in mean_abs.index[:20]:
    if "_svd_" not in col:
        continue
    field, idx = col.split("_svd_")
    weights = comps[field][int(idx)]
    order = np.argsort(weights)
    svd_rows.append({
        "variable": col,
        "shap_medio_abs": round(float(mean_abs[col]), 4),
        "terminos_positivos": ", ".join(vocab[field][order[::-1][:10]]),
        "terminos_negativos": ", ".join(vocab[field][order[:10]]),
    })
pd.DataFrame(svd_rows).to_csv(TAB / "04_terminos_svd.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------------------
# 5. Errores del modelo final: los 3 FP y 3 FN en que más se equivoca
#    (selección determinista, sin escoger casos a mano)
# ---------------------------------------------------------------------------
res = pd.DataFrame({"id": X.index, "y": y.to_numpy(), "p": probs[FINAL]})
res["pred"] = (res["p"] >= THRESHOLD).astype(int)
errors = res[res.y != res.pred]
fps = errors[errors.y == 0].sort_values(["p", "id"], ascending=[False, True]).head(N_ERRORS_PER_TYPE)
fns = errors[errors.y == 1].sort_values(["p", "id"], ascending=[True, True]).head(N_ERRORS_PER_TYPE)
selected = pd.concat([fps, fns])

clean = pd.read_parquet(ROOT / "data" / "interim" / "welfake_clean.parquet").set_index("id")
assert (clean.loc[X.index, "label"].to_numpy() == y.to_numpy()).all()

pos = {article_id: i for i, article_id in enumerate(X.index)}
err_rows, dump = [], ["# Noticias mal clasificadas por la iteración 3 (texto original)\n"]
for r in selected.itertuples():
    i = pos[r.id]
    top = np.argsort(np.abs(sv[i]))[::-1][:5]
    art = clean.loc[r.id]
    kind = "FP" if r.y == 0 else "FN"
    err_rows.append({
        "id": int(r.id), "Tipo": kind, "Real": NAMES[r.y], "Predicha": NAMES[r.pred],
        "P(Fake)": round(float(r.p), 4),
        "Titular": art["title"],
        "Palabras cuerpo": len(art["text"].split()),
        "Variables principales (SHAP)": "; ".join(f"{X.columns[k]} ({sv[i, k]:+.2f})" for k in top),
    })
    dump.append(
        f"## {kind} · id {r.id} · P(Fake) = {r.p:.4f}\n\n**Título:** {art['title']}\n\n"
        f"**Cuerpo ({len(art['text'].split())} palabras, primeros 2,500 caracteres):** {art['text'][:2500]}\n"
    )
table8 = pd.DataFrame(err_rows)
save_table(table8, "tabla08_errores")
(TAB / "04_errores_texto.md").write_text("\n".join(dump), encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. Figuras 11 y 12: waterfall de un acierto claro y de un error
# ---------------------------------------------------------------------------
def waterfall(article_id: int, path: Path) -> None:
    plt.figure()
    shap.plots.waterfall(explanation[pos[article_id]], max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    fig.canvas.draw()
    for ax in fig.axes:  # shap rotula en inglés el bloque de variables agrupadas
        labels = [t.get_text() for t in ax.get_yticklabels()]
        if any("other features" in t for t in labels):
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([t.replace("other features", "variables restantes") for t in labels])
    plt.savefig(path)
    plt.close("all")


# Acierto claro: verdadero positivo con mayor P(Fake). Error: el de mayor distancia al umbral.
hit = res[(res.y == 1) & (res.pred == 1)].sort_values(["p", "id"], ascending=[False, True]).iloc[0]
miss = selected.assign(dist=(selected.p - THRESHOLD).abs()).sort_values(["dist", "id"], ascending=[False, True]).iloc[0]
waterfall(int(hit.id), FIG / "fig11_shap_waterfall_acierto.png")
waterfall(int(miss.id), FIG / "fig12_shap_waterfall_error.png")


def case_info(r) -> dict:
    return {"id": int(r.id), "real": NAMES[int(r.y)], "p_fake": float(r.p),
            "titulo": clean.loc[int(r.id), "title"]}


# ---------------------------------------------------------------------------
# 7. Diagnósticos y log
# ---------------------------------------------------------------------------
X_train = pd.read_parquet(PROC / "X_train.parquet")
body_cols = groups["SVD cuerpo (100)"]
train_keys = set(map(tuple, X_train[body_cols].to_numpy().round(6)))
same_body = np.array([tuple(v) in train_keys for v in X[body_cols].to_numpy().round(6)])

final_pred = res.pred.to_numpy()
confidence = np.maximum(res.p, 1 - res.p)
log = {
    "n_test": int(len(y)),
    "clases_test": {"Real": int((y == 0).sum()), "Fake": int((y == 1).sum())},
    "umbral": THRESHOLD,
    "metricas_test": table7.to_dict(orient="records"),
    "errores_iter3": {
        "total": int(len(errors)), "FP": int((errors.y == 0).sum()), "FN": int((errors.y == 1).sum()),
        "con_confianza_mayor_0.9": int(((res.y != res.pred) & (confidence > 0.9)).sum()),
        "con_confianza_menor_0.7": int(((res.y != res.pred) & (confidence < 0.7)).sum()),
    },
    "shap": {
        "base_value_log_odds": float(np.ravel(explanation.base_values)[0]),
        "error_max_aditividad": additivity_error,
        "top20_media_abs": {k: float(v) for k, v in mean_abs.head(20).items()},
        "proporcion_por_grupo": group_share,
    },
    "waterfall_acierto": case_info(hit),
    "waterfall_error": case_info(miss),
    "solapamiento_cuerpo_train_test": {
        "descripcion": "Artículos de Test cuyo vector SVD del cuerpo (redondeado a 6 decimales) también está en Train",
        "n": int(same_body.sum()),
        "exactitud_en_ellos": float((final_pred[same_body] == y.to_numpy()[same_body]).mean()) if same_body.any() else None,
        "f1_sin_ellos": float(f1_score(y.to_numpy()[~same_body], final_pred[~same_body])),
    },
}
(TAB / "04_eval_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: log[k] for k in ("errores_iter3", "shap", "waterfall_acierto", "waterfall_error",
                                      "solapamiento_cuerpo_train_test")}, indent=2, ensure_ascii=False))
print(table8.drop(columns=["Titular"]).to_string(index=False))
