"""
03_modeling_tuning.py — Integrante 3: Marco algorítmico, modelado iterativo y optimización

Entrada : data/processed/X_train.parquet, y_train.parquet
          data/processed/X_test.parquet, y_test.parquet (solo validación de contrato)
Salida  : models/xgb_iter1_baseline.json
          models/xgb_iter2_features.json
          models/xgb_iter3_tuned.json
          reports/tables/cv_results.csv
          reports/tables/tabla05_resultados_cv.{csv,md}
          reports/tables/tabla06_hiperparametros.{csv,md}
          reports/tables/03_modeling_log.json
          reports/figures/fig07_cv_comparison.png (300 dpi)
          reports/figures/fig08_feature_importance.png (300 dpi)

Protocolo de evaluación
----------------------
Todas las métricas de esta etapa provienen de validación cruzada estratificada
de 5 folds sobre Train (StratifiedKFold, shuffle=True, random_state=42).
X_test se carga únicamente para verificar el contrato (forma, columnas e
índices); NO se calcula ninguna métrica sobre él: esa evaluación corresponde
a la etapa 4.

Convención de etiquetas (verificada empíricamente, ver CLAUDE.md):
    label = 0 -> Real   |   label = 1 -> Fake (clase positiva)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from matplotlib.patches import Patch
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MOD = ROOT / "models"
TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"
for d in (MOD, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

DPI = 300
N_SPLITS = 5
N_JOBS = 5
N_ITER = 30

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Contrato de columnas
# ---------------------------------------------------------------------------
TITLE_COLS = [f"title_svd_{i:02d}" for i in range(50)]
TEXT_COLS = [f"text_svd_{i:02d}" for i in range(100)]
STRUCT_COLS = [
    "title_char_len",
    "title_word_len",
    "text_char_len",
    "text_word_len",
    "title_caps_ratio",
    "title_excl_count",
    "title_quest_count",
    "lexical_diversity",
    "title_sentiment",
    "text_sentiment",
]
ALL_COLS = TITLE_COLS + TEXT_COLS + STRUCT_COLS

EXPECTED_SHAPES = {"X_train": (50106, 160), "X_test": (12527, 160)}

# Métricas solicitadas: clave de scikit-learn -> etiqueta en español
METRICS = {
    "accuracy": "Exactitud",
    "precision": "Precisión",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC_AUC",
    "average_precision": "PR_AUC",
}
SCORING = list(METRICS.keys())

CV = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Hiperparámetros por defecto de XGBClassifier usados en las iteraciones 1 y 2
DEFAULT_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.3,
    "n_estimators": 100,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

# Espacio de búsqueda de la iteración 3 (RandomizedSearchCV)
SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
    "n_estimators": [200, 300, 400, 500, 600, 800],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1, 2, 5],
    "reg_lambda": [0.5, 1, 2, 5, 10, 20],
}
PARAM_ORDER = list(SEARCH_SPACE.keys())

GROUP_COLORS = {
    "SVD título": "#2E6F9E",
    "SVD cuerpo": "#C0392B",
    "Estructural": "#D68910",
}
METRIC_COLORS = {
    "f1": "#2E6F9E",
    "roc_auc": "#D68910",
    "average_precision": "#1E8449",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8")
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")


def to_python(obj):
    """Convierte tipos de numpy a tipos nativos serializables por json."""
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def format_value(value) -> str:
    """Presenta hiperparámetros sin decimales superfluos."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def feature_group(name: str) -> str:
    if name.startswith("title_svd_"):
        return "SVD título"
    if name.startswith("text_svd_"):
        return "SVD cuerpo"
    return "Estructural"


def make_estimator(**overrides) -> XGBClassifier:
    """XGBClassifier con los hiperparámetros por defecto de las iteraciones 1–2."""
    params = {
        **DEFAULT_PARAMS,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def metric_summary(cv_result: dict) -> dict:
    """Resume media y DE muestral (ddof=1) de las 6 métricas en los 5 folds."""
    summary = {}
    for key, label in METRICS.items():
        values = np.asarray(cv_result[f"test_{key}"], dtype=float)
        summary[key] = {
            "label": label,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "folds": [float(v) for v in values],
        }
    return summary


# ---------------------------------------------------------------------------
# Carga y validación del contrato
# ---------------------------------------------------------------------------
def load_and_validate():
    print("[1/7] Cargando y validando el contrato de datos")
    X_train = pd.read_parquet(PROC / "X_train.parquet")
    X_test = pd.read_parquet(PROC / "X_test.parquet")
    y_train_df = pd.read_parquet(PROC / "y_train.parquet")
    y_test_df = pd.read_parquet(PROC / "y_test.parquet")

    assert X_train.shape == EXPECTED_SHAPES["X_train"], f"X_train inesperado: {X_train.shape}"
    assert X_test.shape == EXPECTED_SHAPES["X_test"], f"X_test inesperado: {X_test.shape}"
    assert list(X_train.columns) == ALL_COLS, "Columnas de X_train fuera de contrato"
    assert list(X_test.columns) == ALL_COLS, "Columnas de X_test fuera de contrato"
    assert list(y_train_df.columns) == ["label"] and list(y_test_df.columns) == ["label"]
    assert len(y_train_df) == len(X_train) and len(y_test_df) == len(X_test)
    assert X_train.index.equals(y_train_df.index), "Índices Train desalineados"
    assert X_test.index.equals(y_test_df.index), "Índices Test desalineados"
    assert set(X_train.index).isdisjoint(set(X_test.index)), "Fuga de IDs entre Train y Test"
    assert np.isfinite(X_train.to_numpy()).all(), "NaN o inf en X_train"
    assert np.isfinite(X_test.to_numpy()).all(), "NaN o inf en X_test"
    assert y_train_df["label"].isin([0, 1]).all() and y_test_df["label"].isin([0, 1]).all()

    y_train = y_train_df["label"].to_numpy(dtype=np.int8).ravel()
    y_test = y_test_df["label"].to_numpy(dtype=np.int8).ravel()

    print(f"      X_train {X_train.shape} | X_test {X_test.shape} | 160 columnas verificadas")
    print(
        "      Train: Real=0 -> "
        f"{int((y_train == 0).sum()):,} | Fake=1 -> {int((y_train == 1).sum()):,}"
    )
    print("      X_test: solo validación de contrato; sin métricas en esta etapa")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Iteraciones 1 y 2
# ---------------------------------------------------------------------------
def run_iteration(name: str, cols: list[str], X: pd.DataFrame, y: np.ndarray, tag: str):
    print(f"      CV {N_SPLITS} folds sobre {len(cols)} variables ({tag}) ...", flush=True)
    estimator = make_estimator()
    t0 = time.perf_counter()
    cv_result = cross_validate(
        estimator,
        X[cols],
        y,
        cv=CV,
        scoring=SCORING,
        n_jobs=N_JOBS,
        error_score="raise",
    )
    cv_seconds = time.perf_counter() - t0
    metrics = metric_summary(cv_result)
    print(
        f"      {name}: F1={metrics['f1']['mean']:.4f}±{metrics['f1']['std']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']['mean']:.4f} | CV {cv_seconds:.1f}s",
        flush=True,
    )

    print(f"      Ajustando {name} en todo Train y guardando modelo ...", flush=True)
    final_model = make_estimator()
    t0 = time.perf_counter()
    final_model.fit(X[cols], y)
    fit_seconds = time.perf_counter() - t0
    model_path = MOD / f"xgb_{name}.json"
    final_model.save_model(model_path)

    return {
        "metrics": metrics,
        "cv_seconds": cv_seconds,
        "fit_seconds": fit_seconds,
        "n_features": len(cols),
        "model_path": model_path,
        "cv_fit_time_mean": float(np.mean(cv_result["fit_time"])),
        "model": final_model,
    }


# ---------------------------------------------------------------------------
# Iteración 3: búsqueda aleatoria + CV final
# ---------------------------------------------------------------------------
def run_search(X: pd.DataFrame, y: np.ndarray):
    print(
        f"      RandomizedSearchCV: {N_ITER} candidatos × {N_SPLITS} folds = "
        f"{N_ITER * N_SPLITS} ajustes (n_jobs={N_JOBS}) ...",
        flush=True,
    )
    search = RandomizedSearchCV(
        estimator=make_estimator(),
        param_distributions=SEARCH_SPACE,
        n_iter=N_ITER,
        scoring="f1",
        cv=CV,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        error_score="raise",
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    search_seconds = time.perf_counter() - t0
    print(
        f"      Búsqueda finalizada en {search_seconds:.1f}s | "
        f"mejor F1 CV={search.best_score_:.4f}",
        flush=True,
    )

    best_params = {k: v for k, v in search.best_params_.items()}
    print(f"      Mejores hiperparámetros: {best_params}", flush=True)

    best_model = search.best_estimator_
    best_model.save_model(MOD / "xgb_iter3_tuned.json")

    print("      Cross-validate final del mejor estimador con el mismo CV ...", flush=True)
    t0 = time.perf_counter()
    cv_result = cross_validate(
        best_model,
        X,
        y,
        cv=CV,
        scoring=SCORING,
        n_jobs=N_JOBS,
        error_score="raise",
    )
    cv_seconds = time.perf_counter() - t0
    metrics = metric_summary(cv_result)
    print(
        f"      Iteración 3: F1={metrics['f1']['mean']:.4f}±{metrics['f1']['std']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']['mean']:.4f} | CV {cv_seconds:.1f}s",
        flush=True,
    )

    return {
        "search": search,
        "best_params": best_params,
        "search_seconds": search_seconds,
        "cv_seconds": cv_seconds,
        "metrics": metrics,
        "model": best_model,
        "n_features": int(X.shape[1]),
        "cv_fit_time_mean": float(np.mean(cv_result["fit_time"])),
    }


# ---------------------------------------------------------------------------
# Importancias gain
# ---------------------------------------------------------------------------
def gain_importance(model: XGBClassifier) -> pd.DataFrame:
    booster = model.get_booster()
    names = booster.feature_names
    if names is None or all(str(n).startswith("f") and str(n)[1:].isdigit() for n in names):
        names = ALL_COLS
    raw = booster.get_score(importance_type="gain")
    gains = np.array([float(raw.get(name, 0.0)) for name in names], dtype=float)
    total = float(gains.sum())
    df = pd.DataFrame({
        "Variable": names,
        "Grupo": [feature_group(str(n)) for n in names],
        "Gain": gains,
    })
    df["Gain_pct"] = np.where(total > 0, 100.0 * df["Gain"] / total, 0.0)
    return df.sort_values("Gain", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tablas y figuras
# ---------------------------------------------------------------------------
def build_table5(results: dict, iter3_seconds: float) -> pd.DataFrame:
    rows = []
    definition = [
        ("Iteración 1 — Baseline", "50 SVD título", "iter1", results["iter1"]["cv_seconds"]),
        ("Iteración 2 — Enriquecida", "160: SVD + estructurales", "iter2", results["iter2"]["cv_seconds"]),
        ("Iteración 3 — Optimizada", "160", "iter3", iter3_seconds),
    ]
    for label, variables, key, seconds in definition:
        metrics = results[key]["metrics"]
        rows.append({
            "Iteración": label,
            "Variables": variables,
            "Exactitud_M": round(metrics["accuracy"]["mean"], 4),
            "Exactitud_DE": round(metrics["accuracy"]["std"], 4),
            "Precisión_M": round(metrics["precision"]["mean"], 4),
            "Precisión_DE": round(metrics["precision"]["std"], 4),
            "Recall_M": round(metrics["recall"]["mean"], 4),
            "Recall_DE": round(metrics["recall"]["std"], 4),
            "F1_M": round(metrics["f1"]["mean"], 4),
            "F1_DE": round(metrics["f1"]["std"], 4),
            "ROC_AUC_M": round(metrics["roc_auc"]["mean"], 4),
            "ROC_AUC_DE": round(metrics["roc_auc"]["std"], 4),
            "PR_AUC_M": round(metrics["average_precision"]["mean"], 4),
            "PR_AUC_DE": round(metrics["average_precision"]["std"], 4),
            "Tiempo_s": round(seconds, 1),
        })
    return pd.DataFrame(rows)


def build_table6(best_params: dict) -> pd.DataFrame:
    rows = []
    for param in PARAM_ORDER:
        rows.append({
            "Hiperparámetro": param,
            "Rango explorado": str(SEARCH_SPACE[param]),
            "Iteración 1–2 (por defecto)": format_value(DEFAULT_PARAMS[param]),
            "Iteración 3 (óptima)": format_value(best_params[param]),
        })
    return pd.DataFrame(rows)


def plot_cv_comparison(table5: pd.DataFrame, results: dict) -> None:
    iter_keys = ["iter1", "iter2", "iter3"]
    x_labels = [
        "Iteración 1\nBaseline",
        "Iteración 2\nEnriquecida",
        "Iteración 3\nOptimizada",
    ]
    plot_metrics = [
        ("f1", "F1", METRIC_COLORS["f1"]),
        ("roc_auc", "ROC-AUC", METRIC_COLORS["roc_auc"]),
        ("average_precision", "PR-AUC", METRIC_COLORS["average_precision"]),
    ]
    x = np.arange(len(iter_keys))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    all_lows = []
    all_highs = []
    for j, (key, label, color) in enumerate(plot_metrics):
        means = np.array([results[k]["metrics"][key]["mean"] for k in iter_keys])
        stds = np.array([results[k]["metrics"][key]["std"] for k in iter_keys])
        all_lows.extend(means - stds)
        all_highs.extend(means + stds)
        positions = x + (j - 1) * width
        ax.bar(
            positions,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=label,
            error_kw={"elinewidth": 1.1, "ecolor": "#333333"},
        )
        for pos, mean, std in zip(positions, means, stds):
            ax.text(
                pos,
                mean - 0.008,
                f"{mean:.4f}",
                ha="center",
                va="top",
                fontsize=7.6,
                color="white",
                fontweight="bold",
            )

    low = min(all_lows)
    high = max(all_highs)
    bottom = max(0.0, np.floor((low - 0.02) * 50.0) / 50.0)
    ax.set_ylim(bottom, min(1.0, high + 0.02))
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9.5)
    ax.set_ylabel("Puntuación media en validación cruzada (5 folds)")
    ax.set_title(
        "Comparación de las tres iteraciones mediante validación cruzada",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=9)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(
        0.01,
        0.01,
        "Nota. Barras de error = ±1 DE entre folds. Eje y truncado para resaltar diferencias.",
        fontsize=7.8,
        color="#444444",
    )
    fig.savefig(FIG / "fig07_cv_comparison.png")
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame) -> None:
    top = importance.head(15).iloc[::-1].reset_index(drop=True)
    colors = [GROUP_COLORS[g] for g in top["Grupo"]]

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    bars = ax.barh(top["Variable"], top["Gain"], color=colors, height=0.68)
    xmax = float(top["Gain"].max())
    for bar, value in zip(bars, top["Gain"]):
        ax.text(
            value + xmax * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.1f}",
            va="center",
            ha="left",
            fontsize=8.1,
        )
    ax.set_xlim(0, xmax * 1.17)
    ax.set_xlabel("Gain medio (importancia por ganancia)")
    ax.set_ylabel("Variable")
    ax.set_title(
        "Top-15 importancias (gain) del modelo optimizado — Iteración 3",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.tick_params(axis="y", labelsize=8.6)
    handles = [Patch(facecolor=GROUP_COLORS[g], label=g) for g in GROUP_COLORS]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_feature_importance.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main() -> None:
    script_start = time.perf_counter()
    X_train, y_train, X_test, y_test = load_and_validate()

    print("[2/7] Definiendo protocolo de validación cruzada y estimadores")
    print(
        f"      {N_SPLITS} folds estratificados, shuffle=True, random_state={RANDOM_STATE} | "
        f"métricas: {', '.join(SCORING)}"
    )
    print("      Estimadores con n_jobs=1; paralelismo a nivel de folds (n_jobs=5)")

    print("[3/7] Iteración 1 — Baseline (50 SVD de título)")
    iter1 = run_iteration("iter1_baseline", TITLE_COLS, X_train, y_train, "título")

    print("[4/7] Iteración 2 — Enriquecida (160 variables)")
    iter2 = run_iteration("iter2_features", ALL_COLS, X_train, y_train, "completo")

    print("[5/7] Iteración 3 — Optimizada (RandomizedSearchCV sobre 160 variables)")
    iter3 = run_search(X_train, y_train)
    iter3_seconds = iter3["search_seconds"] + iter3["cv_seconds"]

    results = {"iter1": iter1, "iter2": iter2, "iter3": iter3}

    print("[6/7] Guardando tablas, figuras e importancias")
    search = iter3["search"]
    cv_results = (
        pd.DataFrame(search.cv_results_)
        .sort_values("rank_test_score", kind="mergesort")
        .reset_index(drop=True)
    )
    cv_results.to_csv(TAB / "cv_results.csv", index=False, encoding="utf-8")
    print(f"      -> reports/tables/cv_results.csv ({len(cv_results)} candidatos)")

    table5 = build_table5(results, iter3_seconds)
    save_table(table5, "tabla05_resultados_cv")

    table6 = build_table6(iter3["best_params"])
    save_table(table6, "tabla06_hiperparametros")

    importance = gain_importance(iter3["model"])
    plot_cv_comparison(table5, results)
    plot_feature_importance(importance)
    print("      -> fig07_cv_comparison.png y fig08_feature_importance.png (300 dpi)")

    print("[7/7] Validando modelos guardados y escribiendo el log de la etapa 3")
    model_paths = {
        "iter1": MOD / "xgb_iter1_baseline.json",
        "iter2": MOD / "xgb_iter2_features.json",
        "iter3": MOD / "xgb_iter3_tuned.json",
    }
    reload_info = {}
    reload_cols = {"iter1": TITLE_COLS, "iter2": ALL_COLS, "iter3": ALL_COLS}
    for key, path in model_paths.items():
        reloaded = XGBClassifier()
        reloaded.load_model(path)
        n_rounds = int(reloaded.get_booster().num_boosted_rounds())
        proba = reloaded.predict_proba(X_train[reload_cols[key]].iloc[:5])
        assert proba.shape == (5, 2), f"Predicción inesperada al recargar {path.name}"
        reload_info[key] = {
            "path": str(path.relative_to(ROOT)),
            "n_boosted_rounds": n_rounds,
            "n_features": int(reloaded.get_booster().num_features()),
            "reload_ok": True,
        }
        print(f"      {path.name}: recargado ({n_rounds} árboles, {reloaded.get_booster().num_features()} variables)")

    total_seconds = time.perf_counter() - script_start
    class_counts = {
        "train": {
            "Real": int((y_train == 0).sum()),
            "Fake": int((y_train == 1).sum()),
            "total": int(len(y_train)),
            "pct_Fake": round(100.0 * float((y_train == 1).mean()), 2),
        },
        "test": {
            "Real": int((y_test == 0).sum()),
            "Fake": int((y_test == 1).sum()),
            "total": int(len(y_test)),
            "pct_Fake": round(100.0 * float((y_test == 1).mean()), 2),
        },
    }

    log = {
        "random_state": RANDOM_STATE,
        "versions": {
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "class_balance": class_counts,
        "cv_protocol": {
            "estimator": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "scoring": SCORING,
            "error_score": "raise",
            "n_jobs_cv": N_JOBS,
            "estimator_n_jobs": 1,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "metrics": METRICS,
            "std_ddof": 1,
        },
        "n_iter": N_ITER,
        "n_fits_search": N_ITER * N_SPLITS,
        "timings": {
            "iter1_cv_s": round(iter1["cv_seconds"], 2),
            "iter1_final_fit_s": round(iter1["fit_seconds"], 2),
            "iter2_cv_s": round(iter2["cv_seconds"], 2),
            "iter2_final_fit_s": round(iter2["fit_seconds"], 2),
            "iter3_search_s": round(iter3["search_seconds"], 2),
            "iter3_final_cv_s": round(iter3["cv_seconds"], 2),
            "iter3_total_s": round(iter3_seconds, 2),
            "iter3_note": (
                "El tiempo de la iteración 3 incluye 150 ajustes de la búsqueda "
                "(30 candidatos × 5 folds) más el cross-validate final del mejor estimador."
            ),
            "total_script_s": round(total_seconds, 2),
        },
        "iterations": {
            key: {
                "variables": int(results[key]["n_features"]),
                "metrics": {
                    metric: {
                        "mean": round(results[key]["metrics"][metric]["mean"], 6),
                        "std": round(results[key]["metrics"][metric]["std"], 6),
                        "folds": [round(v, 6) for v in results[key]["metrics"][metric]["folds"]],
                    }
                    for metric in METRICS
                },
                "cv_seconds": round(results[key]["cv_seconds"], 2),
            }
            for key in ("iter1", "iter2", "iter3")
        },
        "iter3_search": {
            "best_score_f1": round(float(search.best_score_), 6),
            "best_index": int(search.best_index_),
            "best_params": iter3["best_params"],
        },
        "search_space": SEARCH_SPACE,
        "default_params": DEFAULT_PARAMS,
        "top20_importance_gain": [
            {
                "rank": int(i + 1),
                "variable": str(row.Variable),
                "grupo": str(row.Grupo),
                "gain": round(float(row.Gain), 6),
                "gain_pct": round(float(row.Gain_pct), 4),
            }
            for i, row in importance.head(20).iterrows()
        ],
        "outputs": {
            "script": "src/03_modeling_tuning.py",
            "models": [str(p.relative_to(ROOT)) for p in model_paths.values()],
            "tables": [
                "reports/tables/cv_results.csv",
                "reports/tables/tabla05_resultados_cv.csv",
                "reports/tables/tabla05_resultados_cv.md",
                "reports/tables/tabla06_hiperparametros.csv",
                "reports/tables/tabla06_hiperparametros.md",
                "reports/tables/03_modeling_log.json",
            ],
            "figures": [
                "reports/figures/fig07_cv_comparison.png",
                "reports/figures/fig08_feature_importance.png",
            ],
        },
        "reload_check": reload_info,
        "test_usage": (
            "X_test se cargó únicamente para validar forma, columnas e índices. "
            "No se calcularon métricas ni predicciones sobre Test en esta etapa."
        ),
    }
    (TAB / "03_modeling_log.json").write_text(
        json.dumps(to_python(log), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[Listo] Resultados de validación cruzada (Train)")
    print(table5.to_string(index=False))
    print("\nMejores hiperparámetros de la iteración 3:")
    print(json.dumps(to_python(iter3["best_params"]), indent=2, ensure_ascii=False))
    print(f"\nDuración total: {total_seconds:.1f}s")
    if not (
        results["iter3"]["metrics"]["f1"]["mean"]
        > max(results["iter1"]["metrics"]["f1"]["mean"], results["iter2"]["metrics"]["f1"]["mean"])
    ):
        print("AVISO: la iteración 3 no supera en F1 a las iteraciones previas (reportado sin ajustes).")


if __name__ == "__main__":
    main()
