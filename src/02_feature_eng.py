"""
02_feature_eng.py — Integrante 2: Preprocesamiento NLP e Ingeniería de Características

Entrada : data/interim/welfake_clean.parquet
Salida  : data/processed/{X_train,X_test,y_train,y_test}.parquet
          models/{tfidf_title,tfidf_text,svd_title,svd_text,scaler}.joblib
          reports/tables/tabla03_limpieza_fuentes.{csv,md}
          reports/tables/tabla04_features_resumen.{csv,md}
          reports/tables/02_feature_log.json
          reports/figures/fig05_svd_explained_variance.png (300 dpi)
          reports/figures/fig06_structural_features.png (300 dpi)

Convención de etiquetas (verificada empíricamente, ver CLAUDE.md):
    label = 0 -> Real   |   label = 1 -> Fake
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim" / "welfake_clean.parquet"
PROC = ROOT / "data" / "processed"
MOD = ROOT / "models"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
for d in (PROC, MOD, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {0: "Real", 1: "Fake"}
PALETTE = {"Real": "#2E6F9E", "Fake": "#C0392B"}
ORDER = ["Real", "Fake"]
DPI = 300
TEXT_SENTIMENT_CHARS = 1000

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Reglas de limpieza de marcadores de procedencia (source leakage)
# ---------------------------------------------------------------------------
RE_TITLE_SUFFIX = re.compile(
    r"\s*[-–—|]\s*(?:The New York Times|Breitbart(?:\s+News)?|Reuters)\s*$",
    flags=re.IGNORECASE,
)
RE_REUTERS_DATELINE = re.compile(
    r"^\s*[A-Z][A-Z\s.,/()-]{0,45}\(Reuters\)\s*[-–—:]?\s*"
)
RE_REUTERS_TAG = re.compile(r"\(Reuters\)", flags=re.IGNORECASE)
RE_IMAGE_CREDITS = re.compile(
    r"Featured\s+image(?:\s+via)?[^\n.]{0,120}"
    r"|(?:Photo|Image|Screenshot|Screengrab)\s+(?:by|via)\s+[^\n.]{0,120}"
    r"|(?:via\s+)?Getty\s+Images|via\s+Flickr|Wikimedia\s+Commons",
    flags=re.IGNORECASE,
)
RE_21WIRE = re.compile(
    r"READ\s+MORE\s+[^\n]{0,80}NEWS\s+AT:\s*21st\s+Century\s+Wire[^\n]{0,80}"
    r"|SUPPORT\s+21WIRE[^\n]{0,120}"
    r"|21st\s+Century\s+Wire|21WIRE(?:\.TV)?",
    flags=re.IGNORECASE,
)
RE_URLS = re.compile(
    r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+",
    flags=re.IGNORECASE,
)
RE_MULTI_SPACE = re.compile(r"\s+")
RE_ALPHA_TOKENS = re.compile(r"[a-z]+")

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

STRUCT_LABELS = {
    "title_char_len": "Longitud del título (caracteres)",
    "title_word_len": "Longitud del título (palabras)",
    "text_char_len": "Longitud del cuerpo (caracteres)",
    "text_word_len": "Longitud del cuerpo (palabras)",
    "title_caps_ratio": "Proporción de mayúsculas en título",
    "title_excl_count": "Signos ! en título",
    "title_quest_count": "Signos ? en título",
    "lexical_diversity": "Diversidad léxica (TTR)",
    "title_sentiment": "Sentimiento del título (VADER)",
    "text_sentiment": "Sentimiento del cuerpo (VADER, 1k car.)",
}

STRUCT_SHORT_LABELS = {
    "title_char_len": "Tít. caracteres",
    "title_word_len": "Tít. palabras",
    "text_char_len": "Cuerpo caracteres",
    "text_word_len": "Cuerpo palabras",
    "title_caps_ratio": "Tít. mayúsculas",
    "title_excl_count": "Tít. signos !",
    "title_quest_count": "Tít. signos ?",
    "lexical_diversity": "Diversidad léxica",
    "title_sentiment": "Sent. título",
    "text_sentiment": "Sent. cuerpo",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def iqr(x: pd.Series) -> float:
    return float(x.quantile(0.75) - x.quantile(0.25))


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8")
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")


def class_counts_series(labels: pd.Series) -> dict[str, int]:
    vc = labels.map(LABEL_NAMES).value_counts()
    return {k: int(vc.get(k, 0)) for k in ORDER}


def clean_source_markers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elimina marcadores editoriales/fuente y registra conteos por clase para la Tabla 3."""
    titles = df["title"].astype(str)
    texts = df["text"].astype(str)
    labels = df["label_name"]
    n_total = len(df)

    m_title_suffix = titles.str.contains(RE_TITLE_SUFFIX, regex=True, na=False)
    m_reuters = (
        texts.str.contains(RE_REUTERS_DATELINE, regex=True, na=False)
        | texts.str.contains(RE_REUTERS_TAG, regex=True, na=False)
    )
    m_img = texts.str.contains(RE_IMAGE_CREDITS, regex=True, na=False)
    m_21wire = texts.str.contains(RE_21WIRE, regex=True, na=False)
    m_urls = texts.str.contains(RE_URLS, regex=True, na=False)
    m_any = m_title_suffix | m_reuters | m_img | m_21wire | m_urls

    title_clean = (
        titles.str.replace(RE_TITLE_SUFFIX, "", regex=True)
        .str.replace(RE_MULTI_SPACE, " ", regex=True)
        .str.strip()
    )

    text_clean = texts.str.replace(RE_REUTERS_DATELINE, " ", regex=True)
    text_clean = text_clean.str.replace(RE_REUTERS_TAG, " ", regex=True)
    text_clean = text_clean.str.replace(RE_IMAGE_CREDITS, " ", regex=True)
    text_clean = text_clean.str.replace(RE_21WIRE, " ", regex=True)
    text_clean = text_clean.str.replace(RE_URLS, " ", regex=True)
    text_clean = text_clean.str.replace(RE_MULTI_SPACE, " ", regex=True).str.strip()

    rules = [
        ("Sufijos editoriales en titular (- NYT, - Breitbart, - Reuters)", "title", m_title_suffix),
        ("Cabecera y firma de agencia Reuters ((Reuters))", "text", m_reuters),
        ("Créditos de imagen y agencias fotográficas (Featured image, Getty, Flickr, Wikimedia)", "text", m_img),
        ("Banners y firmas de 21st Century Wire / 21WIRE", "text", m_21wire),
        ("Enlaces web y multimedia (http/https, www, pic.twitter.com)", "text", m_urls),
        ("Al menos un marcador eliminado (título o cuerpo)", "title / text", m_any),
    ]

    rows = []
    for nombre, campo, mask in rules:
        n_aff = int(mask.sum())
        n_real = int((mask & (labels == "Real")).sum())
        n_fake = int((mask & (labels == "Fake")).sum())
        rows.append({
            "Regla de limpieza": nombre,
            "Campo": campo,
            "Artículos afectados": n_aff,
            "% del corpus": round(100.0 * n_aff / n_total, 2),
            "Real": n_real,
            "Fake": n_fake,
        })

    out_df = df.copy()
    out_df["title_clean"] = title_clean
    out_df["text_clean"] = text_clean
    return out_df, pd.DataFrame(rows)


def compute_structural_features(
    df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer
) -> pd.DataFrame:
    """Calcula las 10 variables lingüístico-estructurales y de sentimiento sin estado."""
    title_s = df["title_clean"].astype(str)
    text_s = df["text_clean"].astype(str)

    letters = title_s.str.count(r"[A-Za-z]")
    upper = title_s.str.count(r"[A-Z]")
    title_caps = (upper / letters.replace(0, np.nan)).fillna(0.0)

    lex_div = np.empty(len(df), dtype=np.float64)
    for idx, t in enumerate(text_s):
        tokens = RE_ALPHA_TOKENS.findall(t.lower())
        lex_div[idx] = (len(set(tokens)) / len(tokens)) if tokens else 0.0

    title_sent = np.empty(len(df), dtype=np.float64)
    for idx, t in enumerate(title_s):
        title_sent[idx] = analyzer.polarity_scores(t)["compound"] if t else 0.0

    text_sent = np.empty(len(df), dtype=np.float64)
    for idx, t in enumerate(text_s):
        lead = t[:TEXT_SENTIMENT_CHARS]
        text_sent[idx] = analyzer.polarity_scores(lead)["compound"] if lead else 0.0

    feats = pd.DataFrame(
        {
            "title_char_len": title_s.str.len().to_numpy(dtype=np.float64),
            "title_word_len": title_s.str.split().str.len().fillna(0).to_numpy(dtype=np.float64),
            "text_char_len": text_s.str.len().to_numpy(dtype=np.float64),
            "text_word_len": text_s.str.split().str.len().fillna(0).to_numpy(dtype=np.float64),
            "title_caps_ratio": title_caps.to_numpy(dtype=np.float64),
            "title_excl_count": title_s.str.count("!").to_numpy(dtype=np.float64),
            "title_quest_count": title_s.str.count(r"\?").to_numpy(dtype=np.float64),
            "lexical_diversity": lex_div,
            "title_sentiment": title_sent,
            "text_sentiment": text_sent,
        },
        index=df["id"].astype(np.int64).to_numpy(),
    )
    feats.index.name = "id"
    assert not feats.isna().any().any(), "Se detectaron NaN en características estructurales"
    return feats


def build_table4(struct_train: pd.DataFrame, y_train_names: pd.Series) -> pd.DataFrame:
    """Construye la Tabla 4 (descriptivos en Train sin escalar + Mann-Whitney U)."""
    rows = []
    real_mask = (y_train_names == "Real").values
    fake_mask = (y_train_names == "Fake").values

    for col in STRUCT_COLS:
        r = struct_train.loc[real_mask, col]
        f = struct_train.loc[fake_mask, col]
        u, p = mannwhitneyu(r, f, alternative="two-sided")
        rbc = 1.0 - 2.0 * u / (len(r) * len(f))
        row = {"Característica": STRUCT_LABELS[col], "Columna": col}
        for lab, x in (("Real", r), ("Fake", f)):
            row[f"{lab} M"] = round(float(x.mean()), 3)
            row[f"{lab} DE"] = round(float(x.std()), 3)
            row[f"{lab} Mdn"] = round(float(x.median()), 3)
            row[f"{lab} IQR"] = round(iqr(x), 3)
        row["U"] = float(u)
        row["p"] = "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")
        row["r_rb"] = round(float(rbc), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def top_tfidf_terms_by_class(
    mat, vectorizer: TfidfVectorizer, y_names: pd.Series, top_n: int = 10
) -> dict[str, list[list]]:
    """Extrae los términos con mayor peso TF-IDF medio por clase en Train."""
    terms = np.array(vectorizer.get_feature_names_out())
    out = {}
    for lab in ORDER:
        mask = (y_names == lab).values
        mean_weights = np.asarray(mat[mask].mean(axis=0)).ravel()
        top_idx = np.argsort(mean_weights)[::-1][:top_n]
        out[lab] = [[str(terms[i]), round(float(mean_weights[i]), 5)] for i in top_idx]
    return out


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[1/6] Cargando {INTERIM.relative_to(ROOT)}")
    df = pd.read_parquet(INTERIM)
    assert len(df) == 62633, f"Se esperaban 62,633 filas, se obtuvieron {len(df)}"
    assert df["id"].is_unique, "La columna 'id' debe ser única"

    print("[2/6] Mitigando marcadores de procedencia (source leakage) y generando Tabla 3")
    df, tabla3 = clean_source_markers(df)
    save_table(tabla3, "tabla03_limpieza_fuentes")

    empty_titles_total = int(df["title_clean"].eq("").sum())
    empty_texts_after = int(df["text_clean"].eq("").sum())
    short_texts_after = int((df["text_clean"].str.len() < 20).sum())
    print(
        f"      Títulos vacíos conservados: {empty_titles_total} | "
        f"Cuerpos vacíos tras limpieza: {empty_texts_after} | "
        f"Cuerpos < 20 car. tras limpieza: {short_texts_after}"
    )

    print("[3/6] Partición estratificada Train/Test (80/20, random_state=42)", flush=True)
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    assert set(train_df["id"]).isdisjoint(set(test_df["id"])), "Fuga de IDs entre Train y Test"

    print("[4/6] Calculando 10 características estructurales/sentimiento, Tabla 4 y StandardScaler")
    analyzer = SentimentIntensityAnalyzer()
    struct_train_raw = compute_structural_features(train_df, analyzer)
    struct_test_raw = compute_structural_features(test_df, analyzer)

    tabla4 = build_table4(struct_train_raw, train_df["label_name"])
    save_table(tabla4, "tabla04_features_resumen")

    scaler = StandardScaler()
    struct_train_scaled = pd.DataFrame(
        scaler.fit_transform(struct_train_raw),
        columns=STRUCT_COLS,
        index=struct_train_raw.index,
    )
    struct_test_scaled = pd.DataFrame(
        scaler.transform(struct_test_raw),
        columns=STRUCT_COLS,
        index=struct_test_raw.index,
    )
    joblib.dump(scaler, MOD / "scaler.joblib")

    print("[5/6] Vectorización TF-IDF (n-gramas 1–2, max_features=5000) y TruncatedSVD (50 + 100)")
    tfidf_title = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
        min_df=3,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )
    tfidf_text = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
        min_df=5,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )

    X_title_tr_tfidf = tfidf_title.fit_transform(train_df["title_clean"])
    X_title_te_tfidf = tfidf_title.transform(test_df["title_clean"])

    X_text_tr_tfidf = tfidf_text.fit_transform(train_df["text_clean"])
    X_text_te_tfidf = tfidf_text.transform(test_df["text_clean"])

    svd_title = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
    svd_text = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)

    X_title_tr_svd = svd_title.fit_transform(X_title_tr_tfidf)
    X_title_te_svd = svd_title.transform(X_title_te_tfidf)

    X_text_tr_svd = svd_text.fit_transform(X_text_tr_tfidf)
    X_text_te_svd = svd_text.transform(X_text_te_tfidf)

    joblib.dump(tfidf_title, MOD / "tfidf_title.joblib")
    joblib.dump(tfidf_text, MOD / "tfidf_text.joblib")
    joblib.dump(svd_title, MOD / "svd_title.joblib")
    joblib.dump(svd_text, MOD / "svd_text.joblib")

    title_svd_cols = [f"title_svd_{i:02d}" for i in range(50)]
    text_svd_cols = [f"text_svd_{i:02d}" for i in range(100)]
    all_cols = title_svd_cols + text_svd_cols + STRUCT_COLS

    X_train = pd.DataFrame(
        np.hstack([X_title_tr_svd, X_text_tr_svd, struct_train_scaled.values]).astype(np.float32),
        columns=all_cols,
        index=train_df["id"].astype(np.int64).values,
    )
    X_train.index.name = "id"

    X_test = pd.DataFrame(
        np.hstack([X_title_te_svd, X_text_te_svd, struct_test_scaled.values]).astype(np.float32),
        columns=all_cols,
        index=test_df["id"].astype(np.int64).values,
    )
    X_test.index.name = "id"

    y_train = pd.DataFrame(
        {"label": train_df["label"].astype(np.int8).values},
        index=X_train.index,
    )
    y_test = pd.DataFrame(
        {"label": test_df["label"].astype(np.int8).values},
        index=X_test.index,
    )

    # Validaciones estrictas del contrato
    assert X_train.shape[1] == 160 and X_test.shape[1] == 160
    assert not np.isnan(X_train.values).any() and not np.isnan(X_test.values).any()
    assert not np.isinf(X_train.values).any() and not np.isinf(X_test.values).any()
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert set(X_train.index).isdisjoint(set(X_test.index))

    X_train.to_parquet(PROC / "X_train.parquet")
    X_test.to_parquet(PROC / "X_test.parquet")
    y_train.to_parquet(PROC / "y_train.parquet")
    y_test.to_parquet(PROC / "y_test.parquet")
    print(f"      -> data\\processed\\X_train.parquet {X_train.shape} | X_test.parquet {X_test.shape}")

    print("[6/6] Generando Figuras 5 y 6 (300 dpi) y log de auditoría")

    # Figura 5 — Varianza explicada SVD (título y cuerpo)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panels = [
        (axes[0], svd_title.explained_variance_ratio_ * 100.0, "Titulares (50 componentes SVD)", "#2E6F9E"),
        (axes[1], svd_text.explained_variance_ratio_ * 100.0, "Cuerpo del artículo (100 componentes SVD)", "#C0392B"),
    ]
    for ax, ratios_pct, titulo, color in panels:
        comps = np.arange(1, len(ratios_pct) + 1)
        cum_pct = np.cumsum(ratios_pct)
        ax.bar(comps, ratios_pct, color=color, alpha=0.55, width=0.8, label="Individual (%)")
        ax.set_xlabel("Componente SVD")
        ax.set_ylabel("Varianza individual (%)", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_title(titulo, fontsize=10.5, fontweight="bold")
        ax.set_xlim(0, len(comps) + 1)

        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(comps, cum_pct, color="#1F2937", linewidth=1.8, label="Acumulada (%)")
        ax2.set_ylabel("Varianza acumulada (%)", color="#1F2937")
        ax2.set_ylim(0, max(cum_pct[-1] * 1.25, 25))
        ax2.annotate(
            f"Total: {cum_pct[-1]:.2f}%",
            xy=(comps[-1], cum_pct[-1]),
            xytext=(len(comps) * 0.58, cum_pct[-1] * 1.07),
            arrowprops={"arrowstyle": "->", "color": "#1F2937", "lw": 0.9},
            fontsize=9,
            fontweight="bold",
            color="#1F2937",
        )
    fig.tight_layout()
    fig.savefig(FIG / "fig05_svd_explained_variance.png")
    plt.close(fig)

    # Figura 6 — Correlaciones de Spearman y tamaño del efecto r_rb (Train)
    corr_df = struct_train_raw[STRUCT_COLS].rename(columns=STRUCT_SHORT_LABELS).corr(method="spearman")
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)

    t4_sorted = tabla4.sort_values("r_rb", ascending=True).reset_index(drop=True)
    y_labels = [STRUCT_SHORT_LABELS[c] for c in t4_sorted["Columna"]]
    r_vals = t4_sorted["r_rb"].values
    bar_colors = [PALETTE["Fake"] if v > 0 else PALETTE["Real"] for v in r_vals]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.4,
        annot_kws={"size": 7.5},
        cbar_kws={"shrink": 0.78, "label": "ρ de Spearman"},
        ax=axes[0],
    )
    axes[0].set_title("Correlaciones de Spearman en Train", fontsize=10.5, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=40, labelsize=8.5)
    axes[0].tick_params(axis="y", rotation=0, labelsize=8.5)

    bars = axes[1].barh(y_labels, r_vals, color=bar_colors, height=0.65)
    axes[1].axvline(0.0, color="#333333", linewidth=0.9)
    axes[1].set_xlim(min(r_vals.min() - 0.16, -0.30), max(r_vals.max() + 0.18, 0.85))
    axes[1].set_xlabel("Correlación biserial de rangos ($r_{rb}$: + Fake, − Real)")
    axes[1].set_title("Tamaño del efecto por clase en Train", fontsize=10.5, fontweight="bold")
    axes[1].tick_params(axis="y", labelsize=8.8)
    for b, val in zip(bars, r_vals):
        offset = 0.015 if val >= 0 else -0.015
        ha = "left" if val >= 0 else "right"
        axes[1].text(
            val + offset,
            b.get_y() + b.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha=ha,
            fontsize=8.2,
        )
    fig.tight_layout()
    fig.savefig(FIG / "fig06_structural_features.png")
    plt.close(fig)

    cum_title = np.cumsum(svd_title.explained_variance_ratio_)
    cum_text = np.cumsum(svd_text.explained_variance_ratio_)

    log = {
        "random_state": RANDOM_STATE,
        "split": {
            "total_rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_classes": class_counts_series(train_df["label"]),
            "test_classes": class_counts_series(test_df["label"]),
            "empty_titles_total": empty_titles_total,
            "empty_titles_train": int(train_df["title_clean"].eq("").sum()),
            "empty_titles_test": int(test_df["title_clean"].eq("").sum()),
            "empty_texts_after_cleaning": empty_texts_after,
            "texts_lt_20_chars_after_cleaning": short_texts_after,
        },
        "source_cleaning_table03": tabla3.to_dict(orient="records"),
        "tfidf": {
            "title_vocab_size": int(len(tfidf_title.vocabulary_)),
            "text_vocab_size": int(len(tfidf_text.vocabulary_)),
            "title_nnz_train": int(X_title_tr_tfidf.nnz),
            "text_nnz_train": int(X_text_tr_tfidf.nnz),
            "top_terms_title_train": top_tfidf_terms_by_class(
                X_title_tr_tfidf, tfidf_title, train_df["label_name"], top_n=10
            ),
            "top_terms_text_train": top_tfidf_terms_by_class(
                X_text_tr_tfidf, tfidf_text, train_df["label_name"], top_n=10
            ),
        },
        "svd": {
            "title_n_components": 50,
            "title_cum_var": {
                "k5": round(float(cum_title[4]) * 100, 4),
                "k10": round(float(cum_title[9]) * 100, 4),
                "k25": round(float(cum_title[24]) * 100, 4),
                "k50": round(float(cum_title[49]) * 100, 4),
            },
            "text_n_components": 100,
            "text_cum_var": {
                "k5": round(float(cum_text[4]) * 100, 4),
                "k10": round(float(cum_text[9]) * 100, 4),
                "k25": round(float(cum_text[24]) * 100, 4),
                "k50": round(float(cum_text[49]) * 100, 4),
                "k100": round(float(cum_text[99]) * 100, 4),
            },
        },
        "structural_features_table04": tabla4.to_dict(orient="records"),
        "spearman_corr_train": struct_train_raw[STRUCT_COLS].corr(method="spearman").round(4).to_dict(),
        "output_shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "y_train": list(y_train.shape),
            "y_test": list(y_test.shape),
        },
        "columns": all_cols,
    }
    (TAB / "02_feature_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[Listo]")
    print("\n--- Tabla 3: Limpieza de marcadores de fuente ---")
    print(tabla3.to_string(index=False))
    print("\n--- Tabla 4: Características estructurales y sentimiento (Train) ---")
    print(tabla4.to_string(index=False))


if __name__ == "__main__":
    main()
