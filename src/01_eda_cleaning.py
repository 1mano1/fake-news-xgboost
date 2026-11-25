"""
01_eda_cleaning.py — Integrante 1: Ingeniería de Datos y Análisis Exploratorio (EDA)

Entrada : data/raw/WELFake_Dataset.csv
Salida  : data/interim/welfake_clean.parquet
          reports/tables/tabla01_auditoria.{csv,md}
          reports/tables/tabla02_descriptivos.{csv,md}
          reports/tables/01_audit_log.json
          reports/figures/fig01..fig04 (300 dpi)

Convención de etiquetas (verificada empíricamente, ver CLAUDE.md):
    label = 0 -> Real   |   label = 1 -> Fake
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from wordcloud import STOPWORDS, WordCloud

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "WELFake_Dataset.csv"
INTERIM = ROOT / "data" / "interim"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
for d in (INTERIM, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {0: "Real", 1: "Fake"}
PALETTE = {"Real": "#2E6F9E", "Fake": "#C0392B"}
ORDER = ["Real", "Fake"]
MIN_TEXT_CHARS = 20
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def is_blank(s: pd.Series) -> pd.Series:
    """True si el valor es nulo o una cadena vacía / solo espacios."""
    return s.isna() | s.astype(str).str.strip().eq("")


def caps_ratio(s: pd.Series) -> pd.Series:
    """Proporción de letras mayúsculas sobre el total de letras (0 si no hay letras)."""
    letters = s.str.count(r"[A-Za-z]")
    upper = s.str.count(r"[A-Z]")
    return (upper / letters.replace(0, np.nan)).fillna(0.0)


def word_count(s: pd.Series) -> pd.Series:
    return s.str.split().str.len().fillna(0).astype(int)


def iqr(x: pd.Series) -> float:
    return float(x.quantile(0.75) - x.quantile(0.25))


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8")
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")


def class_counts(df: pd.DataFrame) -> dict:
    vc = df["label"].map(LABEL_NAMES).value_counts()
    return {k: int(vc.get(k, 0)) for k in ORDER}


# ---------------------------------------------------------------------------
# 1. Auditoría inicial
# ---------------------------------------------------------------------------
print(f"[1/5] Cargando {RAW.relative_to(ROOT)}")
raw = pd.read_csv(RAW)

audit = {
    "shape_original": list(raw.shape),
    "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
    "nulos": raw.isna().sum().astype(int).to_dict(),
    "vacios_o_espacios": {
        c: int((raw[c].notna() & raw[c].astype(str).str.strip().eq("")).sum())
        for c in ("title", "text")
    },
    "duplicados_title_text": int(raw.duplicated(subset=["title", "text"]).sum()),
    "duplicados_text": int(raw.loc[raw["text"].notna()].duplicated(subset=["text"]).sum()),
    "clases_original": class_counts(raw),
}
# Textos idénticos con etiquetas contradictorias
txt_lbl = raw.dropna(subset=["text"]).groupby("text")["label"].nunique()
audit["textos_con_etiqueta_contradictoria"] = int((txt_lbl > 1).sum())
# Marcadores de fuente (evidencia de la dirección de la etiqueta y de posible atajo)
txt = raw["text"].fillna("")
audit["marcador_reuters_por_clase"] = {
    LABEL_NAMES[k]: int(v)
    for k, v in raw.loc[txt.str.contains("(Reuters)", regex=False), "label"].value_counts().items()
}

# ---------------------------------------------------------------------------
# 2. Limpieza (cada paso queda registrado para la Tabla 1)
# ---------------------------------------------------------------------------
print("[2/5] Limpieza")
steps: list[dict] = []


def log_step(regla: str, antes: pd.DataFrame, despues: pd.DataFrame) -> None:
    cc = class_counts(despues)
    steps.append({
        "Paso": len(steps),
        "Regla aplicada": regla,
        "Filas eliminadas": len(antes) - len(despues),
        "Filas resultantes": len(despues),
        "Real": cc["Real"],
        "Fake": cc["Fake"],
    })


df = raw.copy()
steps.append({"Paso": 0, "Regla aplicada": "Conjunto original", "Filas eliminadas": 0,
              "Filas resultantes": len(df), **class_counts(df)})

# 2.1 Índice residual -> se conserva como identificador trazable 'id'
df = df.rename(columns={"Unnamed: 0": "id"}) if "Unnamed: 0" in df.columns else df.reset_index(names="id")

# 2.2 Normalización: espacios en blanco en extremos; vacíos -> NaN
for c in ("title", "text"):
    df[c] = df[c].astype("string").str.strip()
    df.loc[is_blank(df[c]), c] = pd.NA

# 2.3 Ambos campos nulos
before = df
df = df[~(df["title"].isna() & df["text"].isna())]
log_step("Eliminar filas con title y text nulos/vacíos", before, df)

# 2.4 Imputación del campo faltante con cadena vacía
n_title_imp = int(df["title"].isna().sum())
n_text_imp = int(df["text"].isna().sum())
df = df.assign(title=df["title"].fillna(""), text=df["text"].fillna(""))

# 2.5 Cuerpos rotos / ruido (< MIN_TEXT_CHARS caracteres; incluye text imputado vacío)
before = df
df = df[df["text"].str.len() >= MIN_TEXT_CHARS]
log_step(f"Eliminar text con < {MIN_TEXT_CHARS} caracteres", before, df)

# 2.6 Duplicados exactos (title + text)
before = df
df = df.drop_duplicates(subset=["title", "text"], keep="first")
log_step("Eliminar duplicados exactos (title + text)", before, df)

# 2.7 Duplicados por cuerpo idéntico (mismo artículo con título distinto).
#     Si un mismo cuerpo aparece con etiquetas contradictorias, se descartan todas
#     sus copias (no hay forma objetiva de decidir la verdadera).
before = df
conflict = df.groupby("text")["label"].transform("nunique") > 1
df = df[~conflict]
log_step("Eliminar cuerpos idénticos con etiqueta contradictoria", before, df)
before = df
df = df.drop_duplicates(subset=["text"], keep="first")
log_step("Eliminar duplicados por text idéntico", before, df)

df = df.astype({"title": str, "text": str, "label": "int8", "id": "int64"})
df["label_name"] = df["label"].map(LABEL_NAMES)
df = df[["id", "title", "text", "label", "label_name"]].reset_index(drop=True)

audit["imputados"] = {"title_vacio": n_title_imp, "text_vacio": n_text_imp}
audit["shape_final"] = list(df.shape)
audit["clases_final"] = class_counts(df)

out = INTERIM / "welfake_clean.parquet"
df.to_parquet(out, index=False)
print(f"      -> {out.relative_to(ROOT)}  {df.shape}")

tabla1 = pd.DataFrame(steps)
save_table(tabla1, "tabla01_auditoria")

# ---------------------------------------------------------------------------
# 3. Métricas exploratorias (solo EDA; no se guardan en el parquet)
# ---------------------------------------------------------------------------
print("[3/5] Métricas exploratorias y Tabla 2")
eda = pd.DataFrame({
    "Clase": df["label_name"],
    "title_char_len": df["title"].str.len(),
    "title_word_count": word_count(df["title"]),
    "text_char_len": df["text"].str.len(),
    "text_word_count": word_count(df["text"]),
    "title_caps_ratio": caps_ratio(df["title"]),
    "title_excl_count": df["title"].str.count("!"),
    "title_quest_count": df["title"].str.count(r"\?"),
})

VARS = {
    "title_word_count": "Longitud del título (palabras)",
    "title_char_len": "Longitud del título (caracteres)",
    "text_word_count": "Longitud del cuerpo (palabras)",
    "text_char_len": "Longitud del cuerpo (caracteres)",
    "title_caps_ratio": "Proporción de mayúsculas en título",
    "title_excl_count": "Signos ! en título",
    "title_quest_count": "Signos ? en título",
}
rows = []
for var, nombre in VARS.items():
    r = eda.loc[eda["Clase"] == "Real", var]
    f = eda.loc[eda["Clase"] == "Fake", var]
    u, p = mannwhitneyu(r, f, alternative="two-sided")
    rbc = 1 - 2 * u / (len(r) * len(f))  # correlación biserial de rangos (tamaño del efecto)
    row = {"Variable": nombre}
    for lab, x in (("Real", r), ("Fake", f)):
        row[f"{lab} M"] = round(x.mean(), 3)
        row[f"{lab} DE"] = round(x.std(), 3)
        row[f"{lab} Mdn"] = round(x.median(), 3)
        row[f"{lab} IQR"] = round(iqr(x), 3)
    row["U"] = float(u)
    row["p"] = "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")
    row["r_rb"] = round(rbc, 3)
    rows.append(row)
tabla2 = pd.DataFrame(rows)
save_table(tabla2, "tabla02_descriptivos")

# ---------------------------------------------------------------------------
# 4. Figuras
# ---------------------------------------------------------------------------
print("[4/5] Figuras")

# Figura 1 — distribución de clases antes / después
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, (titulo, cc) in zip(axes, [("Antes de la limpieza", audit["clases_original"]),
                                    ("Después de la limpieza", audit["clases_final"])]):
    total = sum(cc.values())
    bars = ax.bar(ORDER, [cc[k] for k in ORDER], color=[PALETTE[k] for k in ORDER], width=0.6)
    for b, k in zip(bars, ORDER):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{cc[k]:,}\n({cc[k] / total:.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{titulo} (n = {total:,})", fontsize=10)
    ax.set_xlabel("Clase")
axes[0].set_ylabel("Número de artículos")
axes[0].set_ylim(0, max(audit["clases_original"].values()) * 1.18)
fig.savefig(FIG / "fig01_class_distribution.png")
plt.close(fig)

# Figura 2 — longitudes (recorte al percentil 99 para legibilidad)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, var, lab in [(axes[0], "title_word_count", "Palabras en el título"),
                     (axes[1], "text_word_count", "Palabras en el cuerpo")]:
    cap = eda[var].quantile(0.99)
    sub = eda[eda[var] <= cap]
    sns.violinplot(data=sub, x="Clase", y=var, hue="Clase", order=ORDER, palette=PALETTE,
                   inner="quartile", cut=0, linewidth=1, ax=ax, legend=False)
    ax.set_ylabel(lab)
    ax.set_xlabel("Clase")
    ax.set_title(f"Recortado en P99 = {cap:,.0f}", fontsize=9)
fig.savefig(FIG / "fig02_length_distributions.png")
plt.close(fig)

# Figura 3 — mayúsculas y exclamaciones en titulares
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
cap = eda["title_caps_ratio"].quantile(0.99)
sns.boxplot(data=eda[eda["title_caps_ratio"] <= cap], x="Clase", y="title_caps_ratio",
            hue="Clase", order=ORDER, palette=PALETTE, showfliers=False, ax=axes[0], legend=False)
axes[0].set_ylabel("Proporción de mayúsculas en el título")
axes[0].set_xlabel("Clase")
pct = (eda.assign(tiene=eda["title_excl_count"] > 0)
          .groupby("Clase")["tiene"].mean().reindex(ORDER) * 100)
bars = axes[1].bar(ORDER, pct.values, color=[PALETTE[k] for k in ORDER], width=0.6)
for b, v in zip(bars, pct.values):
    axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
axes[1].set_ylabel("Títulos con al menos un «!» (%)")
axes[1].set_xlabel("Clase")
fig.savefig(FIG / "fig03_punctuation_caps.png")
plt.close(fig)

# Figura 4 — nubes de palabras (stopwords básicas en inglés)
TOKEN = re.compile(r"[a-z]{3,}")
stop = {w.lower() for w in STOPWORDS}


def freqs(texts: pd.Series) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(w for w in TOKEN.findall(t.lower()) if w not in stop)
    return c


fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
top_terms = {}
for ax, lab, cmap in [(axes[0], "Fake", "Reds"), (axes[1], "Real", "Blues")]:
    fr = freqs(df.loc[df["label_name"] == lab, "text"])
    top_terms[lab] = fr.most_common(25)
    # Recortar el colormap a su tramo oscuro para que todas las palabras sean legibles
    base = plt.get_cmap(cmap)
    dark = matplotlib.colors.ListedColormap(base(np.linspace(0.55, 1.0, 256)))
    wc = WordCloud(width=1200, height=900, background_color="white", colormap=dark,
                   max_words=150, random_state=RANDOM_STATE, prefer_horizontal=0.9)
    ax.imshow(wc.generate_from_frequencies(fr), interpolation="bilinear")
    ax.set_title(f"Noticias {'falsas' if lab == 'Fake' else 'reales'}", fontsize=11)
    ax.axis("off")
fig.savefig(FIG / "fig04_wordclouds_comparison.png")
plt.close(fig)
audit["top_terminos_cuerpo"] = {k: [[w, int(n)] for w, n in v] for k, v in top_terms.items()}

# ---------------------------------------------------------------------------
# 5. Log de auditoría
# ---------------------------------------------------------------------------
(TAB / "01_audit_log.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
print("[5/5] Listo.")
print(tabla1.to_string(index=False))
