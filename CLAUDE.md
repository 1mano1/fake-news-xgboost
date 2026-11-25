# CLAUDE.md — Detección de Fake News con XGBoost

Guía para agentes de consola (Claude Code u otros) que trabajen en este repositorio.

## 1. Proyecto

- **Título tentativo:** Detección Automática de Noticias Falsas (Fake News) mediante Análisis Lingüístico-Estructural y Optimización de Gradient Boosting (XGBoost).
- **Paradigma único:** Gradient Boosting (XGBoost). No introducir otros algoritmos como modelo principal.
- **Dataset:** WELFake (~72,134 artículos en inglés). Columnas: `title`, `text`, `label`.
- **⚠ Convención de etiquetas (verificada en los datos):** `label = 0 → Real`, `label = 1 → Fake`. La documentación de Kaggle dice lo contrario, pero 21,266 de 21,282 artículos con el marcador "(Reuters)" tienen `0` y todos los de sitios como "21st Century Wire" tienen `1`. Se conservan los valores originales: **la clase positiva (1) es Fake**, por lo que Precision/Recall/F1 miden la detección de noticias falsas. El parquet limpio incluye `label_name` ("Real"/"Fake").
- **⚠ Atajo de fuente:** marcadores como "(Reuters)", "Featured image", "via Getty" o "21st Century Wire" identifican casi perfectamente la clase. Integrante 2 debe decidir (y documentar) si los elimina del texto para que el modelo no aprenda la fuente en lugar del contenido.
  - Kaggle: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
  - Cita APA 7: Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. (2021). WELFake: Word Embedding Over Linguistic Features for Fake News Detection. *IEEE Transactions on Computational Social Systems, 8*(4), 881–893. https://doi.org/10.1109/TCSS.2021.3068519
- **Idioma:** código y comentarios pueden ir en inglés o español; la redacción académica (`reports/sections/`) va en español.

## 2. Reglas globales (obligatorias)

- `RANDOM_STATE = 42` en todos los scripts (splits, modelos, búsquedas, SVD, etc.).
- Gráficas: exportar desde Python en `.png` a **300 dpi** en `reports/figures/`.
- Datos exactos de cada tabla: guardar en `reports/tables/` (CSV y/o Markdown).
- **No modificar archivos fuera de la carpeta/archivos asignados** a cada integrante ni sobreescribir entregables de otro.
- Evitar *data leakage*: cualquier `fit` de transformadores ocurre solo sobre Train.
- `data/raw/` no se sube a git si el CSV es pesado (añadir a `.gitignore`).

### Formato APA 7ª ed.

- **Tablas:** etiqueta en negrita arriba (**Tabla X**), título descriptivo en cursiva debajo, sin líneas verticales, nota al pie si aplica (*Nota.* Elaboración propia...).
- **Figuras:** etiqueta en negrita arriba (**Figura X**), título descriptivo en cursiva debajo, imagen centrada, nota al pie (*Nota.* Elaboración propia...).
- Respetar la numeración reservada por integrante (ver §4).

### Documento final

1. Portada
2. Introducción (~1.5 págs., no cuenta para el mínimo)
3. Desarrollo (mínimo 20 páginas reales; meta 22–24, ~5.5 por integrante)
4. Conclusiones (~1.5 págs., no cuenta para el mínimo)
5. Referencias APA 7 (no cuenta para el mínimo)

## 3. Estructura del repositorio

```
fake-news-xgboost/
├── CLAUDE.md
├── PROJECT_CONTEXT.md
├── requirements.txt
├── data/
│   ├── raw/          # WELFake_Dataset.csv
│   ├── interim/      # welfake_clean.parquet (Integrante 1)
│   └── processed/    # X_train, X_test, y_train, y_test .parquet (Integrante 2)
├── models/           # transformadores (Integrante 2) + modelos XGB (Integrante 3)
├── src/
│   ├── 01_eda_cleaning.py
│   ├── 02_feature_eng.py
│   ├── 03_modeling_tuning.py
│   └── 04_eval_shap_errors.py
└── reports/
    ├── figures/      # fig01_... a fig12_...
    ├── tables/
    └── sections/
        ├── 01_desarrollo_eda.md
        ├── 02_desarrollo_features.md
        ├── 03_desarrollo_modelos.md
        └── 04_desarrollo_evaluacion.md
```

## 4. Flujo en cascada y contratos (1 → 2 → 3 → 4)

Cada integrante entrega **código/datos** y **su sección redactada (~5.5 págs.)** antes de pasar la estafeta.

| # | Sección | Entrada | Salida | Tablas | Figuras |
|---|---------|---------|--------|--------|---------|
| 1 | 3.1 Ingeniería de Datos y EDA | `data/raw/WELFake_Dataset.csv` | `data/interim/welfake_clean.parquet` | 1–2 | 1–4 |
| 2 | 3.2 Preprocesamiento NLP e Ingeniería de Características | `data/interim/welfake_clean.parquet` | `data/processed/{X_train,X_test,y_train,y_test}.parquet` + `models/{tfidf_title,tfidf_text,svd_text,scaler}.joblib` | 3–4 | 5–6 |
| 3 | 3.3 Marco Algorítmico, Modelado Iterativo y Optimización | `data/processed/*.parquet` | `models/{xgb_iter1_baseline,xgb_iter2_features,xgb_iter3_tuned}.json` + `reports/tables/cv_results.csv` | 5–6 | 7–8 |
| 4 | 3.4 Evaluación, Explicabilidad (SHAP) y Errores | `X_test`, `y_test` + 3 modelos | Métricas en Test, gráficas SHAP, análisis FP/FN | 7–8 | 9–12 |

Al terminar el Integrante 4, el equipo redacta Introducción, Conclusiones, Portada y Referencias.

## 5. Especificación por script

### `src/01_eda_cleaning.py` (Integrante 1)

**Auditoría inicial:** dimensiones, tipos, nulos en `title`/`text`, cadenas vacías o solo espacios, duplicados (`title + text` y `text` idéntico).

**Limpieza (documentar cada regla y su efecto en filas):**
- Eliminar índice residual (`Unnamed: 0` si existe).
- Si un campo es nulo y el otro existe → imputar `""` en el nulo.
- Eliminar filas con ambos nulos o con `text` de menos de 20 caracteres.
- Deduplicación exacta (evita que una misma noticia caiga en Train y Test).

**Métricas EDA (solo análisis):** caracteres y `word_count` de `title` y `text`; `caps_ratio` y conteo de `!` / `?` en `title`.

**Salida:** `data/interim/welfake_clean.parquet` — columnas `id` (índice original, para trazabilidad), `title`, `text`, `label` (int8), `label_name`. Resultado actual: 62,633 filas (Real 34,617 / Fake 28,016). Detalle en `reports/tables/tabla01_auditoria.csv` y `01_audit_log.json`.

**Entregables:**
- Tabla 1: resumen de auditoría y depuración (filas iniciales, nulos eliminados, duplicados removidos, filas finales por clase).
- Tabla 2: descriptivos por clase (media, DE, mediana, IQR, p-valor Mann-Whitney U) para longitud de título, longitud de cuerpo y proporción de mayúsculas.
- `fig01_class_distribution.png`: distribución de clases antes/después de limpieza (barras con conteos y %).
- `fig02_length_distributions.png`: longitud de títulos y artículos en palabras por clase (boxplot/violin, recortando percentiles extremos).
- `fig03_punctuation_caps.png`: mayúsculas y `!` en titulares reales vs. falsos.
- `fig04_wordclouds_comparison.png`: WordClouds lado a lado Fake vs. Real (sin stopwords básicas).

### `src/02_feature_eng.py` (Integrante 2)

- **Regla de oro:** `train_test_split(..., test_size=0.20, stratify=y, random_state=42)` **antes** de ajustar `TfidfVectorizer`, `TruncatedSVD` o `StandardScaler`. `fit` en Train, `transform` en Test.
- Variables estructurales (mínimo 8): `title_char_len`, `title_word_len`, `text_char_len`, `text_word_len`, `title_caps_ratio`, `title_excl_count`, `title_quest_count`, `lexical_diversity` (únicas/totales) y polaridad de sentimiento (VADER o TextBlob).
- Texto: TF-IDF (n-gramas 1–2, `max_features=5000`) → TruncatedSVD de 100 componentes (cuerpo) + 50 componentes (título), concatenado con variables estructurales escaladas.

### `src/03_modeling_tuning.py` (Integrante 3)

- **Iter. 1 (Baseline):** `XGBClassifier(random_state=42)` solo con las 50 columnas SVD del título.
- **Iter. 2 (Enriquecido):** `XGBClassifier(random_state=42)` con título + cuerpo + estructurales.
- **Iter. 3 (Optimizado):** `RandomizedSearchCV` (30 iteraciones, `StratifiedKFold(n_splits=5)`), `scoring="f1"`, sobre `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`.

### `src/04_eval_shap_errors.py` (Integrante 4)

- Evaluar las 3 iteraciones en `X_test`: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
- `shap.TreeExplainer` sobre la Iteración 3: Summary Plot global + 2 Waterfall Plots (un acierto claro y un FP/FN).
- Análisis textual de 6 noticias mal clasificadas.

## 6. Convenciones para agentes

- Rutas relativas a la raíz del repo (usar `pathlib`); nunca rutas absolutas de una máquina concreta.
- Scripts ejecutables de punta a punta desde la raíz: `python src/0X_*.py`.
- Antes de escribir en `data/`, `models/` o `reports/`, verificar que el archivo pertenece al integrante en curso.
- No inventar resultados ni cifras en las secciones redactadas: todo número debe salir de `reports/tables/`.
