# Detección de Fake News con XGBoost

Proyecto académico: clasificación de noticias falsas sobre el dataset **WELFake** usando Gradient Boosting (XGBoost). Las reglas del equipo, los contratos entre integrantes y la numeración de tablas y figuras están en [`CLAUDE.md`](CLAUDE.md).

---

## Si recibiste el proyecto en ZIP

El ZIP ya trae el dataset original (`data/raw/WELFake_Dataset.csv`) y el dataset limpio (`data/interim/welfake_clean.parquet`). Solo falta el entorno de Python, que no se puede copiar entre computadoras porque guarda rutas de la máquina donde se creó.

1. Instala **Python 3.10 o superior** desde https://www.python.org/downloads/ y marca **“Add Python to PATH”**.
2. Descomprime el ZIP y, dentro de la carpeta `fake-news-xgboost`, haz doble clic en **`instalar_entorno.bat`** (Windows). En macOS/Linux sigue el Paso 2 de la sección siguiente.
3. Abre una terminal en la carpeta, activa el entorno con `.venv\Scripts\activate` y ya puedes trabajar.

Para comprobar que todo funciona, ejecuta `python src/01_eda_cleaning.py`: debe terminar mostrando `(62633, 5)`.

---

## Si clonaste el proyecto desde GitHub

El repositorio **no incluye los datos ni el entorno de Python** porque pesan demasiado (el CSV pesa 245 MB). Los generas tú en unos 5 minutos siguiendo estos pasos.

### Paso 0. Requisitos

- **Python 3.10 o superior** (recomendado: **3.12**; el Paso 4 se probó con 3.12.10 y las versiones fijadas en `requirements.txt`). Compruébalo con `python --version`. Si no lo tienes, descárgalo de https://www.python.org/downloads/ y durante la instalación marca la casilla **“Add Python to PATH”**.
- Una cuenta de **Kaggle** (gratis) para descargar el dataset.

### Paso 1. Descomprimir y abrir una terminal en la carpeta

1. Descomprime el ZIP donde quieras (por ejemplo, en `Documentos`).
2. Abre la carpeta `fake-news-xgboost`.
3. En Windows: clic derecho en un espacio vacío de la carpeta → **“Abrir en Terminal”**. También puedes abrir la carpeta en VS Code y usar su terminal.

**Todos los comandos siguientes se ejecutan dentro de esa carpeta.**

### Paso 2. Crear el entorno e instalar las librerías

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sabrás que el entorno está activo porque la terminal muestra `(.venv)` al inicio de la línea. Cada vez que abras una terminal nueva, vuelve a activarlo con `.venv\Scripts\activate`.

> Si PowerShell dice que “la ejecución de scripts está deshabilitada”, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, responde `S` y vuelve a intentar.

### Paso 3. Descargar el dataset

**El dataset correcto es este:**

| Dato | Valor |
|------|-------|
| Nombre | **WELFake** (en Kaggle aparece como *Fake News Classification*) |
| Página | https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification |
| Archivo | `WELFake_Dataset.csv` |
| Tamaño | ~245 MB |
| Filas | 72,134 |
| Columnas | `Unnamed: 0`, `title`, `text`, `label` |
| Artículo original | Verma et al. (2021), *IEEE Transactions on Computational Social Systems*, https://doi.org/10.1109/TCSS.2021.3068519 |

**Opción A, manual (la más fácil):**
1. Entra a la página de Kaggle, inicia sesión y pulsa **Download**.
2. Descomprime el archivo que se descarga.
3. Copia `WELFake_Dataset.csv` a la carpeta `data/raw/` del proyecto.

**Opción B, por comando:**
1. En Kaggle: tu foto de perfil → *Settings* → *API* → **Create New Token**. Se descarga `kaggle.json`.
2. Guárdalo en `C:\Users\<tu_usuario>\.kaggle\kaggle.json` (macOS/Linux: `~/.kaggle/kaggle.json`). **No lo compartas con nadie.**
3. Ejecuta:
   ```bash
   kaggle datasets download -d saurabhshahane/fake-news-classification -p data/raw --unzip
   ```

En cualquiera de las dos opciones, el archivo debe quedar exactamente en:
```
fake-news-xgboost/data/raw/WELFake_Dataset.csv
```

### Paso 4. Generar el dataset limpio

```bash
python src/01_eda_cleaning.py
```

Tarda unos 25 segundos. **Comprueba que te salga esto:**
```
      -> data\interim\welfake_clean.parquet  (62633, 5)
```
Si ves `(62633, 5)`, tienes exactamente el mismo dataset limpio que el resto del equipo. El script también vuelve a generar las Tablas 1–2 y las Figuras 1–4.

### Paso 5. Regenerar los datos procesados (Etapa 2) y trabajar en tu parte

`requirements.txt` ya incluye `scikit-learn`, `joblib` y `vaderSentiment`. Para regenerar localmente la matriz de características (`data/processed/*.parquet`) que dejó lista la Etapa 2:

```bash
python src/02_feature_eng.py
```

Cada integrante crea su script en `src/` y lee lo que dejó el anterior. Revisa en `CLAUDE.md` qué archivos recibes, qué debes entregar y qué numeración de tablas y figuras te toca.

---

## ⚠ Dos avisos antes de empezar

1. **Las etiquetas están invertidas respecto a lo que dice Kaggle.** En los datos reales **`label = 0` es Real y `label = 1` es Fake** (verificado: 21,266 de los 21,282 artículos con la firma “(Reuters)” tienen `0`). Se mantuvieron los valores originales, así que **la clase positiva es Fake**. Si tienes dudas, usa la columna `label_name`.
2. **Hay marcas que delatan la fuente.** Textos como “(Reuters)”, “Featured image”, “via Getty” o “21st Century Wire” revelan la clase casi sin error, y los 482 artículos sin título son todos falsos. El Integrante 2 debe decidir (y documentar) si los elimina antes de vectorizar; si no, el modelo aprende la fuente en vez del contenido.

---

## Orden de ejecución

| Paso | Comando | Responsable | Estado |
|------|---------|-------------|--------|
| 1 | `python src/01_eda_cleaning.py` | Integrante 1 | ✅ Listo |
| 2 | `python src/02_feature_eng.py` | Integrante 2 | ✅ Listo |
| 3 | `python src/03_modeling_tuning.py` | Integrante 3 | ✅ Listo |
| 4 | `python src/04_eval_shap_errors.py` | Integrante 4 | ✅ Listo |

El Paso 4 necesita los datos que generan los Pasos 1 y 2. **No hace falta ejecutar el Paso 3:** los modelos ya vienen en `models/` (su búsqueda de hiperparámetros tardó unos 20 minutos).

> El Paso 2 vuelve a escribir los `.joblib` de `models/` y las Figuras 5 y 6, así que Git puede mostrarlos como modificados. En la comprobación del Paso 4 el vocabulario fue idéntico y las diferencias numéricas, mínimas. Para recuperar las versiones del repositorio: `git restore models/*.joblib reports/figures/fig05_svd_explained_variance.png reports/figures/fig06_structural_features.png`

## Qué hace el Paso 1 (`src/01_eda_cleaning.py`)

1. **Auditoría:** dimensiones, tipos, nulos, cadenas vacías, duplicados y etiquetas contradictorias.
2. **Limpieza** (en este orden):
   - Renombra el índice residual `Unnamed: 0` → `id` (se conserva para rastrear artículos).
   - Recorta espacios; las cadenas vacías se tratan como nulos.
   - Imputa `""` al título o cuerpo faltante.
   - Elimina artículos con `text` de menos de 20 caracteres.
   - Elimina duplicados exactos (`title + text`) y artículos con `text` idéntico (evita *data leakage* entre Train y Test).
3. **Análisis exploratorio:** longitudes, proporción de mayúsculas y signos `!`/`?` por clase, con prueba de Mann-Whitney U y tamaño del efecto.
4. **Exporta** tablas, figuras (300 dpi) y el dataset limpio.

**Resultado:** 72,134 → **62,633** artículos (Real 34,617 · Fake 28,016).

| Archivo | Contenido |
|---------|-----------|
| `data/interim/welfake_clean.parquet` | Dataset limpio: `id`, `title`, `text`, `label`, `label_name` |
| `reports/tables/tabla01_auditoria.{csv,md}` | Tabla 1: filas eliminadas por cada regla |
| `reports/tables/tabla02_descriptivos.{csv,md}` | Tabla 2: descriptivos por clase + Mann-Whitney U |
| `reports/tables/01_audit_log.json` | Todas las cifras de la auditoría |
| `reports/figures/fig01` a `fig04` | Figuras 1–4 a 300 dpi |
| `reports/Proyecto_FakeNews_XGBoost.docx` | Documento del proyecto (portada, introducción y sección 1) |

## Qué hace el Paso 2 (`src/02_feature_eng.py`)

1. **Mitigación de *source leakage*:** aplica 5 reglas regex determinísticas sobre `title` y `text` antes de vectorizar (sufijos editoriales `- NYT`/`- Breitbart`/`- Reuters`, *dateline* y firma `(Reuters)`, créditos de imagen `Featured image`/`Getty`/`Flickr`/`Wikimedia`, banners `21st Century Wire`/`21WIRE`, y URLs/`pic.twitter.com`). Impacta **42,770 artículos (68.29 %)** y conserva los 482 artículos sin título.
2. **Partición estratificada sin *data leakage*:** ejecuta `train_test_split(..., test_size=0.20, stratify=label, random_state=42)` **antes** de ajustar cualquier transformador (`50,106` en Train: 27,693 Real / 22,413 Fake; `12,527` en Test: 6,924 Real / 5,603 Fake).
3. **10 características estructurales y de sentimiento:** `title_char_len`, `title_word_len`, `text_char_len`, `text_word_len`, `title_caps_ratio`, `title_excl_count`, `title_quest_count`, `lexical_diversity` (TTR), `title_sentiment` y `text_sentiment` (VADER sobre el título y los primeros 1,000 caracteres del cuerpo), escaladas con `StandardScaler` ajustado solo en Train.
4. **TF-IDF + TruncatedSVD:** vectoriza título y cuerpo por separado (`ngram_range=(1, 2)`, `max_features=5000`, `sublinear_tf=True`) y reduce dimensionalidad con `TruncatedSVD` (`50` componentes para título: `title_svd_00`..`title_svd_49`, 9.60 % varianza acumulada; `100` componentes para cuerpo: `text_svd_00`..`text_svd_99`, 17.89 % varianza acumulada).
5. **Validación y exportación:** ensambla matrices densas de **160 columnas** (`float32`, índice `id` trazable) sin nulos ni infinitos.

| Archivo | Contenido | En Git / Regenerable |
|---------|-----------|----------------------|
| `data/processed/{X_train,X_test,y_train,y_test}.parquet` | Matrices finales `(50106, 160)` y `(12527, 160)` + etiquetas `int8` | Regenerable localmente (`python src/02_feature_eng.py`) |
| `models/{tfidf_title,tfidf_text,svd_title,svd_text,scaler}.joblib` | Vectorizadores TF-IDF, reductores SVD y escalador ajustados en Train | Subido al repositorio |
| `reports/tables/tabla03_limpieza_fuentes.{csv,md}` | Tabla 3: artículos afectados por cada regla de limpieza de fuentes | Regenerable localmente |
| `reports/tables/tabla04_features_resumen.{csv,md}` | Tabla 4: descriptivos en Train sin escalar + Mann-Whitney U | Regenerable localmente |
| `reports/tables/02_feature_log.json` | Cifras exactas de partición, TF-IDF, SVD y correlaciones | Regenerable localmente |
| `reports/figures/fig05` y `fig06` | Figuras 5–6 a 300 dpi (varianza SVD y correlaciones/efecto) | Regenerable localmente |

> **Nota para el Integrante 3 (`03_modeling_tuning.py`):** para la **Iteración 1 (*Baseline*)** usa únicamente las 50 columnas `title_svd_00`..`title_svd_49`. Para las **Iteraciones 2 y 3** usa las 160 columnas completas (`title_svd_*`, `text_svd_*` y las 10 estructurales). El índice `id` permite rastrear cada artículo en el análisis de errores del Integrante 4.


## Qué hace el Paso 3 (`src/03_modeling_tuning.py`)

1. **Contrato y validación:** carga `X_train`/`X_test`/`y_train`/`y_test` y verifica formas (50,106 × 160 y 12,527 × 160), columnas exactas y finitud; `X_test` se usa solo para validar contrato, nunca para métricas.
2. **Protocolo común:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre Train; métricas accuracy, precision, recall, F1, ROC-AUC y PR-AUC con `cross_validate`; estimadores con `tree_method="hist"`, `eval_metric="logloss"`, `random_state=42`.
3. **Iteración 1 — Baseline:** `XGBClassifier` con hiperparámetros por defecto y solo las 50 columnas SVD del título. F1 = 0.7671 ± 0.0071.
4. **Iteración 2 — Enriquecida:** mismos hiperparámetros por defecto con las 160 columnas (título + cuerpo + estructurales). F1 = 0.9458 ± 0.0029 (+0.1787 sobre la línea base).
5. **Iteración 3 — Optimizada:** `RandomizedSearchCV` (30 configuraciones × 5 pliegues = 150 ajustes, `scoring="f1"`, `random_state=42`) sobre las 160 columnas. Mejor F1 de búsqueda = 0.9542.

**Configuración óptima:** `max_depth=5`, `learning_rate=0.2`, `n_estimators=800`, `subsample=0.9`, `colsample_bytree=1.0`, `min_child_weight=7`, `reg_alpha=0.01`, `reg_lambda=2`. `n_estimators=800` coincide con el límite superior del espacio explorado (200–800): es una limitación del diseño, no un óptimo demostrado.

| Iteración | Variables | Exactitud | Precisión | Recall | F1 | ROC-AUC | PR-AUC | Tiempo (s) |
|-----------|-----------|-----------|-----------|--------|----|---------|--------|------------|
| 1 — Baseline | 50 SVD título | 0.799 (0.006) | 0.797 (0.008) | 0.739 (0.008) | 0.767 (0.007) | 0.884 (0.004) | 0.873 (0.005) | 6.8 |
| 2 — Enriquecida | 160 | 0.952 (0.003) | 0.949 (0.004) | 0.943 (0.004) | 0.946 (0.003) | 0.990 (0.001) | 0.989 (0.001) | 13.5 |
| 3 — Optimizada | 160 | 0.959 (0.002) | 0.956 (0.003) | 0.952 (0.004) | 0.954 (0.003) | 0.993 (0.001) | 0.991 (0.001) | 1,239.7 |

> Los valores son media (DE) entre los 5 pliegues; la clase positiva es Fake. La mejora dominante proviene de las variables (Iter. 1 → 2, +0.1787 en F1), no del ajuste (Iter. 2 → 3, +0.0084 a ~92× el costo computacional).

**Importancias por gain (modelo optimizado):** `title_caps_ratio` domina (12.38 %), seguida de `title_svd_01` (5.60 %), `title_char_len` (4.12 %), `text_svd_09` (3.92 %) y `text_svd_07` (2.68 %). En el top-20: 14 componentes del cuerpo (25.66 %), 5 variables estructurales (20.85 % del top-20; 23.48 % sumando las 10) y 1 SVD del título (5.60 %). El cuerpo, en conjunto, suma el 58.49 % del gain total.

| Archivo | Contenido | En Git / Regenerable |
|---------|-----------|----------------------|
| `models/xgb_iter1_baseline.json` | Modelo Iteración 1 (50 vars, 100 árboles) | Subido al repositorio |
| `models/xgb_iter2_features.json` | Modelo Iteración 2 (160 vars, 100 árboles) | Subido al repositorio |
| `models/xgb_iter3_tuned.json` | Modelo Iteración 3 (160 vars, 800 árboles) | Subido al repositorio |
| `reports/tables/cv_results.csv` | 30 candidatos de la búsqueda aleatoria (21 columnas) | Regenerable localmente |
| `reports/tables/tabla05_resultados_cv.{csv,md}` | Tabla 5: rendimiento CV de las 3 iteraciones | Regenerable localmente |
| `reports/tables/tabla06_hiperparametros.{csv,md}` | Tabla 6: espacio de búsqueda y óptimo | Regenerable localmente |
| `reports/tables/03_modeling_log.json` | Cifras exactas: tiempos, métricas, importancias, balance | Regenerable localmente |
| `reports/figures/fig07_cv_comparison.png` | Figura 7: comparación CV a 300 dpi | Regenerable localmente |
| `reports/figures/fig08_feature_importance.png` | Figura 8: top-15 importancias gain a 300 dpi | Regenerable localmente |
| `reports/sections/03_desarrollo_modelos.md` | Sección 3.3 (2.997 palabras, 3.3.1–3.3.7) | Redacción del Integrante 3 |

> **Nota para el Integrante 4 (`04_eval_shap_errors.py`):** evalúa las 3 iteraciones en `X_test` (12,527 artículos) con Accuracy, Precision, Recall, F1, ROC-AUC y PR-AUC. Usa `shap.TreeExplainer` sobre la Iteración 3 para el Summary Plot global y 2 Waterfall Plots (un acierto y un FP/FN), y analiza 6 noticias mal clasificadas. El índice `id` permite rastrear cada artículo hasta `welfake_clean.parquet`.


## Qué hace el Paso 4 (`src/04_eval_shap_errors.py`)

1. **Evaluación en Test:** carga los 3 modelos guardados, **sin reentrenarlos**, y los evalúa en los 12,527 artículos de `X_test` con umbral fijo de 0.5.
2. **SHAP:** usa `shap.TreeExplainer` sobre la Iteración 3 (valores en *log-odds*; positivo empuja hacia Fake) y comprueba que las contribuciones reconstruyan la salida del modelo. Exporta los términos de los componentes SVD más influyentes.
3. **Explicaciones locales:** 2 *waterfall plots*, el verdadero positivo con mayor probabilidad y el error más extremo.
4. **Errores:** selecciona automáticamente los 3 falsos positivos y los 3 falsos negativos en que el modelo más se equivoca.

| Iteración | Exactitud | Precisión | Recall | F1 | ROC-AUC | PR-AUC |
|-----------|-----------|-----------|--------|----|---------|--------|
| 1 — Baseline | 0.8076 | 0.8058 | 0.7508 | 0.7773 | 0.8925 | 0.8790 |
| 2 — Enriquecida | 0.9544 | 0.9499 | 0.9481 | 0.9490 | 0.9918 | 0.9902 |
| 3 — Optimizada | 0.9622 | 0.9577 | 0.9579 | 0.9578 | 0.9941 | 0.9930 |

La Iteración 3 comete **473 errores** (237 falsos positivos y 236 falsos negativos). La variable más influyente según SHAP es `title_caps_ratio`; por bloques, el SVD del cuerpo suma el 71.7 % de la importancia, las 10 estructurales el 17.2 % y el SVD del título el 11.1 %.

**Limitaciones:** “reuters” sigue en el vocabulario del cuerpo (aparece en `text_svd_87`); además, 153 artículos de Test tienen el mismo vector SVD del cuerpo que alguno de Train (en 33 es todo ceros; excluirlos cambia el F1 de 0.9578 a 0.9573). Las métricas describen el desempeño dentro de WELFake, no en fuentes nuevas.

| Archivo | Contenido |
|---------|-----------|
| `reports/tables/tabla07_metricas_test.{csv,md}` | Tabla 7: métricas y matrices de confusión en Test |
| `reports/tables/tabla08_errores.{csv,md}` | Tabla 8: las 6 noticias mal clasificadas (versión técnica) |
| `reports/tables/04_errores_texto.md` | Título y primeros 2,500 caracteres de esas 6 noticias |
| `reports/tables/04_terminos_svd.csv` | Términos de los componentes SVD influyentes |
| `reports/tables/04_eval_log.json` | Cifras exactas de métricas, errores, SHAP y solapamiento |
| `reports/figures/fig09_test_comparison.png` | Figura 9: curvas ROC y PR, y matriz de confusión |
| `reports/figures/fig10_shap_summary.png` | Figura 10: resumen SHAP global |
| `reports/figures/fig11_shap_waterfall_acierto.png` | Figura 11: explicación de un acierto |
| `reports/figures/fig12_shap_waterfall_error.png` | Figura 12: explicación de un error |
| `reports/sections/04_desarrollo_evaluacion.md` | Sección 3.4 (redacción) |
| `reports/Proyecto_FakeNews_XGBoost_completo.docx` | **Documento más actualizado:** índice, introducción, desarrollo (secciones 1 a 4), conclusiones y referencias |


## Estructura

```
fake-news-xgboost/
├── CLAUDE.md            # Reglas del equipo y contratos entre integrantes
├── README.md            # Este archivo
├── requirements.txt     # Librerías con versiones exactas
├── data/
│   ├── raw/             # Aquí va WELFake_Dataset.csv (lo descargas tú)
│   ├── interim/         # welfake_clean.parquet (lo genera el Paso 1)
│   └── processed/       # Salida del Integrante 2
├── models/              # Transformadores y modelos (Integrantes 2 y 3)
├── src/                 # Scripts 01 → 04
└── reports/
    ├── figures/         # Gráficas a 300 dpi
    ├── tables/          # Datos exactos de cada tabla
    └── sections/        # Redacción por integrante
```

## Problemas comunes

| Error | Solución |
|-------|----------|
| `python` no se reconoce | Reinstala Python marcando “Add Python to PATH”, o usa `py` en lugar de `python`. |
| `No module named ...` | No activaste el entorno. Ejecuta `.venv\Scripts\activate` y vuelve a intentar. |
| `FileNotFoundError: ... WELFake_Dataset.csv` | El CSV no está en `data/raw/` o tiene otro nombre. Revisa el Paso 3. |
| `401 Unauthorized` al usar `kaggle` | El `kaggle.json` no está en la carpeta `.kaggle` o está vencido. Genera uno nuevo o usa la opción manual. |
