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

- **Python 3.10 o superior.** Compruébalo con `python --version`. Si no lo tienes, descárgalo de https://www.python.org/downloads/ y durante la instalación marca la casilla **“Add Python to PATH”**.
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

### Paso 5. Trabajar en tu parte

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
| 2 | `python src/02_feature_eng.py` | Integrante 2 | Pendiente |
| 3 | `python src/03_modeling_tuning.py` | Integrante 3 | Pendiente |
| 4 | `python src/04_eval_shap_errors.py` | Integrante 4 | Pendiente |

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
