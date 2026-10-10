# -*- coding: utf-8 -*-
"""
☕ Dashboard Café Brasil (COE 2023) – Streamlit para Hugging Face Spaces
=======================================================================
Envuelve el análisis exploratorio (EDA) del Excel de café de Brasil:
  - Limpieza de la hoja 'BRAZIL COE_Sin Atr' (encabezado detectado automáticamente)
  - Estadísticas descriptivas, distribuciones y matriz de correlación
  - Ganadores por año, atributos sensoriales vs PONT
  - PCA, K-means (sobre componentes PCA) y MFA (Análisis Factorial Múltiple)

Todo es relativo al repositorio: si el usuario no sube nada, se intenta cargar 'data.xlsx'.
"""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin interfaz gráfica (necesario en servidores)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Dashboard Café Brasil", page_icon="☕", layout="wide")

# =============================================================================
# 1. CONFIGURACIÓN (aquí puedes adaptar nombres de columnas y rutas)
# =============================================================================
# Archivo por defecto (ruta relativa al repositorio)
ARCHIVO_POR_DEFECTO = Path(__file__).parent / "data.xlsx"
HOJA_PRINCIPAL = "BRAZIL COE_Sin Atr"

# Nombres canónicos de las columnas clave
COL_ANIO = "Año"
COL_POS = "POS"
COL_PONT = "PONT"
COL_AREA = "AT(ha)"
COL_ALT = "Alti(msnm)"
COLS_CLAVE = [COL_ANIO, COL_POS, COL_PONT, COL_AREA, "ATCaf(ha)", "TLei(Lb)", "TLei(kg)", COL_ALT]

# Columnas que se revisan para el checkbox de "filas con nulos"
COLS_NULOS_CLAVE = [COL_ANIO, COL_POS, COL_PONT]

# Columnas de texto (identificadores / categorías); el resto se intenta convertir a número
COLS_TEXTO = ["Amostras", "CTG", "Produtor", "Fazenda", "RG", "ES", "CDE"]

# Columnas que NO son atributos sensoriales (misma lista del script original)
COLS_NO_SENSORIALES = [
    "Amostras", "CTG", "Año", "POS", "Produtor", "Fazenda", "RG", "ES", "CDE",
    "RFA", "UTZ", "CMC", "COrg", "4C", "BSCA", "CBio", "FaTr", "CULT",
    "ColhMAN", "ColhMEC", "DesNat", "Nat", "Lav", "Dry", "SDry", "Wet",
    "FerCont", "NNFer", "SecSun", "SecMEC", "PatCon", "CamAfr", "Inver",
    "EquipMEC", "AT(ha)", "ATCaf(ha)", "TLei(Lb)", "TLei(kg)", "Alti(msnm)",
    "NumAtri", "PONT",
]

# Columnas continuas con cola larga (se les puede aplicar log1p en el MFA)
COLS_LOG_MFA = ["AT(ha)", "ATCaf(ha)", "TLei(Lb)", "TLei(kg)"]

# Alias de encabezados -> nombre canónico (clave en minúsculas y sin espacios)
ALIAS_COLUMNAS = {
    "año": COL_ANIO, "ano": COL_ANIO, "year": COL_ANIO, "a\ufffdo": COL_ANIO,
    "pos": COL_POS, "pont": COL_PONT, "overall": COL_PONT,
    "at(ha)": COL_AREA, "atcaf(ha)": "ATCaf(ha)",
    "tlei(lb)": "TLei(Lb)", "tlei(kg)": "TLei(kg)", "alti(msnm)": COL_ALT,
}


# =============================================================================
# 2. FUNCIONES AUXILIARES DE LIMPIEZA
# =============================================================================
def flatten_multiindex_columns(df):
    new_columns = []
    for i, col in enumerate(df.columns):
        if isinstance(col, str):
            new_columns.append(col)
            continue

        level0 = str(col[0]).strip() if pd.notna(col[0]) else ''
        level1 = str(col[1]).strip() if pd.notna(col[1]) else ''

        if level0 and level1:
            new_columns.append(f'{level0}_{level1}')
        elif level0:
            new_columns.append(level0)
        elif level1:
            new_columns.append(level1)
        else:
            current_name = f'Unnamed: {i}'
            new_columns.append(current_name)

    seen = {}
    final_columns = []
    for col_name in new_columns:
        original_name = col_name
        count = seen.get(col_name, 0)
        temp_col_name = col_name
        while temp_col_name in final_columns:
            count += 1
            temp_col_name = f"{original_name}.{count}"
        final_columns.append(temp_col_name)
        seen[original_name] = count

    return final_columns


def _norm(texto):
    """Normaliza un encabezado para compararlo (minúsculas, sin espacios)."""
    return str(texto).strip().lower().replace(" ", "")


def detectar_fila_encabezado(raw):
    """
    Busca (en las primeras 15 filas) la fila de encabezados reales.
    1) Fila con palabras clave (Amostras, POS, PONT, Overall, Año...).
    2) Si no hay, la fila con más celdas de texto.
    Devuelve el índice de fila o None si no encuentra nada razonable.
    """
    claves = {"amostras", "pos", "pont", "overall", "individuals", "año", "ano", "ctg"}
    limite = min(15, len(raw))

    mejor_idx, mejor_score = None, 0
    for i in range(limite):
        celdas = {_norm(v) for v in raw.iloc[i].tolist() if pd.notna(v)}
        score = len(celdas & claves)
        if score > mejor_score:
            mejor_idx, mejor_score = i, score
    if mejor_score >= 2:
        return mejor_idx

    mejor_idx, mejor_n = None, 0
    for i in range(limite):
        n_txt = sum(isinstance(v, str) for v in raw.iloc[i].tolist())
        if n_txt > mejor_n:
            mejor_idx, mejor_n = i, n_txt
    return mejor_idx if mejor_n >= 3 else None


@st.cache_data(show_spinner=False)
def listar_hojas(contenido: bytes):
    """Devuelve los nombres de hojas del Excel (cacheado)."""
    return pd.ExcelFile(io.BytesIO(contenido)).sheet_names


@st.cache_data(show_spinner="Cargando y limpiando el Excel...")
def cargar_y_limpiar(contenido: bytes, hoja: str):
    """
    Lee una hoja, detecta el encabezado, renombra columnas clave, elimina columnas
    vacías / 'Unnamed:' y convierte a numérico. Devuelve (DataFrame, avisos).
    avisos = lista de tuplas ('error' | 'warning' | 'info', mensaje).
    """
    avisos = []
    xls = pd.ExcelFile(io.BytesIO(contenido))
    raw = pd.read_excel(xls, sheet_name=hoja, header=None)
    if raw.empty:
        raise ValueError("La hoja seleccionada está vacía.")

    idx = detectar_fila_encabezado(raw)
    if idx is not None:
        nombres = raw.iloc[idx].tolist()
        df = raw.iloc[idx + 1:].reset_index(drop=True).copy()
        df.columns = [str(n).strip() if pd.notna(n) else f"Unnamed: {i}" for i, n in enumerate(nombres)]
    else:
        # Plan B: encabezado de dos filas aplanado (como en el script original)
        avisos.append(("warning", "No se detectó la fila de encabezado; se usó header=[0,1] aplanado."))
        df = pd.read_excel(xls, sheet_name=hoja, header=[0, 1])
        df.columns = flatten_multiindex_columns(df)

    # Nombres de columna únicos
    vistos, nuevos = {}, []
    for c in df.columns:
        if c in vistos:
            vistos[c] += 1
            nuevos.append(f"{c}.{vistos[c]}")
        else:
            vistos[c] = 0
            nuevos.append(c)
    df.columns = nuevos

    # Renombrar columnas clave a nombres legibles (Año, POS, PONT, ...)
    renombres = {}
    for c in df.columns:
        destino = ALIAS_COLUMNAS.get(_norm(c))
        if destino and destino != c and destino not in df.columns and destino not in renombres.values():
            renombres[c] = destino
    df = df.rename(columns=renombres)

    # Quitar columnas 'Unnamed:' / vacías y filas completamente vacías
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if "Amostras" in df.columns:  # filas de metadatos o vacías
        df = df.dropna(subset=["Amostras"])
    df = df.reset_index(drop=True)

    # Tipos: texto para identificadores, número para el resto
    for c in df.columns:
        if c in COLS_TEXTO:
            df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        if c in COLS_CLAVE or conv.notna().sum() >= 0.5 * max(df[c].notna().sum(), 1):
            df[c] = conv.astype(float)
        else:  # columna de texto residual: evitar tipos mixtos (Arrow)
            df[c] = df[c].where(df[c].isna(), df[c].astype(str))

    # Validaciones de estructura
    if COL_PONT not in df.columns:
        avisos.append(("error", f"No se encontró la columna '{COL_PONT}' (ni 'Overall'). "
                                "Selecciona otra hoja o revisa la estructura del archivo."))
    for c in (COL_ANIO, COL_POS):
        if c not in df.columns:
            avisos.append(("warning", f"Falta la columna '{c}': algunos filtros y pestañas no estarán disponibles."))
    if not any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        avisos.append(("error", "La hoja no contiene columnas numéricas."))
    return df, avisos


def columnas_numericas(df):
    """Columnas numéricas con al menos un dato."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]


def columnas_sensoriales(df):
    """Atributos sensoriales = numéricas que no están en la lista de no-sensoriales."""
    return [c for c in columnas_numericas(df) if c not in COLS_NO_SENSORIALES and c not in COLS_TEXTO]


def mostrar_plotly(fig):
    st.plotly_chart(fig, use_container_width=True)


def mostrar_mpl(fig):
    st.pyplot(fig)
    plt.close(fig)  # libera memoria


def categorias_disponibles(df, candidatas=("CTG", "ES", "RG")):
    return [c for c in candidatas if c in df.columns and df[c].notna().any()]


# =============================================================================
# 3. FUNCIONES DE ANÁLISIS (cacheadas)
# =============================================================================
@st.cache_data(show_spinner=False)
def calcular_pca(X: pd.DataFrame, n_max: int):
    """PCA sobre datos estandarizados. Devuelve (scores, loadings, varianza explicada)."""
    Z = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=n_max, random_state=42)
    comps = pca.fit_transform(Z)
    nombres = [f"PC{i + 1}" for i in range(n_max)]
    scores = pd.DataFrame(comps, index=X.index, columns=nombres)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=nombres)
    return scores, loadings, pca.explained_variance_ratio_


@st.cache_data(show_spinner=False)
def curva_codo(Z: np.ndarray, k_max: int, semilla: int):
    """Inercia (método del codo) y silueta para k = 1..k_max."""
    inercias, siluetas = [], []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=semilla, n_init=10).fit(Z)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(Z, km.labels_) if k >= 2 else np.nan)
    return inercias, siluetas


@st.cache_data(show_spinner=False)
def ajustar_kmeans(Z: np.ndarray, k: int, semilla: int):
    return KMeans(n_clusters=k, random_state=semilla, n_init=10).fit_predict(Z)


@st.cache_data(show_spinner="Calculando MFA...")
def calcular_mfa(grupos: dict, n_comp: int = 10):
    """
    Análisis Factorial Múltiple (MFA) implementado con numpy (sin dependencias extra):
      1) Cada grupo se estandariza (z-score).
      2) Cada grupo se pondera por 1 / (primer valor singular de su PCA).
      3) Se concatenan los grupos y se hace un SVD global.
    Devuelve coordenadas de individuos, correlaciones variable-factor,
    inercia parcial por grupo y valores propios.
    """
    bloques, zs, meta = [], [], []
    indice = None
    for nombre, Xg in grupos.items():
        Xg = Xg.astype(float)
        Xg = Xg.fillna(Xg.mean())                 # imputación por la media
        sd = Xg.std(ddof=0)
        Xg = Xg.loc[:, sd > 0]                    # quita variables constantes
        sd = sd[sd > 0]
        Z = ((Xg - Xg.mean()) / sd).values
        n = Z.shape[0]
        indice = Xg.index
        s1 = np.linalg.svd(Z / np.sqrt(n), compute_uv=False)[0]
        bloques.append(Z / np.sqrt(n) / s1)       # peso del grupo = 1 / s1
        zs.append(Z)
        meta += [(nombre, c) for c in Xg.columns]

    Xw = np.hstack(bloques)
    Zall = np.hstack(zs)
    U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
    k = min(n_comp, len(S))
    n = Xw.shape[0]
    nombres = [f"Dim{i + 1}" for i in range(k)]

    eig = S ** 2
    F = U[:, :k] * S[:k] * np.sqrt(n)            # coordenadas de los individuos
    corr = Zall.T @ F / (n * S[:k])              # correlación variable-factor

    meta_df = pd.DataFrame(meta, columns=["Grupo", "Variable"])
    contrib = pd.DataFrame(Vt[:k].T ** 2 * 100, columns=nombres)  # contribución (%) a cada dim
    inercia_g = (
        pd.DataFrame((S[:k] ** 2) * (Vt[:k].T ** 2), columns=nombres)
        .assign(Grupo=meta_df["Grupo"].values)
        .groupby("Grupo").sum()
    )
    inercia_g_pct = inercia_g.div(eig[:k], axis=1) * 100

    return {
        "coords": pd.DataFrame(F, index=indice, columns=nombres),
        "corr": pd.concat([meta_df, pd.DataFrame(corr, columns=nombres)], axis=1),
        "contrib": pd.concat([meta_df, contrib], axis=1),
        "inercia_grupo_pct": inercia_g_pct,
        "eig": pd.DataFrame({
            "Dimensión": nombres,
            "Valor propio": eig[:k],
            "% inercia": eig[:k] / eig.sum() * 100,
        }).assign(**{"% acumulado": lambda d: d["% inercia"].cumsum()}),
    }


# =============================================================================
# 4. BARRA LATERAL: archivo, hoja y filtros
# =============================================================================
st.title("☕ Dashboard Café Brasil – COE 2023")
st.markdown(
    "Explora el análisis exploratorio de las muestras de café de Brasil: estadísticas, "
    "correlaciones, ganadores por año, **PCA**, **clustering (K-means)** y **MFA**. "
    "Carga tu Excel en la barra lateral o usa el archivo por defecto del repositorio."
)

st.sidebar.header("⚙️ Configuración")
archivo = st.sidebar.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls"])

if archivo is not None:
    contenido, origen = archivo.getvalue(), archivo.name
elif ARCHIVO_POR_DEFECTO.exists():
    contenido, origen = ARCHIVO_POR_DEFECTO.read_bytes(), f"{ARCHIVO_POR_DEFECTO.name} (por defecto)"
else:
    st.info("👈 Sube un archivo Excel (.xlsx / .xls) o añade un 'data.xlsx' en la raíz del repositorio.")
    st.stop()
st.sidebar.caption(f"Archivo en uso: **{origen}**")

try:
    hojas = listar_hojas(contenido)
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

hoja = st.sidebar.selectbox(
    "Hoja del Excel", hojas, index=hojas.index(HOJA_PRINCIPAL) if HOJA_PRINCIPAL in hojas else 0
)

try:
    df, avisos = cargar_y_limpiar(contenido, hoja)
except Exception as e:
    st.error(f"Error al procesar la hoja '{hoja}': {e}")
    st.stop()

hay_error = False
for nivel, msg in avisos:
    if nivel == "error":
        st.error(msg)
        hay_error = True
    elif nivel == "warning":
        st.warning(msg)
    else:
        st.info(msg)
if hay_error:
    st.stop()

st.sidebar.subheader("🔎 Filtros")

# --- Rango de Año
rango_anio, lim_anio = None, None
if COL_ANIO in df.columns and df[COL_ANIO].notna().any():
    a_min, a_max = int(df[COL_ANIO].min()), int(df[COL_ANIO].max())
    lim_anio = (a_min, a_max)
    if a_min < a_max:
        rango_anio = st.sidebar.slider("Rango de Año", a_min, a_max, (a_min, a_max))
    else:
        st.sidebar.caption(f"Año único: {a_min}")

# --- Rango de altitud
rango_alt, lim_alt = None, None
if COL_ALT in df.columns and df[COL_ALT].notna().any():
    h_min, h_max = int(df[COL_ALT].min()), int(df[COL_ALT].max())
    lim_alt = (h_min, h_max)
    if h_min < h_max:
        rango_alt = st.sidebar.slider("Rango de Alti(msnm)", h_min, h_max, (h_min, h_max))
    else:
        st.sidebar.caption(f"Altitud única: {h_min} msnm")

# --- Posición (POS)
pos_sel = []
if COL_POS in df.columns and df[COL_POS].notna().any():
    opciones_pos = sorted(df[COL_POS].dropna().astype(int).unique().tolist())
    pos_sel = st.sidebar.multiselect("POS (vacío = todas)", opciones_pos, default=[])

# --- Nulos en columnas clave
mostrar_nulos = st.sidebar.checkbox(
    "Mostrar filas con nulos en columnas clave",
    value=False,
    help=f"Si está desactivado, se ocultan filas con nulos en: {', '.join(COLS_NULOS_CLAVE)}.",
)


def filtrar_rango(serie, rango, limites, incluir_nulos):
    """Aplica un filtro de rango; conserva los nulos si el rango es completo o se pidió verlos."""
    cond = serie.between(rango[0], rango[1])
    if incluir_nulos or tuple(rango) == tuple(limites):
        cond = cond | serie.isna()
    return cond


mascara = pd.Series(True, index=df.index)
if not mostrar_nulos:
    cols_nulos = [c for c in COLS_NULOS_CLAVE if c in df.columns]
    if cols_nulos:
        mascara &= df[cols_nulos].notna().all(axis=1)
if rango_anio:
    mascara &= filtrar_rango(df[COL_ANIO], rango_anio, lim_anio, mostrar_nulos)
if rango_alt:
    mascara &= filtrar_rango(df[COL_ALT], rango_alt, lim_alt, mostrar_nulos)
if pos_sel:
    mascara &= df[COL_POS].isin(pos_sel)

df_f = df[mascara].copy()
st.sidebar.markdown(f"**Registros filtrados:** {len(df_f):,} de {len(df):,}")

if df_f.empty:
    st.warning("Los filtros actuales no dejan ningún registro. Amplía los rangos o desactiva filtros.")
    st.stop()

sens_cols = columnas_sensoriales(df)
num_cols = columnas_numericas(df_f)


# =============================================================================
# 5. PESTAÑAS
# =============================================================================
def tab_resumen():
    def media(col, dec):
        if col in df_f.columns and df_f[col].notna().any():
            return f"{df_f[col].mean():,.{dec}f}"
        return "N/D"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df_f):,}")
    c2.metric("Promedio de PONT", media(COL_PONT, 2))
    c3.metric("Promedio de AT(ha)", media(COL_AREA, 1))
    c4.metric("Promedio de Alti(msnm)", media(COL_ALT, 0))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("PONT máximo", f"{df_f[COL_PONT].max():.1f}" if df_f[COL_PONT].notna().any() else "N/D")
    c6.metric("Estados (ES)", df_f["ES"].nunique() if "ES" in df_f else "N/D")
    c7.metric("Productores", df_f["Produtor"].nunique() if "Produtor" in df_f else "N/D")
    c8.metric("Atributos sensoriales", len(sens_cols))

    g1, g2 = st.columns(2)
    if COL_ANIO in df_f.columns and df_f[COL_ANIO].notna().any():
        por_anio = df_f[COL_ANIO].dropna().astype(int).value_counts().sort_index().reset_index()
        por_anio.columns = ["Año", "Registros"]
        with g1:
            mostrar_plotly(px.bar(por_anio, x="Año", y="Registros", title="Registros por año"))
    if "CTG" in df_f.columns:
        por_ctg = df_f["CTG"].value_counts().reset_index()
        por_ctg.columns = ["CTG", "Registros"]
        with g2:
            mostrar_plotly(px.bar(por_ctg, x="CTG", y="Registros", title="Registros por categoría (CTG)"))


def tab_estadisticas():
    base = [c for c in COLS_CLAVE if c in df_f.columns]
    extra = st.multiselect("Añadir columnas al resumen", [c for c in num_cols if c not in base], key="est_extra")
    cols = base + extra
    if not cols:
        st.warning("No hay columnas numéricas clave para describir.")
        return
    transponer = st.checkbox("Transponer tabla (una fila por variable)", value=True, key="est_t")
    desc = df_f[cols].describe()
    st.dataframe(desc.T.round(3) if transponer else desc.round(3), use_container_width=True)

    with st.expander("Valores nulos por columna"):
        nulos = df_f.isna().sum().to_frame("Nulos")
        nulos["% nulos"] = (nulos["Nulos"] / len(df_f) * 100).round(1)
        st.dataframe(nulos[nulos["Nulos"] > 0].sort_values("Nulos", ascending=False), use_container_width=True)


def tab_distribuciones():
    if not num_cols:
        st.warning("No hay columnas numéricas para graficar.")
        return
    ordenadas = [c for c in COLS_CLAVE if c in num_cols] + [c for c in num_cols if c not in COLS_CLAVE]
    c1, c2, c3 = st.columns(3)
    col = c1.selectbox("Columna", ordenadas, index=ordenadas.index(COL_PONT) if COL_PONT in ordenadas else 0, key="dist_col")
    bins = c2.slider("Número de bins", 5, 100, 30, key="dist_bins")
    cats = categorias_disponibles(df_f)
    color = c3.selectbox("Colorear por", ["(ninguno)"] + cats, key="dist_color")

    datos = df_f.dropna(subset=[col])
    fig = px.histogram(
        datos, x=col, nbins=bins, marginal="box", opacity=0.8,
        color=None if color == "(ninguno)" else color,
        barmode="overlay", title=f"Distribución de {col}",
    )
    mostrar_plotly(fig)
    s = datos[col]
    st.caption(f"n = {len(s):,} · media = {s.mean():.3f} · mediana = {s.median():.3f} · "
               f"desv. est. = {s.std():.3f} · asimetría = {s.skew():.2f}")


def tab_correlacion():
    c1, c2, c3 = st.columns(3)
    metodo = c1.radio("Método", ["pearson", "spearman"], horizontal=True, key="cor_met")
    min_pct = c2.slider("Mín. % de datos no nulos por columna", 10, 100, 50, 5, key="cor_pct")
    solo_completas = c3.checkbox("Usar solo filas completas (dropna)", value=False, key="cor_dropna")

    c4, c5, c6 = st.columns(3)
    umbral = c4.slider("Umbral mínimo de |correlación|", 0.0, 1.0, 0.3, 0.05, key="cor_umbral")
    modo = c5.radio("Variables relevantes según", ["Cualquier par", f"Relación con {COL_PONT}"], key="cor_modo")
    max_vars = c6.slider("Máx. de variables en el heatmap", 5, 60, 25, key="cor_max")

    # Columnas candidatas: numéricas con > min_pct de datos y varianza > 0
    cands = [c for c in num_cols if df_f[c].notna().mean() * 100 >= min_pct and df_f[c].nunique() > 1]
    if len(cands) < 2:
        st.warning("No hay suficientes columnas numéricas (con varianza y datos) para calcular correlaciones.")
        return

    datos = df_f[cands]
    if solo_completas:
        datos = datos.dropna()
        if len(datos) < 3:
            st.warning("Con 'solo filas completas' quedan menos de 3 filas. Desactiva esa opción.")
            return
    corr = datos.corr(method=metodo).dropna(how="all").dropna(axis=1, how="all")

    if modo.startswith("Relación") and COL_PONT in corr.columns:
        rel = corr[COL_PONT].drop(COL_PONT).abs()
        sel = rel[rel >= umbral].sort_values(ascending=False).head(max_vars - 1).index.tolist() + [COL_PONT]
    else:
        abs_c = corr.abs().copy()
        np.fill_diagonal(abs_c.values, 0)
        mx = abs_c.max(axis=1)
        sel = mx[mx >= umbral].sort_values(ascending=False).head(max_vars).index.tolist()

    if len(sel) < 2:
        st.info("Ninguna variable supera el umbral elegido. Baja el umbral para ver resultados.")
        return

    sub = corr.loc[sel, sel]
    st.caption(f"Se muestran {len(sel)} de {len(corr)} variables · método: {metodo}.")
    lado = max(6, 0.5 * len(sel))
    fig, ax = plt.subplots(figsize=(min(lado + 2, 18), min(lado, 15)))
    sns.heatmap(sub, cmap="coolwarm", vmin=-1, vmax=1, annot=len(sel) <= 25, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de correlación de variables numéricas")
    plt.tight_layout()
    mostrar_mpl(fig)

    # Pares más correlacionados
    mascara_sup = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pares = corr.where(mascara_sup).stack().reset_index()
    pares.columns = ["Variable 1", "Variable 2", "Correlación"]
    pares = pares[pares["Correlación"].abs() >= umbral]
    pares = pares.reindex(pares["Correlación"].abs().sort_values(ascending=False).index).head(30)
    with st.expander(f"Pares con |r| ≥ {umbral:.2f} (top 30)"):
        st.dataframe(pares.round(3).reset_index(drop=True), use_container_width=True)


def tab_ganadores():
    if COL_ANIO not in df_f.columns or COL_POS not in df_f.columns:
        st.warning("Esta pestaña necesita las columnas 'Año' y 'POS'.")
        return
    top_n = st.slider("Top N posiciones por año", 1, 20, 5, key="gan_n")
    d = df_f.dropna(subset=[COL_ANIO, COL_POS]).copy()
    d[COL_ANIO] = d[COL_ANIO].astype(int)
    top = d.sort_values([COL_ANIO, COL_POS]).groupby(COL_ANIO).head(top_n)
    mostrar = [c for c in [COL_ANIO, COL_POS, "Amostras", "CTG", "Produtor", "Fazenda", "ES", "RG", COL_PONT] if c in top]
    st.dataframe(top[mostrar].reset_index(drop=True), use_container_width=True)

    c1, c2 = st.columns(2)
    if "CTG" in top.columns:
        freq = top["CTG"].value_counts().reset_index()
        freq.columns = ["CTG", "Frecuencia"]
        with c1:
            mostrar_plotly(px.bar(freq, x="CTG", y="Frecuencia", title=f"Categorías (CTG) en el top {top_n}"))
    if "Produtor" in top.columns:
        rep = top["Produtor"].value_counts()
        rep = rep[rep >= 2].head(15).reset_index()
        rep.columns = ["Produtor", "Presencias en el top"]
        with c2:
            st.markdown("**Productores con más presencias en el top**")
            if rep.empty:
                st.caption("Ningún productor aparece más de una vez con este filtro.")
            else:
                st.dataframe(rep, use_container_width=True)

    if top[COL_PONT].notna().any():
        mostrar_plotly(px.box(top, x=COL_ANIO, y=COL_PONT, points="all",
                              title=f"PONT del top {top_n} por año", hover_data=[c for c in ["Amostras", COL_POS] if c in top]))
        with st.expander("Estadísticos de PONT y POS por año (top N)"):
            est = top.groupby(COL_ANIO).agg(
                PONT_media=(COL_PONT, "mean"), PONT_mediana=(COL_PONT, "median"),
                PONT_min=(COL_PONT, "min"), PONT_max=(COL_PONT, "max"),
                POS_media=(COL_POS, "mean"), POS_max=(COL_POS, "max"),
            ).round(2)
            st.dataframe(est, use_container_width=True)

    if "CTG" in d.columns and d[COL_PONT].notna().any():
        ev = d.dropna(subset=["CTG", COL_PONT]).groupby([COL_ANIO, "CTG"])[COL_PONT].mean().reset_index()
        mostrar_plotly(px.line(ev, x=COL_ANIO, y=COL_PONT, color="CTG", markers=True,
                               title="PONT promedio por categoría (CTG) a lo largo del tiempo (todas las muestras filtradas)"))
        filas = []
        for ctg, g in ev.groupby("CTG"):
            if len(g) >= 3:
                r = stats.linregress(g[COL_ANIO], g[COL_PONT])
                filas.append({"CTG": ctg, "Pendiente (pts/año)": r.slope, "R²": r.rvalue ** 2, "p-valor": r.pvalue, "n años": len(g)})
        if filas:
            with st.expander("Tendencia lineal del PONT por categoría"):
                st.dataframe(pd.DataFrame(filas).round(4), use_container_width=True)


def tab_atributos():
    if COL_PONT not in df_f.columns or not sens_cols:
        st.warning("No hay atributos sensoriales o falta 'PONT'.")
        return
    cols = [c for c in sens_cols if c in df_f.columns]
    with np.errstate(all="ignore"):
        cp = df_f[cols].corrwith(df_f[COL_PONT]).dropna().sort_values(ascending=False)
    if cp.empty:
        st.warning("No se pudo calcular ninguna correlación con PONT.")
        return
    n = st.slider("Atributos a mostrar por lado (positivos / negativos)", 3, 30, 10, key="atr_n")
    sel = pd.concat([cp.head(n), cp.tail(n)]).drop_duplicates()
    graf = sel.reset_index()
    graf.columns = ["Atributo", "Correlación con PONT"]
    graf["Signo"] = np.where(graf["Correlación con PONT"] >= 0, "Positiva", "Negativa")
    fig = px.bar(graf.sort_values("Correlación con PONT"), x="Correlación con PONT", y="Atributo", color="Signo",
                 orientation="h", color_discrete_map={"Positiva": "#2a9d8f", "Negativa": "#e76f51"},
                 title=f"Top {n} correlaciones positivas y negativas con PONT", height=max(400, 22 * len(graf)))
    mostrar_plotly(fig)
    st.caption("⚠️ Las correlaciones son débiles (|r| < 0.4 típicamente) y los atributos son presencia/ausencia (0/1): "
               "interprétalas como tendencias, no como causalidad.")

    # Tabla con prevalencia y diferencia de PONT
    filas = []
    for a, r in cp.items():
        presente = df_f.loc[df_f[a] > 0, COL_PONT]
        ausente = df_f.loc[df_f[a] == 0, COL_PONT]
        filas.append({
            "Atributo": a, "Correlación": r, "Prevalencia (%)": (df_f[a] > 0).mean() * 100,
            "PONT medio (con)": presente.mean(), "PONT medio (sin)": ausente.mean(),
            "Δ PONT": presente.mean() - ausente.mean(),
        })
    with st.expander("Tabla completa de atributos"):
        st.dataframe(pd.DataFrame(filas).round(3), use_container_width=True)

    atributo = st.selectbox("Explorar un atributo", list(cp.index), key="atr_sel")
    d = df_f.dropna(subset=[atributo, COL_PONT])
    if d[atributo].nunique() <= 2:
        d = d.assign(**{atributo: d[atributo].astype(int).astype(str)})
        mostrar_plotly(px.box(d, x=atributo, y=COL_PONT, points="all", title=f"PONT según presencia de {atributo} (0 = ausente, 1 = presente)"))
    else:
        mostrar_plotly(px.scatter(d, x=atributo, y=COL_PONT, title=f"PONT vs {atributo}"))


def preparar_pca():
    """Prepara la matriz sensorial (NaN -> 0, sin columnas constantes) y ejecuta el PCA."""
    cols = [c for c in sens_cols if c in df_f.columns]
    if len(cols) < 2:
        return None
    X = df_f[cols].fillna(0)
    X = X.loc[:, X.std() > 0]
    if X.shape[0] < 4 or X.shape[1] < 2:
        return None
    n_max = int(min(30, X.shape[0] - 1, X.shape[1]))
    scores, loadings, ratio = calcular_pca(X, n_max)
    return X, scores, loadings, ratio


def tab_pca(res):
    if res is None:
        st.warning("No hay suficientes atributos sensoriales / muestras para el PCA (mín. 4 filas y 2 atributos con varianza).")
        return
    X, scores, loadings, ratio = res
    n_max = len(ratio)

    cum = np.cumsum(ratio) * 100
    fig = go.Figure()
    fig.add_bar(x=list(range(1, n_max + 1)), y=ratio * 100, name="% varianza")
    fig.add_scatter(x=list(range(1, n_max + 1)), y=cum, name="% acumulada", mode="lines+markers")
    fig.update_layout(title="Scree plot: varianza explicada por componente", xaxis_title="Componente", yaxis_title="%")
    mostrar_plotly(fig)

    c1, c2, c3 = st.columns(3)
    opciones = list(range(1, n_max + 1))
    px_ = c1.selectbox("Eje X (PC)", opciones, index=0, key="pca_x")
    py_ = c2.selectbox("Eje Y (PC)", opciones, index=min(1, n_max - 1), key="pca_y")
    color = c3.selectbox("Colorear por", ["(ninguno)"] + categorias_disponibles(df_f) + [c for c in [COL_PONT, COL_ANIO] if c in df_f], key="pca_color")

    plot_df = scores.join(df_f[[c for c in ["Amostras", "CTG", "ES", "RG", COL_PONT, COL_ANIO] if c in df_f]])
    fig = px.scatter(
        plot_df, x=f"PC{px_}", y=f"PC{py_}", color=None if color == "(ninguno)" else color,
        hover_data=[c for c in ["Amostras", COL_PONT, COL_ANIO] if c in plot_df], opacity=0.75,
        title=f"Muestras en el espacio PCA (PC{px_} {ratio[px_ - 1] * 100:.1f}% vs PC{py_} {ratio[py_ - 1] * 100:.1f}%)",
    )
    mostrar_plotly(fig)

    c4, c5 = st.columns([1, 3])
    pc = c4.selectbox("Componente para loadings", opciones, key="pca_load_pc")
    top_l = c4.slider("Nº de atributos", 5, 40, 15, key="pca_load_n")
    ld = loadings[f"PC{pc}"]
    ld = ld.reindex(ld.abs().sort_values(ascending=False).head(top_l).index).reset_index()
    ld.columns = ["Atributo", "Loading"]
    with c5:
        mostrar_plotly(px.bar(ld.iloc[::-1], x="Loading", y="Atributo", orientation="h",
                              title=f"Atributos con mayor |loading| en PC{pc}", height=max(350, 22 * len(ld))))
    st.caption("NaN de atributos sensoriales se rellenan con 0 (ausencia). Datos estandarizados antes del PCA.")


def tab_clustering(res):
    """K-means sobre los componentes PCA. Devuelve la serie de etiquetas (o None)."""
    if res is None:
        st.warning("No hay suficientes datos sensoriales para el clustering.")
        return None
    X, scores, loadings, ratio = res
    n_max = len(ratio)

    c1, c2, c3 = st.columns(3)
    n_pcs = c1.slider("Componentes PCA usados", 2, n_max, min(10, n_max), key="km_pcs")
    k_max_perm = max(2, min(10, len(X) - 1))
    k = c2.slider("Número de clústeres (K)", 2, k_max_perm, min(4, k_max_perm), key="km_k")
    semilla = int(c3.number_input("Semilla aleatoria", 0, 9999, 42, key="km_seed"))

    Z = scores.iloc[:, :n_pcs].values
    inercias, siluetas = curva_codo(Z, k_max_perm, semilla)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ks = list(range(1, k_max_perm + 1))
    fig.add_scatter(x=ks, y=inercias, name="Inercia (codo)", mode="lines+markers", secondary_y=False)
    fig.add_scatter(x=ks[1:], y=siluetas[1:], name="Silueta", mode="lines+markers", secondary_y=True)
    fig.update_layout(title="Elección de K: método del codo y coeficiente de silueta", xaxis_title="K")
    fig.update_yaxes(title_text="Inercia", secondary_y=False)
    fig.update_yaxes(title_text="Silueta", secondary_y=True)
    mostrar_plotly(fig)

    etiquetas = pd.Series(ajustar_kmeans(Z, k, semilla), index=X.index, name="Cluster")
    d = df_f.loc[X.index].join(etiquetas)
    d["Cluster"] = d["Cluster"].astype(int)
    orden = [str(i) for i in range(k)]

    # Tamaños y PONT por clúster
    resumen = d.groupby("Cluster").agg(Muestras=("Cluster", "size"), PONT_medio=(COL_PONT, "mean"), PONT_desv=(COL_PONT, "std")).round(2)
    st.dataframe(resumen, use_container_width=True)
    if (resumen["Muestras"] < 3).any():
        st.warning("Hay clústeres con menos de 3 muestras (posibles outliers). Prueba otro K o más/menos componentes.")

    g1, g2 = st.columns(2)
    sc = scores.assign(Cluster=d["Cluster"].astype(str))
    with g1:
        mostrar_plotly(px.scatter(sc, x="PC1", y="PC2", color="Cluster", category_orders={"Cluster": orden},
                                  title="Clústeres en el espacio PCA (PC1 vs PC2)", opacity=0.75))
    with g2:
        mostrar_plotly(px.box(d.assign(Cluster=d["Cluster"].astype(str)), x="Cluster", y=COL_PONT, points="all",
                              category_orders={"Cluster": orden}, title="Distribución de PONT por clúster"))

    # Pruebas estadísticas
    pont_por = {c: g[COL_PONT].dropna() for c, g in d.groupby("Cluster")}
    validos = [c for c, s in pont_por.items() if len(s) >= 2]
    if len(validos) >= 2:
        st.markdown("**Pruebas sobre PONT entre clústeres**")
        try:
            kw = stats.kruskal(*[pont_por[c] for c in validos])
            st.caption(f"Kruskal-Wallis global: H = {kw.statistic:.3f}, p = {kw.pvalue:.4f} "
                       f"({'diferencias significativas' if kw.pvalue < 0.05 else 'sin diferencias significativas'} al 5%).")
        except ValueError:
            st.caption("No se pudo calcular Kruskal-Wallis (valores idénticos).")
        medias = {c: pont_por[c].mean() for c in validos}
        alto, bajo = max(medias, key=medias.get), min(medias, key=medias.get)
        t1, t2 = st.columns(2)
        ca = t1.selectbox("Clúster A", validos, index=validos.index(alto), key="km_a")
        cb = t2.selectbox("Clúster B", validos, index=validos.index(bajo), key="km_b")
        if ca != cb:
            t, p = stats.ttest_ind(pont_por[ca], pont_por[cb], equal_var=False)
            st.info(f"t de Welch (clúster {ca} vs {cb}): t = {t:.3f}, p = {p:.4f} → "
                    f"{'diferencia significativa' if p < 0.05 else 'sin diferencia significativa'} (α = 0.05). "
                    "Análisis exploratorio: los clústeres se formaron con los mismos datos.")
    else:
        alto = bajo = None

    # Perfil sensorial por clúster
    cols_s = list(X.columns)
    medias_s = d[cols_s].groupby(d["Cluster"]).mean()
    top_var = medias_s.var().sort_values(ascending=False).head(25).index
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * k + 3), 9))
    sns.heatmap(medias_s[top_var].T, cmap="viridis", linewidths=0.5, ax=ax, annot=k <= 6, fmt=".2f")
    ax.set_title("Perfil sensorial por clúster (25 atributos que más los distinguen)")
    plt.tight_layout()
    mostrar_mpl(fig)

    # Radar: clúster de mayor vs menor PONT
    if alto is not None and alto != bajo:
        top_a = medias_s.loc[alto].sort_values(ascending=False).head(10).index
        top_b = medias_s.loc[bajo].sort_values(ascending=False).head(10).index
        atrs = list(dict.fromkeys(list(top_a) + list(top_b)))
        va, vb = medias_s.loc[alto, atrs], medias_s.loc[bajo, atrs]
        mx = pd.concat([va, vb], axis=1).max(axis=1).replace(0, 1)
        va, vb = (va / mx).tolist(), (vb / mx).tolist()
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=va + va[:1], theta=atrs + atrs[:1], fill="toself", name=f"Clúster {alto} (mayor PONT: {medias[alto]:.2f})"))
        radar.add_trace(go.Scatterpolar(r=vb + vb[:1], theta=atrs + atrs[:1], fill="toself", name=f"Clúster {bajo} (menor PONT: {medias[bajo]:.2f})"))
        radar.update_layout(title="Perfil sensorial comparado (normalizado por atributo)", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        mostrar_plotly(radar)

    # Relación clúster - región
    if "ES" in d.columns and d["ES"].notna().any():
        modo = st.radio("Mapa de calor Clúster × Estado (ES)", ["Porcentaje por clúster", "Conteo"], horizontal=True, key="km_heat")
        tabla = pd.crosstab(d["Cluster"], d["ES"])
        if modo.startswith("Porcentaje"):
            tabla = tabla.div(tabla.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * tabla.shape[1] + 2), max(3, 0.8 * k + 1)))
        sns.heatmap(tabla, annot=True, fmt=".1f" if modo.startswith("Porcentaje") else "d", cmap="YlGnBu", linewidths=0.5, ax=ax)
        ax.set_title("Distribución de muestras por clúster y estado")
        plt.tight_layout()
        mostrar_mpl(fig)
    return etiquetas


def tab_mfa():
    st.markdown(
        "**Análisis Factorial Múltiple (MFA)** con dos grupos de variables: "
        "**Localización/cultivo/proceso** vs **Sensorial**. Se ponderan los grupos para que ninguno domine "
        "y se resumen las regiones geográficas por el baricentro de sus muestras."
    )
    # Variables candidatas para el grupo 1 (numéricas no sensoriales, sin resultados del concurso)
    excluidas = {COL_ANIO, COL_POS, COL_PONT, "NumAtri"}
    cand1 = [c for c in df_f.columns
             if c in COLS_NO_SENSORIALES and c not in excluidas and c not in COLS_TEXTO
             and pd.api.types.is_numeric_dtype(df_f[c]) and df_f[c].notna().mean() >= 0.5 and df_f[c].nunique() > 1]
    if "TLei(Lb)" in cand1 and "TLei(kg)" in cand1:
        cand1.remove("TLei(Lb)")  # son equivalentes (lb vs kg)

    c1, c2 = st.columns([2, 1])
    g1_cols = c1.multiselect("Variables del grupo 'Localización/cultivo/proceso'", cand1, default=cand1, key="mfa_g1")
    regiones = [c for c in ["ES", "RG", "CTG"] if c in df_f.columns and df_f[c].notna().any()]
    log1p = c2.checkbox("Aplicar log1p a superficie/producción", value=True, key="mfa_log")
    if not regiones:
        st.warning("No hay columnas de región (ES / RG / CTG) para agrupar.")
        return
    region = c2.selectbox("Región para colorear/agrupar", regiones, key="mfa_region")

    sens = [c for c in sens_cols if c in df_f.columns]
    if len(g1_cols) < 1 or len(sens) < 1:
        st.warning("El MFA necesita al menos una variable en cada grupo.")
        return
    g1 = df_f[g1_cols].copy()
    if log1p:
        for c in COLS_LOG_MFA:
            if c in g1.columns:
                g1[c] = np.log1p(g1[c].clip(lower=0))
    g2 = df_f[sens].fillna(0)
    if len(df_f) < 4:
        st.warning("Se necesitan al menos 4 registros para el MFA.")
        return

    res = calcular_mfa({"Localización/cultivo": g1, "Sensorial": g2}, n_comp=10)
    eig = res["eig"]
    n_dims = len(eig)
    if n_dims < 2:
        st.warning("No hay suficientes dimensiones para graficar.")
        return

    st.dataframe(eig.round(3).head(10), use_container_width=True)
    a, b = st.columns(2)
    dims = list(range(1, n_dims + 1))
    dx = a.selectbox("Dimensión eje X", dims, index=0, key="mfa_dx")
    dy = b.selectbox("Dimensión eje Y", dims, index=1, key="mfa_dy")
    nx, ny = f"Dim{dx}", f"Dim{dy}"

    # Inercia por grupo
    ig = res["inercia_grupo_pct"][[nx, ny]].reset_index().melt(id_vars="Grupo", var_name="Dimensión", value_name="% de la inercia del eje")
    mostrar_plotly(px.bar(ig, x="Dimensión", y="% de la inercia del eje", color="Grupo", barmode="stack",
                          title="Aporte de cada grupo a las dimensiones"))

    # Mapa de individuos + baricentros por región
    coords = res["coords"].join(df_f[[region]])
    min_n = int(st.number_input("Mín. de muestras por región para mostrar baricentro", 1, 100, 5, key="mfa_minn"))
    fig = px.scatter(coords, x=nx, y=ny, color=region, opacity=0.45,
                     title=f"Mapa de individuos (MFA) coloreado por {region}")
    bari = coords.groupby(region)[[nx, ny]].mean()
    bari["n"] = coords.groupby(region).size()
    bari = bari[bari["n"] >= min_n]
    fig.add_trace(go.Scatter(x=bari[nx], y=bari[ny], mode="markers+text", text=bari.index.astype(str),
                             textposition="top center", name="Baricentro",
                             marker=dict(size=14, symbol="diamond", color="black")))
    mostrar_plotly(fig)
    with st.expander(f"Baricentros por {region}"):
        st.dataframe(bari.round(3), use_container_width=True)

    # Círculo de correlaciones
    cr = res["corr"].copy()
    cr["Longitud"] = np.sqrt(cr[nx] ** 2 + cr[ny] ** 2)
    umbral = st.slider("Mostrar etiquetas de variables con longitud ≥", 0.0, 1.0, 0.4, 0.05, key="mfa_umbral")
    fig = px.scatter(cr, x=nx, y=ny, color="Grupo", hover_name="Variable", title="Círculo de correlaciones (variables)")
    vis = cr[cr["Longitud"] >= umbral]
    fig.add_trace(go.Scatter(x=vis[nx], y=vis[ny], mode="text", text=vis["Variable"], textposition="top center", showlegend=False))
    fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1, line=dict(dash="dot"))
    fig.update_xaxes(range=[-1.1, 1.1])
    fig.update_yaxes(range=[-1.1, 1.1], scaleanchor="x", scaleratio=1)
    fig.update_layout(height=650)
    mostrar_plotly(fig)

    # Relación con PONT (variable suplementaria)
    if COL_PONT in df_f.columns:
        sup = coords.join(df_f[COL_PONT]).dropna(subset=[COL_PONT])
        if len(sup) > 3:
            r_x, r_y = sup[nx].corr(sup[COL_PONT]), sup[ny].corr(sup[COL_PONT])
            st.caption(f"PONT como variable suplementaria: r({nx}) = {r_x:.2f}, r({ny}) = {r_y:.2f}.")
    with st.expander("Contribución de variables a las dimensiones (%)"):
        cb = res["contrib"][["Grupo", "Variable", nx, ny]].copy()
        cb["Total"] = cb[nx] + cb[ny]
        st.dataframe(cb.sort_values("Total", ascending=False).head(30).round(2).reset_index(drop=True), use_container_width=True)


def tab_datos(etiquetas):
    d = df_f.copy()
    if etiquetas is not None and st.checkbox("Incluir columna 'Cluster' (K-means de la pestaña Clustering)", value=False, key="raw_cl"):
        d = d.join(etiquetas)
    if st.checkbox("Mostrar todas las columnas", value=True, key="raw_all"):
        cols = list(d.columns)
    else:
        por_defecto = [c for c in ["Amostras", "CTG", COL_ANIO, COL_POS, COL_PONT, "ES", COL_ALT] if c in d.columns]
        cols = st.multiselect("Columnas a mostrar", list(d.columns), default=por_defecto, key="raw_cols") or list(d.columns)
    st.caption(f"{len(d):,} filas × {len(cols)} columnas")
    st.dataframe(d[cols], use_container_width=True, height=500)
    st.download_button(
        "⬇️ Descargar CSV", data=d[cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="cafe_brasil_filtrado.csv", mime="text/csv",
    )


# --- Orden visual de las pestañas (las 5 pedidas primero, luego los análisis adicionales)
tabs = st.tabs([
    "📊 Resumen", "📈 Estadísticas descriptivas", "📉 Distribuciones", "🔥 Matriz de correlación", "🗂️ Datos crudos",
    "🏆 Ganadores por año", "👃 Atributos sensoriales", "🧭 PCA", "🧩 Clustering", "🌎 MFA",
])

with tabs[0]:
    tab_resumen()
with tabs[1]:
    tab_estadisticas()
with tabs[2]:
    tab_distribuciones()
with tabs[3]:
    tab_correlacion()
with tabs[5]:
    tab_ganadores()
with tabs[6]:
    tab_atributos()

resultado_pca = preparar_pca()
with tabs[7]:
    tab_pca(resultado_pca)
with tabs[8]:
    etiquetas_cluster = tab_clustering(resultado_pca)
with tabs[9]:
    tab_mfa()
# 'Datos crudos' se ejecuta al final para poder incluir los clústeres calculados antes
with tabs[4]:
    tab_datos(etiquetas_cluster)

st.caption("Dashboard construido con Streamlit · Datos: Cup of Excellence Brasil 2023")
