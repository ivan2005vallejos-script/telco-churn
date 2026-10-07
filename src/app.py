"""Dashboard de Streamlit para explorar el modelo de churn.

Lee únicamente los reportes en reports/ (no el modelo ni DuckDB).
Se corre con: streamlit run src/app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PREDICTIONS_PATH = "reports/test_predictions.csv"
COEFFICIENTS_PATH = "reports/coefficients.csv"

# Paleta diverging (rojo = sube el riesgo, azul = lo baja)
COLOR_INCREASE = "#e34948"
COLOR_DECREASE = "#2a78d6"

# Estas 7 columnas son colineales: solo valen cuando InternetService_No,
# así que el modelo les da el mismo coeficiente. Las unificamos en una sola barra.
REDUNDANT_NO_INTERNET_FEATURES = [
    "categorical__InternetService_No",
    "categorical__OnlineSecurity_No internet service",
    "categorical__OnlineBackup_No internet service",
    "categorical__DeviceProtection_No internet service",
    "categorical__TechSupport_No internet service",
    "categorical__StreamingTV_No internet service",
    "categorical__StreamingMovies_No internet service",
]


def clean_feature_name(feature: str) -> str:
    """Saca los prefijos categorical__ / numeric__ / passthrough__."""
    return feature.split("__", 1)[-1]


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_coefficients() -> pd.DataFrame:
    df = pd.read_csv(COEFFICIENTS_PATH)

    # Unificamos las 7 columnas redundantes de "sin internet" sumando sus
    # coeficientes: ese es el efecto combinado real para un cliente sin internet
    # (las 7 se activan juntas, no una sola).
    redundant_mask = df["feature"].isin(REDUNDANT_NO_INTERNET_FEATURES)
    redundant_sum = df.loc[redundant_mask, "coefficient"].sum()
    df = df.loc[~redundant_mask].copy()
    combined_row = pd.DataFrame(
        {
            "feature": ["Sin servicio de internet (suma de 7 columnas)"],
            "coefficient": [redundant_sum],
        }
    )
    df = pd.concat([df, combined_row], ignore_index=True)

    df["feature"] = df["feature"].where(
        df["feature"].str.contains("Sin servicio de internet"),
        df["feature"].apply(clean_feature_name),
    )
    return df.sort_values("coefficient", ascending=False)


def reports_exist() -> bool:
    return Path(PREDICTIONS_PATH).exists() and Path(COEFFICIENTS_PATH).exists()


st.title("Dashboard de churn — Telco")
st.markdown(
    "Explora, sobre el **conjunto de test** (clientes que el modelo no vio al "
    "entrenar), cuántos clientes quedarían marcados como riesgo de churn según "
    "el umbral de probabilidad que elijas, y qué variables asocia el modelo con "
    "más o menos riesgo."
)

if not reports_exist():
    st.info(
        "Faltan los reportes en `reports/`. Corré primero `python src/analyze.py` "
        "para generarlos."
    )
    st.stop()

predictions = load_predictions()
coefficients = load_coefficients()

st.header("1. Umbral de decisión")

threshold = st.slider(
    "Umbral de probabilidad de churn",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
)

marked = predictions["churn_proba"] >= threshold
true_positives = (marked & (predictions["churn_real"] == 1)).sum()
false_positives = (marked & (predictions["churn_real"] == 0)).sum()
false_negatives = (~marked & (predictions["churn_real"] == 1)).sum()

n_marked = int(marked.sum())
precision = true_positives / (true_positives + false_positives) if n_marked else 0.0
total_churners = (predictions["churn_real"] == 1).sum()
recall = true_positives / total_churners if total_churners else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Clientes marcados en riesgo", n_marked)
col2.metric("Precision", f"{precision:.2%}")
col3.metric("Recall", f"{recall:.2%}")
col4.metric("Churners reales que se escapan", int(false_negatives))

st.header("2. Clientes marcados en riesgo")

at_risk = (
    predictions.loc[marked, ["customerID", "churn_proba"]]
    .sort_values("churn_proba", ascending=False)
    .reset_index(drop=True)
    .rename(columns={"churn_proba": "Probabilidad de churn"})
)
st.dataframe(at_risk, use_container_width=True, hide_index=True)
st.download_button(
    "Descargar lista como CSV",
    data=at_risk.to_csv(index=False).encode("utf-8"),
    file_name="clientes_en_riesgo.csv",
    mime="text/csv",
)

st.header("3. Qué variables asocia el modelo con el riesgo")
st.caption(
    "Estos coeficientes son asociaciones estadísticas de la regresión logística, "
    "no relaciones de causalidad. Las 7 columnas que solo valen cuando el cliente "
    "no tiene internet (InternetService_No y las versiones 'No internet service' "
    "de OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV y "
    "StreamingMovies) son redundantes entre sí, así que se unificaron en una sola "
    "barra que suma sus 7 coeficientes."
)

top_increase = coefficients.head(10)
top_decrease = coefficients.tail(10).sort_values("coefficient")
chart_data = pd.concat([top_increase, top_decrease]).sort_values("coefficient")

# Fondo transparente y texto en el color del tema activo, para que se vea
# bien tanto en modo claro como en modo oscuro.
theme_base = st.get_option("theme.base") or "light"
text_color = "#f5f5f5" if theme_base == "dark" else "#31333f"

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

colors = [COLOR_INCREASE if c > 0 else COLOR_DECREASE for c in chart_data["coefficient"]]
ax.barh(chart_data["feature"], chart_data["coefficient"], color=colors)
ax.axvline(0, color=text_color, linewidth=1)
ax.set_xlabel("Coeficiente (sube el riesgo →  /  ← baja el riesgo)", color=text_color)
ax.tick_params(colors=text_color)
for spine in ax.spines.values():
    spine.set_color(text_color)
fig.tight_layout()
st.pyplot(fig)
