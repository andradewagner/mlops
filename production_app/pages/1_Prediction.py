import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Ajuste de paths
_PAGE_DIR = Path(__file__).resolve().parent
_APP_DIR = _PAGE_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for _p in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Importa utilidades do seu projeto
from utils.pipeline_utils import preprocessed_raw_inputs
from utils.model_utils import (
    predict_via_rest,
    get_model_ci_params,
    compute_confidence_internal
)

# Configuração da página
st.set_page_config(
    page_title="King County House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 King County House Price Prediction")
st.markdown("Insira os atributos principais da casa para estimar o preço.")

# Sidebar
with st.sidebar:
    st.header("Server Settings")

    model_server_url = st.text_input(
        "MLFlow Model Server URL",
        value="http://localhost:5001",
        help="URL do MLflow models serve (POST /invocations)"
    )

    tracking_server_url = st.text_input(
        "MLFlow Tracking Server URL",
        value="http://localhost:5000",
        help="URL do MLflow tracking server"
    )

    st.divider()
    st.markdown(
        """
        **Servidores necessários:**

        **Tracking Server**
        ```
        mlflow server --backend-store-uri sqlite:////home/wagner/MLOps/mlflow.db --port 5000
        ```

        **Model Server**
        ```
        mlflow models serve -m "models:/kingcounty_price_model/1" -p 5001 --no-conda \
            --tracking-uri sqlite:////home/wagner/MLOps/mlflow.db
        ```
        """
    )

# Inputs principais
st.subheader("House Features")

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=8.0, value=2.0, step=0.25)
    sqft_lot = st.number_input("Lot Size (sqft_lot)", min_value=0, max_value=200000, value=5000, step=50)
    floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)

with col2:
    sqft_living = st.number_input("Living Area (sqft_living)", min_value=200, max_value=10000, value=800, step=50)
    sqft_above = st.number_input("sqft_above", min_value=0, max_value=10000, value=500, step=50)
    sqft_basement = st.number_input("sqft_basement", min_value=0, max_value=5000, value=300, step=50)
    waterfront = st.number_input("Waterfront (0/1)", min_value=0, max_value=1, value=0, step=1)

with col3:
    view = st.number_input("View (0–4)", min_value=0, max_value=4, value=0, step=1)
    condition = st.number_input("Condition (1–5)", min_value=1, max_value=5, value=3, step=1)
    grade = st.number_input("Grade (1–13)", min_value=1, max_value=13, value=7, step=1)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=1980, step=1)

yr_renovated = st.number_input("Year Renovated (0 = nunca)", min_value=0, max_value=2025, value=0, step=1)

lat = st.number_input("Latitude", min_value=47.0, max_value=48.0, value=47.62, step=0.0001, format="%.5f")
long = st.number_input("Longitude", min_value=-123.0, max_value=-121.0, value=-122.33, step=0.0001, format="%.5f")

st.divider()

predict_btn = st.button("Calcular Preço da Casa", type="primary", use_container_width=True)

# Execução
if predict_btn:
    raw_inputs = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "lat": lat,
        "long": long
    }

    # Preprocessamento
    with st.spinner("Executando pipeline de preprocessamento..."):
        try:
            features_df = preprocessed_raw_inputs(raw_inputs)
            print("COLUMNS SENT TO MODEL:", features_df.columns.tolist())
            pipeline_ok = True
        except Exception as e:
            st.error(f"Erro no preprocessamento: {e}")
            pipeline_ok = False

    # Predição
    if pipeline_ok:
        with st.spinner("Consultando o MLflow Model Server..."):
            try:
                y_hat = predict_via_rest(features_df, model_server_url)
                prediction_ok = True
            except Exception as e:
                st.error(
                    f"Erro no servidor de modelo: {e}\n"
                    f"Verifique se o servidor está ativo em {model_server_url}"
                )
                prediction_ok = False

    # Intervalo de confiança
    if pipeline_ok and prediction_ok:
        with st.spinner("Buscando parâmetros de CI no Tracking Server..."):
            try:
                ci_params = get_model_ci_params(tracking_server_url)
                lower, upper = compute_confidence_internal(
                    y_hat=y_hat,
                    cv_rmse_std=ci_params["cv_rmse_std"]
                )
                ci_ok = True
            except Exception as e:
                st.warning(
                    f"Não foi possível obter intervalo de confiança: {e}"
                )
                ci_ok = False

    # Exibir resultados
    if pipeline_ok and prediction_ok:
        st.divider()
        st.subheader("Resultado da Predição")

        colA, colB = st.columns([2,1])

        with colA:
            st.metric(
                label="Preço Previsto",
                value=f"${y_hat:,.0f}"
            )

            if ci_ok:
                st.markdown(
                    f"**95% CI:** ${lower:,.0f} — ${upper:,.0f}"
                )

        with colB:
            if ci_ok:
                st.metric("Holdout RMSE", f"${ci_params['holdout_rmse']:,.0f}")
                st.metric("cv_rmse_std", f"${ci_params['cv_rmse_std']:,.0f}")
                st.caption(f"Model version: {ci_params['model_version']}")
                st.caption(f"Run ID: {ci_params['run_id'][:8]}...")

        with st.expander("Ver features derivadas"):
            st.dataframe(features_df.T.rename(columns={0: "value"}))
