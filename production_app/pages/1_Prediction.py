import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

_PAGE_DIR = Path(__file__).resolve().parent
_APP_DIR = _PAGE_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for _p in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.pipeline_utils import preprocessed_raw_inputs
from utils.model_utils import (
    predict_via_rest,
    get_model_ci_params,
    predict_batch_via_rest,
    compute_confidence_internal
)

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 California Housing Price Prediction")
st.markdown(
    """Enter the original block-group features below"""
)

with st.sidebar:
    st.header("Server Settings")
    model_server_url = st.text_input(
        "MLFlow Model Server URL",
        value="http://localhost:5001",
        help="URL of mlflow models serve (POST/Invocations)"
    )
    tracking_server_url = st.text_input(
        "MLFlow Tracking Server Url",
        value="http://localhost:5000",
        help="URL of mlflow server (for CI parameters)"
    )

    st.divider()
    st.markdown(
    """
    Server:
    # bash
    # Tracking Server
    mlflow server --backend-store-uri mlruns --port 5000

    Model Server
    mlflow models serve -m "models:/california-housing-best/latest" --port 5001 --no-conda
    """
    )

st.subheader("Block-group features")
st.caption(
    "All derived features (log transforms, geo distance, ratios, polynomial "
    "terms, encoding) are computed automatically by the preprocessing pipeline"
)

col1, col2, col3 = st.columns(3)

with col1:
    median_income = st.number_input(
        "Median income (tens of thousands USD)",
        min_value=0.1,
        max_value=20.0,
        value=4.5,
        step=0.1,
        help="E.g. 4.5 means $45,000 median household income"
    )
    housing_median_age = st.number_input(
        "Housing median age (years)",
        min_value=1,
        max_value=52,
        value=25,
        step=1,
        help="Median age of houses in the block. Max = 52 (censored)."
    )
    ocean_proximity = st.selectbox(
        "Ocean proximity",
        options=["ISLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN", "INLAND"],
        index=0,
        help="Categorical location relative to ocean"
    )

with col2:
    total_rooms = st.number_input(
        "Total rooms",
        min_value=1,
        max_value=40000,
        value=2500,
        step=50,
        help="Total number of rooms in the block"
    )
    total_bedrooms = st.number_input(
        "Total bedrooms",
        min_value=1,
        max_value=7000,
        value=500,
        step=10,
        help="Total bedrooms. Leave as 0 to let the pipeline imput it."
    )
    households = st.number_input(
        "Households",
        min_value=1,
        max_value=7000,
        value=450,
        step=10,
        help="Number of households in the block"
    )

with col3:
    population = st.number_input(
        "Population",
        min_value=1,
        max_value=40000,
        value=1200,
        step=50,
        help="Total population of the block"
    )
    latitude = st.number_input(
        "Latitude",
        min_value=32.0,
        max_value=42.0,
        value=37.75,
        step=0.01,
        format="%.4f",
        help="Block-group latitude (California: ~32ºN 42ºN)"
    )
    longitude = st.number_input(
        "Longitude",
        min_value=-125.0,
        max_value=-114.0,
        value=-122.42,
        step=0.01,
        format="%.4f",
        help="Block-group longitude (California: ~-114º -125º)"
    )

st.divider()
predict_btn = st.button("Calculate Housing Price", type="primary", use_container_width=True)

if predict_btn:
    raw_inputs = {
        "median_income": median_income,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": float(total_bedrooms) if total_bedrooms > 0 else float("nan"),
        "population": population,
        "households": households,
        "latitude": latitude,
        "longitude": longitude,
        "ocean_proximity": ocean_proximity
    }

    with st.spinner("Running preprocessing pipeline...."):
        try:
            features_df = preprocessed_raw_inputs(raw_inputs)
            pipeline_ok = True
        except Exception as e:
            st.error(f"Ⓧ Preprocessing failed: {e}")
            pipeline_ok = False

    if pipeline_ok:
        with st.spinner("California MLFlow model server...."):
            try:
                y_hat = predict_via_rest(features_df, model_server_url)
                prediction_ok = True
            except Exception as e:
                st.error(
                    f"Ⓧ Model server error: {e}\n\n"
                    "Make sure the MLFlow model server is running at "
                    f"{model_server_url}"
                )
                prediction_ok=False

    if pipeline_ok and prediction_ok:
        with st.spinner("Fetching CI parameters from tracking server..."):
            try:
                ci_params = get_model_ci_params(tracking_server_url)
                lower, upper = compute_confidence_internal(
                    y_hat=y_hat,
                    cv_rmse_std=ci_params["cv_rmse_std"]
                )
                ci_ok = True
            except Exception as e:
                st.warning(
                    f"Could not fetch CI from tracking server: {e}\n\n"
                    "Prediction is still shown below without confidence interval"
                )
                ci_ok = False

    if pipeline_ok and prediction_ok:
        st.divider()
        st.subheader("Prediction Results")

        res_col1, res_col2, res_col3 = st.columns([2,1,1])

        with res_col1:
            st.metric(
                label="Prediction Median House value",
                value=f"${y_hat:,.0f}",
                help="Point estimate from the MLFlow model server"
            )
            if ci_ok:
                st.markdown(
                    f"""
                    **95% Confidence Interval:**
                    &nbsp:&nbsp; ${lower:,.0f} &nbsp; &nbsp; ${upper:,.0f}

                    *SE = cv\\_rmse\\_std / raiz(5) =
                    {ci_params["cv_rmse_std"]:,.0f} / raiz(5) =
                    {ci_params["cv_rmse_std"] / (5 ** 0.5):,.0f}*
                    """
                )

                ci_range = upper - lower
                lower_pct = max(0, (lower / upper) * 100 - 5)
                st.progress(
                    min(int((y_hat - lower) / (ci_range + 1e-9) * 100), 100),
                    text=f"Prediction within interval | Width: ${ci_range:,.0f}"
                )

        with res_col2:
            if ci_ok:
                st.metric("Lower bound (95% CI)", f"${lower:,.0f}")
                st.metric("Upper bound (95% CI)", f"${upper:,.0f}")


        with res_col3:
            if ci_ok:
                st.metric("Holdout RMSE", f"${ci_params['holdout_rmse']:,.0f}")
                st.metric("cv_rmse_std", f"${ci_params['cv_rmse_std']:,.0f}")
                st.caption(f"Model version: {ci_params['model_version']}")
                st.caption(f"Run ID: {ci_params['run_id'][:8]}...")


        with st.expander("Inspect engineered features"):
            st.caption(
                f"{len(features_df.columns)} features sent to model server"
                f"(original 9 -> {len(features_df.columns)} after full pipeline)"
            )
            st.dataframe(
                features_df.T.rename(columns={0: "value"}).style.format("{:.4f}"),
                use_container_width=True,
                height=600
            )