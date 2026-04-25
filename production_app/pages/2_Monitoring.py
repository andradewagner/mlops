import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

# -----------------------------
# PATH SETUP
# -----------------------------
_PAGE_DIR = Path(__file__).resolve().parent
_APP_DIR = _PAGE_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for _p in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -----------------------------
# IMPORTS DO PROJETO
# -----------------------------
from utils.pipeline_utils import (
    get_feature_parquet,
    get_raw_feature_columns,
    _TARGET_COL
)
from utils.model_utils import predict_batch_via_rest

ROOT = Path(__file__).resolve().parent.parent.parent
raw_path = ROOT / "data/raw/kc_house_data.csv"
out_path = ROOT / "data/processed/kingcounty_processed.parquet"

print("Carregando raw...")
df = pd.read_csv(raw_path)

# Renomear colunas para manter consistência
df = df.rename(columns={
    "sqft_living15": "sqft_living15",
    "sqft_lot15": "sqft_lot15"
})

# Feature engineering (mesmo do pipeline_utils)
df["built_before_1950"] = (df["yr_built"] < 1950).astype(int)
df["bath_per_bed"] = df["bathrooms"] / df["bedrooms"].replace(0, np.nan)
df["sqft_living_per_room"] = df["sqft_living"] / df["bedrooms"].replace(0, np.nan)
df["lot_per_sqft"] = df["sqft_lot"] / df["sqft_living"].replace(0, np.nan)

df["nearest_city_distance"] = np.sqrt(
    (df["lat"] - 47.6062)**2 +
    (df["long"] + 122.3321)**2
)

df["sqft_living_squared"] = df["sqft_living"] ** 2
df["bath_x_bed"] = df["bathrooms"] * df["bedrooms"]

df = df.fillna(df.median(numeric_only=True))

print("Salvando parquet final...")
df.to_parquet(out_path, index=False)

print("OK! Arquivo criado em:", out_path)

# -----------------------------
# CONFIG STREAMLIT
# -----------------------------
st.set_page_config(
    page_title="King County Model Monitoring",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 King County — Model Monitoring Dashboard")
st.markdown(
    """
    Este painel simula **monitoramento em produção**, dividindo amostras em lotes
    sequenciais e avaliando o desempenho do modelo ao longo do tempo.
    Use isso para detectar **drift**, **degradação** ou **viés sistemático**.
    """
)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    model_server_url = st.text_input(
        "MLFlow Model Server URL",
        value="http://localhost:5001",
        help="URL do MLflow models serve (POST /invocations)"
    )

    n_samples = st.slider(
        "Total samples",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    n_batches = st.slider(
        "Number of batches",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )

    rolling_window = st.slider(
        "Rolling average window",
        min_value=2,
        max_value=10,
        value=3
    )

    random_seed = st.number_input("Random seed", value=42, step=1)

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

# -----------------------------
# MÉTRICAS
# -----------------------------
def _compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }

def _usd_formatter(x, _):
    if abs(x) >= 1_000:
        return f"${x/1_000:.0f}k"
    return f"${x:.0f}"

# -----------------------------
# PLOT TIMESERIES
# -----------------------------
def _plot_metric_timeseries(ax, batches, values, rolling, metric_name, color, is_usd=False):
    ax.plot(batches, values, "o-", color=color, alpha=0.5, linewidth=1.5,
            markersize=4, label="Per-batch")
    ax.plot(batches, rolling, "-", color=color, linewidth=2.5,
            label=f"Rolling avg")

    ax.set_xlabel("Batch #", fontsize=9)
    ax.set_title(metric_name.upper(), fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    if is_usd:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))

    std = np.nanstd(values)
    ax.fill_between(
        batches,
        rolling - std,
        rolling + std,
        color=color,
        alpha=0.08,
        label="+/-1 std"
    )

# -----------------------------
# BOTÃO PRINCIPAL
# -----------------------------
run_btn = st.button("▶ Run Monitoring Analysis", type="primary", use_container_width=True)

if run_btn:

    # -----------------------------
    # LOAD FEATURES
    # -----------------------------
    with st.spinner("Loading feature data..."):
        try:
            df_features = get_feature_parquet()
        except Exception as exc:
            st.error(f"❌ Could not load feature parquet: {exc}")
            st.stop()

    # -----------------------------
    # SAMPLE
    # -----------------------------
    rng = np.random.default_rng(int(random_seed))
    sample_idx = rng.choice(len(df_features), size=min(n_samples, len(df_features)), replace=False)
    df_sample = df_features.iloc[sample_idx].reset_index(drop=True)

    y_true_all = df_sample[_TARGET_COL].values

    # Seleciona apenas features originais (antes do pipeline)
    feature_cols_raw = get_raw_feature_columns()
    available_cols = [c for c in feature_cols_raw if c in df_sample.columns]

    X_sample = df_sample[available_cols].copy()
    print("COLUMNS SENT TO MODEL:", X_sample.columns.tolist())

    # -----------------------------
    # PREDIÇÃO VIA REST
    # -----------------------------
    with st.spinner(f"Sending {len(X_sample)} rows to model server..."):
        try:
            y_pred_all = predict_batch_via_rest(X_sample, model_server_url)
            y_pred_all = np.array(y_pred_all)
        except Exception as e:
            st.error(f"❌ Model server error: {e}")
            st.stop()

    # -----------------------------
    # BATCH SPLIT
    # -----------------------------
    batch_size = len(X_sample) // n_batches
    batch_metrics = []

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if end > len(X_sample):
            break

        batch_y_true = y_true_all[start:end]
        batch_y_pred = y_pred_all[start:end]

        m = _compute_metrics(batch_y_true, batch_y_pred)
        m["batch"] = i + 1
        batch_metrics.append(m)

    df_metrics = pd.DataFrame(batch_metrics).set_index("batch")
    df_rolling = df_metrics.rolling(window=rolling_window, min_periods=1).mean()

    # -----------------------------
    # OVERALL METRICS
    # -----------------------------
    overall = _compute_metrics(y_true_all[:len(y_pred_all)], y_pred_all)

    st.divider()
    st.subheader("Overall Metrics")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("RMSE", f"{overall['rmse']:.0f}")
    k2.metric("MAE", f"{overall['mae']:.0f}")
    k3.metric("R2", f"{overall['r2']:.4f}")
    k4.metric("MAPE", f"{overall['mape']:.2f}%")

    # -----------------------------
    # TIME SERIES PLOTS
    # -----------------------------
    st.divider()
    st.subheader("📈 Batch Metrics Over Time")

    batches = df_metrics.index.tolist()
    palette = {"rmse": "#e74c3c", "mae": "#e67e22", "r2": "#27ae60", "mape": "#2980b9"}

    fig_ts, axes_ts = plt.subplots(2, 2, figsize=(14, 7), tight_layout=True)

    metric_pairs = [
        ("rmse", axes_ts[0, 0], True),
        ("mae", axes_ts[0, 1], True),
        ("r2", axes_ts[1, 0], False),
        ("mape", axes_ts[1, 1], False),
    ]

    for metric, ax, is_usd in metric_pairs:
        _plot_metric_timeseries(
            ax=ax,
            batches=batches,
            values=df_metrics[metric].tolist(),
            rolling=df_rolling[metric],
            metric_name=metric,
            color=palette[metric],
            is_usd=is_usd,
        )

    st.pyplot(fig_ts)
    plt.close(fig_ts)

    # -----------------------------
    # RESIDUALS
    # -----------------------------
    st.divider()
    st.subheader("📊 Residuals Distribution")

    residuals = y_true_all[:len(y_pred_all)] - y_pred_all

    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(14, 4), tight_layout=True)

    # Histograma
    sns.histplot(residuals, bins=30, kde=True, ax=axes_hist[0], color="#7e4c3c")
    axes_hist[0].axvline(0, color="black", linestyle="--")
    axes_hist[0].set_title("Residuals Histogram")

    # Scatter
    axes_hist[1].scatter(y_true_all, y_pred_all, alpha=0.3)
    min_val = min(y_true_all.min(), y_pred_all.min())
    max_val = max(y_true_all.max(), y_pred_all.max())
    axes_hist[1].plot([min_val, max_val], [min_val, max_val], "r--")
    axes_hist[1].set_title("Actual vs Predicted")

    st.pyplot(fig_hist)
    plt.close(fig_hist)

    # -----------------------------
    # RAW TABLE
    # -----------------------------
    with st.expander("📋 Raw batch metrics table"):
        display_df = df_metrics.copy()
        display_df["rmse"] = display_df["rmse"].map("${:.0f}".format)
        display_df["mae"] = display_df["mae"].map("${:.0f}".format)
        display_df["r2"] = display_df["r2"].map("{:.4f}".format)
        display_df["mape"] = display_df["mape"].map("{:.2f}%".format)
        display_df.columns = ["RMSE", "MAE", "R2", "MAPE"]
        st.dataframe(display_df, use_container_width=True)
