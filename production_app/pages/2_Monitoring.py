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

_PAGE_DIR = Path(__file__).resolve().parent
_APP_DIR = _PAGE_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for _p in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.pipeline_utils import get_feature_parquet, get_raw_feature_columns, _TARGET_COL
from utils.model_utils import predict_batch_via_rest

st.set_page_config(
    page_title="Model Monitoring",
    page_icon="",
    layout="wide"
)

st.title("Model Monitoring Dashboard")
st.markdown(
    """
    Simulates **batch production monitorin** by taking 200 samples points,
    running the model against them, and computing metics across 20 sequential
    "time batches" (10 samples each). Use this to detect model drift,
    degradation, or systematic biases over time
    """
)

with st.sidebar:
    model_server_url = st.text_input(
        "MLFlow Model Server URL",
        value="http://localhost:5001",
        help="URL of mlflow models serve (POST /invocations)"
    )
    n_samples = st.slider(
        "Total_samples",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="How many points to sample from the feature parquet"
    )
    n_batches = st.slider(
        "Number of batches",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Splits total samples into this many sequential batches"
    )
    rolling_window = st.slider(
        "Rolling average window",
        min_value=2,
        max_value=10,
        value=3,
        help="Window size for moving average on time-series plots"
    )
    random_seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.markdown(
        """
        **Starting Servers:**
        bash
        mlflow models serve -m "models:/california-housing-best/latest" --port 5001 --no-conda
        """
    )

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return {"rmse": rmse, "mae": mae, "R2": r2, "mape": mape}
    
def _usd_formatter(x, _):
    """Matplotlib tick formatter for USD values."""
    if abs(x) >= 1_000:
        return f"${x/1_000:.0f}k"
    return f"${x:.0f}"

def _plot_metric_timeseries(
    ax: plt.Axes,
    batches: list[int],
    values: list[float],
    rolling: pd.Series,
    metric_name: str,
    color: str,
    is_usd: bool = False,
) -> None:
    """Plots raw batch metric + rolling average on a given Axes."""
    ax.plot(batches, values, "o-", color=color, alpha=0.5, linewidth=1.5,
            markersize=4, label="Per-batch")
    ax.plot(batches, rolling, "-", color=color, linewidth=2.5,
            label=f"Rolling avg (w={rolling})")
    ax.set_xlabel("Batch #", fontsize=9)
    ax.set_title(metric_name.upper(), fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    if is_usd:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))

    rolling_arr = rolling.values.astype(float)
    std = np.nanstd(values)
    ax.fill_between(
        batches,
        rolling_arr - std,
        rolling_arr + std,
        color=color,
        alpha=0.08,
        label="+/-1 std"
    )

# Run monitoring
run_btn = st.button("▶ Run Monitoring Analysis", type="primary", use_container_width=True)

if run_btn:
    # — Load feature parquet —
    with st.spinner("Loading feature data..."):
        try:
            df_features = get_feature_parquet()
        except Exception as exc:
            st.error(f"❌ Could not load feature parquet: {exc}")
            st.stop()

    # — Sample & split X / y —
    rng = np.random.default_rng(int(random_seed))
    sample_idx = rng.choice(len(df_features), size=min(n_samples, len(df_features)), replace=False)
    df_sample = df_features.iloc[sample_idx].reset_index(drop=True)

    y_true_all = df_sample[_TARGET_COL].values
    # Use raw (original) column names to select from the parquet, then rename
    feature_cols_raw = get_raw_feature_columns()  # e.g. 'op_<1H OCEAN'

    available_cols = [c for c in feature_cols_raw if c in df_sample.columns]
    missing_cols = [c for c in feature_cols_raw if c not in df_sample.columns]
    if missing_cols:
        st.warning(f"⚠️ Columns not found in parquet (skipped): {missing_cols}")

    X_sample = df_sample[available_cols].copy()

    # Apply XGBoost-safe rename (<, >, [, ]) — same as training
    rename_map = {
        c: c.replace("<", "lt_").replace("[", "(").replace("]", ")")
        for c in X_sample.columns
        if any(ch in c for ch in ("<", "[", "]"))
    }
    if rename_map:
        X_sample = X_sample.rename(columns=rename_map)

    with st.spinner(f"Sending {len(X_sample)} rows to model server....."):
        try:
            y_pred_all = predict_batch_via_rest(X_sample, model_server_url)
            y_pred_all = np.array(y_pred_all)
        except Exception as e:
            st.error(
                f"X Model server error: {e}\n\n"
                f"Make sure the model server is running at {model_server_url}"
            )
            st.stop()

    batch_size = len(X_sample)
    reminder = len(X_sample)

    batch_metrics: list[dict] = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size + (1 if i < reminder else 0)
        if end > len(X_sample):
            break
        batch_y_true = y_true_all[start:end]
        batch_y_pred = y_pred_all[start:end]
        m = _compute_metrics(batch_y_true, batch_y_pred)
        m["batch"] = i + 1
        batch_metrics.append(m)

    df_metrics = pd.DataFrame(batch_metrics).set_index("batch")

    df_rolling = df_metrics.rolling(window=rolling_window, min_periods=1).mean()

    overall = _compute_metrics(y_true_all[: len(y_pred_all)], y_pred_all)

    st.divider()
    st.subheader("Overall Metrics (all 200 samples)")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("RMSE", f"{overall['rmse']:.0f}")
    kpi2.metric("MAE", f"{overall['mae']:.0f}")
    kpi3.metric("R2", f"{overall['r2']:.4f}")
    kpi4.metric("MAPE", f"{overall['mape']:.2f}%")

    # SECTION 2 – Time-series plots
    st.divider()
    st.subheader("📈 Batch Metrics Over Time")
    st.caption(
        f"{n_batches} batches × {batch_size} samples each | "
        f"Rolling average window = {rolling_window} batches"
    )

    batches = df_metrics.index.tolist()
    palette = {"rmse": "#e74c3c", "mae": "#e67e22", "r2": "#27ae60", "mape": "#2980b9"}

    fig_ts, axes_ts = plt.subplots(2, 2, figsize=(14, 7), tight_layout=True)
    fig_ts.patch.set_facecolor("#0E1117")

    metric_pairs = [
        ("rmse", axes_ts[0, 0], True),
        ("mae", axes_ts[0, 1], True),
        ("r2", axes_ts[1, 0], False),
        ("mape", axes_ts[1, 1], False),
    ]

    for metric, ax, is_usd in metric_pairs:
        ax.set_facecolor("#1a1a2e")
        _plot_metric_timeseries(
            ax=ax,
            batches=batches,
            values=df_metrics[metric].tolist(),
            rolling=df_rolling[metric],
            metric_name=metric,
            color=palette[metric],
            is_usd=is_usd,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    st.pyplot(fig_ts, use_container_width=True)
    plt.close(fig_ts)

    # SECTION 3 - Residuals histogram
    #
    st.divider()
    st.subheader("📊 Residuals Distribution")
    st.caption(
        "Residuals = y_true - y_pred"
        "A well-calibrated model shows residuals centred at 0 with no heavy skew."
    )

    residuals = y_true_all[: len(y_pred_all)] - y_pred_all

    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(14, 4), tight_layout=True)
    fig_hist.patch.set_facecolor("#0e1117")

    # Left: histogram + KDE
    ax_hist = axes_hist[0]
    ax_hist.set_facecolor("#1a1a2e")
    sns.histplot(
        residuals,
        bins=30,
        kde=True,
        color="#7e4c3c",
        ax=ax_hist,
        line_kws={"linewidth": 2},
    )

    ax_hist.axvline(0, color="white", linestyle="--", linewidth=1.2, label="Zero error")
    ax_hist.axvline(float(np.mean(residuals)), color="#f1c40f",
                    linestyle="--", linewidth=1.5,
                    label=f"Mean residual: ${np.mean(residuals):,.0f}")
    ax_hist.set_title("Residuals Histogram + KDE", color="white", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Residual (USD)", color="white")
    ax_hist.set_ylabel("Count", color="white")
    ax_hist.tick_params(colors="white")
    for spine in ax_hist.spines.values():
        spine.set_edgecolor("#333")
    ax_hist.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_hist.xaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))

    # Right: scatter true vs predicted
    ax_scatter = axes_hist[1]
    ax_scatter.set_facecolor("#1a1a2e")
    ax_scatter.scatter(
        y_true_all[: len(y_pred_all)],
        y_pred_all,
        alpha=0.35,
        s=12,
        color="#3498db",
        edgecolors="none"
    )

    min_val = min(y_true_all.min(), y_pred_all.min())
    max_val = max(y_true_all.max(), y_pred_all.max())
    ax_scatter.plot(
        [min_val, max_val], [min_val, max_val],
        "r--", linewidth=1.5, label="Perfect prediction"
    )

    ax_scatter.set_title("Actual vs Predicted", color="white", fontsize=11, fontweight="bold")
    ax_scatter.set_xlabel("Actual (USD)", color="white")
    ax_scatter.set_ylabel("Predicted (USD)", color="white")
    ax_scatter.tick_params(colors="white")
    for spine in ax_scatter.spines.values():
        spine.set_edgecolor("#333")
    ax_scatter.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_scatter.xaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))
    ax_scatter.yaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))

    st.pyplot(fig_hist, use_container_width=True)
    plt.close(fig_hist)

    # SECTION 4 – Raw batch metrics table
    with st.expander("📋 Raw batch metrics table"):
        display_df = df_metrics.copy()
        display_df["rmse"] = display_df["rmse"].map("${:.0f}".format)
        display_df["mae"] = display_df["mae"].map("${:.0f}".format)
        display_df["r2"] = display_df["r2"].map("{:.4f}".format)
        display_df["mape"] = display_df["mape"].map("{:.2f}%".format)
        display_df.columns = ["RMSE", "MAE", "R2", "MAPE"]
        st.dataframe(display_df, use_container_width=True)



