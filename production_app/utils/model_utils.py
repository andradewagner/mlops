from __future__ import annotations

import math
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _nan_safe_rows(df: pd.DataFrame) -> list[list]:
    """Converte NaN → None para envio ao MLflow Model Server."""
    rows = df.values.tolist()
    return [
        [None if (isinstance(v, float) and math.isnan(v)) else v for v in row]
        for row in rows
    ]

# ---------------------------------------------------------
# CONFIGURAÇÃO DO SEU PROJETO KING COUNTY
# ---------------------------------------------------------

# Nome do modelo registrado no MLflow Model Registry
_MODEL_NAME = "kingcounty_price_model"

# Número de folds usados no CV do seu pipeline
_N_CV_FOLDS = 3

# Z-score para intervalo de confiança 95%
_Z_95 = 1.96

# ---------------------------------------------------------
# Predição via MLflow REST
# ---------------------------------------------------------

def predict_via_rest(
    features_df: pd.DataFrame,
    model_server_url: str
) -> float:
    """Envia uma única linha de features ao MLflow Model Server."""
    payload = {
        "dataframe_split": {
            "columns": features_df.columns.tolist(),
            "data": _nan_safe_rows(features_df)
        }
    }

    url = f"{model_server_url.rstrip('/')}/invocations"
    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    resp.raise_for_status()

    result = resp.json()

    if isinstance(result, dict) and "predictions" in result:
        return float(result["predictions"][0])

    if isinstance(result, list):
        return float(result[0])

    raise ValueError(f"Unexpected response format from model server: {result}")

def predict_batch_via_rest(
    features_df: pd.DataFrame,
    model_server_url: str
) -> list[float]:
    """Envia várias linhas de features ao MLflow Model Server."""
    payload = {
        "dataframe_split": {
            "columns": features_df.columns.tolist(),
            "data": _nan_safe_rows(features_df)
        }
    }

    url = f"{model_server_url.rstrip('/')}/invocations"
    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    resp.raise_for_status()

    result = resp.json()

    if isinstance(result, dict) and "predictions" in result:
        return [float(v) for v in result["predictions"]]

    if isinstance(result, list):
        return [float(v) for v in result]

    raise ValueError(f"Unexpected response format from model server: {result}")

# ---------------------------------------------------------
# Busca parâmetros do modelo no MLflow Tracking Server
# ---------------------------------------------------------

def get_model_ci_params(tracking_server_url: str) -> dict[str, float]:
    """
    Obtém:
    - cv_rmse_std
    - holdout_rmse
    - run_id
    - model_version

    A partir do MLflow Tracking Server.
    """
    base = tracking_server_url.rstrip("/")

    # Busca versões registradas
    versions_resp = requests.get(
        f"{base}/api/2.0/mlflow/registered-models/get-latest-versions",
        params={"name": _MODEL_NAME},
        timeout=15
    )
    versions_resp.raise_for_status()
    versions = versions_resp.json().get("model_versions", [])

    if not versions:
        raise ValueError(f"No registered versions found for model '{_MODEL_NAME}'.")

    # Preferência: versão em Production
    production_versions = [v for v in versions if v.get("current_stage") == "Production"]
    best_version = production_versions[0] if production_versions else max(
        versions, key=lambda v: int(v.get("version", 0))
    )

    run_id = best_version["run_id"]
    model_version = best_version["version"]

    # Busca métricas do run
    run_resp = requests.get(
        f"{base}/api/2.0/mlflow/runs/get",
        params={"run_id": run_id},
        timeout=15
    )
    run_resp.raise_for_status()

    run_data = run_resp.json().get("run", {}).get("data", {})
    metrics = {m["key"]: m["value"] for m in run_data.get("metrics", [])}

    cv_rmse_std = float(metrics.get("cv_rmse_std", 0.0))
    holdout_rmse = float(metrics.get("holdout_rmse", metrics.get("rmse", 0.0)))

    return {
        "cv_rmse_std": cv_rmse_std,
        "holdout_rmse": holdout_rmse,
        "run_id": run_id,
        "model_version": model_version
    }

# ---------------------------------------------------------
# Intervalo de confiança
# ---------------------------------------------------------

def compute_confidence_internal(
    y_hat: float,
    cv_rmse_std: float,
    n_folds: int = _N_CV_FOLDS,
    z: float = _Z_95
) -> tuple[float, float]:
    """Calcula intervalo de confiança baseado no CV."""
    se = cv_rmse_std / math.sqrt(n_folds)
    margin = z * se
    return (y_hat - margin, y_hat + margin)
