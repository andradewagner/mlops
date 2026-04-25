from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_APP_DIR = _HERE.parent
_PROJECT_ROOT = _APP_DIR.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATA_DIR = _PROJECT_ROOT / "data"

for _p in [str(_PROJECT_ROOT), str(_CONFIG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------
# LOAD CONFIG
# ---------------------------------------------------------
def _load_preprocessing_config() -> dict[str, Any]:
    path = _CONFIG_DIR / "preprocessing.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

_PREP_CFG = _load_preprocessing_config()

_TARGET_COL: str = _PREP_CFG["feature_selection"]["target"]
_FEATURES_TO_KEEP: list[str] = [
    c for c in _PREP_CFG["feature_selection"]["features_to_keep"]
    if c != _TARGET_COL
]

# Parquet correto do King County (contém features + price)
_FEATURES_PARQUET = _DATA_DIR / "processed" / "kingcounty_processed.parquet"

# ---------------------------------------------------------
# PREPROCESSAMENTO PARA O APP
# ---------------------------------------------------------

def preprocessed_raw_inputs(raw: dict[str, Any]) -> pd.DataFrame:
    """
    Converte inputs crus do usuário em features prontas para o modelo,
    aplicando exatamente o mesmo pipeline usado no treinamento.
    """
    print("RAW RECEBIDO:", raw)

    df = pd.DataFrame([raw])

    # Garantir tipos corretos
    df["bedrooms"] = df["bedrooms"].astype(float)
    df["bathrooms"] = df["bathrooms"].astype(float)
    df["sqft_living"] = df["sqft_living"].astype(float)
    df["sqft_lot"] = df["sqft_lot"].astype(float)
    df["sqft_above"] = df["sqft_above"].astype(float)
    df["sqft_basement"] = df["sqft_basement"].astype(float)
    df["floors"] = df["floors"].astype(float)
    df["waterfront"] = df["waterfront"].astype(float)
    df["view"] = df["view"].astype(float)
    df["condition"] = df["condition"].astype(float)
    df["grade"] = df["grade"].astype(float)
    df["yr_built"] = df["yr_built"].astype(float)
    df["yr_renovated"] = df["yr_renovated"].astype(float)
    df["lat"] = df["lat"].astype(float)
    df["long"] = df["long"].astype(float)

    # ---------------------------------------------------------
    # Feature Engineering (EXATAMENTE como no treinamento)
    # ---------------------------------------------------------

    df["built_before_1950"] = (df["yr_built"] < 1950).astype(int)
    df["bath_per_bed"] = df["bathrooms"] / df["bedrooms"].replace(0, np.nan)
    df["sqft_living_per_room"] = df["sqft_living"] / df["bedrooms"].replace(0, np.nan)
    df["lot_per_sqft"] = df["sqft_lot"] / df["sqft_living"].replace(0, np.nan)

    # Distância para a cidade mais próxima (Seattle)
    df["nearest_city_distance"] = np.sqrt(
        (df["lat"] - 47.6062) ** 2 +
        (df["long"] + 122.3321) ** 2
    )

    df["sqft_living_squared"] = df["sqft_living"] ** 2
    df["bath_x_bed"] = df["bathrooms"] * df["bedrooms"]

    # Imputação simples
    df = df.fillna(df.median(numeric_only=True))

    # ---------------------------------------------------------
    # Selecionar features finais
    # ---------------------------------------------------------
    df = df.reindex(columns=_FEATURES_TO_KEEP, fill_value=0)

    return df

# ---------------------------------------------------------
# FUNÇÕES AUXILIARES PARA MONITORAMENTO
# ---------------------------------------------------------

def get_feature_parquet() -> pd.DataFrame:
    """Carrega o parquet de features + price."""
    return pd.read_parquet(_FEATURES_PARQUET)

def get_feature_columns() -> list[str]:
    """Retorna as colunas finais usadas pelo modelo."""
    return list(_FEATURES_TO_KEEP)

def get_raw_feature_columns() -> list[str]:
    """Compatível com o Monitoring."""
    return list(_FEATURES_TO_KEEP)
