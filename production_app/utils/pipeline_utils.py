from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_HERE = Path(__file__).resolve().parent
_APP_DIR = _HERE.parent
_PROJECT_ROOT = _APP_DIR.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATA_DIR = _PROJECT_ROOT / "data"

for _p in [str(_PROJECT_ROOT), str(_CONFIG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.preprocessing import (
    CustomImputer,
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    GeoDistanceTransformer,
    PolynomialFeatureTransformer,
    OceanProximityEncoder,
    FeatureSelector
)

from src.utils.logger import get_logger
logger = get_logger("PipelineUtils")

def _load_preprocessing_config() -> dict[str, Any]:
    path = _CONFIG_DIR / "preprocessing.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
    

_PREP_CFG: dict[str, Any] = _load_preprocessing_config()

_TARGET_COL: str = _PREP_CFG.get("feature_selection", {}).get("target", "median_house_value")
_FEATURES_TO_KEEP: list[str] = [
    c for c in _PREP_CFG.get("feature_selection", {}).get("features_to_keep", [])
    if c != _TARGET_COL
]

_PROCESSED_PARQUET = _DATA_DIR / "processed" / "house_price_predictions.parquet"
_FEATURES_PARQUET = _DATA_DIR / "features" / "house_price_features.parquet"


def _build_fitted_imputer() -> CustomImputer:
    imp_cfg = _PREP_CFG.get("imputation", [{}])[0]
    imputer = CustomImputer(
        group_col=imp_cfg.get("group_by", "ocean_proximity"),
        target_col=imp_cfg.get("column", "total_bedrooms")
    )
    df_train = pd.read_parquet(_PROCESSED_PARQUET)
    imputer.fit(df_train)
    return imputer


_FITTED_IMPUTER: CustomImputer = _build_fitted_imputer()


def preprocessed_raw_inputs(raw: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    df = _FITTED_IMPUTER.transform(df)

    flags_cfg = _PREP_CFG.get("binary_flags", [])
    df = BinaryFlagTransformer(flags=flags_cfg).fit_transform(df)

    ratios_cfg = _PREP_CFG.get("ratio_features", [])
    df = RatioFeatureTransformer(ratios=ratios_cfg).fit_transform(df)

    log_cols = _PREP_CFG.get("log_transformer", {}).get("columns", [])
    df = LogTransformer(columns=log_cols).fit_transform(df)

    get_cfg = _PREP_CFG.get("geo_distance", {})
    df = GeoDistanceTransformer(geo_config=get_cfg, logger=logger).fit_transform(df)

    poly_cfg = _PREP_CFG.get("polynomial_features", [])
    df = PolynomialFeatureTransformer(pol_config=poly_cfg, logger=logger).fit_transform(df)

    enc_cfg = _PREP_CFG.get("categorical_encoding", {})
    print("ENC CFG:", enc_cfg)
    print("DF COLS BEFORE ENCODER:", df.columns.tolist())
    df = OceanProximityEncoder(ope_config=enc_cfg, logger=logger).fit_transform(df)
    print("DF COLS AFTER ENCODER:", df.columns.tolist())

    df = FeatureSelector(features=_FEATURES_TO_KEEP, logger=logger).fit_transform(df)

    df = df.reindex(columns=_FEATURES_TO_KEEP, fill_value=0)

    rename_map = {
        c: c.replace("<", "lt_").replace("[", "(").replace("]", ")")
        for c in df.columns
        if any(ch in c for ch in ("<", "[", "]"))
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def get_feature_parquet() -> pd.DataFrame:
    return pd.read_parquet(_FEATURES_PARQUET)

def get_feature_columns() -> list[str]:
    return [
        c.replace("<", "lt_").replace("[", "(").replace("]", ")")
        for c in _FEATURES_TO_KEEP
    ]


def get_raw_feature_columns() -> list[str]:
    return list(_FEATURES_TO_KEEP)