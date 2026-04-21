import numpy as np
import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            group_col: str,
            target_col: str,
            logger: Any = None
    ) -> None:
        self.group_col = group_col
        self.target_col = target_col
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)
        else:
            print(msg, *args)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CustomImputer":
        self._log("Fitting imputer for column '%s' grouped by '%s'", self.target_col, self.group_col)
        
        missing_cols = [c for c in [self.group_col, self.target_col] if c not in X.columns]
        if missing_cols:
            raise KeyError(f"Columns {missing_cols} not found in input DataFrame")

        self.medians_ = (
            X.groupby(self.group_col)[self.target_col]
             .median()
             .to_dict()
        )
        self.global_median_ = X[self.target_col].median()
        self.impute_values_ = X[self.group_col].map(self.medians_).fillna(self.global_median_)
        self._log("Computed group medians: %s", self.medians_)
        self._log("Computed global median: %s", self.global_median_)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        if not hasattr(self, "medians_"):
            raise RuntimeError("The imputer has not been fitted yet. Please call 'fit' before 'transform'.")
        
        X = X.copy()
        n_before = int(X[self.target_col].isna().sum())
        self._log("Imputing missing values in column '%s'", self.target_col)

        def _fill(row: pd.Series) -> float:
            if pd.isna(row[self.target_col]):
                group_value = row[self.group_col]
                impute_value = self.medians_.get(group_value, self.global_median_)
                self._log(f"Imputing missing value for group {group_value} with value {impute_value}")
                return impute_value
            else:
                return row[self.target_col]
        
        X[self.target_col] = X.apply(_fill, axis=1)
        
        n_after = int(X[self.target_col].isna().sum())
        
        self._log(f"Imputation complete. Missing values before: {n_before}, after: {n_after}")
        
        return X
    
class BinaryFlagTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            flags: list[dict[str, Any]],
            logger: Any = None
    ) -> None:
        self.flags = flags
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)
        else:
            print(msg, *args)
    
    def fit(self, X: pd.DataFrame, y: Any = None) -> "BinaryFlagTransformer":
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for flag in self.flags:
            col = flag['column']
            value = flag['value']
            new_col = flag['new_column']
            if col not in X.columns:
                raise KeyError(f"Column '{col}' not found in input DataFrame")
            self._log("Creating binary flag '%s' for column '%s' with threshold '%s'", new_col, col, value)
            X[new_col] = (X[col] >= value).astype(int)
        return X
    
class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ratios: list[dict], logger: Any = None) -> None:
        self.ratios = ratios
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)
    
    def fit(self, X: pd.DataFrame, y=None) -> "RatioFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        created: list[str] = []

        for spec in self.ratios:
            name = spec["name"]
            num = spec["numerator"]
            den = spec["denominator"]

            if num not in X.columns or den not in X.columns:
                if self.logger:
                    self.logger.warning("RatioFeatureTransformer: colunas '%s' ou '%s' ausentes - '%s' ignorada.", num, den, name)
                continue

            X[name] = (X[num] / X[den].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            created.append(name)

        self._log("RatioFeatureTransformer: features criadas: %s", created)

        return X
    

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], logger: Any =None) -> None:
        self.columns = columns
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "LogTransformer":
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        created: list[str] = []
        skipped: list[str] = []

        for col in self.columns:
            if col not in X.columns:
                skipped.append(col)
                continue

            log_col = f"log_{col}"
            skew_before = float(X[col].dropna().skew())
            X[log_col] = np.log1p(X[col].clip(lower=0))
            skew_after = float(X[log_col].dropna().skew())
            created.append(log_col)

            self._log(
                "LogTransformer: '%s' -> '%s' | skewness: %.2f -> %.2f", col, log_col, skew_before, skew_after,
            )

        if skipped and self.logger:
            self.logger.warning(
                "LogTransformer: colunas nao encontradas (ignoradas): %s", skipped
            )

        return X
    
class GeoDistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, geo_config: dict, logger: Any = None) -> None:
        self.geo_config = geo_config
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "GeoDistanceTransformer":
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        cities = self.geo_config.get("cities", [])
        lat_col = self.geo_config.get("lat_col", "latitude")
        lon_col = self.geo_config.get("lon_col", "longitude")
        nearest_col = self.geo_config.get("nearest_city_column", "nearest_city_distance")

        if lat_col not in X.columns or lon_col not in X.columns:
            if self.logger:
                self.logger.warning(
                    "GeoDistanceTransformer: colunas '%s'/'%s' ausentes - transformaçao ignorada.", lat_col, lon_col
                )
            return X
        
        X = X.copy()
        dist_cols: list[str] = []

        for city in cities:
            name = city["name"]
            col_name = f'dist_{name}'
            X[col_name] = np.sqrt(
                (X[lat_col] - city["lat"]) ** 2 +
                (X[lon_col] - city["lon"]) ** 2
            )
            dist_cols.append(col_name)

        if dist_cols:
            X[nearest_col] = X[dist_cols].min(axis=1)
            self._log(
                "GeoDistanceTransformer: %d distancias calculadas: %s | '%s' adicionado.", len(dist_cols), dist_cols, nearest_col
            )

        return X
    

class PolynomialFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pol_config: dict, logger: Any = None) -> None:
        self.pol_config = pol_config
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "PolynomialFeatureTransformer":
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        created: list[str] = []

        for spec in self.pol_config:
            name = spec["name"]
            cols = spec["columns"]
            missing = [c for c in cols if c not in X.columns]

            if missing:
                if self.logger:
                    self.logger.warning(
                        "PolynomialFeatureTransformer: colunas ausentes %s - '%s' ignorada.", missing, name
                    )
                continue
            if len(cols) == 1:
                X[name] = X[cols[0]] ** 2
            elif len(cols) == 2:
                X[name] = X[cols[0]] * X[cols[1]]
            else:
                if self.logger:
                    self.logger.warning(
                        "PolynomialFeatureTransformer: '%s' tem %d colunas - apenas 1 ou 2 suportadas.", name, len(cols)
                    )
                continue
            created.append(name)

        self._log("PolynomialFeatureTransformer: features criadas: %s", created)
        return X
    
class OceanProximityEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ope_config: dict, logger: Any = None) -> None:
        self.ope_config = ope_config
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "OceanProximityEncoder":
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        column = self.ope_config.get("column", "ocean_proximity")
        ordinal_column = self.ope_config.get("ordinal_columns", "ocean_proximity_encoded")
        ordinal_map: dict = self.ope_config.get("ordinal_map", {})
       
        prefix = self.ope_config.get("one_hot_prefix", "op")
        drop_first: bool = self.ope_config.get("drop_first", False)

        if column not in X.columns:
            if self.logger:
                self.logger.warning(
                    "OceanProximityEncoder: coluna '%s' nao encontrada - encoding ignorado.", column
                )
            return X
        
        X = X.copy()
        X[ordinal_column] = X[column].map(ordinal_map)
        n_unknown = int(X[ordinal_column].isna().sum())
        if n_unknown > 0 and self.logger:
            self.logger.warning(
                "OceanProximityEncoder: %d linhas com valores de '%s' nao mapeados -> NaN", n_unknown, column
            )
        self._log(
            "OceanProximityEncoder: ordinal '%s' criado - mapa: %s", ordinal_column, ordinal_map
        )

        dummies = pd.get_dummies(
            X[column],
            prefix=prefix,
            drop_first=drop_first
        ).astype(int)

        X = pd.concat([X, dummies], axis=1)
        self._log(
            "OceanProximityEncoder: dummies criadas %s", list(dummies.columns)
        )
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str], logger: Any = None) -> None:
        self.features = features
        self.logger = logger
        self.logger.warning(f"Features: {features}")

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        missing = [f for f in self.features if f not in X.columns]
        self.logger.warning(f"FeatureSelector: colunas ausentes {missing} - seleçao falhou.")
        if missing:
            # raise KeyError(f"FeatureSelector: colunas ausentes {missing} - seleçao falhou.")
            return X
        self._log("FeatureSelector: selecionando features %s", self.features)
        return X[self.features].copy()