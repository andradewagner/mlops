from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

_RFE_ESTIMATORS: dict[str, Any] = {
    'ridge': Ridge(alpha=1.0),
    'random_forest': RandomForestRegressor(n_estimators=50, n_jobs=1, random_state=10)
}

def _resolve_rfe_estimator(spec: Any) -> Any:
    if isinstance(spec, str):
        if spec not in _RFE_ESTIMATORS:
            raise ValueError(
                f"rfe_estimator='{spec}' not recognised."
                f"Valid strings: {list(_RFE_ESTIMATORS.keys())}"
            )
        return clone(_RFE_ESTIMATORS[spec])
    return clone(spec)

class FeatureReducer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method: str = 'none',
        n_features_to_select: int = 15,
        rfe_estimator: Any = 'ridge',
        n_components: int = 15,
        kernel: str = 'rbf',
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        logger: Any = None,
    ) -> None:
        self.method = method
        self.n_features_to_select = n_features_to_select
        self.rfe_estimator = rfe_estimator
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)
        else:
            print(msg, *args)

    def _build_inner(self):
        """Instantiates the inner reducer from current params."""
        if self.method == 'none':
            return None
        if self.method == 'rfe':
            estimator = _resolve_rfe_estimator(self.rfe_estimator)
            return RFE(
                estimator=estimator,
                n_features_to_select=self.n_features_to_select,
            )
        if self.method == 'pca':
            return PCA(n_components=self.n_components, random_state=42)
        if self.method == 'kpca':
            return KernelPCA(
                n_components=self.n_components,
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0
            )
        raise ValueError(
            f"FeatureReducer: unknown method='{self.method}'."
            "Valid options: 'none', 'rfe', 'pca', 'kpca'."
        )
    
    def fit(self, X, y=None) -> "FeatureReducer":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None

        if self.method in ('pca', 'kpca'):
            n_features = X.shape[1]
            if self.n_components >= n_features:
                self.n_components = n_features - 1
                self._log(
                    "FeatureReducer.fit: n_components clamped to %d (< n_features=%d).", self.n_components, n_features
                )
        self.reducer_ = self._build_inner()

        if self.reducer_ is None:
            self.feature_names_out_ = self.feature_names_in_
            self._log("FeatureReducer.fit: method='none' - passthrough, no reduction.")
            return self
        
        if self.method == 'rfe':
            if y is None:
                raise ValueError(
                    "FeatureReducer with method='rfe' requires y."
                    "Ensure it is inside a Pipeline that receives y."
                )
            self.reducer_.fit(X, y)
            if self.feature_names_in_ is not None:
                self.feature_names_out_ = [
                    col for col, sel in zip(self.feature_names_in_, self.reducer_.support_) if sel
                ]
            else:
                self.feature_names_out_ = None

            self._log(
                "FeatureReducer.fit: RFE selected %d/%d features: %s", self.n_features_to_select, len(self.feature_names_in_) if self.feature_names_in_ else '?',
                self.feature_names_out_
            )
        elif self.method in ('pca', 'kpca'):
            self.reducer_.fit(X)
            n_out = self.n_components
            self.feature_names_out_ = [f'pc_{i}' for i in range(n_out)]
            explained = None
            if self.method == 'pca' and hasattr(self.reducer_, 'explained_variance_ratio_'):
                explained = float(self.reducer_.explained_variance_ratio_.sum())
            self._log(
                "FeatureReducer.fit: '%s' fitted -> '%d' components '%s'.", self.method.upper(), n_out,
                f' (explained variance: {explained:.3f})' if explained is not None else ''
            )
        return self

    def transform(self, X, y=None):
        if not hasattr(self, 'reducer_'):
            raise RuntimeError(
                "FeatureReducer has not been fitted. Call fit() before transform()."
            )
        
        if self.reducer_ is None:
            return X
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_out = self.reducer_.transform(X_arr)

        if self.feature_names_out_ is not None:
            return pd.DataFrame(X_out, columns=self.feature_names_out_, index=(X.index if isinstance(X, pd.DataFrame) else None))
        return X_out
    
    @property
    def selected_features(self) -> list[str] | None:
        return getattr(self, 'feature_names_out_', None)