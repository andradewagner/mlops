import sys
import json
import time
import warnings
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from pathlib import Path
import optuna
import mlflow
import mlflow.sklearn

from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline as SKPipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# Funções auxiliares
# ============================================================

def _compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }

def _run_cv(model, X, y, cv):
    fold_metrics = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = m.predict(X.iloc[val_idx])
        metrics = _compute_metrics(y.iloc[val_idx], y_pred)
        metrics["fold"] = fold_i + 1
        fold_metrics.append(metrics)
    return fold_metrics

def _aggregate_fold_metrics(fold_metrics):
    df = pd.DataFrame(fold_metrics)
    return {
        f"cv_{col}_mean": float(df[col].mean())
        for col in ["rmse", "mae", "r2", "mape"]
    } | {
        f"cv_{col}_std": float(df[col].std())
        for col in ["rmse", "mae", "r2", "mape"]
    }

def _suggest_param(trial, name, spec):
    t = spec["type"]
    if t == "log_float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Tipo desconhecido: {t}")

def _build_model(model_cfg, extra_params=None):
    module = importlib.import_module(model_cfg["module"])
    cls = getattr(module, model_cfg["class"])
    params = dict(model_cfg.get("default_params") or {})
    if extra_params:
        params.update(extra_params)
    return cls(**params)

def _build_pipeline(model_cfg, model_params, reducer_params):
    return SKPipeline([
        ("reducer", FeatureReducer(**reducer_params)),
        ("estimator", _build_model(model_cfg, model_params)),
    ])

# ============================================================
# Carregar configs
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.feature_reducer import FeatureReducer

config = load_yaml(CONFIG_DIR / "pipeline.yaml")
modeling_cfg = load_yaml(CONFIG_DIR / "modeling.yaml")
config.update(modeling_cfg)

logger = get_logger("Modelagem")
logger.info("=== Modelagem — MLflow + Optuna ===")

# MLflow
mlflow.set_tracking_uri(modeling_cfg["modeling"]["tracking_uri"])
mlflow.set_experiment(modeling_cfg["modeling"]["experiment_name"])

SEED = modeling_cfg["modeling"]["random_seed"]

# ============================================================
# Carregar features
# ============================================================

features_dir = ROOT_DIR / config["paths"]["features_data_dir"]
features_file = features_dir / config["paths"]["features_filename"]

logger.info("Carregando features: %s", features_file)

# Carrega features (X)
X = pq.read_table(features_file).to_pandas()

# Carrega target (y) do processed
processed_file = ROOT_DIR / config["paths"]["processed_data_dir"] / config["paths"]["output_filename"]
df_processed = pq.read_table(processed_file).to_pandas()

target_col = config["feature_selection"]["target"]
y = df_processed[target_col]

# Alinha índices
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

logger.info("Shape X=%s  y=%s", X.shape, y.shape)

# ============================================================
# Holdout
# ============================================================

test_size = config["holdout"]["test_size"]
n_bins = config["holdout"]["stratify_bins"]

y_bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=test_size, random_state=SEED, stratify=y_bins
)

logger.info("Treino: %d | Holdout: %d", len(X_train), len(X_holdout))

# ============================================================
# CV
# ============================================================

cv_cfg = config["cv"]
cv = KFold(
    n_splits=cv_cfg["n_splits"],
    shuffle=cv_cfg["shuffle"],
    random_state=SEED
)

# ============================================================
# Feature Reduction
# ============================================================

feat_red_cfg = config["feature_reduction"]
method = feat_red_cfg["method"]

def _default_reducer_params():
    if method == "none":
        return {"method": "none"}
    return feat_red_cfg[method]

# ============================================================
# Baseline
# ============================================================

models_cfg = config["models"]
all_results = {}

logger.info("=== Baseline ===")

for model_name, model_cfg in models_cfg.items():
    if not model_cfg.get("enabled", True):
        continue

    logger.info("[BASELINE] %s", model_name)

    pipeline = _build_pipeline(
        model_cfg=model_cfg,
        model_params=None,
        reducer_params=_default_reducer_params()
    )

    with mlflow.start_run(run_name=f"baseline_{model_name}"):
        fold_metrics = _run_cv(pipeline, X_train, y_train, cv)
        agg = _aggregate_fold_metrics(fold_metrics)

        mlflow.log_metrics(agg)
        mlflow.log_params(model_cfg.get("default_params") or {})

        all_results[model_name] = {
            "cv_rmse_mean": agg["cv_rmse_mean"],
            "cv_rmse_std": agg["cv_rmse_std"],
            "cv_r2_mean": agg["cv_r2_mean"],
            "fold_metrics": fold_metrics,
            "best_params": model_cfg.get("default_params") or {},
            "reducer_params": _default_reducer_params(),
            "tuned": False,
        }

# ============================================================
# Optuna
# ============================================================

logger.info("=== Optuna ===")

optuna_cfg = config["optuna"]
global_trials = optuna_cfg["default_trials"]

for model_name, model_cfg in models_cfg.items():
    if not model_cfg.get("enabled", True):
        continue

    search_space = model_cfg.get("search_space") or {}
    n_trials = model_cfg.get("optuna_trials", global_trials)

    if not search_space or n_trials <= 1:
        continue

    logger.info("[OPTUNA] %s (%d trials)", model_name, n_trials)

    def objective(trial):
        params = {
            name: _suggest_param(trial, name, spec)
            for name, spec in search_space.items()
        }

        pipeline = _build_pipeline(
            model_cfg=model_cfg,
            model_params=params,
            reducer_params=_default_reducer_params()
        )

        fold_metrics = _run_cv(pipeline, X_train, y_train, cv)
        agg = _aggregate_fold_metrics(fold_metrics)

        return agg["cv_rmse_mean"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    all_results[model_name]["best_params"] = best_params
    all_results[model_name]["tuned"] = True

# ============================================================
# Ensembles (Stacking / Voting)
# ============================================================

logger.info("=== Ensembles ===")

top_n = config["ensembles"]["top_n_base_models"]
sorted_models = sorted(all_results.items(), key=lambda x: x[1]["cv_rmse_mean"])
top_models = sorted_models[:top_n]

logger.info("Top-%d modelos: %s", top_n, [m[0] for m in top_models])

logger.info("=== Avaliação Final no Holdout ===")

best_model_name = min(all_results, key=lambda m: all_results[m]["cv_rmse_mean"])
best_info = all_results[best_model_name]

logger.info("Melhor modelo: %s", best_model_name)

# Reconstruir o melhor pipeline
best_pipeline = _build_pipeline(
    model_cfg=models_cfg[best_model_name],
    model_params=best_info["best_params"],
    reducer_params=best_info["reducer_params"]
)

# Treinar no conjunto completo de treino
best_pipeline.fit(X_train, y_train)

# Prever no holdout
y_pred = best_pipeline.predict(X_holdout)

# Métricas finais
final_metrics = _compute_metrics(y_holdout, y_pred)
logger.info("Métricas no Holdout: %s", final_metrics)

with mlflow.start_run(run_name="final_model", tags={"stage": "final"}) as run:

    # Loga métricas finais
    mlflow.log_metrics({
        "holdout_rmse": final_metrics["rmse"],
        "holdout_mae": final_metrics["mae"],
        "holdout_r2": final_metrics["r2"],
        "holdout_mape": final_metrics["mape"],
    })

    # Loga modelo e captura o caminho correto
    model_info = mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        artifact_path="best_model"
    )

    # Loga hiperparâmetros do melhor modelo
    mlflow.log_params({
        f"best_{k}": v for k, v in best_info["best_params"].items()
    })

    # Loga parâmetros do reducer
    mlflow.log_params({
        f"reducer_{k}": v for k, v in best_info["reducer_params"].items()
    })

    # Salva o modelo final
    input_example = X_train.iloc[:1]

    model_info = mlflow.sklearn.log_model(
        sk_model=best_pipeline, 
        artifact_path="best_model",
        input_example=input_example
    )

    mlflow.log_param("run_id", run.info.run_id)

    logger.info("Modelo final salvo no MLflow. Run ID: %s", run.info.run_id)

    # Plot Predicted vs Actual
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_holdout, y=y_pred, alpha=0.3)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual (Holdout)")
    plot_path = ROOT_DIR / "outputs/modeling/pred_vs_actual.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    mlflow.log_artifact(str(plot_path))

    # Plot Residuals
    residuals = y_holdout - y_pred
    plt.figure(figsize=(6,6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title("Residual Distribution")
    res_path = ROOT_DIR / "outputs/modeling/residuals.png"
    plt.savefig(res_path, dpi=120)
    plt.close()
    mlflow.log_artifact(str(res_path))

mlflow.register_model(
    model_uri=model_info.model_uri,
    name="kingcounty_price_model"
)

# ============================================================
# Finalização
# ============================================================

logger.info("=== Modelagem finalizada ===")
logger.info("Resultados:")

for name, res in all_results.items():
    logger.info("%s — RMSE=%.2f  R2=%.3f  Tuned=%s",
                name, res["cv_rmse_mean"], res["cv_r2_mean"], res["tuned"])