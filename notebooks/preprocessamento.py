import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# -------------------------------------------------------------------
# PATHS E CONFIG
# -------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ROOT_DIR / "config"

sys.path.extend([str(ROOT_DIR), str(CONFIG_DIR)])

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.preprocessing import (
    CustomImputer,
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    GeoDistanceTransformer,
    PolynomialFeatureTransformer,
    FeatureSelector
)

logger = get_logger("Preprocessing")
logger.info("=== Iniciando pré-processamento para King County ===")

# -------------------------------------------------------------------
# CARREGAR CONFIGS
# -------------------------------------------------------------------
pipeline_cfg = load_yaml(CONFIG_DIR / "pipeline.yaml")
prep_cfg = load_yaml(CONFIG_DIR / "preprocessing.yaml")

logger.info("Configurações carregadas com sucesso")

# Caminhos
processed_dir = ROOT_DIR / pipeline_cfg["paths"]["processed_data_dir"]
input_parquet = processed_dir / pipeline_cfg["paths"]["output_filename"]

output_dir = ROOT_DIR / prep_cfg["preprocessing"]["output_dir"]
output_path = output_dir / prep_cfg["preprocessing"]["output_filename"]
compression = prep_cfg["preprocessing"].get("compression", "snappy")

logger.info(f"Lendo arquivo processado: {input_parquet}")

if not input_parquet.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {input_parquet}")

# -------------------------------------------------------------------
# CARREGAR DATAFRAME
# -------------------------------------------------------------------
df = pq.read_table(input_parquet).to_pandas()
logger.info(f"Dataset carregado. Shape: {df.shape}")
logger.info(df.head())

# -------------------------------------------------------------------
# 1. IMPUTAÇÃO
# -------------------------------------------------------------------
imp_cfg = prep_cfg.get("imputation", [])
logger.info(f"Aplicando imputação: {len(imp_cfg)} regras")

for spec in imp_cfg:
    col = spec["column"]
    strategy = spec["strategy"]
    fill_value = spec.get("fill_value")
    group_by = spec.get("group_by")

    logger.info(f"Imputando coluna '{col}' com estratégia '{strategy}'")

    imputer = CustomImputer(
        group_col=group_by,
        target_col=col,
        strategy=strategy,
        fill_value=fill_value,
        logger=logger
    )
    df = imputer.fit_transform(df)

# -------------------------------------------------------------------
# 2. BINARY FLAGS
# -------------------------------------------------------------------
binary_cfg = prep_cfg.get("binary_flags", [])
logger.info(f"Aplicando binary flags: {len(binary_cfg)}")

flag_transformer = BinaryFlagTransformer(binary_cfg, logger=logger)
df = flag_transformer.fit_transform(df)

# -------------------------------------------------------------------
# 3. RATIO FEATURES
# -------------------------------------------------------------------
ratio_cfg = prep_cfg.get("ratio_features", [])
logger.info(f"Aplicando ratio features: {len(ratio_cfg)}")

ratio_transformer = RatioFeatureTransformer(ratio_cfg, logger=logger)
df = ratio_transformer.transform(df)

# -------------------------------------------------------------------
# 4. LOG TRANSFORM
# -------------------------------------------------------------------
log_cols = prep_cfg.get("log_transformer", {}).get("columns", [])
logger.info(f"Aplicando log-transform em: {log_cols}")

log_transformer = LogTransformer(log_cols, logger=logger)
df = log_transformer.transform(df)

# -------------------------------------------------------------------
# 5. GEO DISTANCE
# -------------------------------------------------------------------
geo_cfg = prep_cfg.get("geo_distance", {})
logger.info("Aplicando distâncias geográficas")

geo_transformer = GeoDistanceTransformer(geo_cfg, logger=logger)
df = geo_transformer.transform(df)

# -------------------------------------------------------------------
# 6. POLYNOMIAL FEATURES
# -------------------------------------------------------------------
pol_cfg = prep_cfg.get("polynomial_features", [])
logger.info(f"Aplicando polynomial features: {pol_cfg}")

pol_transformer = PolynomialFeatureTransformer(pol_cfg, logger=logger)
df = pol_transformer.transform(df)

# -------------------------------------------------------------------
# 7. CATEGORICAL ENCODING (ZIPCODE)
# -------------------------------------------------------------------
enc_cfg = prep_cfg.get("categorical_encoding", {})
cat_col = enc_cfg.get("column", "zipcode")
prefix = enc_cfg.get("one_hot_prefix", "zip")

logger.info(f"Aplicando one-hot encoding em '{cat_col}'")

df = pd.get_dummies(df, columns=[cat_col], prefix=prefix, drop_first=False)

# -------------------------------------------------------------------
# 8. FEATURE SELECTION
# -------------------------------------------------------------------
feat_cfg = prep_cfg.get("feature_selection", {})
features_to_keep = feat_cfg.get("features_to_keep", [])

logger.info(f"Selecionando features finais ({len(features_to_keep)}):")
logger.info(features_to_keep)

selector = FeatureSelector(features_to_keep, logger=logger)
df = selector.transform(df)

# -------------------------------------------------------------------
# 9. SALVAR RESULTADO
# -------------------------------------------------------------------
output_dir.mkdir(parents=True, exist_ok=True)

df["bath_per_bed"] = df["bath_per_bed"].fillna(0)
df["sqft_living_per_room"] = df["sqft_living_per_room"].fillna(df["sqft_living_per_room"].median())
df["lot_per_sqft"] = df["lot_per_sqft"].fillna(df["lot_per_sqft"].median())
df["nearest_city_distance"] = df["nearest_city_distance"].fillna(df["nearest_city_distance"].median())

df.to_parquet(output_path, index=False, compression=compression)

logger.info(f"Pré-processamento concluído com sucesso!")
logger.info(f"Arquivo salvo em: {output_path}")
logger.info(f"Shape final: {df.shape}")

print(df.head())