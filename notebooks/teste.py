import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.config_loader import load_yaml    

config = load_yaml(CONFIG_DIR / "pipeline.yaml")

df = pd.read_parquet(ROOT_DIR / config["paths"]["features_data_dir"] / config["paths"]["features_filename"])
print(ROOT_DIR / config["paths"]["features_data_dir"] / config["paths"]["features_filename"])
print(df.columns.tolist())
