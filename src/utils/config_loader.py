import yaml
from pathlib import Path
from typing import Any

def load_yaml(path: Path) -> dict[str, Any]:
    """Carrega um arquivo YAML e retorna seu conteúdo como um dicionário."""
    if not path.is_file():    
        raise FileNotFoundError(f"Config file not found: {path}\n"
                                f"Expected location is: {path.resolve()}")

    with open(path, 'r') as file:
        return yaml.safe_load(file)