import logging
from pathlib import Path
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: str = "logs",
    log_file: str = "pipeline.log",
    logging_config: Optional[dict] = None
) -> logging.Logger:
    
    if logging_config:
        level = logging_config.get("level", level)
        log_to_file = logging_config.get("log_to_file", log_to_file)
        log_dir = logging_config.get("log_dir", log_dir)
        log_file = logging_config.get("log_file", log_file)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        # File handler (opcional)
        if log_to_file:
            log_path = Path(log_dir) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

        logger.propagate = False

    return logger
