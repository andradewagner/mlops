import os
from typing import Any
from dotenv import load_dotenv
from pathlib import Path
from src.utils.logger import get_logger
import logging
import zipfile

def check_kaggle_credentials(secrets_path: Path) -> bool:
    """Verifica se as credenciais do Kaggle estão configuradas."""
    load_dotenv(dotenv_path=str(secrets_path))
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    
    if kaggle_username and kaggle_key:
        return True
    else:
        print("Kaggle credentials not found. Please set KAGGLE_USERNAME and KAGGLE_KEY in your environment variables.")
        return False
    

def list_remote_files(
        dataset: str,
        logging_config: dict[str, Any],
        file_pattern: str = None
        ) -> list[str]:
    
    """Lista os arquivos disponíveis em um dataset do Kaggle, filtrando por um padrão de nome."""

    logger = get_logger(
        name=logging_config.get('name', 'KaggleDownloaderLogger'),
        level=getattr(logging, logging_config.get('level', 'INFO').upper(), logging.INFO),
        log_to_file=logging_config.get('log_to_file', False),
        log_dir=logging_config.get('log_dir', 'logs'),
        log_file=logging_config.get('log_file', 'pipeline.log')
    )

    check_kaggle_credentials()
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        files = api.dataset_list_files(dataset).files
        logger.info(f"Found {len(files)} files in dataset '{dataset}'.")
        if file_pattern:
            filtered_files = [f for f in files if file_pattern in f.name]
            logger.info(f"{len(filtered_files)} files match the pattern '{file_pattern}'.")
            return [f.name for f in filtered_files]
        else:
            return [f.name for f in files]
    except Exception as e:
        logger.error(f"Error listing files from Kaggle dataset '{dataset}': {e}")
        raise

def download_file_from_kaggle(
        dataset: str,
        expected_files: list[str],
        output_dir: Path,
        logging_config: dict[str, Any],
        skip_existing: bool = True,
        force_download: bool = False,
        secrets_path: Path = Path("config/secrets.env"),
        unzip: bool = False
) -> list[Path]:
    """Baixa arquivos específicos de um dataset do Kaggle, com opções para pular arquivos existentes ou forçar o download."""
    
    logger = get_logger(
        name=logging_config.get('name', 'KaggleDownloaderLogger'),
        level=getattr(logging, logging_config.get('level', 'INFO').upper(), logging.INFO),
        log_to_file=logging_config.get('log_to_file', False),
        log_dir=logging_config.get('log_dir', 'logs'),
        log_file=logging_config.get('log_file', 'pipeline.log')
    )

    check_kaggle_credentials(secrets_path)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        downloaded_files = []
        for filename in expected_files:
            output_path = output_dir / filename
            if skip_existing and output_path.is_file():
                logger.info(f"File '{filename}' already exists. Skipping download.")
                downloaded_files.append(output_path)
                continue
            
            if force_download and output_path.is_file():
                logger.info(f"File '{filename}' already exists but force_download is True. Re-downloading.")
            
            logger.info(f"Downloading file '{filename}' from dataset '{dataset}'...")
            api.dataset_download_file(dataset, filename, path=str(output_dir))
            downloaded_files.append(output_path)
        
        return downloaded_files
    except Exception as e:
        logger.error(f"Error downloading files from Kaggle dataset '{dataset}': {e}")
        raise

def _unzip_file(zip_path: Path, extract_to: Path, logger: logging.Logger) -> None:
    """Descompacta um arquivo ZIP para um diretório especificado."""
    
    logger.info(f"Unzipping file '{zip_path}' to '{extract_to}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        logger.info(f"File '{zip_path}' unzipped successfully.")