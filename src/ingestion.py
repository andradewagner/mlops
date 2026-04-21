import time
from pathlib import Path
from typing import Any
from src.utils.logger import get_logger, logging

def ingest_csv_to_parquet(
        raw_dir: Path,
        output_path: Path,
        compression: str = 'snappy',
        chunk_size: int = 50_000,
        validate_schema: bool = True,
        required_columns: list[str] | None = None,
        skip_if_exists: bool = True,
        force: bool = False,
        logging_config: dict[str, Any] = {}
) -> Path:
    """Ingesta arquivos CSV de um diretório, convertendo-os para Parquet com opções de validação e controle de execução."""
    
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.csv as pc

    logger = get_logger(
        name=logging_config.get('name', 'IngestaoLogger'),
        level=getattr(logging, logging_config.get('level', 'INFO').upper(), logging.INFO),
        log_to_file=logging_config.get('log_to_file', False),
        log_dir=logging_config.get('log_dir', 'logs'),
        log_file=logging_config.get('log_file', 'pipeline.log')
    )

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    
    if not force and skip_if_exists and output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Output file '{output_path}' already exists and is not empty. Skipping ingestion.")
        return output_path
    
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in '{raw_dir}'.")
        raise FileNotFoundError(f"No CSV files found in '{raw_dir}'.")

    logger.info(f"Found {len(csv_files)} CSV files to ingest from '{raw_dir}'.")

    for cf in csv_files:
        size_mb = cf.stat().st_size / (1024 * 1024)
        logger.info(f"File Info: {cf.name} ({size_mb:.1f} MB)")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
        
    read_options = pc.ReadOptions(block_size=chunk_size, use_threads=True)

    convert_options = pc.ConvertOptions(
        auto_dict_encode=False,
        include_missing_columns=False
    )

    parse_options = pc.ParseOptions(
        delimiter=',',
        quote_char='"',
        double_quote=True,
        newlines_in_values=False
    )

    start_time = time.monotonic()
    total_rows = 0
    writer: pq.ParquetWriter | None = None

    try:
        logger.info(f"Starting ingestion of CSV files to Parquet with compression='{compression}', chunk_size={chunk_size}, validate_schema={validate_schema}, required_columns={required_columns}")
        for csv_file in csv_files:
            logger.info(f"Processing file '{csv_file}'...")

            with pc.open_csv(csv_file, read_options=read_options, convert_options=convert_options, parse_options=parse_options) as reader:
                schema: pa.Schema = reader.schema

                logger.info(f"Schema for file '{csv_file}': {schema}")

                if writer is None:
                    writer = pq.ParquetWriter(output_path, reader.schema, compression=compression)
                    logger.info(f"Initialized Parquet writer with schema: {reader.schema}")
                    
                for batch in reader:
                    logger.info(f"Read batch with {batch.num_rows} rows from '{csv_file}'")
                    if validate_schema:
                        batch_schema = batch.schema
                        if schema is None:
                            schema = batch_schema

                            logger.info(f"Set reference schema from first batch: {schema}")

                        elif not batch_schema.equals(schema, check_metadata=False):

                            logger.error(f"Schema mismatch detected in file '{csv_file}'. Expected: {schema}, Got: {batch_schema}")

                            raise ValueError(f"Schema mismatch in file '{csv_file}'")

                    if required_columns:
                        logger.info(f"Validating required columns in batch from '{csv_file}'")
                        missing_cols = [col for col in required_columns if col not in batch.schema.names]
                        if missing_cols:
                            logger.error(f"Missing required columns in file '{csv_file}': {missing_cols}")
                            raise ValueError(f"Missing required columns in file '{csv_file}': {missing_cols}")

                    writer.write_batch(batch)
                    total_rows += batch.num_rows
            logger.info(f"Finished processing '{csv_file}'. Total rows so far: {total_rows}")
    finally:
        logger.info(f"Closing Parquet writer.")
        if writer is not None:
            writer.close()

    if validate_schema and required_columns:
        _validate_required_columns(output_path, required_columns, logger)

    elapsed_time = time.monotonic() - start_time
    input_size_mb = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    compression_ratio = input_size_mb / output_size_mb if output_size_mb > 0 else float('inf')

    logger.info(f"Ingestion completed in {elapsed_time:.2f} seconds. Input size is: {input_size_mb}. Output size is: {output_size_mb:.2f} MB. Compression ratio: {compression_ratio:.2f}")

    return output_path

def _validate_required_columns(
        parquet_path: Path,
        required_columns: list[str],
        logger: Any,
) -> None:
    logger.info(f"Validating presence of required columns in output Parquet file '{parquet_path}'...")
    """_summary_

    Args:
        required_columns (list[str]): _description_
        logger (Any): _description_
    """
    import pyarrow.parquet as pq

    schema = pq.read_schema(str(parquet_path))
    actual_columns = set(schema.names)
    missing = [col for col in required_columns if col not in actual_columns]

    if missing:
        raise ValueError(
            f"Missing required columns in output Parquet file: {missing}\n"
            f"Actual columns in Parquet file: {sorted(actual_columns)}\n"
        )
    
    logger.info(f"All {len(required_columns)} required columns are present in the output Parquet file: {required_columns}")