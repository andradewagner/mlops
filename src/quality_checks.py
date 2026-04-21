
import sys
import json
import pandas as pd
from typing import Any
from pathlib import Path
from src.utils.logger import get_logger
from datetime import datetime

logger = get_logger('QualityChecksLogger')
logger.info("Quality checks module initialized.")

def _import_ge():
    logger.info("Attempting to import Great Expectations library for quality checks.")
    try:
        import great_expectations as gx
        import great_expectations.expectations as gxe
        logger.info("Great Expectations library imported successfully.")
        return gx, gxe
    except ImportError as e:
        logger.error("Great Expectations library is not installed. Please install it to run quality checks.")
        raise e
    

def _build_ephemeral_context(df: pd.DataFrame, suite_name: str):
    logger.info("Building ephemeral Great Expectations context for suite: %s", suite_name)
    
    gx, _ = _import_ge()
    
    #criando um contexto efêmero
    context = gx.get_context(
        mode='ephemeral'
    )

    datasource = context.data_sources.add_pandas(name='pipeline_datasource')
    asset = datasource.add_dataframe_asset(name='input_data')
    batch_def = asset.add_batch_definition_whole_dataframe(
              'full_batch'
    )

    suite = context.suites.add(gx.core.expectation_suite.ExpectationSuite(name=suite_name))
    logger.info("Ephemeral context built successfully with datasource and batch defined.")
    return context, batch_def, suite

def _resolve_expectation_class(gxe, expectation_type: str):
    pascal = _snake_to_pascal(expectation_type)
    if hasattr(gxe, pascal):
        return getattr(gxe, pascal)
    elif hasattr(gxe, expectation_type):
        return getattr(gxe, expectation_type)
    else:
        logger.error("Expectation type '%s' not found in Great Expectations library.", expectation_type)
        raise ValueError(f"Expectation type '{expectation_type}' not found in Great Expectations library.")


def _snake_to_pascal(snake_str: str) -> str:
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


def _populate_suite_with_expectations(
        suite,
        table_expectations: list[dict[str, Any]],
        column_expectations: dict[str, list[dict[str, Any]]],
        gxe
) -> None:
    logger.info("Populating expectation suite with table-level and column-level expectations.")
    # Expectativas de nível de tabela
    for exp in (table_expectations or []):
        exp_class = _resolve_expectation_class(gxe, exp['type'])
        suite.add_expectation(exp_class(**exp['kwargs']))
        logger.debug("Added table-level expectation: %s with kwargs: %s", exp['type'], exp['kwargs'])
    
    # Expectativas de nível de coluna
    for col, exp_list in column_expectations.items():
        for exp in exp_list:
            exp_class = _resolve_expectation_class(gxe, exp['type'])
            suite.add_expectation(exp_class(column=col, **exp['kwargs']))
            logger.debug("Added column-level expectation for %s: %s with kwargs: %s", col, exp['type'], exp['kwargs'])

# main
def run_quality_checks(
    df: pd.DataFrame,
    config: dict[str, Any],
    logging_config: dict[str, Any]
) -> dict[str, Any]:
    gx, gxe = _import_ge()
    
    logger = get_logger("QualityChecksLogger", logging_config=logging_config)
    
    # Carregar as expectativas do arquivo de configuração
    suite_name = config.get('suite_name', 'default_suite')
    
    # Se vier False -> False, 
    # se vier "False" (string) -> True, 
    # se vier None -> False
    # O valor False evita que a pipeline falhe caso as verificações de qualidade falhem, 
    # enquanto "False" (string) ou None não causará falha na pipeline, mas ainda assim registrará 
    # os resultados das verificações de qualidade.
    raw = fail_on_error = config.get('fail_pipeline_on_failure', False)
    fail_on_error = (raw is True)
    
    table_expectations = config.get('table_expectations', [])
    column_expectations = config.get('column_expectations', [])

    total_exp = len(table_expectations) + sum(len(v) for v in column_expectations.values())
    logger.info('Iniciando as verificações de qualidade com um total de %d expectativas definidas.', total_exp)

    context, batch_def, suite = _build_ephemeral_context(df, suite_name)

    _populate_suite_with_expectations(suite, table_expectations, column_expectations, gxe)

    logger.info("Running validation for suite: %s", suite_name)

    # Cria e registra a ValidationDefinition
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name=f"{suite_name}_validation",
            data=batch_def,
            suite=suite
        )
    )

    # Executa a validação
    results = validation_def.run(
        batch_parameters={
            'dataframe': df
        }
    )

    success = results.success
    total = len(results.results)
    passed = sum(1 for r in results.results if r.success)
    failed = total - passed

    logger.info("Validation results: %d total, %d passed, %d failed. Success: %s", total, passed, failed, success)
    logger.info(f"fail_on_error = {fail_on_error}")
    logger.info(f"success = {success}")

    if not success and fail_on_error:
        logger.error("Quality checks failed. Failing the pipeline as per configuration.")
        raise ValueError("Quality checks failed. See logs for details.")
    else:
        logger.info("Quality checks completed successfully.")

    logger.info('/'*60)
    logger.info("Resultado de validacao de qualidade:")
    logger.info("   Total checks: %d", total)
    logger.info("   Passed checks: %d", passed)
    logger.info("   Failed checks: %d", failed)
    logger.info('/'*60)

    for r in results.results:
        status = "PASSED" if r.success else "FAILED"
        exp_type = r.expectation_config.type
        column = r.expectation_config.kwargs.get('column', '(tabela)')
        logger.info("   [%s] %-50s  col=%-25s", status, exp_type, column)

    if fail_on_error and not success:
        raise RuntimeError(f"Qualidade de dados REPROVADA: {failed}/{total} checks falharam. \n"
                           "Revise o quality.yaml ou investigue os dados de entrada.\n"
                           "Para continuar, mesmo com falhas, ajuste 'fail_pipeline_on_failure' para False no arquivo de configuração de qualidade.")

    return {
        'success': success,
        'total_expectations': total,
        'passed_expectations': passed,
        'failed_expectations': failed,
        'results': results.to_json_dict()
    }

def save_quality_report(
        summary: dict[str, Any],
        output_path: Path,
        logging_config: dict[str, Any] | None = None
) -> Path:
    logger = get_logger("QualityReportSaver", logging_config=logging_config)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "quality_report.json"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path / f"quality_report_{timestamp}.json"

    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info("Quality report saved to %s", report_path)
    


    details = []
    for r in summary['results']['results']:
        raw_kwargs = dict(r["expectation_config"]["kwargs"])
        column = raw_kwargs.pop('column', None)
        raw_result = r["result"] if isinstance(r["result"], dict) else {}

        details.append({
            'type': r["expectation_config"]["type"],
            'column': column,
            'success': r["success"],
            'kwargs': raw_kwargs,
            'result': {k: v for k, v in raw_result.items() if not isinstance(v, (list, dict)) or len(str(v)) < 500}
        })
        
    report = {
        'success': summary['success'],
        'total_expectations': summary['total_expectations'],
        'passed_expectations': summary['passed_expectations'],
        'failed_expectations': summary['failed_expectations'],
        'details': details
    }

    # Salvar o relatório detalhado em um arquivo JSON
    try:
        with open(report_path, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error("quality_checks.py: Failed to save quality report: %s", e)
        raise e
    
    logger.info("Detailed quality report saved to %s", report_path)
    return report_path