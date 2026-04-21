# teste.py
from mlflow.tracking import MlflowClient
import mlflow

# força o backend sqlite (ajuste o caminho se necessário)
mlflow.set_tracking_uri("sqlite:////home/wagner/MLOps/mlflow.db")
print("tracking uri:", mlflow.get_tracking_uri())

client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

# listar experiments
print("\nExperimentos:")
for exp in client.search_experiments():
    print(exp.experiment_id, "-", exp.name)

# listar runs recentes do experimento 1 (ajuste experiment_id se necessário)
print("\nRuns recentes (experimento 1):")
runs = client.search_runs(experiment_ids=["1"], order_by=["attributes.start_time DESC"], max_results=50)
for r in runs:
    print(r.info.run_id, "| status:", r.info.status, "| start:", r.info.start_time)

# se você já tem um run_id conhecido, cole-o aqui para inspecionar artifacts
run_id = ""  # cole aqui um run_id válido se souber
if run_id:
    run = client.get_run(run_id)
    print("\nartifact_uri:", run.info.artifact_uri)
    print("root artifacts:", client.list_artifacts(run_id, path=""))
    print("artifacts at 'model':", client.list_artifacts(run_id, path="model"))
    print("artifacts at 'plots':", client.list_artifacts(run_id, path="plots"))


mlflow.set_tracking_uri("sqlite:////home/wagner/MLOps/mlflow.db")
client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
run_id = "420da03ffa824dceb87a92740fa326b8"
run = client.get_run(run_id)
print("artifact_uri:", run.info.artifact_uri)
print("root artifacts:", client.list_artifacts(run_id, path=""))
print("artifacts at 'model':", client.list_artifacts(run_id, path="model"))
print("artifacts at 'models':", client.list_artifacts(run_id, path="models"))
print("artifacts at 'plots':", client.list_artifacts(run_id, path="plots"))

