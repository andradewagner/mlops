# mlops

# Como abrir o mlflow (deve ser executado depois do script modelagem.py)
- mlflow ui --backend-store-uri sqlite:////home/wagner/MLOps/mlflow.db

export MLFLOW_TRACKING_URI="sqlite:////home/wagner/MLOps/mlflow.db"
mlflow models serve -m "models:/california-housing-best/latest" --port 5001 --no-conda

streamlit run production_app/app.py
