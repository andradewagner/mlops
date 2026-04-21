 # __  __ _     ___  ____   ____  
# |  \/  | |   / _ \|  _ \ / ___| 
# | |\/| | |  | | | | |_) | |     
# | |  | | |__| |_| |  __/| |___  
# |_|  |_|_____\___/|_|    \____| 
        M L O P S   P I P E L I N E

# mlops

# Como abrir o mlflow (deve ser executado depois do script modelagem.py)
- mlflow ui --backend-store-uri sqlite:////home/{user}/MLOps/mlflow.db

export MLFLOW_TRACKING_URI="sqlite:////home/{user}/MLOps/mlflow.db"
mlflow models serve -m "models:/california-housing-best/latest" --port 5001 --no-conda

streamlit run production_app/app.py

🏡 MLOps – Predição de Preços de Imóveis

Este projeto implementa um pipeline completo de MLOps para treinar, versionar, registrar e servir modelos de Machine Learning usando MLflow, além de uma aplicação Streamlit para consumo do modelo em produção.

O objetivo é demonstrar um fluxo realista de desenvolvimento e deploy de modelos, incluindo experiment tracking, model registry, serving e integração com uma aplicação interativa.
🚀 Funcionalidades

    ✔️ Treinamento de modelos com MLflow Tracking

    ✔️ Registro de modelos no MLflow Model Registry

    ✔️ Servidor de predição via mlflow models serve

    ✔️ Aplicação Streamlit para uso do modelo em produção

    ✔️ Suporte para troca de dataset

    ✔️ Estrutura modular e organizada para estudos de MLOps

📦 Estrutura do Projeto
Código

.
├── production_app/        # Aplicação Streamlit
├── notebooks/              # Scripts de treino e pipelines
├── mlruns/                # Diretório de experimentos do MLflow (não versionar)
├── requirements.txt       # Dependências do projeto
└── README.md

    Importante: mlruns/ e mlflow.db não devem ser versionados.

🧰 Tecnologias Utilizadas

    Python 3.10+

    MLflow (tracking, registry, serving)

    Streamlit (aplicação web)

    scikit-learn (modelos)

    pandas / numpy

    LightGBM / XGBoost (opcional, dependendo do modelo escolhido)

🛠️ Como instalar e rodar
1. Criar ambiente
bash

conda create -n mlops python=3.10 -y
conda activate mlops
pip install -r requirements.txt

2. Configurar MLflow Tracking

Recomenda-se manter o banco fora do repositório:
bash

mkdir -p ~/mlflow_data
export MLFLOW_TRACKING_URI=sqlite:////home/$USER/mlflow_data/mlflow.db

3. Treinar o modelo
bash

python notebooks/ingestao.py
python notebooks/preprocessamento.py
python notebooks/qualidade.py
python notebooks/modelagem.py

4. Abrir a interface do MLflow
bash

- mlflow ui --backend-store-uri sqlite:////home/{user}/MLOps/mlflow.db

Acesse:
👉 http://localhost:5000

5. Servir o modelo
bash

mlflow models serve -m "models:/california-housing-best/latest" --port 5001 --no-conda

6. Rodar a aplicação Streamlit
bash

streamlit run production_app/app.py

🔄 Como trocar o dataset

Se quiser substituir o dataset de exemplo por outro:

    Coloque o novo dataset fora do repositório:

bash

mkdir -p ~/mlflow_data/datasets
mv novo_dataset.csv ~/mlflow_data/datasets/

    Atualize o caminho no script de treino:

python

data_path = "/home/SEU_USUARIO/mlflow_data/datasets/novo_dataset.csv"

    Adicione ao .gitignore:

Código

mlflow.db
mlflow_data/
data/*.csv

    Re-treine o modelo e registre a nova versão.

🔐 Segurança e Boas Práticas

    Nunca versionar:

        mlflow.db

        datasets reais

        arquivos de credenciais

        backups contendo dados sensíveis

    Use pre-commit + detect-secrets para evitar pushes com segredos:

bash

pip install pre-commit detect-secrets
pre-commit install
detect-secrets scan > .secrets.baseline

    Se um segredo for exposto, revogue imediatamente e reescreva o histórico se necessário.

📚 Dataset

O dataset utilizado pode ser substituído por qualquer dataset público de regressão, como:

    Ames Housing

    King County House Sales

    California Housing (scikit-learn)

    Kaggle House Prices

👨‍💻 Autor

Wagner Andrade  
Projeto desenvolvido para estudos de MLOps e deploy de modelos de Machine Learning.
