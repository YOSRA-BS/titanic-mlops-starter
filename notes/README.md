# Titanic MLOps Pipeline

Pipeline MLOps complet avec CI/CD/CT pour la prédiction de survie du Titanic.

## Architecture

```
┌─────────────┐
│   GitHub    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  CI: Tests + Validation         │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Training + MLflow Tracking     │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  CD: Deploy to Production       │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  CT: Monitoring + Retrain       │
└─────────────────────────────────┘
```

## Installation

```bash
# Cloner le repo
git clone https://github.com/YOSRA-BS/titanic-mlops.git
cd titanic-mlops

# Créer l'environnement
conda env create -f conda.yaml
conda activate titanic-mlops

# Installer les dépendances
pip install -r requirements.txt
```

## Usage

### 1. Lancer MLflow Tracking
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
```

### 2. Entraîner le modèle
```bash
python -m src.models.train --data-path data/processed
```

### 3. Déployer avec Docker
```bash
cd deployment
docker-compose up -d
```

### 4. Tester l'API
```bash
curl -X POST http://localhost:5001/predict \
  -H 'Content-Type: application/json' \
  -d '{"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"}'
```

## Structure du Projet

```
titanic-mlops/
├── .github/workflows/    # CI/CD/CT workflows
├── src/                  # Code source
│   ├── data/            # Data processing
│   ├── models/          # Training & evaluation
│   ├── monitoring/      # Drift & performance
│   └── utils/           # Utilities
├── tests/               # Tests unitaires
├── deployment/          # Docker & K8s
├── data/                # Données
└── notebooks/           # Explorations
```

## CI/CD/CT

- **CI**: Tests automatiques sur chaque push
- **CD**: Déploiement automatique après validation
- **CT**: Réentraînement quotidien si drift détecté

## Monitoring

Dashboard disponible sur: http://localhost:8501
```bash
streamlit run deployment/dashboard.py
```
