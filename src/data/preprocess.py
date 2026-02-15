"""Fonctions de chargement et de préparation des données Titanic.

Ce module reprend la logique du notebook `mlflow_titanic_poc.ipynb` :
- chargement du CSV Titanic,
- sélection des features,
- nettoyage (valeurs manquantes, encodage du sexe),
- sauvegarde / rechargement d'un fichier pré-traité.

On suppose par défaut que le fichier brut se trouve dans `data/raw/train.csv`.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
TARGET_COL = "Survived"


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    """Charge les données brutes Titanic depuis un CSV."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Fichier brut introuvable : {raw_path}. "
            "Place le fichier Kaggle `train.csv` dans `data/raw/train.csv`."
        )
    return pd.read_csv(raw_path)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le même preprocessing que dans le notebook."""
    # On ne garde que les colonnes utiles pour ce POC
    data = df[FEATURES + [TARGET_COL]].copy()

    # Remplir les valeurs manquantes pour Age et Fare
    age_median = data["Age"].median()
    fare_median = data["Fare"].median()
    data["Age"].fillna(age_median, inplace=True)
    data["Fare"].fillna(fare_median, inplace=True)

    # Encodage du sexe (male -> 0, female -> 1)
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

    # Par sécurité, on vérifie qu'il ne reste plus de NaN
    if data.isnull().sum().any():
        raise ValueError("Il reste des valeurs manquantes après preprocessing.")

    return data


def ensure_processed_dataset(
    processed_dir: Path, raw_path: Path = Path("data/raw/train.csv")
) -> Tuple[pd.DataFrame, pd.Series]:
    """S'assure qu'un dataset pré-traité existe et le renvoie.

    - Si `train_processed.csv` n'existe pas dans `processed_dir`, on le construit
      à partir du fichier brut.
    - On renvoie X (features) et y (target) prêts pour l'entraînement.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / "train_processed.csv"

    if not processed_file.exists():
        # 1) Charger les données brutes
        df_raw = load_raw_data(raw_path)

        # 2) Preprocessing
        df_processed = preprocess_dataframe(df_raw)

        # 3) Sauvegarde
        processed_file.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(processed_file, index=False)

    df = pd.read_csv(processed_file)
    X = df[FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y

