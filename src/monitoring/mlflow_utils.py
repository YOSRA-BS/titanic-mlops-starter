"""Utilitaires de monitoring autour de MLflow.

Ce module regroupe des fonctions inspirées des dernières cellules du
notebook `mlflow_titanic_poc.ipynb` :
- exploration du dossier `mlruns`,
- rechargement d'un modèle depuis le registry MLflow.
"""

import os
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn


def print_mlruns_structure(root: Path = Path("mlruns")) -> None:
    """Affiche une vue arborescente simplifiée du dossier `mlruns`."""
    print(f"📁 Structure du dossier {root} :\n")

    if not root.exists():
        print("❌ Le dossier mlruns n'existe pas encore.")
        print("   Exécute au moins une expérimentation MLflow d'abord.")
        return

    for current_root, dirs, files in os.walk(root):
        level = Path(current_root).relative_to(root).parts
        depth = len(level)
        indent = "  " * depth
        print(f"{indent}{Path(current_root).name}/")
        sub_indent = "  " * (depth + 1)
        for file in files[:5]:
            print(f"{sub_indent}{file}")
        if len(files) > 5:
            print(f"{sub_indent}... et {len(files) - 5} autres fichiers")


def load_registered_model(
    model_name: str = "titanic_rf_classifier",
    stage_or_version: Optional[str] = "1",
):
    """Recharge un modèle depuis le MLflow Model Registry.

    Parameters
    ----------
    model_name:
        Nom du modèle tel qu'enregistré dans MLflow.
    stage_or_version:
        Soit un numéro de version (ex: \"1\"), soit un stage (\"Staging\", \"Production\").

    Returns
    -------
    Any
        Le modèle rechargé (souvent un objet scikit-learn).
    """
    mlflow.set_tracking_uri("file:./mlruns")

    # On supporte deux formes :
    # - version numérique: models:/name/1
    # - stage: models:/name@Staging
    if stage_or_version is None:
        uri = f"models:/{model_name}"
    elif stage_or_version.isdigit():
        uri = f"models:/{model_name}/{stage_or_version}"
    else:
        uri = f"models:/{model_name}@{stage_or_version}"

    print(f"💾 Chargement du modèle depuis : {uri}")
    try:
        model = mlflow.sklearn.load_model(uri)
    except Exception as exc:  # pragma: no cover - simple helper
        print(f"❌ Erreur lors du chargement du modèle : {exc}")
        raise

    print("✅ Modèle rechargé avec succès.")
    return model

