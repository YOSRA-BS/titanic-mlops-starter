"""Script d'entraînement du modèle Titanic avec MLflow.

Ce module est appelé par MLflow via :

    python -m src.models.train --data-path data/processed ...

Il reprend la logique principale du notebook `mlflow_titanic_poc.ipynb` :
- configuration de MLflow,
- chargement et preprocessing des données,
- split train/test,
- entraînement d'un RandomForestClassifier,
- calcul et log des métriques,
- log et enregistrement du modèle dans MLflow.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data.preprocess import ensure_processed_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement modèle Titanic avec MLflow")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed",
        help="Dossier contenant le dataset pré-traité (ou à créer).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Nombre d'arbres pour le Random Forest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Profondeur maximale des arbres.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed pour la reproductibilité.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Configuration de MLflow comme dans le notebook
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("titanic_survival_prediction")

    processed_dir = Path(args.data_path)

    # Chargement / génération des données pré-traitées
    X, y = ensure_processed_dataset(processed_dir)

    # Split train / test (même logique que le notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )

    run_name = f"RF_n{args.n_estimators}_d{args.max_depth}"

    with mlflow.start_run(run_name=run_name):
        # 1) Log des paramètres
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", 0.2)

        # 2) Création et entraînement du modèle
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # 3) Prédictions
        y_pred = model.predict(X_test)

        # 4) Métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 5) Log des métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # 6) Sauvegarde du modèle
        mlflow.sklearn.log_model(
            model,
            "random_forest_model",
            registered_model_name="titanic_rf_classifier",
        )

        # 7) Quelques tags utiles
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "Titanic")

        run_id = mlflow.active_run().info.run_id
        print("==============================================")
        print("✅ Entraînement terminé avec succès")
        print(f"Run MLflow ID : {run_id}")
        print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"F1-score : {f1:.4f}")
        print("==============================================")


if __name__ == "__main__":
    main(parse_args())

