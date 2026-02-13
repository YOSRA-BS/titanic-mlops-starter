"""Script d'évaluation d'un modèle MLflow sur le dataset Titanic.

Ce module est appelé par MLflow via :

    python -m src.models.evaluate --model-uri <uri> --data-path data/processed

Il recharge un modèle depuis MLflow (run ou registry) et calcule
les mêmes métriques que dans le notebook.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data.preprocess import ensure_processed_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évaluation d'un modèle MLflow sur Titanic")
    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help=(
            "URI du modèle MLflow à charger, par ex. "
            "`runs:/<run_id>/random_forest_model` ou "
            "`models:/titanic_rf_classifier/1`."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed",
        help="Dossier contenant le dataset pré-traité (ou à créer).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed pour la reproductibilité (split train/test).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("titanic_survival_prediction_evaluation")

    processed_dir = Path(args.data_path)
    X, y = ensure_processed_dataset(processed_dir)

    # On refait un split train/test pour évaluer sur la même logique que l'entraînement
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )

    with mlflow.start_run(run_name="evaluation"):
        # Log pour traçabilité de ce qui est évalué
        mlflow.log_param("model_uri", args.model_uri)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", args.random_state)

        # Chargement du modèle depuis MLflow
        model = mlflow.sklearn.load_model(args.model_uri)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print("==============================================")
        print("✅ Évaluation terminée")
        print(f"Model URI : {args.model_uri}")
        print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print("==============================================")


if __name__ == "__main__":
    main(parse_args())

