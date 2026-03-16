from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "artifacts/model.joblib"
TARGET_COLUMN = "species"

CONFUSION_MATRIX_PATH = "artifacts/confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = "artifacts/classification_report.json"

MLFLOW_EXPERIMENT = "dvc-csv-pipeline"


def main():
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_param("test_path", TEST_PATH)
        mlflow.log_param("model_path", MODEL_PATH)
        mlflow.log_param("target_column", TARGET_COLUMN)

        test_df = pd.read_csv(TEST_PATH)
        model = joblib.load(MODEL_PATH)

        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("test_rows", len(test_df))
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision_weighted", precision)
        mlflow.log_metric("test_recall_weighted", recall)
        mlflow.log_metric("test_f1_weighted", f1)

        labels = sorted(y_test.unique().tolist())
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, xticks_rotation=45)
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH)
        plt.close(fig)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        mlflow.log_artifact(CONFUSION_MATRIX_PATH, artifact_path="evaluation")
        mlflow.log_artifact(CLASSIFICATION_REPORT_PATH, artifact_path="evaluation")

        print("Evaluation completed.")
        print(f"Test accuracy: {acc:.4f}")
        print(f"Test F1: {f1:.4f}")


if __name__ == "__main__":
    main()