from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


PARAMS_PATH = "params.yaml"


def load_params(params_path: str) -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    params = load_params(PARAMS_PATH)

    train_path = params["data"]["train_path"]
    valid_path = params["data"]["valid_path"]
    target_column = params["data"]["target_column"]

    model_type = params["model"]["type"]
    model_path = params["model"]["model_path"]
    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]
    random_state = params["model"]["random_state"]

    experiment_name = params["mlflow"]["experiment_name"]
    run_name = params["mlflow"]["run_name"]

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("train_path", train_path)
        mlflow.log_param("valid_path", valid_path)
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_valid = valid_df.drop(columns=[target_column])
        y_valid = valid_df[target_column]

        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("valid_rows", len(valid_df))
        mlflow.log_metric("num_features", X_train.shape[1])

        if model_type != "RandomForestClassifier":
            raise ValueError(f"Unsupported model type: {model_type}")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)

        train_acc = accuracy_score(y_train, train_pred)
        valid_acc = accuracy_score(y_valid, valid_pred)

        train_f1 = f1_score(y_train, train_pred, average="weighted")
        valid_f1 = f1_score(y_valid, valid_pred, average="weighted")

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("valid_accuracy", valid_acc)
        mlflow.log_metric("train_f1_weighted", train_f1)
        mlflow.log_metric("valid_f1_weighted", valid_f1)

        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.sklearn.log_model(model, artifact_path="mlflow_model")

        print("Training completed.")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Valid accuracy: {valid_acc:.4f}")


if __name__ == "__main__":
    main()