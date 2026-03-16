from pathlib import Path
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_DATA_PATH = "data/raw/dataset.csv"
TRAIN_PATH = "data/processed/train.csv"
VALID_PATH = "data/processed/valid.csv"
TEST_PATH = "data/processed/test.csv"
PREPROCESSOR_PATH = "artifacts/preprocessor.joblib"

TARGET_COLUMN = "species"
TEST_SIZE = 0.2
VALID_SIZE = 0.2
RANDOM_STATE = 42

MLFLOW_EXPERIMENT = "dvc-csv-pipeline"


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="preprocess"):
        mlflow.log_param("raw_data_path", RAW_DATA_PATH)
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("valid_size", VALID_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        df = pd.read_csv(RAW_DATA_PATH)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset columns: {list(df.columns)}")

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        mlflow.log_metric("raw_num_rows", len(df))
        mlflow.log_metric("raw_num_features", X.shape[1])
        mlflow.log_metric("num_numeric_features", len(numeric_features))
        mlflow.log_metric("num_categorical_features", len(categorical_features))

        numeric_transformer = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # train / temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=TEST_SIZE + VALID_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # temp -> valid / test
        valid_ratio_in_temp = VALID_SIZE / (TEST_SIZE + VALID_SIZE)
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - valid_ratio_in_temp,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )

        X_train_processed = preprocessor.fit_transform(X_train)
        X_valid_processed = preprocessor.transform(X_valid)
        X_test_processed = preprocessor.transform(X_test)

        feature_names = preprocessor.get_feature_names_out().tolist()

        train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        train_df[TARGET_COLUMN] = y_train.reset_index(drop=True)

        valid_df = pd.DataFrame(X_valid_processed, columns=feature_names)
        valid_df[TARGET_COLUMN] = y_valid.reset_index(drop=True)

        test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        test_df[TARGET_COLUMN] = y_test.reset_index(drop=True)

        train_df.to_csv(TRAIN_PATH, index=False)
        valid_df.to_csv(VALID_PATH, index=False)
        test_df.to_csv(TEST_PATH, index=False)

        joblib.dump(preprocessor, PREPROCESSOR_PATH)

        schema_info = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "processed_feature_count": len(feature_names),
            "processed_feature_names": feature_names,
        }
        mlflow.log_text(json.dumps(schema_info, indent=2), "preprocess_schema.json")

        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("valid_rows", len(valid_df))
        mlflow.log_metric("test_rows", len(test_df))
        mlflow.log_metric("processed_feature_count", len(feature_names))

        mlflow.log_artifact(TRAIN_PATH, artifact_path="processed_data")
        mlflow.log_artifact(VALID_PATH, artifact_path="processed_data")
        mlflow.log_artifact(TEST_PATH, artifact_path="processed_data")
        mlflow.log_artifact(PREPROCESSOR_PATH, artifact_path="preprocessor")

        print("Preprocessing completed.")
        print(f"Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")


if __name__ == "__main__":
    main()