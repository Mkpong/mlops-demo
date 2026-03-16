import os
import hashlib
from pathlib import Path

import mlflow
import pandas as pd
import requests


DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
OUTPUT_PATH = "data/raw/dataset.csv"
MLFLOW_EXPERIMENT = "dvc-csv-pipeline"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="dataset-download"):
        mlflow.log_param("data_url", DATA_URL)
        mlflow.log_param("output_path", OUTPUT_PATH)

        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()

        with open(OUTPUT_PATH, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(OUTPUT_PATH)

        mlflow.log_metric("num_rows", len(df))
        mlflow.log_metric("num_columns", len(df.columns))

        file_size = os.path.getsize(OUTPUT_PATH)
        mlflow.log_metric("file_size_bytes", file_size)

        file_hash = sha256_file(OUTPUT_PATH)
        mlflow.log_text(file_hash, "raw_dataset_sha256.txt")

        mlflow.log_artifact(OUTPUT_PATH, artifact_path="raw_data")

        print(f"Downloaded dataset to: {OUTPUT_PATH}")
        print(f"Shape: {df.shape}")
        print(f"SHA256: {file_hash}")


if __name__ == "__main__":
    main()