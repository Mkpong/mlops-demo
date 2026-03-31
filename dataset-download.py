import os
import mlflow
import yaml
from torchvision import datasets

RAW_DIR = "data/raw"

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    exp_name = params["mlflow"]["experiment_name"]
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="dataset_download"):
        train_dataset = datasets.CIFAR10(root=RAW_DIR, train=True, download=True)
        test_dataset = datasets.CIFAR10(root=RAW_DIR, train=False, download=True)

        mlflow.log_param("raw_dir", RAW_DIR)
        mlflow.log_param("train_size", len(train_dataset))
        mlflow.log_param("test_size", len(test_dataset))

        print("CIFAR-10 다운로드 완료")
        print(f"Train size: {len(train_dataset)}")
        print(f"Test size: {len(test_dataset)}")

if __name__ == "__main__":
    main()
