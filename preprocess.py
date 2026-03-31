import os
import json
import yaml
import mlflow
from torchvision import datasets

RAW_DIR = "data/raw"
ARTIFACTS_DIR = "artifacts"
PROCESSED_DIR = "data/processed"

# data 관련 설정은 코드에서 관리
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    exp_name = params["mlflow"]["experiment_name"]
    run_name = params["mlflow"].get("run_name", "preprocess")
    class_path = params["artifacts"]["class_path"]

    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=f"{run_name}_preprocess"):
        train_dataset = datasets.CIFAR10(root=RAW_DIR, train=True, download=False)

        with open(class_path, "w", encoding="utf-8") as f:
            json.dump(train_dataset.classes, f, ensure_ascii=False, indent=2)

        info = {
            "image_size": IMAGE_SIZE,
            "normalize_mean": NORMALIZE_MEAN,
            "normalize_std": NORMALIZE_STD,
            "note": "실제 transform은 train/evaluate 단계에서 적용"
        }

        info_path = os.path.join(PROCESSED_DIR, "preprocess_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("normalize_mean", NORMALIZE_MEAN)
        mlflow.log_param("normalize_std", NORMALIZE_STD)
        mlflow.log_artifact(class_path)
        mlflow.log_artifact(info_path)

        print("전처리 설정 기록 완료")
        print(f"Saved: {info_path}")
        print(f"Saved: {class_path}")


if __name__ == "__main__":
    main()
