import os
import json
import yaml
import mlflow
from torchvision import datasets

RAW_DIR = "data/raw"
ARTIFACTS_DIR = "artifacts"
PROCESSED_DIR = "data/processed"

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    image_size = params["data"]["image_size"]
    exp_name = params["mlflow"]["experiment_name"]
    class_path = params["artifacts"]["class_path"]

    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="preprocess"):
        train_dataset = datasets.CIFAR10(root=RAW_DIR, train=True, download=False)

        with open(class_path, "w", encoding="utf-8") as f:
            json.dump(train_dataset.classes, f, ensure_ascii=False, indent=2)

        info = {
            "image_size": image_size,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "note": "실제 transform은 train/evaluate 단계에서 적용"
        }

        info_path = os.path.join(PROCESSED_DIR, "preprocess_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        mlflow.log_param("image_size", image_size)
        mlflow.log_artifact(class_path)
        mlflow.log_artifact(info_path)

        print("전처리 설정 기록 완료")
        print(f"Saved: {info_path}")
        print(f"Saved: {class_path}")

if __name__ == "__main__":
    main()
