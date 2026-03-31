import json
import yaml
import mlflow
import torch

from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_utils import build_model

# data 설정은 코드에서 관리
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_CLASSES = 10

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    pretrained = params["train"]["pretrained"]
    freeze_features = params["train"]["freeze_features"]

    exp_name = params["mlflow"]["experiment_name"]
    run_name = params["mlflow"].get("run_name", "run")

    model_path = params["artifacts"]["model_path"]
    metrics_path = params["artifacts"]["metrics_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD
        )
    ])

    test_dataset = datasets.CIFAR10(
        root="data/raw",
        train=False,
        download=False,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        freeze_features=freeze_features
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu()

            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=f"{run_name}_evaluate"):
        mlflow.log_param("eval_device", str(device))
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_workers", NUM_WORKERS)
        mlflow.log_param("num_classes", NUM_CLASSES)

        mlflow.log_metric("test_accuracy", acc)

        print(f"Test Accuracy: {acc:.4f}")
        print(report)

        mlflow.log_artifact(model_path)

    output = {
        "test_accuracy": acc,
        "device": str(device),
        "classification_report": report
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"평가 결과 저장 완료: {metrics_path}")


if __name__ == "__main__":
    main()
