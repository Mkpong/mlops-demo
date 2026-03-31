import os
import json
import random
import yaml
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_utils import build_model

# data 관련 설정은 코드에서 관리
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_CLASSES = 10

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    epochs = params["train"]["epochs"]
    lr = params["train"]["lr"]
    optimizer_name = params["train"]["optimizer"]
    weight_decay = params["train"]["weight_decay"]
    pretrained = params["train"]["pretrained"]
    freeze_features = params["train"]["freeze_features"]
    seed = params["train"]["seed"]

    exp_name = params["mlflow"]["experiment_name"]
    run_name = params["mlflow"]["run_name"]

    model_path = params["artifacts"]["model_path"]
    metrics_path = params["artifacts"]["metrics_path"]

    os.makedirs("artifacts", exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="data/raw",
        train=True,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        freeze_features=freeze_features
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        optimizer_name,
        filter(lambda p: p.requires_grad, model.parameters()),
        lr,
        weight_decay
    )

    mlflow.set_experiment(exp_name)

    best_loss = float("inf")
    history = []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("device", str(device))
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_workers", NUM_WORKERS)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("pretrained", pretrained)
        mlflow.log_param("freeze_features", freeze_features)
        mlflow.log_param("seed", seed)

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_count = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_count += y.size(0)

            epoch_loss = total_loss / total_count
            epoch_acc = total_correct / total_count

            history.append({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc
            })

            mlflow.log_metric("train_loss", epoch_loss, step=epoch + 1)
            mlflow.log_metric("train_acc", epoch_acc, step=epoch + 1)

            print(f"[Epoch {epoch+1}/{epochs}] loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), model_path)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(metrics_path)

        print(f"학습 완료. Best model saved to {model_path}")
        print(f"사용 장치: {device}")


if __name__ == "__main__":
    main()
