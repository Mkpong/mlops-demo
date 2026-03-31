import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 10, pretrained: bool = True, freeze_features: bool = False):
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
