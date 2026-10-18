import torch
import torch.nn as nn
import torchvision.models as models
from models.dictionary_layer import make_final_head

class FTROJAN(nn.Module):
    def __init__(self, in_channels, num_classes, input_hw=(32, 32)):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(0.4),
        )

        # Infer the flattened size from the real input shape.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_hw[0], input_hw[1])

            was_training = self.features.training
            self.features.eval()  # IMPORTANT: disable BatchNorm stats

            feat = self.features(dummy)
            feature_dim = feat.flatten(1).shape[1]

            self.features.train(was_training)  # restore original state

        self.classifier = make_final_head(feature_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# --------------------------------------------------
# EfficientNet-B0
# --------------------------------------------------
def EFFICIENTNET_B0(in_channels, num_classes):
    model = models.efficientnet_b0(weights=None)
    model.features[0][0] = nn.Conv2d(
        in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
    )

    feature_dim = model.classifier[1].in_features
    model.classifier[1] = make_final_head(feature_dim, num_classes)
    return model

# --------------------------------------------------
# DenseNet121
# --------------------------------------------------
def DENSENET121(in_channels, num_classes):
    model = models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(
        in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    feature_dim = model.classifier.in_features
    model.classifier = make_final_head(feature_dim, num_classes)
    return model

# --------------------------------------------------
# ResNet18
# --------------------------------------------------
def RESNET18(in_channels, num_classes):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    feature_dim = model.fc.in_features
    model.fc = make_final_head(feature_dim, num_classes)
    return model