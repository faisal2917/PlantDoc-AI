"""
PlantCNN: a convolutional neural network trained from scratch (no pretrained weights).
Also contains helpers to save and load trained models.
"""

from pathlib import Path

import torch
from torch import nn

import config


class ConvBlock(nn.Module):
    """Two 3x3 convolutions (each with BatchNorm and ReLU), then 2x2 max pooling to halve the size."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # keeps values in a stable range, helps training from scratch
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PlantCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        # Feature extractor. Comments show the image size for 128x128 input.
        self.features = nn.Sequential(
            ConvBlock(3, 32),     # 128 -> 64
            ConvBlock(32, 64),    # 64 -> 32
            ConvBlock(64, 128),   # 32 -> 16
            ConvBlock(128, 256),  # 16 -> 8
            ConvBlock(256, 256),  # 8 -> 4
        )
        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pooling: 256 x 4 x 4 -> 256 x 1 x 1
            nn.Flatten(),             # -> 256 numbers per image
            nn.Dropout(dropout),      # randomly zero some values during training to reduce overfitting
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw scores ("logits"), not probabilities.
        # CrossEntropyLoss applies softmax internally during training,
        # and predict.py applies softmax to turn scores into confidence percentages.
        return self.head(self.features(x))


def build_model(num_classes: int, dropout: float = config.DROPOUT) -> PlantCNN:
    return PlantCNN(num_classes=num_classes, dropout=dropout)


def save_checkpoint(path: Path, model: nn.Module, class_names: list[str], epoch: int, val_loss: float) -> None:
    """
    Save the weights together with everything needed to use them later.
    predict.py reads image size and normalization from here, so preprocessing always matches training.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "image_size": config.IMAGE_SIZE,
            "dropout": config.DROPOUT,
            "norm_mean": config.NORM_MEAN,
            "norm_std": config.NORM_STD,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> tuple[PlantCNN, dict]:
    """Load a trained model in evaluation mode. Returns (model, checkpoint_info)."""
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Train one first with: python train.py")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(len(checkpoint["class_names"]), checkpoint["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()  # eval() turns off dropout and uses fixed BatchNorm statistics
    return model, checkpoint


if __name__ == "__main__":
    # Self-test: python -m src.model
    from src.utils import count_parameters, get_device

    device = get_device()
    model = build_model(num_classes=10).to(device)
    fake_batch = torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    output = model(fake_batch)
    print(f"Input shape:  {tuple(fake_batch.shape)}")
    print(f"Output shape: {tuple(output.shape)}  (4 images, 10 class scores each)")
    print(f"Trainable parameters: {count_parameters(model):,}")