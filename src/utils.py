"""
Small helpers shared across the project:
random seeds, device selection, class-name files, and training plots.
"""

import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw plots straight to files, no popup windows
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Fix the random number generators so splits, weight initialization,
    and augmentation come out the same each run.

    Note: some GPU operations on MPS are not perfectly deterministic,
    so two training runs can still differ slightly in the final decimals.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Use the Apple GPU (MPS) if available, then NVIDIA (CUDA), then CPU."""
    if torch.backends.mps.is_available():
        print("Device: Apple GPU (MPS)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print(f"Device: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")

    if torch.backends.mps.is_built():
        print("MPS is built into PyTorch but not available. Check that macOS is 12.3 or newer.")
    print("Device: CPU (training will be slow)")
    return torch.device("cpu")


def save_class_names(class_names: list[str], path: Path) -> None:
    """
    Save class names in label order. Index 0 in this list is label 0 in the model,
    so prediction must always use this exact file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)


def load_class_names(path: Path) -> list[str]:
    """Load the class names saved during training."""
    if not path.exists():
        raise FileNotFoundError(
            f"Class names file not found at {path}. Train the model first with: python train.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Plants that have a single class in PlantVillage, so their folder name is only the plant.
# The condition comes from the original PlantVillage dataset.
DISPLAY_NAMES = {
    "Blueberry": "Blueberry: healthy",
    "Orange": "Orange: Citrus greening (Huanglongbing)",
    "Raspberry": "Raspberry: healthy",
    "Soybean": "Soybean: healthy",
    "Squash": "Squash: Powdery mildew",
}


def pretty_class_name(name: str) -> str:
    """
    Turn a folder name into readable text for display.
    Example: 'Tomato___Late_blight' -> 'Tomato: Late blight'
    """
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    return name.replace("___", ": ").replace("_", " ").strip()


def count_parameters(model: torch.nn.Module) -> int:
    """Number of weights the model will learn."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_history(history: dict[str, list[float]], save_path: Path) -> None:
    """
    Save loss and accuracy curves for training and validation.

    `history` needs these keys, each a list with one value per epoch:
    train_loss, val_loss, train_acc, val_acc
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"])) + 1  # epoch with lowest validation loss

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, history["train_loss"], label="Train")
    ax_loss.plot(epochs, history["val_loss"], label="Validation")
    ax_loss.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="Train")
    ax_acc.plot(epochs, history["val_acc"], label="Validation")
    ax_acc.axvline(best_epoch, color="gray", linestyle="--")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    # Quick self-test: python -m src.utils
    import tempfile

    set_seed(42)
    device = get_device()

    x = torch.randn(256, 256, device=device)
    y = x @ x
    print(f"Matrix multiply on {device}: OK, result shape {tuple(y.shape)}")

    with tempfile.TemporaryDirectory() as tmp:
        test_path = Path(tmp) / "classes.json"
        save_class_names(["Tomato___healthy", "Tomato___Late_blight"], test_path)
        print("Class names saved and loaded:", load_class_names(test_path))

    print("Pretty name:", pretty_class_name("Tomato___Late_blight"))