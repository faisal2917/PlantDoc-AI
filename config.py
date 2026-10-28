"""
Central settings for PlantDoc AI.

Values are read from the .env file when it exists. If a value is missing,
the default written here is used. All paths are built relative to this
file, so the project works no matter where you put the folder.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Folder that contains this file (the project root)
PROJECT_ROOT = Path(__file__).resolve().parent

# Load variables from .env into the environment (does nothing if .env is missing)
load_dotenv(PROJECT_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    """Read a whole number from the environment, with a clear error if it is invalid."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} in .env must be a whole number, got '{value}'")


def _get_float(name: str, default: float) -> float:
    """Read a decimal number from the environment."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} in .env must be a number, got '{value}'")


def _get_path(name: str, default: str) -> Path:
    """Read a path. Relative paths are treated as relative to the project root."""
    raw = os.getenv(name, default).strip()
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ---------- Paths ----------
DATASET_DIR = _get_path("DATASET_DIR", "data/plantvillage")
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLITS_DIR = OUTPUTS_DIR / "splits"  # CSV files listing which image went to which split

MODEL_PATH = MODELS_DIR / "plantdoc_cnn.pt"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# ---------- Images ----------
IMAGE_SIZE = _get_int("IMAGE_SIZE", 128)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Pixels go from [0, 1] to [-1, 1] using (pixel - 0.5) / 0.5.
# Training, evaluation, and prediction all import these, so preprocessing always matches.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# ---------- Model ----------
DROPOUT = _get_float("DROPOUT", 0.3)

# ---------- Training ----------
BATCH_SIZE = _get_int("BATCH_SIZE", 32)
EPOCHS = _get_int("EPOCHS", 40)
LEARNING_RATE = _get_float("LEARNING_RATE", 1e-3)
WEIGHT_DECAY = _get_float("WEIGHT_DECAY", 1e-4)  # small penalty on large weights, reduces overfitting
EARLY_STOPPING_PATIENCE = _get_int("EARLY_STOPPING_PATIENCE", 7)
LR_PATIENCE = _get_int("LR_PATIENCE", 3)
LR_FACTOR = _get_float("LR_FACTOR", 0.5)
SEED = _get_int("SEED", 42)
NUM_WORKERS = _get_int("NUM_WORKERS", 4)

# ---------- Splits ----------
VAL_SPLIT = _get_float("VAL_SPLIT", 0.15)
TEST_SPLIT = _get_float("TEST_SPLIT", 0.15)
# Folder names to skip completely, comma-separated in .env (empty = use everything)
EXCLUDE_DIRS = {name.strip() for name in os.getenv("EXCLUDE_DIRS", "").split(",") if name.strip()}

# Classes with fewer distinct images than this are dropped (too few to split fairly)
MIN_IMAGES_PER_CLASS = 10

if VAL_SPLIT + TEST_SPLIT >= 1.0:
    raise ValueError("VAL_SPLIT + TEST_SPLIT must be less than 1.0 so training has data.")


def ensure_dirs() -> None:
    """Create output folders if they don't exist yet."""
    for folder in (MODELS_DIR, OUTPUTS_DIR, SPLITS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Run `python config.py` to see the settings currently in use
    print(f"Project root:   {PROJECT_ROOT}")
    print(f"Dataset folder: {DATASET_DIR}  (exists: {DATASET_DIR.exists()})")
    print(f"Model file:     {MODEL_PATH}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE} | Batch size: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE} | Weight decay: {WEIGHT_DECAY} | Dropout: {DROPOUT}")
    print(f"Early stop after {EARLY_STOPPING_PATIENCE} bad epochs | Seed: {SEED} | Workers: {NUM_WORKERS}")
    print(f"Splits: train {1 - VAL_SPLIT - TEST_SPLIT:.2f} / val {VAL_SPLIT} / test {TEST_SPLIT}")