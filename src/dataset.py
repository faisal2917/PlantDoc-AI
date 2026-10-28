"""
Everything to do with the data.

1. Find every image and work out its class from the folder it's in
2. Skip files that can't be opened
3. Detect exact and near-duplicate images so copies never end up in different splits
4. Split into train / val / test (stratified by class) and save the split to a CSV,
   so train.py and evaluate.py always use the same test set
5. Build PyTorch DataLoaders (augmentation for training images only)

Check it without training:  python -m src.dataset
"""

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import config

# Folder names that mean "a split", not "a class"
SPLIT_FOLDER_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing"}
# Per-plant folders in this dataset look like 'stage3_apple'
STAGE_PLANT_FOLDER = re.compile(r"^stage\d+_(.+)$", re.IGNORECASE)
SPLITS_CSV = config.SPLITS_DIR / "splits.csv"


# ---------------------------------------------------------------------------
# 1-2. Finding and checking images
# ---------------------------------------------------------------------------

def _dhash(image: Image.Image) -> str:
    """
    Perceptual "difference hash". Near-identical images (resized, re-saved,
    recompressed) get the same 16-character code; different leaves almost never do.
    """
    small = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.int16)
    brighter = pixels[:, 1:] > pixels[:, :-1]  # is each pixel brighter than its left neighbour?
    return np.packbits(brighter.flatten()).tobytes().hex()


def _group_folder(image_path: Path) -> str:
    """
    The folder above the class folder, skipping split folders like train/ and test/.
    Example: .../stage3_apple/train/healthy/leaf.jpg -> 'stage3_apple'
    """
    for folder in image_path.parents[1:]:
        if folder == config.DATASET_DIR:
            return ""
        if folder.name.lower() not in SPLIT_FOLDER_NAMES:
            return folder.name
    return ""


def _find_image_files() -> list[Path]:
    root = config.DATASET_DIR
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\n"
            "Run 'python download_data.py' or set DATASET_DIR in .env"
        )
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in config.VALID_EXTENSIONS:
            continue
        if any(part in config.EXCLUDE_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No .jpg/.jpeg/.png images found under {root}")
    return sorted(files)


def scan_dataset() -> pd.DataFrame:
    """Open every image once: make sure it isn't corrupted and compute its hashes."""
    rows, bad_files = [], []

    for path in tqdm(_find_image_files(), desc="Checking images"):
        if path.parent == config.DATASET_DIR or path.parent.name.lower() in SPLIT_FOLDER_NAMES:
            bad_files.append((path, "not inside a class folder"))
            continue
        try:
            data = path.read_bytes()
            image = Image.open(io.BytesIO(data))
            image.draft("L", (64, 64))  # lets JPEGs decode at a smaller size, much faster
            image.load()  # fails here if the file is corrupted or cut off
            rows.append({
                "path": path.relative_to(config.DATASET_DIR).as_posix(),
                "class_folder": path.parent.name,
                "group": _group_folder(path),
                "md5": hashlib.md5(data).hexdigest(),  # identical files share this
                "dhash": _dhash(image),                 # near-identical images share this
            })
        except Exception as error:  # a broken file is skipped, not fatal
            bad_files.append((path, f"{type(error).__name__}: {error}"))

    if bad_files:
        print(f"\nSkipped {len(bad_files)} unusable files. First few:")
        for path, reason in bad_files[:10]:
            print(f"  {path.relative_to(config.DATASET_DIR)}  ({reason})")
    if not rows:
        raise RuntimeError("No usable images found. Check the dataset folder.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Labels and duplicates
# ---------------------------------------------------------------------------

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    The class is the folder name. Inside per-plant folders like 'stage3_apple',
    the plant name is added when missing: 'healthy' -> 'Apple___healthy'.
    """
    labels = []
    for class_folder, group in zip(df["class_folder"], df["group"]):
        match = STAGE_PLANT_FOLDER.match(group)
        if match:
            plant = match.group(1)
            plant = plant[:1].upper() + plant[1:]
            if not class_folder.lower().startswith(plant.lower()):
                labels.append(f"{plant}___{class_folder}")
                continue
        labels.append(class_folder)
    return df.assign(label=labels)


def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Same-looking image filed under two different labels: keep the more specific
       label, meaning the one with fewer images (a disease class is smaller than its plant).
    2. Identical files: keep one copy.
    """
    label_sizes = df.groupby("label")["md5"].nunique()
    df = df.assign(label_size=df["label"].map(label_sizes))

    conflicts = int((df.groupby("dhash")["label"].nunique() > 1).sum())
    chosen = df.sort_values(["label_size", "label"]).groupby("dhash")["label"].first()
    df = df[df["label"] == df["dhash"].map(chosen)]

    before = len(df)
    df = df.drop_duplicates("md5").drop(columns="label_size")
    print(f"Duplicate images found under two different labels: {conflicts:,}")
    print(f"Exact duplicate files removed: {before - len(df):,}")
    return df.reset_index(drop=True)


def drop_small_classes(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby("label")["dhash"].nunique()
    small = sizes[sizes < config.MIN_IMAGES_PER_CLASS]
    if len(small):
        print(f"Dropping {len(small)} classes with under {config.MIN_IMAGES_PER_CLASS} images: {list(small.index)}")
        df = df[~df["label"].isin(small.index)]
    return df.reset_index(drop=True)


def warn_overlapping_labels(class_names: list[str]) -> None:
    """Warn if a class like 'Apple' sits next to 'Apple___scab' (plant-level mixed with disease-level)."""
    for name in class_names:
        finer = [o for o in class_names if o != name and o.lower().startswith(name.lower() + "_")]
        if finer:
            print(f"WARNING: class '{name}' looks like a broader version of {finer[:3]}")


# ---------------------------------------------------------------------------
# 4. Splitting
# ---------------------------------------------------------------------------

def split_dataset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Split groups of near-identical images (not single images), so look-alikes
    always land in the same split. Stratified so every class keeps its proportions.
    """
    group_labels = df.groupby("dhash")["label"].first()
    # Plain NumPy arrays: newer pandas text columns don't work with train_test_split's indexing
    group_ids = group_labels.index.to_numpy(dtype=object)
    labels = group_labels.to_numpy(dtype=object)

    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        group_ids, labels, test_size=config.TEST_SPLIT, stratify=labels, random_state=seed,
    )
    val_fraction = config.VAL_SPLIT / (1 - config.TEST_SPLIT)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_fraction, stratify=train_val_labels, random_state=seed,
    )

    split_of = {
        **{g: "train" for g in train_ids},
        **{g: "val" for g in val_ids},
        **{g: "test" for g in test_ids},
    }
    return df.assign(split=df["dhash"].map(split_of))


def print_summary(df: pd.DataFrame) -> None:
    table = pd.crosstab(df["label"], df["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    print(f"\n{len(table)} classes, {len(df):,} images\n")
    print(table.to_string())
    print(f"\nTotal: train {table['train'].sum():,} | val {table['val'].sum():,} | test {table['test'].sum():,}")


def prepare_splits(seed: int = config.SEED) -> tuple[pd.DataFrame, list[str]]:
    """Build the split once and save it; later runs reuse the same file."""
    if SPLITS_CSV.exists():
        print(f"Using the saved split: {SPLITS_CSV}\n(delete the outputs/splits folder to rebuild it)")
        df = pd.read_csv(SPLITS_CSV, dtype=str, keep_default_na=False)
        missing = [p for p in df["path"] if not (config.DATASET_DIR / p).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images in {SPLITS_CSV} no longer exist (e.g. {missing[0]}). "
                "Delete outputs/splits/ and run again."
            )
    else:
        print(f"Scanning {config.DATASET_DIR}\nThis happens once and can take a few minutes.")
        df = scan_dataset()
        df = assign_labels(df)
        df = drop_broad_labels(df) 
        df = resolve_duplicates(df)
        df = drop_small_classes(df)
        df = split_dataset(df, seed)
        config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(SPLITS_CSV, index=False)
        print(f"Split saved to {SPLITS_CSV}")

    class_names = sorted(df["label"].unique())  # sorted, so label numbers are stable
    warn_overlapping_labels(class_names)
    print_summary(df)
    return df, class_names


# ---------------------------------------------------------------------------
# 5. PyTorch Dataset and DataLoaders
# ---------------------------------------------------------------------------

def get_transforms(
    train: bool,
    image_size: int = config.IMAGE_SIZE,
    mean: tuple = config.NORM_MEAN,
    std: tuple = config.NORM_STD,
) -> transforms.Compose:
    """
    Training: random crops, flips, rotation and colour changes, so the model
    sees a slightly different version of each leaf every epoch.
    Val/test/prediction: only resize and normalize, so results are repeatable.
    """
    to_tensor = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            *to_tensor,
        ])
    return transforms.Compose([transforms.Resize((image_size, image_size)), *to_tensor])


class LeafDataset(Dataset):
    """Loads one image at a time from disk, so the whole dataset never sits in memory."""

    def __init__(self, paths: list[str], labels: list[int], transform: transforms.Compose):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(config.DATASET_DIR / self.paths[index]) as image:
            image = image.convert("RGB")  # handles grayscale or transparent PNGs
        return self.transform(image), self.labels[index]


def build_dataloaders(
    seed: int = config.SEED, batch_size: int = config.BATCH_SIZE
) -> tuple[dict[str, DataLoader], list[str]]:
    """Returns ({'train': ..., 'val': ..., 'test': ...}, class_names)."""
    df, class_names = prepare_splits(seed)
    label_to_index = {name: i for i, name in enumerate(class_names)}

    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        dataset = LeafDataset(
            paths=part["path"].tolist(),
            labels=[label_to_index[label] for label in part["label"]],
            transform=get_transforms(train=(split == "train")),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=config.NUM_WORKERS,
            persistent_workers=config.NUM_WORKERS > 0,
            generator=torch.Generator().manual_seed(seed) if split == "train" else None,
        )
    return loaders, class_names

def drop_broad_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove plant-only classes (like 'Apple') when disease-level classes for the same
    plant exist ('Apple___Scab', 'Apple___healthy'). A plant-only folder mixes healthy
    and diseased leaves under one name, so the model can't learn it.
    Plants that only have a plant-level folder (like 'Blueberry') are kept.
    """
    labels = set(df["label"])
    broad = sorted(
        name for name in labels
        if any(other.lower().startswith(name.lower() + "___") for other in labels)
    )
    if broad:
        print(f"Removing plant-only classes that overlap with disease classes: {broad}")
        df = df[~df["label"].isin(broad)]
    return df.reset_index(drop=True)

if __name__ == "__main__":
    # Check the data without training
    loaders, class_names = build_dataloaders()
    images, labels = next(iter(loaders["train"]))
    print(f"\nOne training batch: images {tuple(images.shape)}, labels {tuple(labels.shape)}")
    print(f"Pixel values after normalization: {images.min():.2f} to {images.max():.2f}")