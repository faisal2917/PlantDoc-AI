"""
Download the Balanced PlantVillage dataset from Kaggle and copy it into
the folder set by DATASET_DIR (data/plantvillage by default).

Run once:  python download_data.py
"""

import shutil
from pathlib import Path

import kagglehub

from config import DATASET_DIR

KAGGLE_DATASET = "kanivan2203/balanced-plantvillage-dataset"


def show_structure(folder: Path, max_depth: int = 3, max_items: int = 5, depth: int = 0) -> None:
    """Print the first few folders at each level so we can see how the dataset is organized."""
    if depth >= max_depth:
        return
    items = sorted(p for p in folder.iterdir() if not p.name.startswith("."))
    for item in items[:max_items]:
        indent = "    " * depth
        if item.is_dir():
            count = sum(1 for _ in item.iterdir())
            print(f"{indent}{item.name}/  ({count} items)")
            show_structure(item, max_depth, max_items, depth + 1)
        else:
            print(f"{indent}{item.name}")
    if len(items) > max_items:
        print(f"{'    ' * depth}... and {len(items) - max_items} more")


def main() -> None:
    print("Downloading from Kaggle. The first download can take several minutes...")
    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Kaggle saved the files here: {cache_path}")

    # Only copy if the project folder is empty, so running this twice doesn't duplicate work
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"{DATASET_DIR} already contains files, skipping the copy.")
    else:
        print(f"Copying into {DATASET_DIR} ...")
        shutil.copytree(cache_path, DATASET_DIR, dirs_exist_ok=True)
        print("Copy finished.")

    print("\nDataset structure:")
    show_structure(DATASET_DIR)


if __name__ == "__main__":
    main()