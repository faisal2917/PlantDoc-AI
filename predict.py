"""
Predict the disease for one leaf image.

  python predict.py path/to/leaf.jpg
  python predict.py --random-test      (random image from the test set, shows the true class too)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

import config
from src.dataset import get_transforms
from src.model import load_model
from src.utils import get_device, pretty_class_name


class Predictor:
    """Loads the model once, then predicts as many images as you like."""

    def __init__(self, model_path: Path = config.MODEL_PATH, device: torch.device | None = None):
        self.device = device or get_device()
        self.model, checkpoint = load_model(model_path, self.device)
        self.class_names: list[str] = checkpoint["class_names"]
        # Same preprocessing as validation/testing, using the values saved at training time
        self.transform = get_transforms(
            train=False,
            image_size=checkpoint["image_size"],
            mean=tuple(checkpoint["norm_mean"]),
            std=tuple(checkpoint["norm_std"]),
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        """Returns the top_k (class_name, probability) pairs, most likely first."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)  # add batch dimension
        probs = torch.softmax(self.model(tensor), dim=1)[0].cpu()
        top_probs, top_ids = probs.topk(min(top_k, len(self.class_names)))
        return [(self.class_names[int(i)], float(p)) for p, i in zip(top_probs, top_ids)]


def open_image(path: Path) -> Image.Image:
    """Open an image file with clear errors for missing or unsupported files."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in config.VALID_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Use JPG, JPEG or PNG.")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"{path} is not a readable image.")


def random_test_image() -> tuple[Path, str]:
    splits_csv = config.SPLITS_DIR / "splits.csv"
    if not splits_csv.exists():
        raise FileNotFoundError("No saved split found. Run python train.py first.")
    test = pd.read_csv(splits_csv, dtype=str, keep_default_na=False).query("split == 'test'")
    row = test.sample(1).iloc[0]
    return config.DATASET_DIR / row["path"], row["label"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict plant disease from a leaf image.")
    parser.add_argument("image", nargs="?", help="path to a JPG or PNG leaf image")
    parser.add_argument("--random-test", action="store_true", help="use a random test-set image")
    parser.add_argument("--top", type=int, default=3, help="how many predictions to show")
    args = parser.parse_args()

    try:
        if args.random_test:
            path, true_label = random_test_image()
        elif args.image:
            path, true_label = Path(args.image), None
        else:
            parser.error("give an image path or use --random-test")

        image = open_image(path)
        predictor = Predictor()
        results = predictor.predict(image, top_k=args.top)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    print(f"\nImage: {path}")
    if true_label:
        print(f"True class: {pretty_class_name(true_label)}")
    print("\nPredictions:")
    for rank, (name, prob) in enumerate(results, start=1):
        print(f"  {rank}. {pretty_class_name(name):45s} {prob:6.1%}")


if __name__ == "__main__":
    main()