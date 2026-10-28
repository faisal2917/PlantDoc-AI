"""
Extra detail shown with a prediction in the web app:
photo quality checks, the Grad-CAM overlay, how reliable the model was for each class
on the test set, and a downloadable text report.
"""

from dataclasses import dataclass
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import config
from src.disease_info import SINGLE_CLASS_PLANTS, DiseaseInfo
from src.utils import pretty_class_name

HIGH_CONFIDENCE = 0.90
LOW_CONFIDENCE = 0.60  # below this, the app tells the user the result is uncertain


def confidence_band(prob: float) -> tuple[str, str]:
    """(label, explanation) for a confidence value, matching the bands in the README."""
    if prob >= HIGH_CONFIDENCE:
        return "High", "The image closely matches what the model learned for this class."
    if prob >= LOW_CONFIDENCE:
        return "Medium", "Probably right, but check the second prediction, especially if it's the same plant."
    return "Low", "The model is unsure. The photo may be unclear, or show a plant or problem it doesn't know."


# ---------------------------------------------------------------------------
# Photo quality
# ---------------------------------------------------------------------------

@dataclass
class PhotoCheck:
    name: str
    value: str
    ok: bool
    tip: str = ""


def photo_quality(image: Image.Image) -> list[PhotoCheck]:
    """Simple checks for things that usually make predictions worse."""
    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = gray.mean() / 255

    # Sharpness: variance of edges on a fixed-size copy, so it doesn't depend on the photo's resolution
    small = image.convert("L").resize((256, 256))
    sharpness = float(np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32).var())

    # Share of pixels that look like plant tissue (green, yellow or brown hues with some colour)
    hsv = np.asarray(image.convert("HSV").resize((128, 128)), dtype=np.float32)
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    leafy = ((hue >= 20) & (hue <= 170) & (sat > 0.2) & (val > 0.15)).mean()

    return [
        PhotoCheck("Resolution", f"{width}×{height}", min(width, height) >= config.IMAGE_SIZE,
                   f"Use a photo at least {config.IMAGE_SIZE}px on each side."),
        PhotoCheck("Brightness", f"{brightness:.0%}", bool(0.2 <= brightness <= 0.85),
                   "Too dark." if brightness < 0.2 else "Overexposed; avoid direct glare."),
        PhotoCheck("Sharpness", "Sharp" if sharpness >= 220 else "Blurry", sharpness >= 220,
                   "The photo looks blurry. Hold the camera steady and tap to focus on the leaf."),
        PhotoCheck("Leaf coverage", f"{leafy:.0%}", bool(leafy >= 0.10),
                   "The leaf fills little of the frame. Move closer so one leaf fills most of the photo."),
    ]


# ---------------------------------------------------------------------------
# Grad-CAM overlay
# ---------------------------------------------------------------------------

def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Colour the heatmap and blend it over the photo, at the photo's display size."""
    base = image.convert("RGB")
    base.thumbnail((640, 640))
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(base.size, Image.Resampling.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(cam_img) / 255)[..., :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(base, colored, alpha)


# ---------------------------------------------------------------------------
# Per-class reliability from evaluate.py
# ---------------------------------------------------------------------------

@dataclass
class ClassReliability:
    precision: float      # when the model predicts this class, how often it is right
    recall: float         # of the test images of this class, how many it found
    test_images: int
    mistaken_for: list[tuple[str, int]]  # true classes that were wrongly predicted as this class


def load_class_reliability(class_names: list[str]) -> dict[str, ClassReliability]:
    """
    Read the test-set reports written by evaluate.py. Rows follow the model's class order,
    so they are matched by position. Returns {} if the reports are missing or don't match the model.
    """
    report_path = config.OUTPUTS_DIR / "classification_report.csv"
    matrix_path = config.OUTPUTS_DIR / "confusion_matrix.csv"
    if not (report_path.exists() and matrix_path.exists()):
        return {}

    report = pd.read_csv(report_path, index_col=0).iloc[: len(class_names)]
    matrix = pd.read_csv(matrix_path, index_col=0).to_numpy()
    if len(report) != len(class_names) or matrix.shape != (len(class_names),) * 2:
        return {}

    result = {}
    for i, name in enumerate(class_names):
        column = matrix[:, i]  # true classes predicted as class i
        mistakes = [(class_names[j], int(column[j])) for j in np.argsort(-column) if j != i and column[j] > 0]
        row = report.iloc[i]
        result[name] = ClassReliability(
            precision=float(row["precision"]), recall=float(row["recall"]),
            test_images=int(row["support"]), mistaken_for=mistakes[:3],
        )
    return result


def caveats(class_name: str, prob: float, second_prob: float, same_plant_second: bool) -> list[str]:
    """Plain-language warnings that apply to this particular result."""
    notes = []
    if class_name in SINGLE_CLASS_PLANTS:
        notes.append(
            "The training data has only one class for this plant, so the model learned to recognise the plant, "
            "not the condition. A leaf of this plant will get this label whatever its real condition."
        )
    if prob < LOW_CONFIDENCE:
        notes.append("Confidence is low. Try a sharper photo of a single leaf on a plain background.")
    if second_prob >= 0.15:
        where = "another condition of the same plant" if same_plant_second else "a different plant"
        notes.append(f"The second prediction ({second_prob:.0%}) is {where}; compare the symptoms of both.")
    return notes


# ---------------------------------------------------------------------------
# Downloadable report
# ---------------------------------------------------------------------------

def build_report(file_name: str, results: list[tuple[str, float]], info: DiseaseInfo,
                 checks: list[PhotoCheck], notes: list[str]) -> str:
    """Markdown summary of one diagnosis, for the download button."""
    top_name, top_prob = results[0]
    band, _ = confidence_band(top_prob)
    lines = [
        "# PlantDoc AI diagnosis report",
        "",
        f"- Image: {file_name}",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Plant: {info.plant}",
        f"- Result: {info.condition}",
        f"- Confidence: {top_prob:.1%} ({band})",
        f"- Type: {info.category}" + (f" ({info.pathogen})" if info.pathogen else ""),
        f"- Typical impact: {info.impact}",
        "",
    ]
    if info.summary:
        lines += ["## About", "", info.summary, ""]
    for title, items in (("Symptoms", info.symptoms), ("What makes it spread", info.conditions),
                         ("Treatment", info.treatment), ("Prevention", info.prevention)):
        if items:
            lines += [f"## {title}", "", *[f"- {item}" for item in items], ""]
    lines += ["## Top predictions", "", *[f"{i}. {pretty_class_name(n)}: {p:.1%}" for i, (n, p) in enumerate(results, 1)], ""]
    lines += ["## Photo checks", "", *[f"- {c.name}: {c.value} ({'OK' if c.ok else c.tip})" for c in checks], ""]
    if notes:
        lines += ["## Notes", "", *[f"- {n}" for n in notes], ""]
    lines += [
        "---",
        "Predictions are model estimates based on lab photos from the PlantVillage dataset. "
        "They don't replace a diagnosis from an agricultural expert or plant pathologist. "
        "Check product labels and local regulations before using any pesticide.",
    ]
    return "\n".join(lines)
