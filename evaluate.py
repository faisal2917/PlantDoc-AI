"""
Evaluate the trained model on the test set: images it never saw during training
or model selection. Run after training finishes:  python evaluate.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import LeafDataset, get_transforms, prepare_splits
from src.model import load_model
from src.utils import get_device, pretty_class_name


def predict_all(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probabilities for every image and class, true labels)."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            logits = model(images.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def plot_confusion_matrix(cm: np.ndarray, names: list[str], path) -> None:
    """Each row shows how the images of one true class were predicted, as shares of that row."""
    row_totals = cm.sum(axis=1, keepdims=True)
    shares = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals > 0)

    size = max(8, len(names) * 0.4)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        shares, cmap="Greens", vmin=0, vmax=1, square=True,
        xticklabels=names, yticklabels=names,
        cbar_kws={"label": "Share of the true class"}, ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (test set)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    out = config.OUTPUTS_DIR
    device = get_device()

    model, checkpoint = load_model(config.MODEL_PATH, device)
    class_names = checkpoint["class_names"]
    print(f"Loaded best model from epoch {checkpoint['epoch']} (val loss {checkpoint['val_loss']:.4f})")

    # Same saved split as training, so the test set is exactly the held-out images
    df, split_class_names = prepare_splits()
    if split_class_names != class_names:
        raise RuntimeError(
            "The classes in outputs/splits/splits.csv don't match the trained model. "
            "The split was probably rebuilt after training. Run python train.py again."
        )
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    label_to_index = {name: i for i, name in enumerate(class_names)}

    # Preprocessing settings come from the checkpoint, so they match training exactly
    transform = get_transforms(
        train=False,
        image_size=checkpoint["image_size"],
        mean=tuple(checkpoint["norm_mean"]),
        std=tuple(checkpoint["norm_std"]),
    )
    dataset = LeafDataset(test_df["path"].tolist(), [label_to_index[l] for l in test_df["label"]], transform)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    probs, labels = predict_all(model, loader, device)
    preds = probs.argmax(axis=1)

    # Cross-entropy loss computed from the probabilities of the true class
    true_class_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    test_loss = float(-np.log(true_class_probs).mean())
    test_acc = float(accuracy_score(labels, preds))

    names = [pretty_class_name(n) for n in class_names]
    class_ids = list(range(len(class_names)))

    # Precision, recall, F1 for every class
    report_text = classification_report(labels, preds, labels=class_ids, target_names=names, digits=3, zero_division=0)
    report_dict = classification_report(labels, preds, labels=class_ids, target_names=names, output_dict=True, zero_division=0)
    (out / "classification_report.txt").write_text(report_text)
    pd.DataFrame(report_dict).T.to_csv(out / "classification_report.csv")

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=class_ids)
    pd.DataFrame(cm, index=names, columns=names).to_csv(out / "confusion_matrix.csv")
    plot_confusion_matrix(cm, names, out / "confusion_matrix.png")

    # Most commonly confused pairs (true class -> predicted class)
    mistakes = cm.copy()
    np.fill_diagonal(mistakes, 0)
    pairs = [
        (names[i], names[j], int(mistakes[i, j]), mistakes[i, j] / cm[i].sum())
        for i, j in zip(*np.nonzero(mistakes))
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    pd.DataFrame(pairs, columns=["true", "predicted", "count", "share_of_true_class"]).to_csv(
        out / "confused_pairs.csv", index=False
    )

    # Every wrong prediction, most confident mistakes first (worth looking at by hand)
    wrong = np.nonzero(preds != labels)[0]
    pd.DataFrame({
        "path": test_df["path"].to_numpy()[wrong],
        "true": [class_names[labels[i]] for i in wrong],
        "predicted": [class_names[preds[i]] for i in wrong],
        "confidence": probs[wrong, preds[wrong]],
    }).sort_values("confidence", ascending=False).to_csv(out / "wrong_predictions.csv", index=False)

    summary = {
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
        "test_images": int(len(labels)),
        "wrong_predictions": int(len(wrong)),
        "model_epoch": checkpoint["epoch"],
    }
    (out / "test_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + report_text)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f} | Wrong: {len(wrong)} of {len(labels)}")
    print("\nMost confused classes:")
    for true, predicted, count, share in pairs[:10]:
        print(f"  {true}  ->  {predicted}: {count} images ({share:.1%} of that class)")
    print(f"\nReports saved in {out}")


if __name__ == "__main__":
    main()