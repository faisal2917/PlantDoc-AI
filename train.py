"""
Train PlantCNN from scratch.
Run: python train.py
"""

import time

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

import config
from src.dataset import build_dataloaders
from src.model import build_model, save_checkpoint
from src.utils import count_parameters, get_device, plot_history, save_class_names, set_seed


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, float]:
    """
    One pass over a dataset. With an optimizer it trains; without one it only measures.
    Returns (average loss, accuracy).
    """
    training = optimizer is not None
    model.train(training)  # train mode: dropout on. eval mode: dropout off.
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):  # skip gradient bookkeeping during validation
        for images, labels in tqdm(loader, desc="  train" if training else "  val", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad()  # clear gradients from the previous batch
                loss.backward()        # compute how each weight affected the loss
                optimizer.step()       # nudge the weights to reduce the loss

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += batch_size

    return total_loss / seen, correct / seen


def main() -> None:
    config.ensure_dirs()
    set_seed(config.SEED)
    device = get_device()

    # Loads images and creates train/val/test splits (the test set is not touched here)
    loaders, class_names = build_dataloaders(seed=config.SEED)
    save_class_names(class_names, config.CLASS_NAMES_PATH)

    model = build_model(len(class_names), config.DROPOUT).to(device)
    print(f"Classes: {len(class_names)} | Trainable parameters: {count_parameters(model):,}")

    # CrossEntropyLoss works with integer labels (the PyTorch version of sparse categorical cross-entropy)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    # Halve the learning rate when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.LR_FACTOR, patience=config.LR_PATIENCE
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    bad_epochs = 0

    print(f"\nTraining for up to {config.EPOCHS} epochs. Press Ctrl+C to stop early; the best model is kept.\n")
    try:
        for epoch in range(1, config.EPOCHS + 1):
            start = time.time()
            train_loss, train_acc = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
            val_loss, val_acc = run_epoch(model, loaders["val"], loss_fn, device)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc, current_lr)):
                history[key].append(value)

            print(
                f"Epoch {epoch:2d}/{config.EPOCHS} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
                f"lr {current_lr:.1e} | {time.time() - start:.0f}s"
            )

            # Checkpointing: keep only the model with the lowest validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                save_checkpoint(config.MODEL_PATH, model, class_names, epoch, val_loss)
                print("  -> new best model saved")
            else:
                bad_epochs += 1
                if bad_epochs >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\nNo improvement for {bad_epochs} epochs. Stopping early.")
                    break

            # Redraw the curves every epoch so you can watch progress in outputs/
            plot_history(history, config.OUTPUTS_DIR / "training_curves.png")

    except KeyboardInterrupt:
        print("\nTraining stopped by you.")

    if history["train_loss"]:
        plot_history(history, config.OUTPUTS_DIR / "training_curves.png")
        pd.DataFrame(history, index=range(1, len(history["train_loss"]) + 1)).to_csv(
            config.OUTPUTS_DIR / "training_history.csv", index_label="epoch"
        )

    if best_val_loss == float("inf"):
        print("No epoch finished, so no model was saved.")
        return

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to:  {config.MODEL_PATH}")
    print(f"Curves saved to: {config.OUTPUTS_DIR / 'training_curves.png'}")
    print("Next step: python evaluate.py")


# Required on macOS: DataLoader workers start fresh Python processes that re-import this file,
# and this check stops them from starting training again.
if __name__ == "__main__":
    main()