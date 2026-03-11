"""
Visualization helpers: draw bounding boxes and plot loss curves.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

from src.config import METRICS_DIR
from src.utils import ensure_dirs


# ── Colour palette (one per letter) ───────────────────────────────────────────
_CMAP = plt.cm.get_cmap("tab20", 26)
COLORS = {chr(ord("A") + i): _CMAP(i) for i in range(26)}


def draw_boxes(image, boxes, labels, scores, save_path=None):
    """Draw predicted boxes on a PIL image and optionally save."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        color = COLORS.get(label, "red")
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5, f"{label} {score:.2f}",
            color="white", fontsize=10,
            bbox=dict(facecolor=color, alpha=0.7, pad=1),
        )
    ax.axis("off")
    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_loss_curve(history_path=None, save_path=None):
    """Plot training (and optionally validation) loss from the saved JSON history."""
    if history_path is None:
        history_path = os.path.join(METRICS_DIR, "train_history.json")
    with open(history_path) as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train loss")

    # Plot validation loss if present
    if "val_loss" in history[0]:
        val_losses = [h["val_loss"] for h in history]
        ax.plot(epochs, val_losses, marker="s", label="Val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Faster R-CNN Training & Validation Loss")
    ax.legend()
    ax.grid(True)
    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig
