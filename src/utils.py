"""
Utility helpers: seeding, directory creation, checkpoint I/O, collate_fn.
"""
import os
import random
import numpy as np
import torch
from src.config import SEED


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def collate_fn(batch):
    """Custom collate for detection – returns tuple of lists.

    Filters out ``None`` entries that may arise from corrupted images,
    then groups images and targets into separate tuples so that
    Faster R-CNN receives a list[Tensor] and list[dict].
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0)
