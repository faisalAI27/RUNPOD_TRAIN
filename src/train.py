"""
Phase 6 — Training Pipeline
============================
Full training loop with:
    • per-epoch training + validation loss
    • best-model checkpoint (lowest validation loss) + latest checkpoint
    • JSON history & auto-generated loss curve
    • reproducible seeding
    • hyperparameters tuned for ~9,100 train images (500 per class)

Training strategy
---------------------------------------------------------
    1.  **Transfer learning** – start from COCO-pretrained weights;
        all backbone layers are trainable for maximum adaptation.
    2.  **Small learning rate** (0.005 with SGD) so pretrained features
        aren't destroyed.
    3.  **StepLR decay** – cut LR by 10× after epoch 20 to stabilise.
    4.  **Weight-decay** (5e-4) acts as L2 regularisation.
    5.  **Light augmentation only** – no flips, no heavy crop (Phase 3).
    6.  **Early-stopping awareness** – the validation loss is tracked and
        the *best* checkpoint is saved so you can always recover the
        least-overfit model.
    7.  **Batch size** (4) – fits in memory and provides stable gradients.

Hyper-parameters
-----------------
    BATCH_SIZE  = 4        (9,100 train images, 500 per class)
    LR          = 0.005    (moderate; pretrained backbone)
    EPOCHS      = 40       (sufficient for convergence)
    LR_STEP     = 20       (drop LR after initial convergence)
    WEIGHT_DECAY= 5e-4     (mild regularisation)

Usage
-----
    cd sign_language_frcnn
    PYTHONPATH=. python3 -m src.train
"""
import json
import os
import time

import torch

from src.config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA, DEVICE, NUM_WORKERS,
    CHECKPOINT_DIR, METRICS_DIR,
)
from src.dataloaders import get_train_loader, get_valid_loader
from src.model import get_model
from src.utils import set_seed, ensure_dirs, save_checkpoint
from src.visualize import plot_loss_curve


# ═══════════════════════════════════════════════════════════════════════════════
#  Core loops
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, optimizer, loader, device):
    """Run one training epoch; return the **mean batch loss**."""
    model.train()
    running_loss = 0.0
    num_batches = len(loader)
    for i, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        print(f"  [Train] batch {i}/{num_batches}  loss={losses.item():.4f}", flush=True)

    return running_loss / num_batches


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    """Run one validation epoch; return the **mean batch loss**.

    Faster R-CNN only returns losses when *targets* are supplied AND the
    model is in **train** mode.  We temporarily switch to train mode but
    wrap everything in ``torch.no_grad()`` so that no gradients are
    computed and batch-norm / dropout statistics are still updated as if
    in eval mode (Faster R-CNN has no dropout and uses FrozenBatchNorm2d,
    so this is safe).
    """
    model.train()  # required for loss computation
    running_loss = 0.0
    num_batches = len(loader)
    for i, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()
        print(f"  [Valid] batch {i}/{num_batches}  loss={losses.item():.4f}", flush=True)

    return running_loss / len(loader)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Reproducibility ────────────────────────────────────────────────────
    set_seed()

    # ── Directories ────────────────────────────────────────────────────────
    ensure_dirs(CHECKPOINT_DIR, METRICS_DIR)

    # ── DataLoaders (Phase 4) ──────────────────────────────────────────────
    train_loader = get_train_loader()
    valid_loader = get_valid_loader()
    print(f"Train batches : {len(train_loader)}  ({len(train_loader.dataset)} images)")
    print(f"Valid batches : {len(valid_loader)}  ({len(valid_loader.dataset)} images)")

    # ── Model (Phase 5) ───────────────────────────────────────────────────
    model = get_model().to(DEVICE)
    print(f"Device        : {DEVICE}")

    # Only optimise parameters that require grad (frozen backbone excluded)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE)
        val_loss = validate_one_epoch(model, valid_loader, DEVICE)
        lr_scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:>3d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"lr={lr_now:.6f}  "
            f"time={elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "lr": lr_now,
        })

        # ── Save latest checkpoint every epoch ─────────────────────────────
        save_checkpoint(
            model, optimizer, epoch,
            os.path.join(CHECKPOINT_DIR, "frcnn_latest.pth"),
        )

        # ── Save best checkpoint (lowest validation loss) ──────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(CHECKPOINT_DIR, "frcnn_best.pth"),
            )
            print(f"  ↳ Best model saved (val_loss={val_loss:.4f})")

    # ── Save training history as JSON ──────────────────────────────────────
    history_path = os.path.join(METRICS_DIR, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved  → {history_path}")

    # ── Auto-generate loss curve ───────────────────────────────────────────
    curve_path = os.path.join(METRICS_DIR, "loss_curve.png")
    plot_loss_curve(history_path, save_path=curve_path)
    print(f"Loss curve     → {curve_path}")

    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
