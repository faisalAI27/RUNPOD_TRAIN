"""
Phase 5 — Model Creation
=========================
Builds a **Faster R-CNN** with a **ResNet-50 + FPN** backbone from torchvision.

Why NUM_CLASSES = 27 (26 + 1)?
    Torchvision detection models treat class index **0** as "background".
    Our dataset has 26 ASL letters (A–Z) mapped to indices 1–26, so the
    classifier head must output 26 + 1 = 27 logits.  If we used 26 the
    model would have no slot for background and every region proposal
    would be forced onto a letter, causing massive false positives.

Transfer-learning strategy
    • Load COCO-pretrained weights (``weights="DEFAULT"``).
    • All ``TRAINABLE_BACKBONE_LAYERS`` (default 5) are fine-tuned
      for maximum adaptation to the ASL domain.
    • **Replace** the box-classification head (``FastRCNNPredictor``) so
      that the output dimension matches our 27 classes.

CPU / GPU
    The model is device-agnostic.  Call ``model.to(config.DEVICE)`` after
    creation – ``DEVICE`` is auto-detected in ``config.py``.

Checkpoint loading
    Use ``load_model_from_checkpoint()`` to restore a saved ``.pth`` file
    (works on both CPU and GPU).
"""
import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.config import (
    NUM_CLASSES,
    TRAINABLE_BACKBONE_LAYERS,
    CHECKPOINT_DIR,
    DEVICE,
)
from src.utils import load_checkpoint


def get_model(num_classes: int = NUM_CLASSES):
    """Return a Faster R-CNN with a ResNet-50 FPN backbone.

    Parameters
    ----------
    num_classes : int
        Number of output classes **including background** (default 27).

    Returns
    -------
    model : torchvision FasterRCNN
    """
    # Load COCO-pretrained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
        box_score_thresh=0.05,
        box_nms_thresh=0.3,
        box_detections_per_img=5,
    )

    # Replace the classification + bounding-box head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_model_from_checkpoint(checkpoint_path: str = None,
                               device=DEVICE,
                               num_classes: int = NUM_CLASSES):
    """Build the model and restore weights from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str or None
        Path to a ``.pth`` file.  If ``None``, the latest checkpoint in
        ``CHECKPOINT_DIR`` is used automatically.
    device : torch.device
        Where to place the model after loading.
    num_classes : int
        Must match the checkpoint.

    Returns
    -------
    model : torchvision FasterRCNN  (in eval mode, on *device*)
    epoch : int  (the epoch at which the checkpoint was saved)
    """
    model = get_model(num_classes).to(device)

    if checkpoint_path is None:
        ckpts = sorted(
            f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
        checkpoint_path = os.path.join(CHECKPOINT_DIR, ckpts[-1])

    epoch = load_checkpoint(checkpoint_path, model)
    model.eval()
    return model, epoch


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = get_model()
    print(f"Model class  : {type(m).__name__}")
    print(f"Num classes  : {NUM_CLASSES}  (26 letters + 1 background)")
    print(f"Box predictor: {m.roi_heads.box_predictor}")

    # Quick forward pass sanity check (random data, train mode)
    m.train()
    dummy_img = [torch.randn(3, 480, 640)]
    dummy_tgt = [{"boxes": torch.tensor([[100., 100., 300., 300.]]),
                  "labels": torch.tensor([1])}]
    loss_dict = m(dummy_img, dummy_tgt)
    print(f"Loss keys    : {list(loss_dict.keys())}")
    print(f"Total loss   : {sum(loss_dict.values()).item():.4f}")

    # Eval-mode forward pass
    m.eval()
    with torch.no_grad():
        preds = m(dummy_img)
    print(f"Pred keys    : {list(preds[0].keys())}")
    print("✓ Model creation OK.")
