"""
Central configuration for the Sign Language Faster R-CNN project.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
FEATURE_MAPS_DIR = os.path.join(OUTPUT_DIR, "feature_maps")

# ── Classes ────────────────────────────────────────────────────────────────────
CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]
NUM_CLASSES = len(CLASSES) + 1  # +1 for background

# ── Training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE = 4
NUM_EPOCHS = 40
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 20
LR_GAMMA = 0.1

# ── Model ──────────────────────────────────────────────────────────────────────
BACKBONE = "resnet50"           # used by torchvision Faster R-CNN
PRETRAINED = True
TRAINABLE_BACKBONE_LAYERS = 5

# ── Image ──────────────────────────────────────────────────────────────────────
IMG_WIDTH = 300
IMG_HEIGHT = 300

# ── Device ─────────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Misc ───────────────────────────────────────────────────────────────────────
import platform
NUM_WORKERS = 0 if platform.system() == "Darwin" else 2  # macOS spawn issue
SEED = 42
CONFIDENCE_THRESHOLD = 0.35
