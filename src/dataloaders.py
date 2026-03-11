"""
Phase 4 — DataLoaders
=====================
Creates train / valid / test ``DataLoader`` objects with:
    • the custom ``SignLanguageDataset``  (Phase 1)
    • detection-safe transforms           (Phase 3)
    • custom ``collate_fn``               (required by torchvision detection models)
    • configurable batch size via ``config.BATCH_SIZE``

Why a custom collate_fn?
    Default ``DataLoader`` stacks tensors into a single batch tensor, but
    detection models need a **list of images** (each may differ in #objects).
    ``collate_fn`` keeps them as a tuple-of-lists instead.

Usage
-----
    from src.dataloaders import get_loaders
    train_loader, valid_loader, test_loader = get_loaders()
"""
from torch.utils.data import DataLoader

from src.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    BATCH_SIZE, NUM_WORKERS, DEVICE,
)
from src.dataset import SignLanguageDataset
from src.transforms import get_train_transforms, get_valid_transforms, get_test_transforms
from src.utils import collate_fn

# pin_memory only helps with CUDA, not MPS or CPU
_PIN_MEMORY = DEVICE.type == "cuda"


# ── Factory functions ──────────────────────────────────────────────────────────

def get_train_loader(batch_size: int = BATCH_SIZE,
                     num_workers: int = NUM_WORKERS,
                     data_dir: str = TRAIN_DIR) -> DataLoader:
    """Return a **shuffled** DataLoader for the training split."""
    dataset = SignLanguageDataset(data_dir, transforms=get_train_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=_PIN_MEMORY,
    )


def get_valid_loader(batch_size: int = BATCH_SIZE,
                     num_workers: int = NUM_WORKERS,
                     data_dir: str = VALID_DIR) -> DataLoader:
    """Return a deterministic DataLoader for the validation split."""
    dataset = SignLanguageDataset(data_dir, transforms=get_valid_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=_PIN_MEMORY,
    )


def get_test_loader(batch_size: int = BATCH_SIZE,
                    num_workers: int = NUM_WORKERS,
                    data_dir: str = TEST_DIR) -> DataLoader:
    """Return a deterministic DataLoader for the test split."""
    dataset = SignLanguageDataset(data_dir, transforms=get_test_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=_PIN_MEMORY,
    )


def get_loaders(batch_size: int = BATCH_SIZE,
                num_workers: int = NUM_WORKERS):
    """Convenience: return (train_loader, valid_loader, test_loader) in one call."""
    return (
        get_train_loader(batch_size, num_workers),
        get_valid_loader(batch_size, num_workers),
        get_test_loader(batch_size, num_workers),
    )


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building loaders …")
    train_ld, valid_ld, test_ld = get_loaders()

    for name, loader in [("Train", train_ld), ("Valid", valid_ld), ("Test", test_ld)]:
        images, targets = next(iter(loader))
        print(f"\n{'─'*40}")
        print(f" {name} loader")
        print(f"{'─'*40}")
        print(f"  Batch size requested : {BATCH_SIZE}")
        print(f"  Images in batch      : {len(images)}")
        print(f"  Image tensor shape   : {images[0].shape}")
        print(f"  Target keys          : {list(targets[0].keys())}")
        print(f"  Boxes shape (img 0)  : {targets[0]['boxes'].shape}")
        print(f"  Labels (img 0)       : {targets[0]['labels']}")
        print(f"  Area (img 0)         : {targets[0]['area']}")
        print(f"  iscrowd (img 0)      : {targets[0]['iscrowd']}")
        print(f"  Dataset length       : {len(loader.dataset)}")
        print(f"  Num batches          : {len(loader)}")

    print("\n✓ All loaders created successfully.")
