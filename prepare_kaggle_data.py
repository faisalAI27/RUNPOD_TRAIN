"""
Prepare the Kaggle ASL Alphabet dataset for Faster R-CNN object detection.

The Kaggle dataset (grassknoted/asl-alphabet) is a classification dataset
with images organized by class folders. This script:
    1. Downloads the dataset via kagglehub (if not already cached)
    2. Samples N images per class (A-Z only, skipping 'del', 'nothing', 'space')
    3. Creates Pascal VOC XML annotations with bounding boxes
       (hand fills most of the 200x200 image → use a margin-based bbox)
    4. Splits into train / valid / test sets
    5. Copies images + XMLs into data/train, data/valid, data/test

Usage:
    cd sign_language_frcnn
    python prepare_kaggle_data.py
"""
import os
import random
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
SEED = 42
IMAGES_PER_CLASS = 500          # sample from the 3000 available per class
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15              # remainder

CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]  # A-Z

# Bounding box margins — randomized per image for training diversity
BBOX_MARGIN_MIN = 5   # px from each side (tight crop)
BBOX_MARGIN_MAX = 35  # px from each side (loose crop)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def download_dataset():
    """Download the Kaggle ASL Alphabet dataset and return the path."""
    import kagglehub
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    print(f"Dataset path: {path}")
    return path


def create_voc_xml(img_filename, img_width, img_height, class_name,
                   xmin, ymin, xmax, ymax):
    """Create a Pascal VOC XML annotation string."""
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "filename").text = img_filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"

    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = class_name
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(int(xmin))
    ET.SubElement(bndbox, "ymin").text = str(int(ymin))
    ET.SubElement(bndbox, "xmax").text = str(int(xmax))
    ET.SubElement(bndbox, "ymax").text = str(int(ymax))

    rough_string = ET.tostring(annotation, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def prepare_data():
    random.seed(SEED)

    # ── Download ──────────────────────────────────────────────────────────
    kaggle_path = download_dataset()
    train_src = os.path.join(kaggle_path, "asl_alphabet_train", "asl_alphabet_train")

    if not os.path.isdir(train_src):
        raise FileNotFoundError(f"Expected folder not found: {train_src}")

    # ── Remove old symlinks / data ────────────────────────────────────────
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(DATA_DIR, split)
        if os.path.islink(split_dir):
            os.unlink(split_dir)
        elif os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

    # ── Process each class ────────────────────────────────────────────────
    stats = {"train": 0, "valid": 0, "test": 0}

    for cls in CLASSES:
        cls_dir = os.path.join(train_src, cls)
        if not os.path.isdir(cls_dir):
            print(f"  WARNING: class folder '{cls}' not found – skipping")
            continue

        # List all images
        all_imgs = sorted(
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        # Sample
        n = min(IMAGES_PER_CLASS, len(all_imgs))
        sampled = random.sample(all_imgs, n)

        # Split
        n_train = int(n * TRAIN_RATIO)
        n_valid = int(n * VALID_RATIO)
        train_imgs = sampled[:n_train]
        valid_imgs = sampled[n_train:n_train + n_valid]
        test_imgs  = sampled[n_train + n_valid:]

        for split, img_list in [("train", train_imgs),
                                ("valid", valid_imgs),
                                ("test", test_imgs)]:
            split_dir = os.path.join(DATA_DIR, split)
            for img_name in img_list:
                src_path = os.path.join(cls_dir, img_name)

                # Rename to avoid collisions: A_A1.jpg
                new_name = f"{cls}_{img_name}"
                dst_path = os.path.join(split_dir, new_name)

                # Copy image
                shutil.copy2(src_path, dst_path)

                # Create XML annotation with randomized bbox margins
                img = Image.open(dst_path)
                w, h = img.size
                if split == "train":
                    margin = random.randint(BBOX_MARGIN_MIN, BBOX_MARGIN_MAX)
                else:
                    margin = 10  # fixed for val/test consistency
                xmin = margin
                ymin = margin
                xmax = w - margin
                ymax = h - margin

                xml_content = create_voc_xml(
                    new_name, w, h, cls, xmin, ymin, xmax, ymax,
                )
                xml_name = os.path.splitext(new_name)[0] + ".xml"
                xml_path = os.path.join(split_dir, xml_name)
                with open(xml_path, "w") as f:
                    f.write(xml_content)

                stats[split] += 1

        print(f"  {cls}: {len(train_imgs)} train, {len(valid_imgs)} valid, {len(test_imgs)} test")

    print(f"\n{'='*50}")
    print(f"  Dataset prepared successfully!")
    print(f"  Train: {stats['train']} images")
    print(f"  Valid: {stats['valid']} images")
    print(f"  Test : {stats['test']} images")
    print(f"  Total: {sum(stats.values())} images")
    print(f"{'='*50}")
    print(f"  Data directory: {DATA_DIR}")


if __name__ == "__main__":
    prepare_data()
