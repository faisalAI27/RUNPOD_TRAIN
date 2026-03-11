"""
Pascal-VOC XML dataset for Faster R-CNN sign-language detection.

Returns per-image targets with all fields required by torchvision Faster R-CNN:
    boxes      – FloatTensor[N, 4]  (x1, y1, x2, y2)
    labels     – Int64Tensor[N]
    image_id   – Int64Tensor[1]
    area       – FloatTensor[N]
    iscrowd    – UInt8Tensor[N]
"""
import logging
import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import CLASSES

logger = logging.getLogger(__name__)


class SignLanguageDataset(Dataset):
    """Reads images + Pascal VOC XML annotations (Kaggle ASL Alphabet dataset).

    Invalid / degenerate boxes (width or height <= 0, unknown class labels)
    are silently dropped so that downstream training never crashes.
    """

    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        # Only keep images that have a matching XML annotation
        all_imgs = sorted(f for f in os.listdir(root_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.imgs = []
        for fname in all_imgs:
            xml_name = os.path.splitext(fname)[0] + ".xml"
            if os.path.isfile(os.path.join(root_dir, xml_name)):
                self.imgs.append(fname)
            else:
                logger.warning("No annotation for %s – skipped", fname)

        # class name → integer label (1-indexed; 0 is background)
        self.class_to_idx = {c: i + 1 for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.imgs)

    # ── XML parsing ────────────────────────────────────────────────────────
    def _parse_xml(self, xml_path):
        """Parse a Pascal VOC XML file.

        Returns
        -------
        boxes  : list[list[float]]   – valid [x1, y1, x2, y2] boxes
        labels : list[int]           – corresponding class indices
        """
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            logger.error("Malformed XML: %s", xml_path)
            return [], []

        root = tree.getroot()
        boxes, labels = [], []

        for obj in root.findall("object"):
            # --- class label --------------------------------------------------
            name_el = obj.find("name")
            if name_el is None or name_el.text is None:
                logger.warning("Missing <name> in %s – object skipped", xml_path)
                continue
            label_text = name_el.text.strip()
            if label_text not in self.class_to_idx:
                logger.warning("Unknown class '%s' in %s – object skipped", label_text, xml_path)
                continue

            # --- bounding box -------------------------------------------------
            bb = obj.find("bndbox")
            if bb is None:
                logger.warning("Missing <bndbox> in %s – object skipped", xml_path)
                continue
            try:
                xmin = float(bb.find("xmin").text)
                ymin = float(bb.find("ymin").text)
                xmax = float(bb.find("xmax").text)
                ymax = float(bb.find("ymax").text)
            except (AttributeError, TypeError, ValueError):
                logger.warning("Bad bbox values in %s – object skipped", xml_path)
                continue

            # Ignore degenerate boxes (zero or negative area)
            if xmax <= xmin or ymax <= ymin:
                logger.warning(
                    "Degenerate box [%.1f, %.1f, %.1f, %.1f] in %s – skipped",
                    xmin, ymin, xmax, ymax, xml_path,
                )
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[label_text])

        return boxes, labels

    # ── __getitem__ ────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        xml_path = os.path.join(
            self.root_dir, os.path.splitext(img_name)[0] + ".xml",
        )

        image = Image.open(img_path).convert("RGB")
        boxes, labels = self._parse_xml(xml_path)

        # Convert to tensors ------------------------------------------------
        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            # Empty annotations – still need correct shape for Faster R-CNN
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)

        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        # area = (x2 - x1) * (y2 - y1)
        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area":     area,
            "iscrowd":  torch.zeros(len(labels), dtype=torch.uint8),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
