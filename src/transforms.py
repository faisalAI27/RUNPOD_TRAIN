"""
Detection-safe image transforms / augmentations.

Every transform is a callable that receives ``(image, target)`` and returns
``(image, target)`` so that bounding boxes stay consistent with the image.

Avoided (harmful for sign-language detection):
    - Horizontal / vertical flip  (changes letter meaning, e.g. J ↔ mirrored J)
    - Aggressive random crop      (may cut the hand)
    - Strong colour / geometric distortion
"""
import math
import random

import torch
import torchvision.transforms.functional as F
from PIL import Image

from src.config import IMG_WIDTH, IMG_HEIGHT


# ═══════════════════════════════════════════════════════════════════════════════
#  Primitive detection-safe transforms
# ═══════════════════════════════════════════════════════════════════════════════

class Compose:
    """Chain transforms that operate on (image, target) pairs."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert a PIL Image to a float32 tensor in [0, 1]."""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """ImageNet-style normalisation (applied after ToTensor)."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    """Resize image *and* scale bounding boxes accordingly.

    Keeps the exact target size (no aspect-ratio preservation) to match
    the dataset's annotation dimensions.
    """
    def __init__(self, width=IMG_WIDTH, height=IMG_HEIGHT):
        self.width = width
        self.height = height

    def __call__(self, image, target):
        orig_w, orig_h = image.size  # PIL (w, h)
        image = F.resize(image, [self.height, self.width])

        if "boxes" in target and target["boxes"].numel() > 0:
            sx = self.width / orig_w
            sy = self.height / orig_h
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            target["boxes"] = boxes
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return image, target


class LightColorJitter:
    """Mild brightness / contrast / saturation jitter (no geometric change)."""
    def __init__(self, brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        image = F.adjust_brightness(image, 1.0 + random.uniform(-self.brightness, self.brightness))
        image = F.adjust_contrast(image, 1.0 + random.uniform(-self.contrast, self.contrast))
        image = F.adjust_saturation(image, 1.0 + random.uniform(-self.saturation, self.saturation))
        image = F.adjust_hue(image, random.uniform(-self.hue, self.hue))
        return image, target


class SlightRotation:
    """Small random rotation (±degrees) with box re-computation.

    Only the axis-aligned bounding rectangle of the rotated corners is kept,
    so slight angles (≤ 5°) are recommended.
    """
    def __init__(self, max_degrees=5):
        self.max_degrees = max_degrees

    def __call__(self, image, target):
        angle = random.uniform(-self.max_degrees, self.max_degrees)
        image = F.rotate(image, angle, expand=False, fill=0)

        if "boxes" in target and target["boxes"].numel() > 0:
            w, h = image.size  # PIL size after rotate (expand=False → same)
            cx, cy = w / 2.0, h / 2.0
            rad = math.radians(-angle)  # PIL rotates counter-clockwise
            cos_a, sin_a = math.cos(rad), math.sin(rad)

            boxes = target["boxes"]
            # four corners per box -> rotate -> axis-aligned enclosing box
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            corners_x = torch.stack([x1, x2, x2, x1], dim=1) - cx
            corners_y = torch.stack([y1, y1, y2, y2], dim=1) - cy
            rot_x = corners_x * cos_a - corners_y * sin_a + cx
            rot_y = corners_x * sin_a + corners_y * cos_a + cy

            new_x1 = rot_x.min(dim=1).values.clamp(min=0)
            new_y1 = rot_y.min(dim=1).values.clamp(min=0)
            new_x2 = rot_x.max(dim=1).values.clamp(max=w)
            new_y2 = rot_y.max(dim=1).values.clamp(max=h)

            new_boxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)

            # Drop boxes that collapsed after clipping
            valid = (new_boxes[:, 2] > new_boxes[:, 0]) & (new_boxes[:, 3] > new_boxes[:, 1])
            target["boxes"] = new_boxes[valid]
            target["labels"] = target["labels"][valid]
            target["area"] = (
                (target["boxes"][:, 2] - target["boxes"][:, 0])
                * (target["boxes"][:, 3] - target["boxes"][:, 1])
            )
            target["iscrowd"] = target["iscrowd"][valid]

        return image, target


class RandomGaussianBlur:
    """Apply Gaussian blur with random kernel size (helps generalization)."""
    def __init__(self, p=0.3, kernel_sizes=(3, 5)):
        self.p = p
        self.kernel_sizes = kernel_sizes

    def __call__(self, image, target):
        if random.random() < self.p:
            k = random.choice(self.kernel_sizes)
            image = F.gaussian_blur(image if isinstance(image, torch.Tensor) else F.to_tensor(image), [k, k])
            if isinstance(image, torch.Tensor) and not isinstance(image, Image.Image):
                image = F.to_pil_image(image)
        return image, target


class RandomScaleJitter:
    """Randomly scale the image slightly (0.85-1.15x) and adjust boxes."""
    def __init__(self, min_scale=0.85, max_scale=1.15):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, target):
        scale = random.uniform(self.min_scale, self.max_scale)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = F.resize(image, [new_h, new_w])

        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes *= scale
            # Clamp to new image size
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)
            target["boxes"] = boxes
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return image, target


# ═══════════════════════════════════════════════════════════════════════════════
#  Public factory functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_train_transforms():
    """Training: scale jitter → resize → colour jitter → rotation → blur → tensor."""
    return Compose([
        RandomScaleJitter(),
        Resize(),
        LightColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.08),
        SlightRotation(max_degrees=8),
        ToTensor(),
    ])


def get_valid_transforms():
    """Validation: deterministic resize → tensor."""
    return Compose([
        Resize(),
        ToTensor(),
    ])


def get_test_transforms():
    """Test / inference: identical to validation (no randomness)."""
    return Compose([
        Resize(),
        ToTensor(),
    ])
