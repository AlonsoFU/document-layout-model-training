"""Dataset, RFS sampler, and augmenter for layout fine-tuning.

Adapted from training/train_round4.py — that script's DocLayoutDataset
is the reference. Differences here:
- Path-based init (takes a COCO file + image dir) instead of holding
  the entire coco_data dict in memory.
- RepeatFactorSampler implements the standard RFS formula based on
  the rarest class per image.
- Augmenter is a separate class so it can be turned off for val/eval.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


class CocoDocDataset(Dataset):
    def __init__(
        self,
        coco_path: Path,
        images_dir: Path,
        image_ids: list[int],
        processor=None,
        augmenter=None,
    ):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.augmenter = augmenter

        with open(coco_path) as f:
            coco = json.load(f)
        self.id_to_img = {img["id"]: img for img in coco["images"]}
        self.img_to_anns: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)
        self.image_ids = list(image_ids)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _ensure_processor(self):
        if self.processor is None:
            from transformers import RTDetrImageProcessor
            self.processor = RTDetrImageProcessor.from_pretrained(
                "docling-project/docling-layout-heron"
            )

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.id_to_img[img_id]
        image = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        anns = self.img_to_anns[img_id]
        iw, ih = info["width"], info["height"]
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([(x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih])
            labels.append(ann["category_id"] - 1)  # 0-based
        if self.augmenter and boxes:
            image, boxes = self.augmenter(image, boxes)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "class_labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
        }
        self._ensure_processor()
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), target


class RepeatFactorSampler:
    """RFS: oversample images that contain rare classes.

    For each class c, frequency f(c) = (#images with c) / (total images).
    For each image i, factor r(i) = max(1, max_{c in i} sqrt(threshold / f(c))).
    Floor to int (we don't fractionally repeat).
    """

    def __init__(self, factors: dict[int, int]):
        self.factors = dict(factors)

    @classmethod
    def from_coco(cls, coco_path: Path, image_ids: list[int], threshold: float = 0.5):
        with open(coco_path) as f:
            coco = json.load(f)
        ids_set = set(image_ids)
        # classes per image
        img_classes: dict[int, set[int]] = defaultdict(set)
        for ann in coco["annotations"]:
            if ann["image_id"] in ids_set:
                img_classes[ann["image_id"]].add(ann["category_id"])
        # class frequency
        n = len(image_ids)
        class_count: dict[int, int] = defaultdict(int)
        for classes in img_classes.values():
            for c in classes:
                class_count[c] += 1
        class_freq = {c: cnt / n for c, cnt in class_count.items()}
        # per-image factor
        factors: dict[int, int] = {}
        for img_id in image_ids:
            classes = img_classes.get(img_id, set())
            if not classes:
                factors[img_id] = 1
                continue
            ratios = [math.sqrt(threshold / class_freq[c]) for c in classes if class_freq[c] > 0]
            r = max(1.0, max(ratios) if ratios else 1.0)
            factors[img_id] = max(1, math.ceil(r))
        return cls(factors)

    def __iter__(self) -> Iterator[int]:
        out: list[int] = []
        for img_id, factor in self.factors.items():
            out.extend([img_id] * factor)
        random.shuffle(out)
        return iter(out)

    def __len__(self) -> int:
        return sum(self.factors.values())


class DocumentAugmenter:
    """Image-only stochastic augmentation. Boxes are returned unchanged
    (legacy script also does not transform boxes for these augmentations)."""

    def __call__(self, image: Image.Image, boxes: list):
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            image = image.rotate(random.uniform(-3, 3), fillcolor=(255, 255, 255))
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return image, boxes


def collate_fn(batch):
    """Stack pixel_values; keep targets as a list (variable-length)."""
    pixels = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return pixels, targets
