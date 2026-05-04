"""Tests for core.lib.data — COCO dataset, RFS sampler, augmenter."""
import json
from pathlib import Path

import pytest
from PIL import Image

from core.lib.data import (
    CocoDocDataset,
    DocumentAugmenter,
    RepeatFactorSampler,
    collate_fn,
)


@pytest.fixture()
def fake_coco(tmp_path):
    """Build a tiny COCO file + 3 PNG images. 3 images, 4 anns, 2 classes."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in (1, 2, 3):
        Image.new("RGB", (100, 200), (255, 255, 255)).save(img_dir / f"p{i}.png")
    coco = {
        "categories": [
            {"id": 1, "name": "Common"},   # appears in 3/3 images
            {"id": 2, "name": "Rare"},     # appears in 1/3 images
        ],
        "images": [
            {"id": 1, "file_name": "p1.png", "width": 100, "height": 200},
            {"id": 2, "file_name": "p2.png", "width": 100, "height": 200},
            {"id": 3, "file_name": "p3.png", "width": 100, "height": 200},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 4, "image_id": 3, "category_id": 2, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
        ],
    }
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps(coco))
    return coco_path, img_dir


def test_dataset_len_matches_image_count(fake_coco):
    coco_path, img_dir = fake_coco
    ds = CocoDocDataset(coco_path, img_dir, image_ids=[1, 2, 3])
    assert len(ds) == 3


def test_dataset_returns_pixel_values_and_target(fake_coco):
    """Item is (pixel_values, target). Target has boxes (CXCYWH normalized) and 0-based class_labels."""
    coco_path, img_dir = fake_coco
    ds = CocoDocDataset(coco_path, img_dir, image_ids=[3])  # has both classes
    px, target = ds[0]
    assert px.ndim == 3  # CHW
    assert "boxes" in target and "class_labels" in target
    assert target["boxes"].shape[0] == 2  # 2 anns on image 3
    assert target["boxes"].shape[1] == 4  # CXCYWH
    assert (target["boxes"] >= 0).all() and (target["boxes"] <= 1).all()
    # class_labels are 0-based (model expects 0-based; COCO is 1-based)
    assert set(target["class_labels"].tolist()) == {0, 1}


def test_dataset_filters_to_specified_image_ids(fake_coco):
    coco_path, img_dir = fake_coco
    ds = CocoDocDataset(coco_path, img_dir, image_ids=[1])
    assert len(ds) == 1


def test_repeat_factor_sampler_oversamples_rare(fake_coco):
    coco_path, _ = fake_coco
    sampler = RepeatFactorSampler.from_coco(coco_path, image_ids=[1, 2, 3], threshold=0.5)
    # Image 3 has the rare class (Rare appears in 1/3 = 33% < threshold 50%).
    # Image 1 and 2 have only Common (appears in 100% >= threshold), so factor = 1.
    factors = sampler.factors
    assert factors[1] == 1
    assert factors[2] == 1
    assert factors[3] > 1  # oversampled


def test_repeat_factor_sampler_iter_repeats_correctly():
    sampler = RepeatFactorSampler(factors={1: 1, 2: 3, 3: 2})
    indices = list(sampler)
    assert indices.count(1) == 1
    assert indices.count(2) == 3
    assert indices.count(3) == 2
    assert len(sampler) == 6


def test_collate_fn_stacks_batch():
    import torch
    px1 = torch.zeros(3, 10, 10)
    px2 = torch.zeros(3, 10, 10)
    t1 = {"boxes": torch.zeros((1, 4)), "class_labels": torch.tensor([0])}
    t2 = {"boxes": torch.zeros((2, 4)), "class_labels": torch.tensor([0, 1])}
    pixels, targets = collate_fn([(px1, t1), (px2, t2)])
    assert pixels.shape == (2, 3, 10, 10)
    assert len(targets) == 2


def test_augmenter_returns_image_and_boxes():
    """Augmenter is stochastic; just verify shape preservation."""
    import random
    random.seed(0)
    aug = DocumentAugmenter()
    img = Image.new("RGB", (50, 50), (128, 128, 128))
    boxes = [[0.5, 0.5, 0.2, 0.2]]
    out_img, out_boxes = aug(img, boxes)
    assert out_img.size == (50, 50)
    assert out_boxes == boxes  # boxes are not transformed (only image)
