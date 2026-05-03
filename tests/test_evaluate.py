"""Tests for core.evaluate."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.evaluate import _group_images_by_pdf, _per_group_mAP


def test_group_images_by_pdf_uses_subdir_name(tmp_path):
    """Verify the helper that maps image_id → pdf-stem-based group."""
    images_root = tmp_path / "images"
    (images_root / "DocA").mkdir(parents=True)
    (images_root / "DocB").mkdir(parents=True)
    (images_root / "DocA" / "pagina-001.png").write_bytes(b"")
    (images_root / "DocB" / "pagina-001.png").write_bytes(b"")
    coco = {
        "images": [
            {"id": 1, "file_name": "pagina-001.png"},
            {"id": 2, "file_name": "pagina-001_1.png"},
        ],
    }
    groups = _group_images_by_pdf(coco, images_root)
    assert groups == {1: "DocA", 2: "DocB"}


def test_per_group_mAP_with_one_group():
    import torch

    preds = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}]
    gts = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]
    out = _per_group_mAP(preds, gts, ["DocA"])
    # 1 group with perfect prediction → mAP=1.0
    assert "DocA" in out
    assert abs(out["DocA"]["mAP"] - 1.0) < 1e-6


def test_per_group_mAP_aggregates_separately():
    import torch

    preds = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])},  # DocA
        {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)},  # DocB
    ]
    gts = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])},
    ]
    out = _per_group_mAP(preds, gts, ["DocA", "DocB"])
    assert abs(out["DocA"]["mAP"] - 1.0) < 1e-6
    assert out["DocB"]["mAP"] == 0.0
