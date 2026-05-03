"""Tests for core.lib.eval — mAP@[.5:.95] computation."""
import torch

from core.lib.eval import box_iou, compute_ap, compute_map


def test_box_iou_identity():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    iou = box_iou(a, a)
    assert iou.shape == (1, 1)
    assert abs(iou[0, 0].item() - 1.0) < 1e-6


def test_box_iou_disjoint():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    iou = box_iou(a, b)
    assert iou[0, 0].item() == 0.0


def test_compute_ap_perfect_prediction_returns_one():
    """1 GT box, 1 prediction matching exactly with high score."""
    preds = [{
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
    }]
    gts = [{
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([0]),
    }]
    ap = compute_ap(preds, gts, class_id=0, iou_threshold=0.5)
    assert ap == 1.0


def test_compute_ap_no_predictions_returns_zero():
    preds = [{
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros(0),
        "labels": torch.zeros(0, dtype=torch.long),
    }]
    gts = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]
    assert compute_ap(preds, gts, class_id=0, iou_threshold=0.5) == 0.0


def test_compute_map_aggregates_classes_and_thresholds():
    """Sanity: a perfect prediction yields mAP=1.0 across all IoUs since boxes are identical."""
    preds = [{
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
    }]
    gts = [{
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([0]),
    }]
    out = compute_map(preds, gts)
    # Perfect overlap survives all 10 IoU thresholds (0.5..0.95)
    assert abs(out["mAP"] - 1.0) < 1e-6


def test_compute_map_returns_zero_when_no_gt():
    preds = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}]
    gts = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}]
    assert compute_map(preds, gts)["mAP"] == 0.0
