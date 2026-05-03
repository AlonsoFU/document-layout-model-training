"""mAP@[.5:.95] evaluation. Adapted from training/train_round4.py."""
from __future__ import annotations

from typing import Iterable

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two box sets (xyxy)."""
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    ix1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    return inter / (a1[:, None] + a2 - inter + 1e-6)


def compute_ap(predictions, ground_truth, class_id: int, iou_threshold: float) -> float:
    """11-point interpolated average precision for one class at one IoU threshold."""
    scores: list[float] = []
    tps: list[int] = []
    n_gt = 0
    for p, t in zip(predictions, ground_truth):
        gm = (t["labels"] == class_id)
        gb = t["boxes"][gm]
        n_gt += int(gm.sum().item())
        pm = (p["labels"] == class_id)
        pb = p["boxes"][pm]
        ps = p["scores"][pm]
        if len(pb) == 0:
            continue
        order = ps.argsort(descending=True)
        pb = pb[order]
        ps = ps[order]
        matched = set()
        for i in range(len(pb)):
            scores.append(float(ps[i].item()))
            if len(gb) == 0:
                tps.append(0)
                continue
            ious = box_iou(pb[i].unsqueeze(0), gb)[0]
            best_iou, best_idx = ious.max(0)
            bi = int(best_idx.item())
            if best_iou.item() >= iou_threshold and bi not in matched:
                tps.append(1)
                matched.add(bi)
            else:
                tps.append(0)
    if n_gt == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_tps = [tps[i] for i in order]
    tp = fp = 0
    precs: list[float] = []
    recs: list[float] = []
    for t in sorted_tps:
        tp += t
        fp += 1 - t
        precs.append(tp / (tp + fp))
        recs.append(tp / n_gt)
    ap = 0.0
    for r in [i / 10.0 for i in range(11)]:
        candidates = [p for p, rc in zip(precs, recs) if rc >= r]
        ap += (max(candidates) if candidates else 0.0) / 11.0
    return min(1.0, ap)


def compute_map(
    predictions,
    ground_truth,
    iou_thresholds: Iterable[float] | None = None,
) -> dict:
    """Mean AP over (classes x IoU thresholds). COCO-style mAP@[.5:.95] by default.

    Returns:
        dict with keys 'mAP' (float), 'per_threshold' (dict[iou] -> mAP), 'per_class@0.5' (dict[cls] -> AP).
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    iou_thresholds = list(iou_thresholds)

    classes: set[int] = set()
    for t in ground_truth:
        classes.update(int(c) for c in t["labels"].tolist())
    if not classes:
        return {"mAP": 0.0, "per_threshold": {iou: 0.0 for iou in iou_thresholds}, "per_class@0.5": {}}

    classes_sorted = sorted(classes)

    per_threshold = {}
    for iou_t in iou_thresholds:
        aps = [compute_ap(predictions, ground_truth, c, iou_t) for c in classes_sorted]
        per_threshold[iou_t] = sum(aps) / len(aps)

    per_class_05 = {}
    for c in classes_sorted:
        per_class_05[c] = compute_ap(predictions, ground_truth, c, 0.5)

    mean_map = sum(per_threshold.values()) / len(per_threshold)
    return {"mAP": mean_map, "per_threshold": per_threshold, "per_class@0.5": per_class_05}
