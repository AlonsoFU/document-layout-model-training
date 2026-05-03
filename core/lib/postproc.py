"""Geometric post-processing for layout detections.

Detection format: dict with keys 'box' (tuple x1,y1,x2,y2), 'label' (str), 'score' (float).
All functions are pure (no I/O, no torch) and return new lists.
"""
from __future__ import annotations

import copy
from typing import Iterable


Box = tuple[float, float, float, float]
Detection = dict


# ---------- Geometry primitives ----------

def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Box, b: Box) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Fraction of `inner` that lies inside `outer` (0..1)."""
    inner_area = box_area(inner)
    if inner_area <= 0:
        return 0.0
    return intersection_area(inner, outer) / inner_area


# ---------- Filters ----------

def apply_thresholds(
    detections: Iterable[Detection],
    thresholds: dict[str, float],
    default: float,
) -> list[Detection]:
    """Keep detections whose score >= threshold for their label (or `default`)."""
    out = []
    for d in detections:
        thr = thresholds.get(d["label"], default)
        if d["score"] >= thr:
            out.append(d)
    return out


def full_page_picture_filter(
    detections: Iterable[Detection],
    page_w: float,
    page_h: float,
    min_fraction: float = 0.9,
) -> list[Detection]:
    """Drop Picture detections that cover >= min_fraction of the page."""
    page_area = page_w * page_h
    return [
        d for d in detections
        if not (d["label"] == "Picture" and box_area(d["box"]) >= min_fraction * page_area)
    ]


# ---------- NMS ----------

def nms_per_category(
    detections: Iterable[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Greedy NMS within each label class. Higher score wins."""
    by_label: dict[str, list[Detection]] = {}
    for d in detections:
        by_label.setdefault(d["label"], []).append(d)

    kept: list[Detection] = []
    for label, dets in by_label.items():
        dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
        survivors: list[Detection] = []
        for d in dets_sorted:
            if all(iou(d["box"], s["box"]) <= iou_threshold for s in survivors):
                survivors.append(d)
        kept.extend(survivors)
    return kept


# ---------- Wrapper killer ----------

def kill_wrappers(
    detections: Iterable[Detection],
    containment_threshold: float = 0.7,
) -> list[Detection]:
    """Drop boxes that contain >=2 other boxes (each with containment > threshold)."""
    dets = list(detections)
    n = len(dets)
    kill = set()
    for i in range(n):
        children = 0
        for j in range(n):
            if i == j:
                continue
            if containment(dets[j]["box"], dets[i]["box"]) > containment_threshold:
                children += 1
                if children >= 2:
                    break
        if children >= 2:
            kill.add(i)
    return [d for k, d in enumerate(dets) if k not in kill]


# ---------- Crop ----------

def crop_box(to_crop: Box, fixed: Box) -> Box | None:
    """Crop `to_crop` to avoid intersecting `fixed`; pick the largest remaining rectangle.

    Returns None if all candidate slices are too small (<15% of original area)
    or have an aspect ratio worse than 12:1.
    """
    x1, y1, x2, y2 = to_crop
    fx1, fy1, fx2, fy2 = fixed

    if x1 >= fx2 or x2 <= fx1 or y1 >= fy2 or y2 <= fy1:
        return to_crop  # no actual intersection, return as-is

    original_area = (x2 - x1) * (y2 - y1)
    if original_area <= 0:
        return None

    candidates: list[Box] = []
    if fy2 < y2:
        candidates.append((x1, fy2, x2, y2))      # bottom strip
    if fy1 > y1:
        candidates.append((x1, y1, x2, fy1))      # top strip
    if fx2 < x2:
        candidates.append((fx2, y1, x2, y2))      # right strip
    if fx1 > x1:
        candidates.append((x1, y1, fx1, y2))      # left strip

    if not candidates:
        return None

    best = max(candidates, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
    bw, bh = best[2] - best[0], best[3] - best[1]
    if bw * bh < 0.15 * original_area:
        return None
    if bw > 0 and bh > 0 and max(bw / bh, bh / bw) > 12:
        return None
    return best


# ---------- Resolve overlaps ----------

def resolve_overlaps(
    detections: Iterable[Detection],
    uncropable: set[str],
    near_duplicate_iou: float = 0.8,
    containment_keep: float = 0.7,
    partial_iou: float = 0.3,
) -> list[Detection]:
    """Phase 2 of clean_overlaps_v3: resolve pairwise overlaps within each frame.

    Strategy (uncropable classes — Table, Picture, Formula, Code, Form,
    Key-value-region — are preserved when possible):
    - IoU > near_duplicate_iou (0.8): drop the lower-priority box.
    - containment > containment_keep (0.7): one is mostly inside the other —
      drop or crop the croppable one.
    - partial_iou < IoU < near_duplicate_iou: crop the lower-priority box.
    - else: keep both.
    """
    dets = [copy.copy(d) for d in detections]
    n = len(dets)
    if n == 0:
        return []

    # Priority: uncropable first, then by score (descending).
    order = sorted(
        range(n),
        key=lambda k: (
            1 if dets[k]["label"] in uncropable else 0,
            dets[k]["score"],
        ),
        reverse=True,
    )

    drop = set()
    for ai, i in enumerate(order):
        if i in drop:
            continue
        for bj in range(ai + 1, len(order)):
            j = order[bj]
            if j in drop:
                continue
            pi, pj = dets[i]["box"], dets[j]["box"]
            inter = intersection_area(pi, pj)
            if inter == 0:
                continue

            iou_val = iou(pi, pj)
            ci_in_j = containment(pi, pj)
            cj_in_i = containment(pj, pi)

            # Near-duplicates -> drop lower priority (j by ordering).
            if iou_val > near_duplicate_iou:
                drop.add(j)
                continue

            # j mostly inside i.
            if cj_in_i > containment_keep:
                name_i = dets[i]["label"]
                name_j = dets[j]["label"]
                if name_i in uncropable and name_j not in uncropable:
                    drop.add(j)
                elif name_j in uncropable and name_i not in uncropable:
                    cropped = crop_box(pi, pj)
                    if cropped is None:
                        drop.add(i)
                        break
                    dets[i]["box"] = cropped
                else:
                    cropped = crop_box(pi, pj)
                    if cropped is None:
                        drop.add(i)
                        break
                    dets[i]["box"] = cropped
                continue

            # i mostly inside j.
            if ci_in_j > containment_keep:
                name_i = dets[i]["label"]
                name_j = dets[j]["label"]
                if name_j in uncropable and name_i not in uncropable:
                    drop.add(i)
                    break
                if name_i in uncropable and name_j not in uncropable:
                    cropped = crop_box(pj, pi)
                    if cropped is None:
                        drop.add(j)
                    else:
                        dets[j]["box"] = cropped
                else:
                    cropped = crop_box(pj, pi)
                    if cropped is None:
                        drop.add(j)
                    else:
                        dets[j]["box"] = cropped
                continue

            # Partial overlap.
            if iou_val > partial_iou:
                if dets[j]["label"] not in uncropable:
                    cropped = crop_box(pj, pi)
                    if cropped is None:
                        drop.add(j)
                    else:
                        dets[j]["box"] = cropped
                elif dets[i]["label"] not in uncropable:
                    cropped = crop_box(pi, pj)
                    if cropped is None:
                        drop.add(i)
                        break
                    dets[i]["box"] = cropped
    return [d for k, d in enumerate(dets) if k not in drop]
