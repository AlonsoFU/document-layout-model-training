# Plan 03 — Pre-annotation + Post-processing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar `dlmf predict --pre-annotate` (corre Docling Heron baseline sobre las imágenes de un proyecto y produce un COCO de predicciones para que `dlmf cvat-push --coco=…` lo use como pre-labels) más `core/lib/postproc.py` (limpieza estándar de overlaps: NMS per-cat, kill wrappers, resolve overlaps, full-page picture filter).

**Architecture:** Dos módulos nuevos en `core/lib/` (`model.py` carga Heron para inferencia, sin LoRA — eso es Plan 04; `postproc.py` son funciones puras de limpieza geométrica). `core/predict.py` orquesta: carga modelo → infiere por página → aplica thresholds + NMS → escribe COCO. CLI `dlmf predict --project=eaf --pre-annotate` lo invoca.

**Tech Stack:** PyTorch, transformers (`RTDetrV2ForObjectDetection`, `RTDetrImageProcessor`), torchvision (NMS), Pillow.

**Salida verificable:**
- `dlmf predict --project=eaf --pre-annotate` corre sobre las 561 imágenes de EAF y escribe `projects/eaf/cvat/pre_annotations/<YYYY-MM-DD-HHMMSS>.json`.
- Ese COCO consumido por `dlmf cvat-push --project=eaf --coco=<file>` carga las pre-anotaciones en CVAT.
- `pytest` pasa los 33 (planes 01-02) más los nuevos (~20).
- Tag `plan-03-predict-postproc` apunta al último commit.

**Out of scope:**
- Plan 04: `core/train.py` + `core/lib/{data,model.py extension for LoRA, tracking}.py` + MLflow. Este plan implementa solo el path de inferencia de `model.py` (carga modelo, ejecuta forward pass). LoRA application/loading se añade en Plan 04 sin breaking changes.
- Plan 05: `core/predict.py --output=anotado.pdf` (visualización). Este plan solo implementa `--pre-annotate`.
- Inferencia con un modelo ya fine-tuneado (LoRA loaded). Por ahora siempre usa el baseline Heron. Plan 04 añade `--model=<run-name>` o similar.

**Referencias (scripts legacy a migrar):**
- `generate_heron_coco.py` — lógica de inferencia + COCO writing. Se elimina al final del plan.
- `clean_overlaps_v3.py` — algoritmos de post-procesamiento (`box_area`, `iou`, `containment`, `crop_box`, kill wrappers, resolve overlaps). Las funciones puras se migran a `core/lib/postproc.py`. Se elimina al final del plan.

---

## File Structure

### Files to CREATE

| Path | Responsabilidad |
|---|---|
| `core/lib/postproc.py` | Funciones puras: `box_area`, `intersection_area`, `iou`, `containment`, `crop_box`, `nms_per_category`, `kill_wrappers`, `resolve_overlaps`, `full_page_picture_filter`, `apply_thresholds`. Sin dependencias de torch — solo Python + listas/tuples. |
| `core/lib/model.py` | `load_heron_for_inference(device)` returns `(model, processor)`. Contiene el mapeo `MODEL_INDEX_TO_LABEL_NAME` (0=Caption, 1=Footnote, ..., 16=Key-value-region — el orden nativo del modelo, distinto del orden CVAT). Plan 04 lo extenderá con `apply_lora`. |
| `core/predict.py` | `predict(project_slug, mode="pre-annotate")`. Invoca model + postproc, escribe COCO a `projects/<slug>/cvat/pre_annotations/<timestamp>.json`. |
| `tests/test_postproc.py` | Tests de las funciones puras de postproc (geometría: iou, containment, crop, kill_wrappers; threshold filter; NMS). |
| `tests/test_model.py` | Tests del label mapping (no carga el modelo real — solo verifica el dict). |
| `tests/test_predict.py` | Tests con MagicMock del modelo + processor + 1-2 imágenes fixture. Verifica que el COCO de salida tenga la forma esperada. |

### Files to MODIFY

| Path | Cambio |
|---|---|
| `core/cli.py` | Añadir comando `predict` con flags `--project`, `--pre-annotate` (bool), `--threshold` (float, override del config), `--limit` (int, opcional, para smoke testing). |
| `tests/test_cli.py` | Añadir test que `dlmf predict --help` lista `--project`, `--pre-annotate`, `--threshold`, `--limit`. |

### Files to DELETE (al final, después de validar paridad)

| Path | Razón |
|---|---|
| `generate_heron_coco.py` | Reemplazado por `dlmf predict --pre-annotate`. |
| `clean_overlaps_v3.py` | Las funciones puras se migran a `core/lib/postproc.py`. La parte de upload-a-CVAT ya está en `dlmf cvat-push --coco=<file>`. |

---

## Tasks

### Task 1: Implement `core/lib/postproc.py` with TDD

**Files:**
- Create: `core/lib/postproc.py`, `tests/test_postproc.py`

The module exports pure functions (no I/O, no torch). Boxes are 4-tuples `(x1, y1, x2, y2)` in image coordinates. Detections are dicts: `{"box": (x1,y1,x2,y2), "label": "Caption", "score": 0.9}`.

Functions to implement:
- `box_area(box) -> float`
- `intersection_area(a, b) -> float`
- `iou(a, b) -> float`
- `containment(inner, outer) -> float` — fraction of `inner` that is inside `outer` (0..1)
- `apply_thresholds(detections, thresholds: dict[str, float], default: float) -> list` — keep only detections whose score ≥ threshold for their label
- `full_page_picture_filter(detections, page_w, page_h, min_fraction=0.9) -> list` — drop `Picture` boxes that cover ≥ `min_fraction` of the page
- `nms_per_category(detections, iou_threshold=0.5) -> list` — pure-Python greedy NMS within each label class
- `kill_wrappers(detections, containment_threshold=0.7) -> list` — drop boxes that contain ≥2 other boxes (each with containment > threshold)
- `resolve_overlaps(detections, uncropable: set[str], iou_thresholds=(0.3, 0.8)) -> list` — combined logic from `clean_overlaps_v3.py`'s phase 2 (drop near-duplicates, prefer uncropable, crop the rest). Returns the list with some boxes possibly mutated and some removed.
- `crop_box(to_crop, fixed) -> tuple | None` — crop `to_crop` to avoid `fixed`; returns the largest remaining rectangle, or None if too small / wrong aspect.

- [ ] **Step 1: Write failing tests**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_postproc.py`:

```python
"""Tests for core.lib.postproc — geometric cleanup of layout detections."""
import pytest

from core.lib.postproc import (
    apply_thresholds,
    box_area,
    containment,
    crop_box,
    full_page_picture_filter,
    intersection_area,
    iou,
    kill_wrappers,
    nms_per_category,
    resolve_overlaps,
)


# ---- Geometry primitives --------------------------------------------------

def test_box_area_basic():
    assert box_area((0, 0, 10, 20)) == 200


def test_box_area_zero_or_negative():
    assert box_area((10, 10, 5, 5)) == 0  # inverted
    assert box_area((10, 10, 10, 20)) == 0  # zero-width


def test_intersection_area():
    assert intersection_area((0, 0, 10, 10), (5, 5, 20, 20)) == 25
    assert intersection_area((0, 0, 5, 5), (10, 10, 20, 20)) == 0


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # Half overlap
    val = iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert abs(val - 50 / 150) < 1e-9


def test_containment():
    # Fully inside
    assert containment((2, 2, 8, 8), (0, 0, 10, 10)) == 1.0
    # Half outside
    assert containment((5, 0, 15, 10), (0, 0, 10, 10)) == 0.5
    # No overlap
    assert containment((20, 20, 30, 30), (0, 0, 10, 10)) == 0.0


# ---- Threshold filter -----------------------------------------------------

def test_apply_thresholds_keeps_above_threshold():
    dets = [
        {"box": (0, 0, 10, 10), "label": "Caption", "score": 0.6},
        {"box": (0, 0, 10, 10), "label": "Caption", "score": 0.4},
        {"box": (0, 0, 10, 10), "label": "Title", "score": 0.5},
    ]
    out = apply_thresholds(dets, thresholds={"Caption": 0.5, "Title": 0.45}, default=0.5)
    assert len(out) == 2
    assert all(d["score"] >= 0.45 for d in out)


def test_apply_thresholds_uses_default_for_unknown_labels():
    dets = [
        {"box": (0, 0, 1, 1), "label": "MysteryLabel", "score": 0.4},
        {"box": (0, 0, 1, 1), "label": "MysteryLabel", "score": 0.6},
    ]
    out = apply_thresholds(dets, thresholds={}, default=0.5)
    assert len(out) == 1
    assert out[0]["score"] == 0.6


# ---- Full-page Picture filter ---------------------------------------------

def test_full_page_picture_filter_drops_huge_pictures():
    dets = [
        {"box": (0, 0, 100, 100), "label": "Picture", "score": 0.8},  # 100% page
        {"box": (0, 0, 50, 50), "label": "Picture", "score": 0.8},     # 25% page
        {"box": (0, 0, 100, 100), "label": "Table", "score": 0.8},     # full but Table
    ]
    out = full_page_picture_filter(dets, page_w=100, page_h=100, min_fraction=0.9)
    assert len(out) == 2  # small Picture + Table kept; full Picture dropped


# ---- NMS ------------------------------------------------------------------

def test_nms_per_category_keeps_higher_score():
    dets = [
        {"box": (0, 0, 10, 10), "label": "Caption", "score": 0.7},
        {"box": (1, 1, 11, 11), "label": "Caption", "score": 0.9},  # near-overlap, higher score
        {"box": (0, 0, 10, 10), "label": "Title", "score": 0.6},     # diff label, kept
    ]
    out = nms_per_category(dets, iou_threshold=0.5)
    # 1 caption (the 0.9) + 1 title
    assert len(out) == 2
    captions = [d for d in out if d["label"] == "Caption"]
    assert len(captions) == 1
    assert captions[0]["score"] == 0.9


def test_nms_per_category_no_overlap_keeps_all():
    dets = [
        {"box": (0, 0, 10, 10), "label": "Caption", "score": 0.5},
        {"box": (50, 50, 60, 60), "label": "Caption", "score": 0.6},
    ]
    out = nms_per_category(dets, iou_threshold=0.5)
    assert len(out) == 2


# ---- Wrapper killing ------------------------------------------------------

def test_kill_wrappers_drops_box_containing_two_children():
    dets = [
        {"box": (0, 0, 100, 100), "label": "Text", "score": 0.5},     # wrapper
        {"box": (10, 10, 30, 30), "label": "Table", "score": 0.8},     # child 1
        {"box": (40, 40, 60, 60), "label": "Picture", "score": 0.8},   # child 2
    ]
    out = kill_wrappers(dets, containment_threshold=0.7)
    labels = {d["label"] for d in out}
    assert labels == {"Table", "Picture"}


def test_kill_wrappers_preserves_box_with_single_child():
    dets = [
        {"box": (0, 0, 100, 100), "label": "Text", "score": 0.5},
        {"box": (10, 10, 30, 30), "label": "Table", "score": 0.8},
    ]
    out = kill_wrappers(dets, containment_threshold=0.7)
    assert len(out) == 2  # only 1 child, not killed


# ---- Crop -----------------------------------------------------------------

def test_crop_box_picks_largest_remaining_rectangle():
    # to_crop is wider; fixed sits in the middle vertically — crop top OR bottom
    cropped = crop_box((0, 0, 100, 100), (40, 30, 60, 70))
    # Expect either top portion or bottom portion (whichever is larger; both are 30 tall)
    assert cropped is not None
    x1, y1, x2, y2 = cropped
    assert x1 == 0 and x2 == 100
    # Either the top (0,0..100,30) or the bottom (0,70..100,100): areas equal
    assert (y1, y2) in {(0, 30), (70, 100), (0, 30), (70, 100)}


def test_crop_box_returns_none_when_remainder_too_small():
    # `to_crop` is fully contained in `fixed` => no remainder
    assert crop_box((10, 10, 20, 20), (0, 0, 100, 100)) is None


def test_crop_box_returns_none_for_extreme_aspect_ratio():
    # Cropping leaves a 1x100 sliver — should be rejected
    assert crop_box((0, 0, 1, 100), (0, 0, 1, 99)) is None


# ---- Resolve overlaps -----------------------------------------------------

def test_resolve_overlaps_drops_near_duplicate():
    dets = [
        {"box": (0, 0, 100, 100), "label": "Text", "score": 0.7},
        {"box": (1, 1, 100, 100), "label": "Text", "score": 0.9},
    ]
    out = resolve_overlaps(dets, uncropable={"Table", "Picture"})
    assert len(out) == 1
    assert out[0]["score"] == 0.9


def test_resolve_overlaps_prefers_uncropable_over_crop():
    """Text wrapping a Table → drop the Text wrapper, keep the Table."""
    dets = [
        {"box": (0, 0, 100, 100), "label": "Text", "score": 0.6},
        {"box": (10, 10, 90, 90), "label": "Table", "score": 0.8},
    ]
    out = resolve_overlaps(dets, uncropable={"Table", "Picture"})
    labels = {d["label"] for d in out}
    # Table must be kept; Text either dropped or cropped
    assert "Table" in labels


def test_resolve_overlaps_empty_input():
    assert resolve_overlaps([], uncropable={"Table"}) == []
```

- [ ] **Step 2: Run tests; expect failure (no module)**

```bash
uv run pytest tests/test_postproc.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.lib.postproc'`.

- [ ] **Step 3: Implement `core/lib/postproc.py`**

Use `clean_overlaps_v3.py` as reference for the geometric primitives (`box_area`, `intersection_area`, `iou`, `containment`, `crop_box`, the wrapper-kill phase, the resolve-overlaps phase). Adapt to:
- Pure Python (no requests, no CVAT API calls).
- Operate on lists of detection dicts `{"box": (x1,y1,x2,y2), "label": str, "score": float}` instead of CVAT shape dicts.
- Return new lists (don't mutate input — for `resolve_overlaps` it's OK to mutate the dicts inside a copy, since `crop_box` requires updating `box`).

Write `/home/alonso/Documentos/Github/document-layout-model-training/core/lib/postproc.py`:

```python
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
        candidates.append((x1, fy2, x2, y2))
    if fy1 > y1:
        candidates.append((x1, y1, x2, fy1))
    if fx2 < x2:
        candidates.append((fx2, y1, x2, y2))
    if fx1 > x1:
        candidates.append((x1, y1, fx1, y2))

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
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
uv run pytest tests/test_postproc.py -v
```
Expected: 16 tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/lib/postproc.py tests/test_postproc.py
git commit -m "feat(core): add postproc — pure-Python NMS, wrapper killer, overlap resolver

core.lib.postproc exposes geometric primitives (box_area, intersection_area,
iou, containment, crop_box) and four cleanup pipelines:
- apply_thresholds: per-class score gating with default fallback.
- full_page_picture_filter: drop Picture boxes covering >=90% of the page.
- nms_per_category: greedy NMS within each label class.
- kill_wrappers: drop boxes that contain >=2 children (containment > 0.7).
- resolve_overlaps: pairwise overlap resolution with priority for uncropable
  classes (Table, Picture, Formula, Code, Form, Key-value-region) and
  geometric crop for the rest.

Migrated and generalized from clean_overlaps_v3.py — that script also
talked to CVAT directly; the upload responsibility now lives in
core.cvat_sync.push (Plan 02), and these are pure functions.

16 tests in tests/test_postproc.py cover geometry, threshold, filter,
NMS, wrapper-kill, crop edge-cases, and overlap resolution.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Implement `core/lib/model.py` for Heron inference

**Files:**
- Create: `core/lib/model.py`, `tests/test_model.py`

The model module:
1. Defines `MODEL_INDEX_TO_LABEL_NAME` — a fixed list mapping the 17 native model output indices (0..16) to label names. Order matches the legacy `generate_heron_coco.py` LABELS dict (DocLayNet's pre-training order, NOT the project's CVAT order).
2. Provides `load_heron(device="auto") -> tuple[model, processor]`.
3. (Plan 04 will add `apply_lora`, `load_lora_state` — out of scope here.)

The label index → name mapping (DocLayNet pre-training order):
```
0: Caption           1: Footnote          2: Formula
3: List-item         4: Page-footer       5: Page-header
6: Picture           7: Section-header    8: Table
9: Text             10: Title            11: Document Index
12: Code            13: Checkbox-selected 14: Checkbox-unselected
15: Form            16: Key-value-region
```

- [ ] **Step 1: Write failing tests**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_model.py`:

```python
"""Tests for core.lib.model — label mapping (no real model loading)."""
import pytest

from core.lib.model import MODEL_INDEX_TO_LABEL_NAME, label_name_for_index


def test_mapping_has_17_entries():
    assert len(MODEL_INDEX_TO_LABEL_NAME) == 17


def test_mapping_is_in_doclaynet_native_order():
    # First 5 entries match DocLayNet pre-training order
    assert MODEL_INDEX_TO_LABEL_NAME[0] == "Caption"
    assert MODEL_INDEX_TO_LABEL_NAME[1] == "Footnote"
    assert MODEL_INDEX_TO_LABEL_NAME[2] == "Formula"
    assert MODEL_INDEX_TO_LABEL_NAME[3] == "List-item"
    assert MODEL_INDEX_TO_LABEL_NAME[8] == "Table"
    assert MODEL_INDEX_TO_LABEL_NAME[16] == "Key-value-region"


def test_label_name_for_index_returns_string():
    assert label_name_for_index(0) == "Caption"
    assert label_name_for_index(8) == "Table"


def test_label_name_for_index_raises_on_out_of_range():
    with pytest.raises(IndexError):
        label_name_for_index(17)
    with pytest.raises(IndexError):
        label_name_for_index(-1)
```

- [ ] **Step 2: Run, expect import failure**

```bash
uv run pytest tests/test_model.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `core/lib/model.py`**

Write `/home/alonso/Documentos/Github/document-layout-model-training/core/lib/model.py`:

```python
"""Heron model loader for inference (Plan 03 — baseline only).

Plan 04 will extend this module with apply_lora() and load_lora_state()
for fine-tuned model paths. The label-index mapping is fixed by the
model architecture (DocLayNet pre-training order), so it stays here.
"""
from __future__ import annotations

from typing import Tuple

# DocLayNet native order — fixed by the Heron checkpoint, do not reorder.
MODEL_INDEX_TO_LABEL_NAME: tuple[str, ...] = (
    "Caption",            # 0
    "Footnote",           # 1
    "Formula",            # 2
    "List-item",          # 3
    "Page-footer",        # 4
    "Page-header",        # 5
    "Picture",            # 6
    "Section-header",     # 7
    "Table",              # 8
    "Text",               # 9
    "Title",              # 10
    "Document Index",     # 11
    "Code",               # 12
    "Checkbox-selected",  # 13
    "Checkbox-unselected",# 14
    "Form",               # 15
    "Key-value-region",   # 16
)

DEFAULT_BASE_MODEL = "docling-project/docling-layout-heron"


def label_name_for_index(idx: int) -> str:
    if idx < 0 or idx >= len(MODEL_INDEX_TO_LABEL_NAME):
        raise IndexError(f"label index {idx} out of range (0..{len(MODEL_INDEX_TO_LABEL_NAME)-1})")
    return MODEL_INDEX_TO_LABEL_NAME[idx]


def load_heron(model_name: str = DEFAULT_BASE_MODEL, device: str = "auto") -> Tuple[object, object]:
    """Load Heron model + processor for inference.

    Args:
        model_name: HuggingFace model identifier.
        device: "auto" (use CUDA if available), "cuda", or "cpu".

    Returns:
        (model, processor) tuple. Model is moved to device and put in eval mode.
    """
    import torch
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrV2ForObjectDetection.from_pretrained(model_name)
    model.to(device).eval()
    return model, processor
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
uv run pytest tests/test_model.py -v
```
Expected: 4 tests pass. (No real model loading — those tests are intentionally absent; integration-level loading is verified in Task 4's smoke test.)

- [ ] **Step 5: Commit**

```bash
git add core/lib/model.py tests/test_model.py
git commit -m "feat(core): add model loader for Heron inference

core.lib.model:
- MODEL_INDEX_TO_LABEL_NAME: tuple of 17 strings in DocLayNet pre-training
  order (0=Caption, 8=Table, 16=Key-value-region). This order is fixed by
  the Heron checkpoint and DIFFERENT from CVAT's 1-based label IDs.
- label_name_for_index(idx): translates a model output index to a label name.
- load_heron(model_name, device): loads RTDetrV2 + processor for inference,
  moves to device, puts in eval mode. Plan 04 will add apply_lora here.

4 tests in tests/test_model.py cover the mapping. Real model loading is
exercised in Plan 03 Task 4's smoke test (network + GPU dependent).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Implement `core/predict.py` and wire CLI

**Files:**
- Create: `core/predict.py`, `tests/test_predict.py`
- Modify: `core/cli.py` (add `predict` command), `tests/test_cli.py` (add help test)

The `predict` function in pre-annotate mode:
1. Loads project config.
2. Loads Heron model + processor (from `core.lib.model`).
3. For each `projects/<slug>/data/images/<task>/pagina-NNN.png`:
   a. Run inference.
   b. Translate model outputs to detection dicts (via `MODEL_INDEX_TO_LABEL_NAME`).
   c. Apply per-class thresholds (from config).
   d. Apply `nms_per_category`.
   e. Apply `full_page_picture_filter`.
4. Build a project-level COCO with all images and annotations.
5. Translate label names to category_ids using the project's `labels` list (1-based, CVAT order).
6. Write to `projects/<slug>/cvat/pre_annotations/<YYYYMMDD-HHMMSS>.json`.

Note: this Plan 03 only applies threshold + NMS + full-page filter. The full `kill_wrappers` + `resolve_overlaps` pipeline is run by Docling itself at inference time AND by `clean_overlaps_v3.py` legacy script. For Plan 03 we keep the pre-annotation pipeline simple: threshold + NMS + page filter — that matches what `generate_heron_coco.py` did. Plan 04+ can add an optional `--clean` flag that also runs `kill_wrappers + resolve_overlaps`.

- [ ] **Step 1: Write failing tests**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_predict.py`:

```python
"""Tests for core.predict — pre-annotation pipeline.

Mocks the model and processor; uses small fake images.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from core.predict import predict


@pytest.fixture()
def fake_project(tmp_path, monkeypatch):
    proj = tmp_path / "projects" / "test"
    (proj / "data" / "pdfs").mkdir(parents=True)
    (proj / "data" / "images" / "doc-A").mkdir(parents=True)
    (proj / "cvat").mkdir(parents=True)
    # Create 2 fake PNGs (real RGB images so PIL.open works)
    from PIL import Image
    for i in (1, 2):
        Image.new("RGB", (100, 200), color=(255, 255, 255)).save(
            proj / "data" / "images" / "doc-A" / f"pagina-{i:03d}.png"
        )
    (proj / "data" / "pdfs" / "doc-A.pdf").write_bytes(b"%PDF")
    (proj / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "project": {"slug": "test"},
                "labels": ["Caption", "Picture", "Table", "Text"],
                "postprocess": {
                    "thresholds": {"default": 0.5},
                    "nms_iou": 0.5,
                    "full_page_picture_filter": 0.9,
                },
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    return proj


def _fake_load_heron(model_name, device):
    """Returns a (model, processor) pair where post_process_object_detection
    returns one Caption box at score 0.9 per image."""
    import torch

    model = MagicMock()
    processor = MagicMock()

    def _processor_call(images, return_tensors):
        return {"pixel_values": torch.zeros(1, 3, 100, 100)}

    processor.side_effect = _processor_call

    def _post_process(outputs, target_sizes, threshold):
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),  # 0 = Caption (DocLayNet order)
            }
        ]

    processor.post_process_object_detection = _post_process

    model_called = MagicMock()
    model.return_value = model_called  # forward() returns object, used in `outputs`

    # Make the processor callable
    processor.__call__ = _processor_call
    return model, processor


def test_predict_writes_coco_with_correct_label_id(fake_project, monkeypatch):
    monkeypatch.setattr("core.predict.load_heron", _fake_load_heron)
    # No torch.cuda needed — _fake_load_heron avoids that path
    predict("test", mode="pre-annotate")

    # Find the produced COCO
    out_files = sorted((fake_project / "cvat" / "pre_annotations").glob("*.json"))
    assert len(out_files) == 1
    coco = json.loads(out_files[0].read_text())

    # Should have 2 images, 2 annotations (one Caption per image)
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 2
    # category_id is the 1-based position of "Caption" in the project's labels list
    # labels = ["Caption", "Picture", "Table", "Text"] -> Caption is id 1
    assert all(ann["category_id"] == 1 for ann in coco["annotations"])
    assert all(ann["score"] == pytest.approx(0.9) for ann in coco["annotations"])


def test_predict_drops_below_threshold(fake_project, monkeypatch):
    """Lower the score and confirm filtering."""
    def _low_score_load(model_name, device):
        m, p = _fake_load_heron(model_name, device)
        import torch

        def _post(outputs, target_sizes, threshold):
            return [{
                "boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]),
                "scores": torch.tensor([0.3]),  # below default 0.5
                "labels": torch.tensor([0]),
            }]
        p.post_process_object_detection = _post
        return m, p

    monkeypatch.setattr("core.predict.load_heron", _low_score_load)
    predict("test", mode="pre-annotate")

    out_files = sorted((fake_project / "cvat" / "pre_annotations").glob("*.json"))
    coco = json.loads(out_files[-1].read_text())
    assert coco["annotations"] == []
```

- [ ] **Step 2: Run, confirm import failure**

```bash
uv run pytest tests/test_predict.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.predict'`.

- [ ] **Step 3: Implement `core/predict.py`**

Write `/home/alonso/Documentos/Github/document-layout-model-training/core/predict.py`:

```python
"""Pre-annotation: run Heron baseline over a project's images, write COCO predictions.

Output goes to `projects/<slug>/cvat/pre_annotations/<timestamp>.json`.
This file is then consumed by `dlmf cvat-push --coco=<file>` to pre-load
predictions into CVAT for human review.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
from pathlib import Path
from typing import Any

from core.lib.config import load_config
from core.lib.model import MODEL_INDEX_TO_LABEL_NAME, load_heron
from core.lib.postproc import (
    apply_thresholds,
    full_page_picture_filter,
    nms_per_category,
)

PROJECTS_ROOT = Path("projects")


def predict(
    project_slug: str,
    mode: str = "pre-annotate",
    threshold: float | None = None,
    limit: int | None = None,
) -> Path:
    """Run inference and write a COCO file. Returns the output path.

    `mode='pre-annotate'` is the only mode in Plan 03. Plan 05 adds 'visualize'.
    `threshold` overrides the config's `postprocess.thresholds.default` if given.
    `limit` is an integer that caps the number of images processed (smoke testing).
    """
    if mode != "pre-annotate":
        raise NotImplementedError(f"mode={mode!r} not implemented yet")

    project_dir = PROJECTS_ROOT / project_slug
    cfg = load_config(project_dir / "config.yaml")
    labels: list[str] = list(cfg["labels"])
    label_to_cat_id = {name: i + 1 for i, name in enumerate(labels)}

    pp_cfg = cfg.get("postprocess", {})
    thresholds = dict(pp_cfg.get("thresholds", {}))
    default_thr = float(threshold if threshold is not None else thresholds.pop("default", 0.5))
    nms_iou = float(pp_cfg.get("nms_iou", 0.5))
    fullpage_frac = float(pp_cfg.get("full_page_picture_filter", 0.9))

    images_root = project_dir / "data" / "images"
    image_dirs = sorted(d for d in images_root.iterdir() if d.is_dir())
    if not image_dirs:
        raise FileNotFoundError(f"no image directories in {images_root}")

    # Collect all images across all task dirs.
    all_pngs: list[Path] = []
    for d in image_dirs:
        all_pngs.extend(sorted(d.glob("pagina-*.png")))
    if limit is not None:
        all_pngs = all_pngs[:limit]
    if not all_pngs:
        raise FileNotFoundError(f"no PNGs under {images_root}")

    print(f"[predict] loading Heron model...")
    model, processor = load_heron()

    import torch
    from PIL import Image

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 1

    for img_idx, png in enumerate(all_pngs, start=1):
        image = Image.open(png).convert("RGB")
        w, h = image.size
        coco_images.append({
            "id": img_idx,
            "width": w,
            "height": h,
            "file_name": png.name,
        })

        inputs = processor(images=image, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        if hasattr(model, "device"):
            target_sizes = target_sizes.to(model.device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=default_thr
        )[0]

        # Translate to detection dicts.
        dets = []
        for box, score, lbl in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            name = MODEL_INDEX_TO_LABEL_NAME[int(lbl.item())]
            dets.append({
                "box": (x1, y1, x2, y2),
                "label": name,
                "score": float(score.item()),
            })

        # Postprocess: per-class threshold (we already passed default_thr to
        # post_process, but apply per-class for the ones above default), NMS,
        # full-page picture filter.
        dets = apply_thresholds(dets, thresholds, default=default_thr)
        dets = nms_per_category(dets, iou_threshold=nms_iou)
        dets = full_page_picture_filter(dets, page_w=w, page_h=h, min_fraction=fullpage_frac)

        # Convert to COCO annotations.
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            cat_id = label_to_cat_id.get(d["label"])
            if cat_id is None:
                continue  # label not in this project's vocabulary
            bw, bh = x2 - x1, y2 - y1
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": cat_id,
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "segmentation": [],
                "score": round(d["score"], 4),
                "attributes": {"occluded": False, "rotation": 0.0},
            })
            ann_id += 1

        # Cleanup per page to keep VRAM use low.
        del outputs, inputs, image
        if (img_idx % 25) == 0:
            print(f"[predict]   {img_idx}/{len(all_pngs)} pages")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    # Build COCO.
    categories = [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(labels)]
    coco = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "description": f"Heron baseline pre-annotations for {project_slug}",
            "date_created": dt.datetime.now().isoformat(),
        },
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    out_dir = project_dir / "cvat" / "pre_annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{ts}.json"
    out_path.write_text(json.dumps(coco))
    print(f"[predict] wrote {out_path}: {len(coco_images)} images, {len(coco_annotations)} annotations")
    return out_path
```

- [ ] **Step 4: Add CLI command in `core/cli.py`**

Add (after the `cvat_pull_cmd` function, before the `if __name__ == "__main__":`):

```python
@app.command(name="predict")
def predict_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug."),
    pre_annotate: bool = typer.Option(
        False,
        "--pre-annotate",
        help="Generate COCO predictions to projects/<slug>/cvat/pre_annotations/<timestamp>.json.",
    ),
    threshold: float = typer.Option(
        None, "--threshold", help="Override the default confidence threshold."
    ),
    limit: int = typer.Option(
        None, "--limit", help="Cap the number of images (smoke testing)."
    ),
) -> None:
    """Run the Heron baseline (or, in later plans, a fine-tuned model) over the project's images."""
    if not pre_annotate:
        raise typer.BadParameter("must pass --pre-annotate (other modes added in later plans)")
    from core.predict import predict

    predict(project, mode="pre-annotate", threshold=threshold, limit=limit)
```

- [ ] **Step 5: Update `tests/test_cli.py`**

Append:
```python
def test_cli_predict_help_mentions_required_flags():
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout
    assert "--pre-annotate" in result.stdout
    assert "--threshold" in result.stdout
    assert "--limit" in result.stdout
```

Also update `test_cli_help_lists_subcommands` to include `predict`:
```python
def test_cli_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "render" in result.stdout
    assert "cvat-push" in result.stdout
    assert "cvat-pull" in result.stdout
    assert "predict" in result.stdout
```

- [ ] **Step 6: Run all relevant tests**

```bash
uv run pytest tests/test_predict.py tests/test_cli.py -v
```
Expected: 2 predict tests + 5 CLI tests pass.

- [ ] **Step 7: Commit**

```bash
git add core/predict.py core/cli.py tests/test_predict.py tests/test_cli.py
git commit -m "feat(core): add 'dlmf predict --pre-annotate' command

core.predict.predict(project_slug, mode='pre-annotate', threshold=None,
limit=None) does:
- Load project config + Heron baseline (via core.lib.model.load_heron).
- Iterate all PNGs in projects/<slug>/data/images/<task>/.
- Run inference, translate model indices to label names via
  MODEL_INDEX_TO_LABEL_NAME (DocLayNet native order).
- Apply per-class thresholds (from config), NMS per category, and
  full-page-picture filter.
- Build COCO with category_ids matching the project's 1-based labels list.
- Write to projects/<slug>/cvat/pre_annotations/<YYYYMMDD-HHMMSS>.json.

CLI command 'dlmf predict --project=eaf --pre-annotate' invokes it.
The optional --limit caps the number of pages (smoke testing) and
--threshold overrides the config's default confidence cutoff.

2 unit tests in tests/test_predict.py mock the model + processor; 1 new
CLI help test ensures the flags are discoverable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Smoke test against the real EAF data

This task runs `dlmf predict` against the actual EAF images, on a small subset first, then validates the output. No new code is written; this is a verification step.

- [ ] **Step 1: Smoke test with --limit=5**

```bash
uv run dlmf predict --project=eaf --pre-annotate --limit=5 2>&1 | tail -20
```
Expected:
- `[predict] loading Heron model...` (downloads ~170MB the first time, cached after)
- Iterates 5 pages.
- `[predict] wrote projects/eaf/cvat/pre_annotations/<ts>.json: 5 images, N annotations` (N typically ~20-40 for 5 pages of EAF).

If this fails for memory reasons (4GB GPU is tight): try CPU mode with `CUDA_VISIBLE_DEVICES="" uv run dlmf predict --project=eaf --pre-annotate --limit=5`. Document which mode was used.

- [ ] **Step 2: Validate the smoke output**

```bash
ls projects/eaf/cvat/pre_annotations/
SMOKE=$(ls -t projects/eaf/cvat/pre_annotations/*.json | head -1)
python3 -c "
import json
d = json.load(open('$SMOKE'))
print(f'images: {len(d[\"images\"])}')
print(f'annotations: {len(d[\"annotations\"])}')
print(f'categories: {len(d[\"categories\"])}')
print(f'sample annotations:')
for a in d['annotations'][:3]:
    print(f'  cat={a[\"category_id\"]} score={a[\"score\"]} bbox={a[\"bbox\"]}')
"
```
Expected: 5 images, ≥1 annotation, 17 categories. Each annotation has bbox, score, category_id.

- [ ] **Step 3: Full run (all 561 images)** — only if Step 1 succeeded

```bash
uv run dlmf predict --project=eaf --pre-annotate 2>&1 | tail -20
```
Expected: ~5-15 minutes on GPU (depends on warm-up + I/O). Final output: `561 images, M annotations` where M is in the few-thousands range.

If this is too slow or runs out of memory, skip and document. Step 1 + 2 are sufficient as smoke validation.

- [ ] **Step 4: Commit the smoke output**

```bash
ls -lh projects/eaf/cvat/pre_annotations/
git add projects/eaf/cvat/pre_annotations/
git commit -m "chore(eaf): add Heron baseline pre-annotations from Plan 03 smoke test

Generated by 'dlmf predict --project=eaf --pre-annotate' as proof
that the pipeline works end-to-end against the real EAF images.

These pre-annotations are the input to 'dlmf cvat-push --coco=<file>'
when bootstrapping a new CVAT round (model-assisted labeling).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Delete superseded scripts

- [ ] **Step 1: Verify scripts are fully replaced**

```bash
uv run dlmf predict --help | head -3
ls generate_heron_coco.py clean_overlaps_v3.py
```
Expected: predict CLI has its help; both legacy scripts still exist.

- [ ] **Step 2: Search for references**

```bash
grep -rn "generate_heron_coco\|clean_overlaps_v3" \
  --include="*.py" --include="*.toml" --include="*.yaml" --include="*.md" \
  --exclude-dir=.venv --exclude-dir=.git --exclude-dir=docs/superpowers \
  2>/dev/null | head -10
```
Expected: hits ONLY in `README.md` and possibly `conversacion_claude.md` (legacy doc). No hits in active `.py` files.

- [ ] **Step 3: Delete**

```bash
git rm generate_heron_coco.py clean_overlaps_v3.py
```

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove superseded inference and overlap-cleaning scripts

These are now fully replaced by the dlmf CLI:
- generate_heron_coco.py  → dlmf predict --pre-annotate
- clean_overlaps_v3.py    → core.lib.postproc (pure functions; CVAT
                            upload via dlmf cvat-push --coco=<file>)

The README still references them in narrative sections; that will
be cleaned up when the README is fully rewritten in Plan 06.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Final verification + tag

- [ ] **Step 1: Run all tests**

```bash
uv run pytest -v 2>&1 | tail -20
```
Expected: 33 (planes 01-02) + 16 (postproc) + 4 (model) + 2 (predict) + 1 (new CLI test) = **56 tests** all green.

- [ ] **Step 2: Verify CLI**

```bash
uv run dlmf --help
uv run dlmf predict --help
```
Expected: predict listed in main help; predict --help shows --project, --pre-annotate, --threshold, --limit.

- [ ] **Step 3: Verify deletions**

```bash
ls generate_heron_coco.py clean_overlaps_v3.py 2>&1 | grep -E "No such|cannot|no se puede"
```
Expected: both missing.

- [ ] **Step 4: Verify the pre-annotation file was committed**

```bash
git log --diff-filter=A --name-only -- 'projects/eaf/cvat/pre_annotations/*' | head
```
Expected: shows the smoke test JSON.

- [ ] **Step 5: Tag**

```bash
git tag -a plan-03-predict-postproc -m "Plan 03 complete: dlmf predict --pre-annotate runs Heron baseline + postproc end-to-end."
git tag -l "plan-*"
```

- [ ] **Step 6: Summary**

```bash
git log $(git rev-list -n 1 plan-02-render-cvat-cli)..HEAD --oneline
git log master..HEAD --oneline | wc -l
```

---

## Self-Review

**Spec coverage:**
- ✅ Spec section 5 (CLI): `dlmf predict --pre-annotate` shipped.
- ✅ Spec section 6 (post-processing): `kill_wrappers`, `resolve_overlaps`, `full_page_picture_filter`, `nms_per_category`, `apply_thresholds` all implemented as pure functions.
- ✅ Spec section 4 (config drives behavior): `predict` reads `postprocess.thresholds`, `postprocess.nms_iou`, `postprocess.full_page_picture_filter` from project config.
- 🔜 LoRA loading for fine-tuned models: deferred to Plan 04 (predict still works for inference; the `--model=<run>` flag will be added when there's a trained model to point at).
- 🔜 PDF visualization (`predict --output=anotado.pdf`): deferred to Plan 05.

**Placeholder scan:** No `TBD`/`TODO`. All commands have full code.

**Type/name consistency:**
- `MODEL_INDEX_TO_LABEL_NAME` (model.py) ↔ DocLayNet native order ↔ used in predict.py.
- `Detection` schema (`box` tuple, `label` str, `score` float) consistent across postproc + predict.
- `category_id` in COCO output matches the project's 1-based labels list (CVAT-compatible).
