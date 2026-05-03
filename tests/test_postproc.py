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
