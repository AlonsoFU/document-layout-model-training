"""Tests for core.train — only the pure helpers (real training is in Tasks 6-8 smoke)."""
import pytest

from core.train import _data_split


def test_data_split_is_deterministic_with_seed():
    image_ids = list(range(100))
    s1 = _data_split(image_ids, val_fraction=0.2, seed=42)
    s2 = _data_split(image_ids, val_fraction=0.2, seed=42)
    assert s1 == s2


def test_data_split_proportions():
    image_ids = list(range(100))
    train, val = _data_split(image_ids, val_fraction=0.15, seed=42)
    assert len(val) == 15
    assert len(train) == 85
    assert set(train).isdisjoint(val)


def test_data_split_handles_small_sets():
    train, val = _data_split([1, 2, 3], val_fraction=0.5, seed=42)
    assert len(train) + len(val) == 3
    assert len(val) >= 1
