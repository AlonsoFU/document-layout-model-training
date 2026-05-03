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
