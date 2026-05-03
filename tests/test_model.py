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


def test_apply_lora_replaces_target_linears():
    """Smoke: build a tiny model with named linears matching the patterns and confirm they're replaced."""
    import torch
    from core.lib.model import LoRALinear, apply_lora

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(10, 10)
            self.k_proj = torch.nn.Linear(10, 10)
            self.v_proj = torch.nn.Linear(10, 10)
            self.unrelated = torch.nn.Linear(10, 10)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.ModuleList([FakeAttention()])

    model = FakeModel()
    apply_lora(model, rank=4, alpha=8, target_substrings=["q_proj", "k_proj", "v_proj"])

    decoder_layer = model.decoder[0]
    assert isinstance(decoder_layer.q_proj, LoRALinear)
    assert isinstance(decoder_layer.k_proj, LoRALinear)
    assert isinstance(decoder_layer.v_proj, LoRALinear)
    # unrelated should not be wrapped
    assert not isinstance(decoder_layer.unrelated, LoRALinear)


def test_lora_state_dict_extracts_only_lora_weights():
    import torch
    from core.lib.model import LoRALinear, apply_lora, lora_state_dict

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(10, 10)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.ModuleList([FakeAttention()])

    model = FakeModel()
    apply_lora(model, rank=4, alpha=8, target_substrings=["q_proj"])
    sd = lora_state_dict(model)
    # Should have lora_A and lora_B for the wrapped layer
    assert any("lora_A" in k for k in sd.keys())
    assert any("lora_B" in k for k in sd.keys())
    # Should NOT have the original weights
    assert not any("original.weight" in k for k in sd.keys())
