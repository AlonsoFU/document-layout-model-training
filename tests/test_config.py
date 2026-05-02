from pathlib import Path

import pytest

from core.lib.config import apply_overrides, load_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_config_reads_basic_yaml():
    cfg = load_config(FIXTURES / "sample_config.yaml")
    assert cfg["project"]["slug"] == "testdoc"
    assert cfg["training"]["lr"] == 1.0e-4


def test_load_config_resolves_include_relative_to_file():
    cfg = load_config(FIXTURES / "sample_config.yaml")
    assert cfg["labels"] == ["LabelOne", "LabelTwo", "LabelThree"]


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(FIXTURES / "does_not_exist.yaml")


def test_apply_overrides_sets_nested_key():
    cfg = {"training": {"lora": {"rank": 32}}}
    out = apply_overrides(cfg, ["training.lora.rank=64"])
    assert out["training"]["lora"]["rank"] == 64
    # Original is untouched (returns a copy).
    assert cfg["training"]["lora"]["rank"] == 32


def test_apply_overrides_parses_floats_and_ints():
    cfg = {"training": {"lr": 1e-4, "batch_size": 1}}
    out = apply_overrides(cfg, ["training.lr=5e-5", "training.batch_size=2"])
    assert out["training"]["lr"] == 5e-5
    assert out["training"]["batch_size"] == 2


def test_apply_overrides_parses_booleans():
    cfg = {"augmentation": {"enabled": False}}
    out = apply_overrides(cfg, ["augmentation.enabled=true"])
    assert out["augmentation"]["enabled"] is True


def test_apply_overrides_unknown_key_raises():
    cfg = {"training": {"lr": 1e-4}}
    with pytest.raises(KeyError, match="training.typo"):
        apply_overrides(cfg, ["training.typo=5"])


def test_apply_overrides_no_overrides_returns_equal_dict():
    cfg = {"a": 1, "b": {"c": 2}}
    out = apply_overrides(cfg, [])
    assert out == cfg
