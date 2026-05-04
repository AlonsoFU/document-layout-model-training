"""YAML config loader with !include resolution and CLI override application."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader extension that resolves `!include <path>` relative to the YAML file."""


def _construct_include(loader: _IncludeLoader, node: yaml.Node) -> Any:
    rel = loader.construct_scalar(node)
    base = Path(loader.name).parent if loader.name else Path.cwd()
    target = (base / rel).resolve()
    with target.open("r", encoding="utf-8") as f:
        sub_loader = _IncludeLoader(f)
        sub_loader.name = str(target)
        try:
            return sub_loader.get_single_data()
        finally:
            sub_loader.dispose()


_IncludeLoader.add_constructor("!include", _construct_include)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config from `path`, resolving any `!include` directives."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        loader = _IncludeLoader(f)
        loader.name = str(p)
        try:
            data = loader.get_single_data()
        finally:
            loader.dispose()
    return data or {}


def _coerce(value: str) -> Any:
    """Parse override value as YAML scalar (handles ints, floats, bools, strings).

    PyYAML 6.x does not recognise bare scientific notation like ``5e-5`` as a
    float (it needs a leading digit and decimal point, e.g. ``5.0e-5``).  We
    fall back to Python's ``float()`` when yaml.safe_load returns a string
    that Python itself can parse as a float.
    """
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        try:
            return float(parsed)
        except ValueError:
            pass
    return parsed


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a deep copy of `cfg` with dotted-key overrides applied.

    Overrides are strings of the form `a.b.c=value`. The leaf must already
    exist in `cfg`; missing keys raise KeyError to catch typos.
    """
    out = copy.deepcopy(cfg)
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"override missing '=': {raw!r}")
        dotted, value_str = raw.split("=", 1)
        keys = dotted.split(".")
        node = out
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise KeyError(dotted)
            node = node[k]
        leaf = keys[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(dotted)
        node[leaf] = _coerce(value_str)
    return out
