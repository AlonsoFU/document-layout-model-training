"""Heron model loader for inference (Plan 03 — baseline only).

Plan 04 will extend this module with apply_lora() and load_lora_state()
for fine-tuned model paths. The label-index mapping is fixed by the
model architecture (DocLayNet pre-training order), so it stays here.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

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


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around an existing nn.Linear.

    During forward: out = original(x) + (lora_B @ lora_A @ dropout(x)) * (alpha/rank).
    The original layer's weights are frozen; only lora_A and lora_B train.
    """

    def __init__(self, original: nn.Linear, rank: int = 32, alpha: int = 64, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_lora(
    model: nn.Module,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    target_substrings: list[str] | None = None,
    scope: str = "decoder",
) -> int:
    """Replace target nn.Linear layers under `scope` with LoRALinear wrappers.

    A layer is wrapped iff its qualified name contains `scope` AND any of
    `target_substrings` (default: q_proj, k_proj, v_proj). Returns the number
    of replacements done.
    """
    if target_substrings is None:
        target_substrings = ["q_proj", "k_proj", "v_proj"]

    replaced = 0
    # We can't mutate during iteration; collect first.
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if scope not in name or not isinstance(module, nn.Linear):
            continue
        if not any(sub in name for sub in target_substrings):
            continue
        # Find parent.
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        targets.append((parent, parts[-1], module))

    for parent, leaf, original in targets:
        setattr(parent, leaf, LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1
    return replaced


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA weights (lora_A, lora_B) — typically <10MB for r=32."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora_state(model: nn.Module, state: dict, strict: bool = False) -> None:
    """Load LoRA-only state dict into a model that has been apply_lora'd."""
    own = model.state_dict()
    for k, v in state.items():
        if k in own:
            own[k].copy_(v)
        elif strict:
            raise KeyError(f"LoRA key {k} not found in model")
