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
