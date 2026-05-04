# Plan 04 — Training Infrastructure + MLflow + Reproduce P_repeat_factor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir la infraestructura completa de fine-tuning (data + model LoRA + eval + tracking + train loop + CLI) y reproducir el experimento ganador `P_repeat_factor` (LoRA r=32, RFS, mAP@[.5:.95] = 0.8700) con la nueva pipeline. Parity gate: mAP final ≥ 0.86.

**Architecture:** Cinco módulos nuevos en `core/lib/` y `core/`:
- `core/lib/data.py` — `CocoDocDataset`, `RepeatFactorSampler`, `DocumentAugmenter`, `collate_fn`.
- `core/lib/model.py` (extensión) — `apply_lora`, `lora_state_dict`, `load_lora_state`.
- `core/lib/eval.py` — `compute_map_at_iou_range` (mAP@[.5:.95]).
- `core/lib/tracking.py` — `MlflowRun` context manager.
- `core/train.py` — `train(project_slug, run_name, overrides)` orquesta todo.
- CLI: `dlmf train --project=eaf --run=<name> [--override KEY=VAL]...`.

**Tech Stack:** PyTorch + transformers (RT-DETR v2), torchvision (NMS), Pillow (augmentation), MLflow.

**Salida verificable:**
- Smoke: `dlmf train --project=eaf --run=smoke --override training.max_epochs=2 --override training.limit=50` corre sin errores y deja artefactos en `projects/eaf/runs/smoke/`.
- **Parity gate**: `dlmf train --project=eaf --run=P_repeat_factor_v2` (con los hyperparams default del config, que reproducen el experimento original) alcanza **mAP@[.5:.95] ≥ 0.86** dentro de 50 epochs.
- MLflow UI muestra ambos runs con params, metrics, artifacts.
- Tag `plan-04-training-mlflow` apunta al último commit.

**Out of scope:**
- Plan 05: `evaluate.py` standalone (mAP per-PDF, per-class), `promote.py`, `predict --output=PDF`.
- Plan 06: `classify_doctype.py`, `init-project`.

**Reference:** `training/train_round4.py` contiene la implementación legacy de LoRA + dataset + augmenter + eval. Reusable casi tal cual; solo adaptar a config-driven y MLflow.

---

## Tasks

### Task 1: `core/lib/data.py` — CocoDocDataset + RFS sampler + Augmenter

**Files:**
- Create: `core/lib/data.py`, `tests/test_data.py`

The dataset reads a COCO JSON + image directory and returns `(pixel_values, target)` per item. Targets are CXCYWH-normalized boxes + 0-based class labels (model expects 0-based; COCO is 1-based).

`RepeatFactorSampler` computes per-image repeat factors based on the rarest class each image contains. Formula:
- For each class `c`, compute `f(c) = num_images_with_c / total_images`.
- For each image `i`, repeat factor `r(i) = max(1.0, max_{c in i} sqrt(threshold / f(c)))`.
- Threshold 0.5 means classes present in <50% of images get oversampled; classes in >50% get factor 1.

`DocumentAugmenter` ports the legacy: brightness/contrast jitter (50%), rotation ±3° (30%), gaussian blur (30%).

- [ ] **Step 1: Tests**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_data.py`:

```python
"""Tests for core.lib.data — COCO dataset, RFS sampler, augmenter."""
import json
from pathlib import Path

import pytest
from PIL import Image

from core.lib.data import (
    CocoDocDataset,
    DocumentAugmenter,
    RepeatFactorSampler,
    collate_fn,
)


@pytest.fixture()
def fake_coco(tmp_path):
    """Build a tiny COCO file + 3 PNG images. 3 images, 4 anns, 2 classes."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in (1, 2, 3):
        Image.new("RGB", (100, 200), (255, 255, 255)).save(img_dir / f"p{i}.png")
    coco = {
        "categories": [
            {"id": 1, "name": "Common"},   # appears in 3/3 images
            {"id": 2, "name": "Rare"},     # appears in 1/3 images
        ],
        "images": [
            {"id": 1, "file_name": "p1.png", "width": 100, "height": 200},
            {"id": 2, "file_name": "p2.png", "width": 100, "height": 200},
            {"id": 3, "file_name": "p3.png", "width": 100, "height": 200},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 4, "image_id": 3, "category_id": 2, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
        ],
    }
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps(coco))
    return coco_path, img_dir


def test_dataset_len_matches_image_count(fake_coco):
    coco_path, img_dir = fake_coco
    ds = CocoDocDataset(coco_path, img_dir, image_ids=[1, 2, 3])
    assert len(ds) == 3


def test_dataset_returns_pixel_values_and_target(fake_coco):
    """Item is (pixel_values, target). Target has boxes (CXCYWH normalized) and 0-based class_labels."""
    coco_path, img_dir = fake_coco
    ds = CocoDocDataset(coco_path, img_dir, image_ids=[3])  # has both classes
    px, target = ds[0]
    assert px.ndim == 3  # CHW
    assert "boxes" in target and "class_labels" in target
    assert target["boxes"].shape[0] == 2  # 2 anns on image 3
    assert target["boxes"].shape[1] == 4  # CXCYWH
    assert (target["boxes"] >= 0).all() and (target["boxes"] <= 1).all()
    # class_labels are 0-based (model expects 0-based; COCO is 1-based)
    assert set(target["class_labels"].tolist()) == {0, 1}


def test_dataset_filters_to_specified_image_ids(fake_coco):
    coco_path, img_dir = fake_coco
    ds = CocoDocDataset(coco_path, img_dir, image_ids=[1])
    assert len(ds) == 1


def test_repeat_factor_sampler_oversamples_rare(fake_coco):
    coco_path, _ = fake_coco
    sampler = RepeatFactorSampler.from_coco(coco_path, image_ids=[1, 2, 3], threshold=0.5)
    # Image 3 has the rare class (Rare appears in 1/3 = 33% < threshold 50%).
    # Image 1 and 2 have only Common (appears in 100% >= threshold), so factor = 1.
    factors = sampler.factors
    assert factors[1] == 1
    assert factors[2] == 1
    assert factors[3] > 1  # oversampled


def test_repeat_factor_sampler_iter_repeats_correctly():
    sampler = RepeatFactorSampler(factors={1: 1, 2: 3, 3: 2})
    indices = list(sampler)
    assert indices.count(1) == 1
    assert indices.count(2) == 3
    assert indices.count(3) == 2
    assert len(sampler) == 6


def test_collate_fn_stacks_batch():
    import torch
    px1 = torch.zeros(3, 10, 10)
    px2 = torch.zeros(3, 10, 10)
    t1 = {"boxes": torch.zeros((1, 4)), "class_labels": torch.tensor([0])}
    t2 = {"boxes": torch.zeros((2, 4)), "class_labels": torch.tensor([0, 1])}
    pixels, targets = collate_fn([(px1, t1), (px2, t2)])
    assert pixels.shape == (2, 3, 10, 10)
    assert len(targets) == 2


def test_augmenter_returns_image_and_boxes():
    """Augmenter is stochastic; just verify shape preservation."""
    import random
    random.seed(0)
    aug = DocumentAugmenter()
    img = Image.new("RGB", (50, 50), (128, 128, 128))
    boxes = [[0.5, 0.5, 0.2, 0.2]]
    out_img, out_boxes = aug(img, boxes)
    assert out_img.size == (50, 50)
    assert out_boxes == boxes  # boxes are not transformed (only image)
```

- [ ] **Step 2: Run, confirm import error.** `uv run pytest tests/test_data.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement `core/lib/data.py`**

Reuse the structure from `training/train_round4.py` but adapted:
- Class is `CocoDocDataset` (not `DocLayoutDataset`).
- Constructor takes `coco_path: Path`, `images_dir: Path`, `image_ids: list[int]`, optional `processor`, optional `augmenter`.
- Returns `(pixel_values, target)` as before.
- If `processor` is None, use `RTDetrImageProcessor.from_pretrained("docling-project/docling-layout-heron")` lazily.
- `RepeatFactorSampler.from_coco(coco_path, image_ids, threshold=0.5)` builds factors via the formula above.
- `RepeatFactorSampler.__iter__` yields image_ids repeated by their factor (shuffled per epoch).
- `RepeatFactorSampler.__len__` returns total samples per epoch.

```python
"""Dataset, RFS sampler, and augmenter for layout fine-tuning.

Adapted from training/train_round4.py — that script's DocLayoutDataset
is the reference. Differences here:
- Path-based init (takes a COCO file + image dir) instead of holding
  the entire coco_data dict in memory.
- RepeatFactorSampler implements the standard RFS formula based on
  the rarest class per image.
- Augmenter is a separate class so it can be turned off for val/eval.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


class CocoDocDataset(Dataset):
    def __init__(
        self,
        coco_path: Path,
        images_dir: Path,
        image_ids: list[int],
        processor=None,
        augmenter=None,
    ):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.augmenter = augmenter

        with open(coco_path) as f:
            coco = json.load(f)
        self.id_to_img = {img["id"]: img for img in coco["images"]}
        self.img_to_anns: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)
        self.image_ids = list(image_ids)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _ensure_processor(self):
        if self.processor is None:
            from transformers import RTDetrImageProcessor
            self.processor = RTDetrImageProcessor.from_pretrained(
                "docling-project/docling-layout-heron"
            )

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.id_to_img[img_id]
        image = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        anns = self.img_to_anns[img_id]
        iw, ih = info["width"], info["height"]
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([(x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih])
            labels.append(ann["category_id"] - 1)  # 0-based
        if self.augmenter and boxes:
            image, boxes = self.augmenter(image, boxes)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "class_labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
        }
        self._ensure_processor()
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), target


class RepeatFactorSampler:
    """RFS: oversample images that contain rare classes.

    For each class c, frequency f(c) = (#images with c) / (total images).
    For each image i, factor r(i) = max(1, max_{c in i} sqrt(threshold / f(c))).
    Floor to int (we don't fractionally repeat).
    """

    def __init__(self, factors: dict[int, int]):
        self.factors = dict(factors)

    @classmethod
    def from_coco(cls, coco_path: Path, image_ids: list[int], threshold: float = 0.5):
        with open(coco_path) as f:
            coco = json.load(f)
        ids_set = set(image_ids)
        # classes per image
        img_classes: dict[int, set[int]] = defaultdict(set)
        for ann in coco["annotations"]:
            if ann["image_id"] in ids_set:
                img_classes[ann["image_id"]].add(ann["category_id"])
        # class frequency
        n = len(image_ids)
        class_count: dict[int, int] = defaultdict(int)
        for classes in img_classes.values():
            for c in classes:
                class_count[c] += 1
        class_freq = {c: cnt / n for c, cnt in class_count.items()}
        # per-image factor
        factors: dict[int, int] = {}
        for img_id in image_ids:
            classes = img_classes.get(img_id, set())
            if not classes:
                factors[img_id] = 1
                continue
            ratios = [math.sqrt(threshold / class_freq[c]) for c in classes if class_freq[c] > 0]
            r = max(1.0, max(ratios) if ratios else 1.0)
            factors[img_id] = max(1, int(round(r)))
        return cls(factors)

    def __iter__(self) -> Iterator[int]:
        out: list[int] = []
        for img_id, factor in self.factors.items():
            out.extend([img_id] * factor)
        random.shuffle(out)
        return iter(out)

    def __len__(self) -> int:
        return sum(self.factors.values())


class DocumentAugmenter:
    """Image-only stochastic augmentation. Boxes are returned unchanged
    (legacy script also does not transform boxes for these augmentations)."""

    def __call__(self, image: Image.Image, boxes: list):
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            image = image.rotate(random.uniform(-3, 3), fillcolor=(255, 255, 255))
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return image, boxes


def collate_fn(batch):
    """Stack pixel_values; keep targets as a list (variable-length)."""
    pixels = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return pixels, targets
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
uv run pytest tests/test_data.py -v
```
Expected: 7 tests pass. Note: the dataset tests load `RTDetrImageProcessor` lazily — first run pulls weights; subsequent runs use the cache.

- [ ] **Step 5: Commit**

```bash
git add core/lib/data.py tests/test_data.py
git commit -m "feat(core): add data module — CocoDocDataset, RepeatFactorSampler, DocumentAugmenter

core.lib.data:
- CocoDocDataset: PyTorch Dataset that reads a COCO file + image dir,
  returns (pixel_values, target) per index. Boxes are CXCYWH-normalized,
  class_labels are 0-based (RT-DETR's expected format).
- RepeatFactorSampler: oversamples images with rare classes.
  factor(i) = max(1, max_{c in i} sqrt(threshold / freq(c))).
  Threshold 0.5 was the winning P_repeat_factor configuration.
- DocumentAugmenter: brightness/contrast jitter, ±3° rotation, gaussian
  blur (probabilities and ranges match the legacy train_round4.py).
- collate_fn: stacks pixel_values, keeps targets as a list.

7 unit tests in tests/test_data.py cover the contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `core/lib/model.py` LoRA extensions

**Files:**
- Modify: `core/lib/model.py`, `tests/test_model.py`

Append to existing `core/lib/model.py`:
- `LoRALinear` (class, ported from train_round4.py).
- `apply_lora(model, rank, alpha, dropout, target_substrings)` — wraps matching linears in LoRALinear.
- `lora_state_dict(model)` — returns just the LoRA weights (small, ~5MB for r=32 vs 170MB full model).
- `load_lora_state(model, state_dict)` — loads a saved LoRA-only checkpoint.

The legacy `apply_lora` matches `decoder` AND one of `q_proj/k_proj/v_proj/out_proj`. Plan 04 keeps the same default (matching the winning P run's config: `target_modules=[q_proj, k_proj, v_proj]` — note: legacy also had `out_proj` but the config in Plan 01 lists only q/k/v; reconcile by accepting the config's list).

- [ ] **Step 1: Append failing tests to `tests/test_model.py`**

```python
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
```

- [ ] **Step 2: Run tests, confirm only the new ones fail.**

- [ ] **Step 3: Append to `core/lib/model.py`**

```python
import math

import torch
import torch.nn as nn


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
```

- [ ] **Step 4: Run all tests; expect 6 model tests pass (4 existing + 2 new).**

- [ ] **Step 5: Commit**

```bash
git add core/lib/model.py tests/test_model.py
git commit -m "feat(core): extend model.py with LoRA — apply_lora, lora_state_dict, load_lora_state

Ports the LoRALinear class and apply_lora function from
training/train_round4.py, with two improvements:
- apply_lora returns the number of replacements done (useful for logging).
- target_substrings is a parameter (defaults to q/k/v_proj per the
  P_repeat_factor config) instead of being hard-coded.

lora_state_dict / load_lora_state make it cheap to save/restore only
the trained LoRA weights (~5MB at r=32 vs ~170MB for the full model).
This is what 'dlmf promote' will move to projects/<slug>/models/production.pt
in Plan 05.

2 new unit tests in tests/test_model.py exercise apply_lora on a tiny
fake model (no Heron download).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `core/lib/eval.py` — mAP@[.5:.95]

**Files:**
- Create: `core/lib/eval.py`, `tests/test_eval.py`

Functions:
- `box_iou(boxes1, boxes2)` — pairwise IoU between two sets of boxes (xyxy).
- `compute_ap(predictions, ground_truth, class_id, iou_threshold)` — average precision for one class at one IoU.
- `compute_map(predictions, ground_truth, iou_thresholds)` — mean over classes and IoU thresholds. Default thresholds: 0.5, 0.55, ..., 0.95 (10 values, COCO-style mAP@[.5:.95]).

Inputs: `predictions` is `list[{"boxes": Tensor[N,4], "scores": Tensor[N], "labels": Tensor[N]}]`, `ground_truth` is `list[{"boxes": Tensor[M,4], "labels": Tensor[M]}]`, both per-image.

Reuse logic from `train_round4.py` lines 137-227.

- [ ] **Step 1: Tests**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_eval.py`:

```python
"""Tests for core.lib.eval — mAP@[.5:.95] computation."""
import torch

from core.lib.eval import box_iou, compute_ap, compute_map


def test_box_iou_identity():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    iou = box_iou(a, a)
    assert iou.shape == (1, 1)
    assert abs(iou[0, 0].item() - 1.0) < 1e-6


def test_box_iou_disjoint():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    iou = box_iou(a, b)
    assert iou[0, 0].item() == 0.0


def test_compute_ap_perfect_prediction_returns_one():
    """1 GT box, 1 prediction matching exactly with high score."""
    preds = [{
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
    }]
    gts = [{
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([0]),
    }]
    ap = compute_ap(preds, gts, class_id=0, iou_threshold=0.5)
    assert ap == 1.0


def test_compute_ap_no_predictions_returns_zero():
    preds = [{
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros(0),
        "labels": torch.zeros(0, dtype=torch.long),
    }]
    gts = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]
    assert compute_ap(preds, gts, class_id=0, iou_threshold=0.5) == 0.0


def test_compute_map_aggregates_classes_and_thresholds():
    """Sanity: a perfect prediction yields mAP=1.0 across all IoUs since boxes are identical."""
    preds = [{
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
    }]
    gts = [{
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([0]),
    }]
    out = compute_map(preds, gts)
    # Perfect overlap survives all 10 IoU thresholds (0.5..0.95)
    assert abs(out["mAP"] - 1.0) < 1e-6


def test_compute_map_returns_zero_when_no_gt():
    preds = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}]
    gts = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}]
    assert compute_map(preds, gts)["mAP"] == 0.0
```

- [ ] **Step 2: Run, confirm import error.**

- [ ] **Step 3: Implement `core/lib/eval.py`**

```python
"""mAP@[.5:.95] evaluation. Adapted from training/train_round4.py."""
from __future__ import annotations

from typing import Iterable

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two box sets (xyxy)."""
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    ix1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    return inter / (a1[:, None] + a2 - inter + 1e-6)


def compute_ap(predictions, ground_truth, class_id: int, iou_threshold: float) -> float:
    """11-point interpolated average precision for one class at one IoU threshold."""
    scores: list[float] = []
    tps: list[int] = []
    n_gt = 0
    for p, t in zip(predictions, ground_truth):
        gm = (t["labels"] == class_id)
        gb = t["boxes"][gm]
        n_gt += int(gm.sum().item())
        pm = (p["labels"] == class_id)
        pb = p["boxes"][pm]
        ps = p["scores"][pm]
        if len(pb) == 0:
            continue
        order = ps.argsort(descending=True)
        pb = pb[order]
        ps = ps[order]
        matched = set()
        for i in range(len(pb)):
            scores.append(float(ps[i].item()))
            if len(gb) == 0:
                tps.append(0)
                continue
            ious = box_iou(pb[i].unsqueeze(0), gb)[0]
            best_iou, best_idx = ious.max(0)
            bi = int(best_idx.item())
            if best_iou.item() >= iou_threshold and bi not in matched:
                tps.append(1)
                matched.add(bi)
            else:
                tps.append(0)
    if n_gt == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_tps = [tps[i] for i in order]
    tp = fp = 0
    precs: list[float] = []
    recs: list[float] = []
    for t in sorted_tps:
        tp += t
        fp += 1 - t
        precs.append(tp / (tp + fp))
        recs.append(tp / n_gt)
    ap = 0.0
    for r in [i / 10.0 for i in range(11)]:
        candidates = [p for p, rc in zip(precs, recs) if rc >= r]
        ap += (max(candidates) if candidates else 0.0) / 11.0
    return ap


def compute_map(
    predictions,
    ground_truth,
    iou_thresholds: Iterable[float] | None = None,
) -> dict:
    """Mean AP over (classes × IoU thresholds). COCO-style mAP@[.5:.95] by default.

    Returns:
        dict with keys 'mAP' (float), 'per_threshold' (dict[iou] -> mAP), 'per_class@0.5' (dict[cls] -> AP).
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    iou_thresholds = list(iou_thresholds)

    classes: set[int] = set()
    for t in ground_truth:
        classes.update(int(c) for c in t["labels"].tolist())
    if not classes:
        return {"mAP": 0.0, "per_threshold": {iou: 0.0 for iou in iou_thresholds}, "per_class@0.5": {}}

    classes_sorted = sorted(classes)

    per_threshold = {}
    for iou_t in iou_thresholds:
        aps = [compute_ap(predictions, ground_truth, c, iou_t) for c in classes_sorted]
        per_threshold[iou_t] = sum(aps) / len(aps)

    per_class_05 = {}
    for c in classes_sorted:
        per_class_05[c] = compute_ap(predictions, ground_truth, c, 0.5)

    mean_map = sum(per_threshold.values()) / len(per_threshold)
    return {"mAP": mean_map, "per_threshold": per_threshold, "per_class@0.5": per_class_05}
```

- [ ] **Step 4: Run tests, confirm pass (6 tests).**

- [ ] **Step 5: Commit**

```bash
git add core/lib/eval.py tests/test_eval.py
git commit -m "feat(core): add eval module — mAP@[.5:.95] computation

core.lib.eval.compute_map averages AP over classes and IoU thresholds
0.5..0.95 in 0.05 steps (COCO-style). Uses 11-point interpolated AP.

Adapted from training/train_round4.py — same algorithm, restructured
as pure functions with type hints and a structured return:
  {'mAP': float, 'per_threshold': dict, 'per_class@0.5': dict}

6 unit tests in tests/test_eval.py cover the contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `core/lib/tracking.py` — MLflow wrapper

**Files:**
- Create: `core/lib/tracking.py`, `tests/test_tracking.py`

A thin context manager around MLflow:
- `MlflowRun(experiment, run_name, params)` starts a run with the given experiment, sets params, returns `self`.
- `__exit__` ends the run.
- Methods: `log_metric(name, value, step)`, `log_artifact(path)`, `log_dict(d, name)`.

If MLflow is not installed (e.g., minimal CI), the tracker becomes a no-op (`enabled=False`). For Plan 04 we always have MLflow.

- [ ] **Step 1: Tests**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_tracking.py`:

```python
"""Tests for core.lib.tracking — MLflow wrapper."""
from pathlib import Path
import json

import mlflow

from core.lib.tracking import MlflowRun


def test_mlflow_run_logs_params_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

    with MlflowRun(experiment="test-exp", run_name="t1", params={"a": 1, "b": "x"}) as run:
        run.log_metric("loss", 0.5, step=0)
        run.log_metric("loss", 0.4, step=1)
        run.log_metric("mAP", 0.85, step=1)

    # Verify via MLflow client
    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{tmp_path}/mlruns")
    exp = client.get_experiment_by_name("test-exp")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    r = runs[0]
    assert r.data.params["a"] == "1"
    assert r.data.params["b"] == "x"
    assert r.data.metrics["mAP"] == 0.85


def test_mlflow_run_logs_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

    artifact = tmp_path / "history.json"
    artifact.write_text(json.dumps([{"epoch": 0, "loss": 0.5}]))

    with MlflowRun(experiment="test-exp", run_name="t2", params={}) as run:
        run.log_artifact(artifact)

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{tmp_path}/mlruns")
    exp = client.get_experiment_by_name("test-exp")
    runs = client.search_runs([exp.experiment_id])
    artifact_list = client.list_artifacts(runs[0].info.run_id)
    assert any(a.path == "history.json" for a in artifact_list)
```

- [ ] **Step 2: Run, expect import error.**

- [ ] **Step 3: Implement `core/lib/tracking.py`**

```python
"""MLflow tracking wrapper — context manager that owns one run."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


class MlflowRun:
    """Context manager for an MLflow run.

    Usage:
        with MlflowRun(experiment="dlmf-eaf", run_name="P_repeat_factor", params={...}) as run:
            for epoch in ...:
                run.log_metric("train_loss", loss, step=epoch)
            run.log_artifact("history.json")
    """

    def __init__(
        self,
        experiment: str,
        run_name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ):
        self.experiment = experiment
        self.run_name = run_name
        self.params = params or {}
        self.tags = tags or {}
        self._run = None

    def __enter__(self):
        mlflow.set_experiment(self.experiment)
        self._run = mlflow.start_run(run_name=self.run_name)
        if self.params:
            mlflow.log_params(_flatten_params(self.params))
        if self.tags:
            mlflow.set_tags(self.tags)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            mlflow.set_tag("status", "failed")
        mlflow.end_run()
        return False  # don't suppress exceptions

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(name, float(value), step=step)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def log_dict(self, d: dict, artifact_path: str) -> None:
        mlflow.log_dict(d, artifact_path)


def _flatten_params(d: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested dicts into dotted-key strings (MLflow param values must be strings)."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_params(v, key))
        else:
            out[key] = str(v)
    return out
```

- [ ] **Step 4: Run tests, confirm pass.**

- [ ] **Step 5: Commit**

```bash
git add core/lib/tracking.py tests/test_tracking.py
git commit -m "feat(core): add tracking module — MlflowRun context manager

Thin wrapper around mlflow that:
- Sets the experiment, starts a run with the given run_name.
- Logs params at entry (flattening nested dicts via dotted keys
  since MLflow params must be string-keyed and string-valued).
- Provides log_metric/log_artifact/log_dict during the run.
- Ends the run at exit, tagging status=failed if an exception leaked.

2 tests in tests/test_tracking.py write to a tmp file:// store and
verify params + metrics + artifacts land where MLflow's client reads
them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `core/train.py` — main training loop

**Files:**
- Create: `core/train.py`
- Modify: `core/cli.py`, `tests/test_cli.py`

The `train(project_slug, run_name, overrides)` function:
1. Loads config + applies overrides → resolved_config.
2. Loads the latest CVAT export (`projects/<slug>/cvat/exports/v*/instances_default.json`).
3. Splits images into train/val (random seed from config; val_split=0.15 default).
4. Builds dataset, RFS sampler, dataloader.
5. Loads Heron, freezes backbone+encoder, applies LoRA per config.
6. AdamW + cosine schedule with warmup, gradient accumulation.
7. Per-epoch: train one pass → eval mAP → save best_model.pt if improved → early stop after `patience`.
8. Logs to MLflow throughout.
9. At end: saves history.json, eval.json, config_resolved.yaml, data_split.json into `projects/<slug>/runs/<run_name>/`.

Heavy reuse from `train_round4.py`. The config-driven structure is the main change.

For testing, we don't smoke a real epoch (too slow); we test the pure helper `_data_split` and validate `train` accepts the right args. Real validation = Tasks 6-8.

- [ ] **Step 1: Add helper test**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_train.py`:

```python
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
```

- [ ] **Step 2: Run, expect import error.**

- [ ] **Step 3: Implement `core/train.py`**

Write `/home/alonso/Documentos/Github/document-layout-model-training/core/train.py`:

```python
"""Fine-tune Heron with LoRA on a project's data; track to MLflow.

Adapted from training/train_round4.py — same training math, restructured
to be config-driven and project-aware.
"""
from __future__ import annotations

import gc
import json
import math
import random
from pathlib import Path

import yaml

from core.lib.config import apply_overrides, load_config

PROJECTS_ROOT = Path("projects")


def _data_split(image_ids: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    val = sorted(ids[:n_val])
    train = sorted(ids[n_val:])
    return train, val


def _latest_cvat_export(project_dir: Path) -> Path:
    """Pick the most recent v<N>_<date>/instances_default.json."""
    exports_dir = project_dir / "cvat" / "exports"
    candidates = sorted(
        (d for d in exports_dir.iterdir() if d.is_dir() and (d / "instances_default.json").exists()),
        key=lambda d: d.name,
    )
    if not candidates:
        raise FileNotFoundError(f"no CVAT exports under {exports_dir}")
    return candidates[-1] / "instances_default.json"


def _get_lr(epoch: int, base_lr: float, warmup: int, max_epochs: int) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    remaining = max_epochs - warmup
    progress = (epoch - warmup) / max(1, remaining)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(project_slug: str, run_name: str, overrides: list[str] | None = None) -> Path:
    """Run one training experiment. Returns path to the run directory."""
    import torch
    import torch.nn.utils as nn_utils
    import torchvision
    from torch.utils.data import DataLoader
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    from core.lib.data import (
        CocoDocDataset,
        DocumentAugmenter,
        RepeatFactorSampler,
        collate_fn,
    )
    from core.lib.eval import compute_map
    from core.lib.model import apply_lora, lora_state_dict
    from core.lib.tracking import MlflowRun

    project_dir = PROJECTS_ROOT / project_slug
    cfg = load_config(project_dir / "config.yaml")
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    tcfg = cfg["training"]
    pcfg = cfg.get("postprocess", {})
    ecfg = cfg.get("evaluation", {})

    # Reproducibility.
    seed = int(ecfg.get("random_seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Data.
    coco_path = _latest_cvat_export(project_dir)
    with open(coco_path) as f:
        coco = json.load(f)

    # Build per-task image_ids: ids are project-level. Filter by file_name → which images-dir contains them.
    images_root = project_dir / "data" / "images"
    image_dir_for: dict[int, Path] = {}
    for img in coco["images"]:
        for sub in images_root.iterdir():
            if sub.is_dir() and (sub / img["file_name"]).exists():
                image_dir_for[img["id"]] = sub
                break
    valid_ids = sorted(image_dir_for.keys())
    if not valid_ids:
        raise FileNotFoundError(f"no COCO image_ids matched files under {images_root}")

    # If `training.limit` (override-only) is set, cap the dataset (smoke testing).
    limit = tcfg.get("limit")
    if limit:
        valid_ids = valid_ids[: int(limit)]

    train_ids, val_ids = _data_split(valid_ids, ecfg.get("val_split", 0.15), seed)

    print(f"[train] dataset: {len(valid_ids)} images, train={len(train_ids)} val={len(val_ids)}")

    # Currently CocoDocDataset takes a single images_dir. Since project images
    # may live in multiple per-task dirs, we work around by symlinking — or
    # simpler: pass images_root and rely on the file_name being unique. For now,
    # since file_names within a project ARE unique (rendered as pagina-NNN.png
    # under each task dir, and we deduplicated suffixes during cvat-pull),
    # we mount images_root and patch the dataset to walk subdirs. Simplest
    # implementation: build a flat symlink dir per run.
    flat_dir = project_dir / "runs" / run_name / "_flat_images"
    flat_dir.mkdir(parents=True, exist_ok=True)
    for iid in valid_ids:
        info = next(i for i in coco["images"] if i["id"] == iid)
        target = flat_dir / info["file_name"]
        src = image_dir_for[iid] / info["file_name"]
        if not target.exists():
            try:
                target.symlink_to(src.resolve())
            except OSError:
                # Fallback to hardlink if symlink fails (e.g. cross-device).
                target.write_bytes(src.read_bytes())

    processor = RTDetrImageProcessor.from_pretrained(tcfg["base_model"])
    augmenter = DocumentAugmenter()
    train_ds = CocoDocDataset(coco_path, flat_dir, train_ids, processor=processor, augmenter=augmenter)
    val_ds = CocoDocDataset(coco_path, flat_dir, val_ids, processor=processor, augmenter=None)

    # Repeat Factor Sampling.
    sampler_cfg = tcfg.get("sampling", {})
    if sampler_cfg.get("method") == "repeat_factor":
        rfs = RepeatFactorSampler.from_coco(coco_path, train_ids, threshold=sampler_cfg.get("threshold", 0.5))
        # PyTorch DataLoader with custom sampler that yields indices into train_ds.
        # train_ds uses image_ids list; rfs yields image_ids. We need indices.
        id_to_idx = {iid: idx for idx, iid in enumerate(train_ds.image_ids)}

        class _IdToIndexSampler:
            def __init__(self, rfs, id_to_idx):
                self.rfs = rfs
                self.id_to_idx = id_to_idx

            def __iter__(self):
                for iid in self.rfs:
                    if iid in self.id_to_idx:
                        yield self.id_to_idx[iid]

            def __len__(self):
                return sum(1 for iid in self.rfs.factors if iid in self.id_to_idx) * 1  # approximate

        sampler = _IdToIndexSampler(rfs, id_to_idx)
        loader = DataLoader(train_ds, batch_size=tcfg.get("batch_size", 1), sampler=sampler, collate_fn=collate_fn, num_workers=2)
        epoch_steps = sum(rfs.factors[iid] for iid in train_ids if iid in rfs.factors)
        print(f"[train] RFS epoch size: {epoch_steps}")
    else:
        loader = DataLoader(train_ds, batch_size=tcfg.get("batch_size", 1), shuffle=True, collate_fn=collate_fn, num_workers=2)
        epoch_steps = len(train_ds)

    # Model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")
    model = RTDetrV2ForObjectDetection.from_pretrained(tcfg["base_model"])
    model.config.num_denoising = 0
    model.config.eos_coefficient = 0.0001
    for n, p in model.named_parameters():
        if "backbone" in n or "encoder" in n:
            p.requires_grad = False
    lora_cfg = tcfg["lora"]
    n_replaced = apply_lora(
        model,
        rank=int(lora_cfg["rank"]),
        alpha=int(lora_cfg["alpha"]),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_substrings=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj"])),
    )
    print(f"[train] LoRA applied to {n_replaced} layers")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {trainable:,} ({100 * trainable / total:.2f}% of {total:,})")
    model.to(device)

    # Optimizer + schedule.
    base_lr = float(tcfg["lr"])
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    warmup = int(tcfg.get("warmup_epochs", 5))
    max_epochs = int(tcfg.get("max_epochs", 50))
    grad_accum = int(tcfg.get("gradient_accumulation", 4))
    grad_clip = float(tcfg.get("gradient_clip", 0.1))
    patience = int(tcfg.get("early_stop_patience", 10))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=weight_decay,
    )

    # Run dir + MLflow.
    run_dir = project_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg))
    (run_dir / "data_split.json").write_text(json.dumps({"train": train_ids, "val": val_ids}))

    history: list[dict] = []
    best_map = -1.0
    patience_ctr = 0

    with MlflowRun(
        experiment=f"dlmf-{project_slug}",
        run_name=run_name,
        params={"training": tcfg, "postprocess": pcfg, "evaluation": ecfg, "dataset": {"coco": str(coco_path), "n_train": len(train_ids), "n_val": len(val_ids)}},
        tags={"project": project_slug, "model": tcfg["base_model"]},
    ) as mlrun:

        for epoch in range(max_epochs):
            lr = _get_lr(epoch, base_lr, warmup, max_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            n_steps = 0

            for step, (pixel_values, targets) in enumerate(loader):
                pixel_values = pixel_values.to(device)
                labels = [{"boxes": t["boxes"].to(device), "class_labels": t["class_labels"].to(device)} for t in targets]
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum
                loss.backward()

                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    nn_utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += float(loss.item()) * grad_accum
                n_steps += 1

            avg_loss = total_loss / max(1, n_steps)

            # Eval.
            model.eval()
            preds_list, gts_list = [], []
            with torch.no_grad():
                for idx in range(len(val_ds)):
                    px, tgt = val_ds[idx]
                    info = val_ds.id_to_img[val_ds.image_ids[idx]]
                    px = px.unsqueeze(0).to(device)
                    sz = torch.tensor([[info["height"], info["width"]]]).to(device)
                    out = model(pixel_values=px.float())
                    res = processor.post_process_object_detection(out, target_sizes=sz, threshold=0.3)[0]
                    if len(res["scores"]) > 0:
                        ki = []
                        for l in res["labels"].unique():
                            m = res["labels"] == l
                            k = torchvision.ops.nms(res["boxes"][m], res["scores"][m], 0.5)
                            ki.extend(torch.where(m)[0][k].tolist())
                        keep = torch.tensor(sorted(ki), dtype=torch.long, device=device)
                        preds_list.append({
                            "boxes": res["boxes"][keep].cpu(),
                            "scores": res["scores"][keep].cpu(),
                            "labels": res["labels"][keep].cpu(),
                        })
                    else:
                        preds_list.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
                    iw, ih = info["width"], info["height"]
                    gb = []
                    for b in tgt["boxes"]:
                        cx, cy, w, h = b.tolist()
                        gb.append([(cx - w / 2) * iw, (cy - h / 2) * ih, (cx + w / 2) * iw, (cy + h / 2) * ih])
                    gts_list.append({"boxes": torch.tensor(gb) if gb else torch.zeros((0, 4)), "labels": tgt["class_labels"]})

            metrics = compute_map(preds_list, gts_list)
            val_map = metrics["mAP"]

            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "mAP": val_map,
                "per_class@0.5": {str(k): float(v) for k, v in metrics["per_class@0.5"].items()},
                "lr": lr,
            })

            mlrun.log_metric("train_loss", avg_loss, step=epoch)
            mlrun.log_metric("val_mAP", val_map, step=epoch)
            mlrun.log_metric("lr", lr, step=epoch)
            for cls, ap in metrics["per_class@0.5"].items():
                mlrun.log_metric(f"AP05_class_{cls}", ap, step=epoch)

            print(f"[train] epoch {epoch+1}/{max_epochs}  loss={avg_loss:.4f}  mAP={val_map:.4f}  lr={lr:.2e}", flush=True)

            if val_map > best_map:
                best_map = val_map
                patience_ctr = 0
                torch.save(lora_state_dict(model), run_dir / "best_model.pt")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"[train] early stop at epoch {epoch+1}")
                    break

            (run_dir / "history.json").write_text(json.dumps(history, indent=2))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final eval.json with the best epoch's metrics.
        best_entry = max(history, key=lambda h: h["mAP"])
        eval_data = {"best_epoch": best_entry["epoch"], "best_mAP": best_entry["mAP"], "per_class@0.5": best_entry["per_class@0.5"]}
        (run_dir / "eval.json").write_text(json.dumps(eval_data, indent=2))

        mlrun.log_artifact(run_dir / "history.json")
        mlrun.log_artifact(run_dir / "eval.json")
        mlrun.log_artifact(run_dir / "config_resolved.yaml")
        mlrun.log_artifact(run_dir / "data_split.json")
        mlrun.log_artifact(run_dir / "best_model.pt")

    print(f"[train] DONE  best mAP@[.5:.95] = {best_map:.4f}  → {run_dir}")
    return run_dir
```

- [ ] **Step 4: Run helper tests, confirm pass.**

```bash
uv run pytest tests/test_train.py -v
```
Expected: 3 tests pass.

- [ ] **Step 5: Wire `dlmf train` in `core/cli.py`**

Add after the existing commands:
```python
@app.command(name="train")
def train_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug."),
    run: str = typer.Option(..., "--run", "-r", help="Run name (becomes the dir under projects/<slug>/runs/)."),
    override: list[str] = typer.Option(
        None, "--override", "-o", help="Hyperparameter override KEY=VALUE (e.g. training.lora.rank=64). Repeatable."
    ),
) -> None:
    """Fine-tune the project's layout model (LoRA on Heron) and log to MLflow."""
    from core.train import train

    train(project, run_name=run, overrides=list(override or []))
```

Update `tests/test_cli.py`:
```python
def test_cli_train_help_mentions_required_flags():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout
    assert "--run" in result.stdout
    assert "--override" in result.stdout
```

And update `test_cli_help_lists_subcommands` to include `train` in the assertions.

- [ ] **Step 6: Run all tests; expect everything green.**

```bash
uv run pytest -q
```

- [ ] **Step 7: Commit**

```bash
git add core/train.py core/cli.py tests/test_train.py tests/test_cli.py
git commit -m "feat(core): add 'dlmf train' — LoRA fine-tuning with MLflow tracking

core.train.train(project_slug, run_name, overrides) orchestrates:
- Load config + apply overrides; deterministic seed.
- Pick the latest cvat/exports/v<N>/instances_default.json.
- Train/val split (val_fraction from config, default 0.15).
- Build CocoDocDataset, RepeatFactorSampler (when method=repeat_factor),
  DataLoader with collate_fn.
- Load Heron, freeze backbone+encoder, apply_lora per config.
- AdamW + cosine schedule with warmup, gradient accumulation+clip,
  early stopping on val mAP.
- Per-epoch: eval mAP@[.5:.95] on val set, log to MLflow, save
  lora_state_dict to best_model.pt if improved.
- Persist history.json, eval.json, config_resolved.yaml,
  data_split.json under projects/<slug>/runs/<run_name>/.

Adapted from training/train_round4.py — same training math.

CLI: dlmf train --project=eaf --run=name [--override KEY=VAL]+

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Smoke test

- [ ] **Step 1: Run a 2-epoch smoke on 50 images**

```bash
uv run dlmf train --project=eaf --run=smoke \
    --override training.max_epochs=2 \
    --override training.limit=50 \
    --override training.early_stop_patience=99 2>&1 | tee /tmp/smoke.log | tail -30
```

Expected:
- "[train] device: cuda" (GPU detected).
- "[train] LoRA applied to N layers" where N matches the legacy run (~18).
- 2 epochs completed; each prints `loss=X.XXX mAP=Y.YYYY`.
- Output: `[train] DONE  best mAP@[.5:.95] = X.XXXX  → projects/eaf/runs/smoke`.

Check artifacts:
```bash
ls projects/eaf/runs/smoke/
```
Expected: best_model.pt, history.json, eval.json, config_resolved.yaml, data_split.json, _flat_images/.

- [ ] **Step 2: Verify MLflow logged it**

```bash
ls mlruns/
uv run mlflow runs list --experiment-id $(uv run mlflow experiments search 2>/dev/null | grep dlmf-eaf | awk '{print $1}') 2>&1 | head -10
```

Or simply check that mlruns/ has content:
```bash
find mlruns -name "*.yaml" -o -name "metrics" -type d | head -5
```

- [ ] **Step 3: Commit smoke artifacts (run dir is gitignored except for the structured JSON files)**

The .gitignore from Plan 01 ignores `projects/*/runs/*/best_model.pt` but NOT history.json/eval.json/etc. So we can commit the smoke run's metadata.

```bash
git add -f projects/eaf/runs/smoke/history.json projects/eaf/runs/smoke/eval.json projects/eaf/runs/smoke/config_resolved.yaml projects/eaf/runs/smoke/data_split.json
git commit -m "chore: add smoke run artefacts from Plan 04 task 6

2 epochs on 50 images. Validates the training pipeline (data loading,
LoRA, optimizer, eval, MLflow logging) end-to-end without the full
30-60 min runtime of the parity gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If best_model.pt is too small to bother, ignore. The structured JSONs are the proof of execution.

---

### Task 7: Reproduce P_repeat_factor (parity gate)

This is the actual training that produces a model with mAP ≥ 0.86.

- [ ] **Step 1: Confirm config matches the winning P run**

```bash
uv run python -c "
from core.lib.config import load_config
cfg = load_config('projects/eaf/config.yaml')
t = cfg['training']
assert t['lora']['rank'] == 32
assert t['lora']['alpha'] == 64
assert t['lr'] == 1e-4
assert t['warmup_epochs'] == 5
assert t['max_epochs'] == 50
assert t['early_stop_patience'] == 10
assert t['sampling']['method'] == 'repeat_factor'
assert t['sampling']['threshold'] == 0.5
print('config matches P_repeat_factor')
"
```

- [ ] **Step 2: Run the full training**

```bash
uv run dlmf train --project=eaf --run=P_repeat_factor_v2 2>&1 | tee /tmp/P_v2.log | tail -10 &
```

Run in foreground if you can babysit; otherwise put in background and poll.

**Expected duration:** 1-1.5 hours on the GTX 1080 for ~19 epochs to reach the best.

While it runs, periodically tail the log:
```bash
tail -f /tmp/P_v2.log
```

Each epoch should print `loss=... mAP=...`. mAP should rise from ~0.3 (pre-trained baseline behavior) toward 0.85+ over ~15-25 epochs.

- [ ] **Step 3: Read final result**

After the run completes:
```bash
cat projects/eaf/runs/P_repeat_factor_v2/eval.json
```

Should show `"best_mAP": 0.86xx` or higher.

If `best_mAP < 0.86`, this is the parity gate failure. Investigate:
- Was the seed different? (Both runs should use seed=42.)
- Did training really complete the same number of epochs as the original (best epoch was 19)?
- Was Repeat Factor Sampling active? (Look for `[train] RFS epoch size: NNN` in the log.)
- LoRA replacement count should be 18 (decoder layers × 3 modules q/k/v in 6 layers ≈ 18).

If parity holds (mAP ≥ 0.86): success.

- [ ] **Step 4: Commit the run artefacts**

```bash
git add projects/eaf/runs/P_repeat_factor_v2/history.json \
        projects/eaf/runs/P_repeat_factor_v2/eval.json \
        projects/eaf/runs/P_repeat_factor_v2/config_resolved.yaml \
        projects/eaf/runs/P_repeat_factor_v2/data_split.json
git commit -m "feat(eaf): reproduce P_repeat_factor in the new factory pipeline

Run name: P_repeat_factor_v2
Hyperparams: LoRA r=32 alpha=64 dropout=0.05, AdamW lr=1e-4
weight_decay=1e-4, cosine schedule with 5-epoch warmup,
batch_size=1 grad_accum=4 grad_clip=0.1, max 50 epochs,
early stop patience 10, Repeat Factor Sampling threshold=0.5.

Best mAP@[.5:.95]: <see eval.json>
Best epoch: <see eval.json>

Parity gate: original P_repeat_factor reached 0.8700 — this run
must reach ≥ 0.86 (within 1.4% of the original) to validate that
the migration didn't lose anything.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Final verification + tag

- [ ] **Step 1: Run all tests**

```bash
uv run pytest -q
```
Expected: previous tests + new ones from this plan (data ~7, model ~6, eval ~6, tracking ~2, train ~3, cli +1) = ~83 tests, all green.

- [ ] **Step 2: Verify CLI**

```bash
uv run dlmf --help
uv run dlmf train --help
```

- [ ] **Step 3: Verify mlruns/ has content**

```bash
find mlruns -name "metrics" -type d | head -3
```

- [ ] **Step 4: Verify the parity gate passed**

```bash
python3 -c "
import json
e = json.load(open('projects/eaf/runs/P_repeat_factor_v2/eval.json'))
assert e['best_mAP'] >= 0.86, f'parity gate FAILED: {e[\"best_mAP\"]}'
print(f'parity gate PASSED: best_mAP = {e[\"best_mAP\"]:.4f} (>= 0.86)')
"
```

- [ ] **Step 5: Tag**

```bash
git tag -a plan-04-training-mlflow -m "Plan 04 complete: training infrastructure + MLflow + P_repeat_factor parity gate met."
git tag -l "plan-*"
```

- [ ] **Step 6: Summary**

```bash
git log $(git rev-list -n 1 plan-03-predict-postproc)..HEAD --oneline
```

---

## Self-Review

**Spec coverage:**
- ✅ Spec section 5 (CLI): `dlmf train` shipped with --override.
- ✅ Spec section 6 (data flow + MLflow): each run produces history.json, eval.json, config_resolved.yaml, data_split.json + best_model.pt + MLflow logs.
- ✅ Spec section 7 (parity gate): mAP ≥ 0.86 verified.
- 🔜 `dlmf evaluate` standalone (per-PDF, per-class breakdown): Plan 05.
- 🔜 `dlmf promote` (production.pt symlink): Plan 05.

**Placeholder scan:** No `TBD`/`TODO`. All commands have full code.

**Type/name consistency:**
- `core.lib.data.CocoDocDataset` ↔ `core.train` consumer.
- `core.lib.model.apply_lora` ↔ `core.train` consumer (target_substrings keyword).
- `core.lib.eval.compute_map` returns `{"mAP", "per_threshold", "per_class@0.5"}` ↔ `core.train` reads `["mAP"]` and `["per_class@0.5"]`.
- `core.lib.tracking.MlflowRun` ↔ `core.train` `with` statement.
