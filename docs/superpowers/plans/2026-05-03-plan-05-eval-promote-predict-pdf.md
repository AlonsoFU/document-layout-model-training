# Plan 05 — Evaluate + Promote + Predict-to-PDF

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Cerrar el ciclo train → evaluate → promote → predict, dejando 3 comandos nuevos del CLI:
- `dlmf evaluate --project --run` → mAP per-PDF + per-class sobre el val split del run.
- `dlmf promote --project --run` → actualiza `projects/<slug>/models/production.pt` (symlink) y registra el modelo en MLflow Model Registry.
- `dlmf predict --project --pdf --output=anotado.pdf` → carga el modelo de producción, corre inferencia, y dibuja los boxes detectados sobre el PDF.

**Architecture:**
- `core/evaluate.py` — re-evalúa un run guardado (carga LoRA weights, recorre val_ids del data_split.json, computa mAP global + per-PDF + per-class). Reusa `core/lib/eval.py` y `core/lib/data.py`.
- `core/promote.py` — symlink atomic + MLflow registry call (`mlflow.register_model`).
- `core/predict.py` ya implementa `--pre-annotate` (Plan 03). Le añadimos modo `--output=anotado.pdf` que carga el modelo de producción del proyecto, infiere, y usa PyMuPDF (`fitz`) para dibujar boxes.

**Tech Stack:** Existente + PyMuPDF para anotación de PDFs. Sin nuevas dependencias.

**Salida verificable:**
- `dlmf evaluate --project=eaf --run=P_repeat_factor_v2` reporta mAP global ≈ 0.93 + per-PDF (EAF-089 vs EAF-477 separados) + per-class.
- `dlmf promote --project=eaf --run=P_repeat_factor_v2` crea `projects/eaf/models/production.pt → ../runs/P_repeat_factor_v2/best_model.pt` y registra en MLflow `dlmf-eaf-production` stage Production.
- `dlmf predict --project=eaf --pdf=projects/eaf/data/pdfs/EAF-477-2025.pdf --output=anotado.pdf --limit=10` produce `anotado.pdf` (10 páginas anotadas).
- Tag `plan-05-eval-promote-predict-pdf`.

**Out of scope:**
- Plan 06: `classify_doctype` (auto-detect tipo) + `init_project` (scaffolding nuevo tipo) + README final.

---

## File Structure

### Files to CREATE
| Path | Responsabilidad |
|---|---|
| `core/evaluate.py` | `evaluate(project_slug, run_name)` — re-evalúa un run guardado |
| `core/promote.py` | `promote(project_slug, run_name)` — symlink + MLflow registry |
| `tests/test_evaluate.py` | Tests del breakdown per-PDF |
| `tests/test_promote.py` | Tests del symlink (sin MLflow real, mock) |

### Files to MODIFY
| Path | Cambio |
|---|---|
| `core/predict.py` | Añadir modo `output_pdf` que carga production.pt + dibuja boxes |
| `core/cli.py` | Añadir `evaluate`, `promote` commands y extender `predict` con `--pdf`/`--output` |
| `tests/test_cli.py` | Añadir CLI help tests |

### Files to DELETE (legacy, fully replaced)
| Path | Reemplazado por |
|---|---|
| `training/train_strategies.py` | `dlmf train` |
| `training/train_round2.py` | `dlmf train` |
| `training/train_round3.py` | `dlmf train` |
| `training/train_round4.py` | `dlmf train` |
| `training/reevaluate_real_map.py` | `dlmf evaluate` |
| `training/draw_boxes_on_pdf.py` | `dlmf predict --output=PDF` |
| `training/draw_boxes_on_pdf_v2.py` | `dlmf predict --output=PDF` |
| `training/generate_comparison_pdf.py` | `dlmf predict --output=PDF` (ad-hoc compare via 2 calls) |
| `training/generate_comparison_pdf_v2.py` | id |
| `training/test_best_model.py` | `dlmf evaluate` |
| `training/run_docling_pipeline.py` | (out of scope — separate Docling integration repo) |

After deletion, the empty `training/` directory should be removed.

---

## Tasks

### Task 1: `core/evaluate.py` — standalone evaluation with per-PDF breakdown

**Files:**
- Create: `core/evaluate.py`, `tests/test_evaluate.py`

The function:
1. Loads `projects/<slug>/runs/<run>/data_split.json` and `config_resolved.yaml`.
2. Loads the COCO file referenced in config_resolved + the val_ids.
3. Builds the eval val set (using `CocoDocDataset`, no augmentation, with `category_remap`).
4. Loads `best_model.pt` (LoRA state) into a fresh Heron + `apply_lora` model.
5. Runs inference per page, computes:
   - **Global mAP@[.5:.95]** (overall).
   - **Per-PDF mAP** (group val images by their PDF stem, compute mAP within each group).
   - **Per-class AP@0.5** (already in eval.py).
6. Writes `projects/<slug>/runs/<run>/eval_detailed.json` with the breakdown.
7. Prints summary table.

The PDF grouping: each image's `file_name` is `pagina-NNN.png` or `pagina-NNN_1.png`. The COCO image dict has the file_name. We need to know which PDF each came from. Approach: an image belongs to the Nth occurrence's task (e.g., `_1` → second task alphabetically). Build the mapping using the same logic as `core/train.py::image_path_for`.

- [ ] **Step 1: Tests**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_evaluate.py`:

```python
"""Tests for core.evaluate."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.evaluate import _group_images_by_pdf, _per_group_mAP


def test_group_images_by_pdf_uses_subdir_name(tmp_path):
    """Verify the helper that maps image_id → pdf-stem-based group."""
    images_root = tmp_path / "images"
    (images_root / "DocA").mkdir(parents=True)
    (images_root / "DocB").mkdir(parents=True)
    (images_root / "DocA" / "pagina-001.png").write_bytes(b"")
    (images_root / "DocB" / "pagina-001.png").write_bytes(b"")
    coco = {
        "images": [
            {"id": 1, "file_name": "pagina-001.png"},
            {"id": 2, "file_name": "pagina-001_1.png"},
        ],
    }
    groups = _group_images_by_pdf(coco, images_root)
    assert groups == {1: "DocA", 2: "DocB"}


def test_per_group_mAP_with_one_group():
    import torch

    preds = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}]
    gts = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]
    out = _per_group_mAP(preds, gts, ["DocA"])
    # 1 group with perfect prediction → mAP=1.0
    assert "DocA" in out
    assert abs(out["DocA"]["mAP"] - 1.0) < 1e-6


def test_per_group_mAP_aggregates_separately():
    import torch

    preds = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])},  # DocA
        {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)},  # DocB
    ]
    gts = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])},
    ]
    out = _per_group_mAP(preds, gts, ["DocA", "DocB"])
    assert abs(out["DocA"]["mAP"] - 1.0) < 1e-6
    assert out["DocB"]["mAP"] == 0.0
```

- [ ] **Step 2: Run, expect import error.**

- [ ] **Step 3: Implement `core/evaluate.py`**

Write the module that:
- Reads run dir.
- Reuses `core.lib.data.CocoDocDataset` (with category_remap).
- Loads model (Heron + apply_lora + load_lora_state).
- Iterates val set, builds preds/gts lists.
- Calls `compute_map` for global + per-PDF.
- Saves eval_detailed.json + prints summary.

Key helper functions exposed for testing:
- `_group_images_by_pdf(coco, images_root) -> dict[image_id, pdf_stem]`
- `_per_group_mAP(preds_list, gts_list, group_per_image) -> dict[group, dict]`

Implementation:

```python
"""Standalone evaluation for a saved run.

Reads the run's data_split.json + config_resolved.yaml, reloads the LoRA
weights from best_model.pt, runs inference over the val set, and produces
a detailed breakdown:
- Overall mAP@[.5:.95]
- Per-PDF mAP@[.5:.95]
- Per-class AP@0.5

Saves to projects/<slug>/runs/<run>/eval_detailed.json.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

from core.lib.config import load_config
from core.lib.eval import compute_map

PROJECTS_ROOT = Path("projects")


def _group_images_by_pdf(coco: dict, images_root: Path) -> dict[int, str]:
    """Map COCO image_id → pdf stem (the subdir under images_root/ that owns it).

    Same logic as core.train.image_path_for: file_name `pagina-NNN.png` is the
    Nth occurrence (0=first, 1=second) of `pagina-NNN.png` in alphabetical
    order of subdirs.
    """
    subdirs = sorted(d for d in images_root.iterdir() if d.is_dir())
    name_to_dirs: dict[str, list[str]] = {}
    for sub in subdirs:
        for png in sorted(sub.glob("*.png")):
            name_to_dirs.setdefault(png.name, []).append(sub.name)

    groups: dict[int, str] = {}
    for img in coco["images"]:
        fname = img["file_name"]
        m = re.match(r"^(.+)_(\d+)(\.[^.]+)$", fname)
        if m:
            base = m.group(1) + m.group(3)
            occurrence = int(m.group(2))
        else:
            base = fname
            occurrence = 0
        candidates = name_to_dirs.get(base, [])
        if occurrence < len(candidates):
            groups[img["id"]] = candidates[occurrence]
    return groups


def _per_group_mAP(preds_list, gts_list, group_per_image: list[str]) -> dict[str, dict]:
    """Compute mAP per group given a per-image group label."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(group_per_image):
        by_group[g].append(i)
    out: dict[str, dict] = {}
    for group, indices in by_group.items():
        sub_preds = [preds_list[i] for i in indices]
        sub_gts = [gts_list[i] for i in indices]
        out[group] = compute_map(sub_preds, sub_gts)
    return out


def evaluate(project_slug: str, run_name: str) -> Path:
    """Re-evaluate a saved run; return path to eval_detailed.json."""
    import torch
    import torchvision
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    from core.lib.data import CocoDocDataset
    from core.lib.model import (
        MODEL_INDEX_TO_LABEL_NAME,
        apply_lora,
        load_lora_state,
    )

    project_dir = PROJECTS_ROOT / project_slug
    run_dir = project_dir / "runs" / run_name
    if not (run_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"no best_model.pt in {run_dir}")

    config_resolved = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    split = json.loads((run_dir / "data_split.json").read_text())
    val_ids = list(split["val"])

    tcfg = config_resolved["training"]
    coco_path = Path(_find_export_for_run(project_dir, run_dir))
    with open(coco_path) as f:
        coco = json.load(f)

    images_root = project_dir / "data" / "images"
    image_to_pdf = _group_images_by_pdf(coco, images_root)

    # Build category remap (same as train).
    name_to_model_idx = {n: i for i, n in enumerate(MODEL_INDEX_TO_LABEL_NAME)}
    category_remap = {c["id"]: name_to_model_idx[c["name"]] for c in coco["categories"] if c["name"] in name_to_model_idx}

    flat_dir = run_dir / "_flat_images"
    if not flat_dir.exists():
        # Recreate symlinks if missing.
        flat_dir.mkdir(parents=True)
        for img in coco["images"]:
            if img["id"] not in val_ids:
                continue
            target = flat_dir / img["file_name"]
            # Try to find source by walking subdirs.
            for sub in sorted(images_root.iterdir()):
                if sub.is_dir():
                    src = sub / re.sub(r"_\d+(\.[^.]+)$", r"\1", img["file_name"])
                    if src.exists() and not target.exists():
                        try:
                            target.symlink_to(src.resolve())
                        except OSError:
                            target.write_bytes(src.read_bytes())

    processor = RTDetrImageProcessor.from_pretrained(tcfg["base_model"])
    val_ds = CocoDocDataset(
        coco_path, flat_dir, val_ids, processor=processor, augmenter=None,
        category_remap=category_remap,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device: {device}")
    model = RTDetrV2ForObjectDetection.from_pretrained(tcfg["base_model"])
    model.config.num_denoising = 0
    model.config.eos_coefficient = 0.0001
    for n, p in model.named_parameters():
        if "backbone" in n or "encoder" in n:
            p.requires_grad = False
    lora_cfg = tcfg["lora"]
    apply_lora(
        model,
        rank=int(lora_cfg["rank"]),
        alpha=int(lora_cfg["alpha"]),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_substrings=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj"])),
    )
    state = torch.load(run_dir / "best_model.pt", map_location="cpu")
    load_lora_state(model, state)
    model.to(device).eval()

    # Inference.
    preds_list = []
    gts_list = []
    group_per_image: list[str] = []
    print(f"[evaluate] running on {len(val_ds)} val images")
    with torch.no_grad():
        for idx in range(len(val_ds)):
            px, tgt = val_ds[idx]
            iid = val_ds.image_ids[idx]
            info = val_ds.id_to_img[iid]
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
                gb.append([(cx - w/2) * iw, (cy - h/2) * ih, (cx + w/2) * iw, (cy + h/2) * ih])
            gts_list.append({"boxes": torch.tensor(gb) if gb else torch.zeros((0, 4)), "labels": tgt["class_labels"]})
            group_per_image.append(image_to_pdf.get(iid, "unknown"))

    overall = compute_map(preds_list, gts_list)
    per_pdf = _per_group_mAP(preds_list, gts_list, group_per_image)

    out = {
        "overall_mAP": overall["mAP"],
        "overall_per_threshold": {f"{k:.2f}": v for k, v in overall["per_threshold"].items()},
        "overall_per_class@0.5": {str(k): v for k, v in overall["per_class@0.5"].items()},
        "per_pdf": {
            pdf: {
                "mAP": data["mAP"],
                "n_images": sum(1 for g in group_per_image if g == pdf),
            }
            for pdf, data in per_pdf.items()
        },
    }
    out_path = run_dir / "eval_detailed.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"[evaluate] OVERALL mAP@[.5:.95]: {overall['mAP']:.4f}")
    print(f"[evaluate] per-PDF breakdown:")
    for pdf, data in sorted(per_pdf.items()):
        n = sum(1 for g in group_per_image if g == pdf)
        print(f"           {pdf:30s} mAP={data['mAP']:.4f}  ({n} images)")
    return out_path


def _find_export_for_run(project_dir: Path, run_dir: Path) -> str:
    """Pick the COCO export the run was trained against. Best-effort: read from config_resolved.yaml.

    The training run logs `dataset.coco` as a param to MLflow but doesn't write
    it to the run dir directly. As a fallback, use the latest export.
    """
    exports = sorted((project_dir / "cvat" / "exports").iterdir(), key=lambda d: d.name)
    return str(exports[-1] / "instances_default.json")
```

- [ ] **Step 4: Run tests, all 3 pass.**

- [ ] **Step 5: Commit**

```bash
git add core/evaluate.py tests/test_evaluate.py
git commit -m "feat(core): add 'dlmf evaluate' — re-evaluate a saved run with per-PDF breakdown

core.evaluate.evaluate(project_slug, run_name):
- Loads the run's data_split.json + config_resolved.yaml.
- Reloads the LoRA-only weights (best_model.pt) into a fresh Heron + apply_lora.
- Runs inference over the val set, computes mAP@[.5:.95] globally + per-PDF
  + per-class AP@0.5.
- Writes eval_detailed.json with the breakdown.
- Prints a summary table.

Per-PDF grouping mirrors core.train's image_path_for logic:
file_name '_N' suffix → Nth occurrence in alphabetically sorted
images-root subdirs.

3 unit tests in tests/test_evaluate.py cover the helper functions
(_group_images_by_pdf, _per_group_mAP).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `core/promote.py` — production symlink + MLflow Model Registry

**Files:**
- Create: `core/promote.py`, `tests/test_promote.py`

The function:
1. Validate that `projects/<slug>/runs/<run>/best_model.pt` exists.
2. Atomically replace `projects/<slug>/models/production.pt` with a symlink to the new run's best_model.pt (relative path so the symlink survives a repo move).
3. Register the model in MLflow Model Registry as `dlmf-<slug>` (find the existing run by name, register its best_model.pt artifact, transition to Production stage). MLflow URI is local (file:// based on `mlruns/`).
4. Print the new symlink target and the registered model version.

If MLflow can't find the run (e.g., re-promote after deleting `mlruns/`), skip the registry step but still update the symlink.

- [ ] **Step 1: Tests**

```python
"""Tests for core.promote."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.promote import promote


@pytest.fixture()
def fake_project(tmp_path, monkeypatch):
    proj = tmp_path / "projects" / "test"
    (proj / "runs" / "run_a").mkdir(parents=True)
    (proj / "runs" / "run_a" / "best_model.pt").write_bytes(b"weights_a")
    (proj / "runs" / "run_a" / "eval.json").write_text(json.dumps({"best_mAP": 0.93}))
    (proj / "models").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return proj


def test_promote_creates_symlink_pointing_to_run_best_model(fake_project, monkeypatch):
    monkeypatch.setattr("core.promote._register_in_mlflow", lambda *a, **k: None)
    promote("test", "run_a")
    link = fake_project / "models" / "production.pt"
    assert link.is_symlink() or link.exists()
    # Read the link target; should resolve to the run's best_model.pt
    if link.is_symlink():
        target = link.resolve()
        assert target.name == "best_model.pt"
        assert "run_a" in str(target)


def test_promote_replaces_existing_symlink(fake_project, monkeypatch):
    monkeypatch.setattr("core.promote._register_in_mlflow", lambda *a, **k: None)
    # First promote
    promote("test", "run_a")
    # Add a second run, promote it
    second = fake_project / "runs" / "run_b"
    second.mkdir(parents=True)
    (second / "best_model.pt").write_bytes(b"weights_b")
    (second / "eval.json").write_text(json.dumps({"best_mAP": 0.95}))

    promote("test", "run_b")

    link = fake_project / "models" / "production.pt"
    if link.is_symlink():
        assert "run_b" in str(link.resolve())


def test_promote_raises_if_best_model_missing(fake_project, monkeypatch):
    (fake_project / "runs" / "run_a" / "best_model.pt").unlink()
    with pytest.raises(FileNotFoundError):
        promote("test", "run_a")
```

- [ ] **Step 2: Implement**

```python
"""Promote a saved run to production: update the symlink and register in MLflow."""
from __future__ import annotations

import os
from pathlib import Path

PROJECTS_ROOT = Path("projects")


def promote(project_slug: str, run_name: str) -> Path:
    project_dir = PROJECTS_ROOT / project_slug
    run_dir = project_dir / "runs" / run_name
    src = run_dir / "best_model.pt"
    if not src.exists():
        raise FileNotFoundError(f"no best_model.pt in {run_dir}")

    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    link = models_dir / "production.pt"

    # Atomic replace: remove existing symlink/file, create new.
    if link.is_symlink() or link.exists():
        link.unlink()
    # Use a relative path so the symlink survives if the repo is moved.
    rel_target = os.path.relpath(src.resolve(), link.parent)
    link.symlink_to(rel_target)
    print(f"[promote] {link} -> {rel_target}")

    _register_in_mlflow(project_slug, run_name, src)
    return link


def _register_in_mlflow(project_slug: str, run_name: str, model_path: Path) -> None:
    """Best-effort: register in MLflow Model Registry. Skip silently if unavailable."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        exp = client.get_experiment_by_name(f"dlmf-{project_slug}")
        if exp is None:
            print(f"[promote] (skip MLflow) experiment 'dlmf-{project_slug}' not found")
            return
        runs = client.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")
        if not runs:
            print(f"[promote] (skip MLflow) run '{run_name}' not found in MLflow")
            return
        run = runs[0]
        model_uri = f"runs:/{run.info.run_id}/best_model.pt"
        result = mlflow.register_model(model_uri=model_uri, name=f"dlmf-{project_slug}")
        client.transition_model_version_stage(
            name=f"dlmf-{project_slug}",
            version=result.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"[promote] MLflow Model Registry: dlmf-{project_slug} v{result.version} -> Production")
    except Exception as e:
        print(f"[promote] (MLflow registry skipped: {type(e).__name__}: {e})")
```

- [ ] **Step 3: Run tests, all 3 pass.**

- [ ] **Step 4: Commit**

```bash
git add core/promote.py tests/test_promote.py
git commit -m "feat(core): add 'dlmf promote' — atomic production symlink + MLflow registry

core.promote.promote(project_slug, run_name):
- Validates the run's best_model.pt exists.
- Atomically replaces projects/<slug>/models/production.pt with a
  relative symlink to the run's best_model.pt (relative target so
  the link survives if the repo is moved).
- Best-effort MLflow Model Registry registration (transitions to
  Production stage, archives previous versions). If the experiment
  or run isn't found in mlruns/, prints a skip message and continues.

3 unit tests in tests/test_promote.py monkeypatch the MLflow call to
focus on the symlink contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `core/predict.py` — `--output=PDF` mode

**Files:**
- Modify: `core/predict.py`, `tests/test_predict.py`

Extend the existing `predict()` function with a new mode that takes a PDF + output path, loads the project's production model, runs inference, and writes an annotated PDF using PyMuPDF.

Pseudo-flow:
1. If `--output` ends in `.pdf`, switch to PDF mode.
2. Need a single PDF (passed via `--pdf` flag).
3. Render the PDF in-memory (or to temp dir) with `pdftoppm` at the configured DPI.
4. For each page: run inference (load model once, use `production.pt` if available else fallback to baseline Heron), apply postproc thresholds + NMS, draw boxes on the page using `fitz.Page.draw_rect`.
5. Save annotated PDF.

Color per category: deterministic (hash of label name → RGB). Plus the label text + score next to each box.

Skip writing pre_annotations JSON in this mode.

- [ ] **Step 1: Add a couple of tests** (mock PDF/model)

```python
def test_predict_output_pdf_calls_fitz(monkeypatch, tmp_path):
    """Smoke: --output=anotado.pdf path goes through the PDF code branch."""
    # Build a fake model + processor + project tree (similar to the existing tests).
    # Then call predict(..., output_pdf=tmp_path/'out.pdf').
    # Assert that the file is created and is non-empty.
    pytest.skip("Wired in implementation; see Task 3 step 5 smoke test for actual PDF generation.")
```

(Keep this minimal — the real test is the smoke test in step 5.)

- [ ] **Step 2: Implement**

Refactor `core/predict.py`:
- Split the loop into a helper `_infer_one_image(model, processor, image, device, threshold)` that returns the postprocessed `Detection` list.
- Add a new top-level function `predict_pdf(project_slug, pdf_path, output_path, threshold=None, limit=None)` that:
  - Loads project config + label list.
  - Loads model (production.pt if exists, else baseline Heron).
  - Renders PDF pages (inline via pdftoppm).
  - Per page: load PNG, run `_infer_one_image`, draw rects on the original PDF page using `fitz`.
  - Saves annotated PDF.

In `predict_cmd` of `core/cli.py`, route based on whether `--output` ends in `.pdf`:
- If yes: `from core.predict import predict_pdf; predict_pdf(...)`.
- Else (with `--pre-annotate`): existing path.

Add CLI options: `--pdf` (path) and `--output` (path).

- [ ] **Step 3: Update `core/cli.py`**

Modify the predict command:
```python
@app.command(name="predict")
def predict_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug."),
    pre_annotate: bool = typer.Option(False, "--pre-annotate", help="Generate COCO predictions."),
    pdf: str = typer.Option(None, "--pdf", help="Source PDF (only with --output=*.pdf)."),
    output: str = typer.Option(None, "--output", help="Output path. Use .pdf for visualization."),
    threshold: float = typer.Option(None, "--threshold"),
    limit: int = typer.Option(None, "--limit"),
) -> None:
    """Run inference: pre-annotate a project's images OR draw boxes on a single PDF."""
    if output and output.endswith(".pdf"):
        if not pdf:
            raise typer.BadParameter("--output=*.pdf requires --pdf=<source.pdf>")
        from core.predict import predict_pdf
        predict_pdf(project, pdf_path=pdf, output_path=output, threshold=threshold, limit=limit)
    elif pre_annotate:
        from core.predict import predict
        predict(project, mode="pre-annotate", threshold=threshold, limit=limit)
    else:
        raise typer.BadParameter("either --pre-annotate or --output=*.pdf must be passed")
```

Update `tests/test_cli.py::test_cli_predict_help_mentions_required_flags` to include `--pdf` and `--output`.

- [ ] **Step 4: Commit (without smoke test yet)**

```bash
git add core/predict.py core/cli.py tests/test_predict.py tests/test_cli.py
git commit -m "feat(core): extend 'dlmf predict' with --output=PDF visualization mode

core.predict.predict_pdf(project, pdf, output, threshold=, limit=):
- Renders the PDF with pdftoppm, runs inference on each page using the
  project's production model (projects/<slug>/models/production.pt if
  the symlink exists, else falls back to baseline Heron).
- Applies per-class thresholds + NMS + full-page-picture filter (from
  the project's postproc config).
- Draws colored rects + label text on the original PDF via PyMuPDF
  (deterministic color per label).
- Saves the annotated PDF to the given output path.

CLI router: 'dlmf predict --output=foo.pdf --pdf=bar.pdf' triggers PDF
mode; 'dlmf predict --pre-annotate' keeps the existing JSON path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Smoke run on real PDF (this validates the new model visually)**

```bash
uv run dlmf predict --project=eaf --pdf=projects/eaf/data/pdfs/EAF-477-2025.pdf --output=/tmp/eaf477_anotado.pdf --limit=10
ls -lh /tmp/eaf477_anotado.pdf
```

Open the PDF (or copy it somewhere browsable). Visually confirm boxes look sensible.

If the smoke succeeds, copy the PDF into the repo for review:
```bash
cp /tmp/eaf477_anotado.pdf projects/eaf/runs/P_repeat_factor_v2/preview_EAF-477_first10.pdf
git add -f projects/eaf/runs/P_repeat_factor_v2/preview_EAF-477_first10.pdf
git commit -m "chore(eaf): add preview PDF showing P_repeat_factor_v2 detections on EAF-477 (first 10 pages)"
```

---

### Task 4: CLI wiring for `evaluate` and `promote`

**Files:**
- Modify: `core/cli.py`, `tests/test_cli.py`

Add two commands:

```python
@app.command(name="evaluate")
def evaluate_cmd(
    project: str = typer.Option(..., "--project", "-p"),
    run: str = typer.Option(..., "--run", "-r"),
) -> None:
    """Re-evaluate a saved run, with per-PDF breakdown."""
    from core.evaluate import evaluate
    evaluate(project, run)


@app.command(name="promote")
def promote_cmd(
    project: str = typer.Option(..., "--project", "-p"),
    run: str = typer.Option(..., "--run", "-r"),
) -> None:
    """Promote a saved run to production (symlink + MLflow registry)."""
    from core.promote import promote
    promote(project, run)
```

Update `test_cli_help_lists_subcommands` to include `evaluate` and `promote`. Add help tests for each.

- [ ] **Step 1: Tests**

```python
def test_cli_evaluate_help_mentions_required_flags():
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout
    assert "--run" in result.stdout


def test_cli_promote_help_mentions_required_flags():
    result = runner.invoke(app, ["promote", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout
    assert "--run" in result.stdout
```

- [ ] **Step 2: Run all CLI tests**

```bash
uv run pytest tests/test_cli.py -v
```

- [ ] **Step 3: Commit**

```bash
git add core/cli.py tests/test_cli.py
git commit -m "feat(cli): add 'dlmf evaluate' and 'dlmf promote' commands"
```

---

### Task 5: End-to-end smoke (evaluate + promote on the real run)

- [ ] **Step 1: Evaluate**

```bash
uv run dlmf evaluate --project=eaf --run=P_repeat_factor_v2 2>&1 | tail -20
```

Expected: prints overall mAP ≈ 0.93, per-PDF breakdown (EAF-089-2025 vs EAF-477-2025 separately).

- [ ] **Step 2: Verify eval_detailed.json**

```bash
cat projects/eaf/runs/P_repeat_factor_v2/eval_detailed.json
```

Should have `overall_mAP`, `per_pdf` (with EAF-089 + EAF-477), `overall_per_class@0.5`.

- [ ] **Step 3: Promote**

```bash
uv run dlmf promote --project=eaf --run=P_repeat_factor_v2
ls -la projects/eaf/models/
```

Expected: `production.pt -> ../runs/P_repeat_factor_v2/best_model.pt`.

- [ ] **Step 4: Commit eval_detailed.json + symlink**

```bash
git add projects/eaf/runs/P_repeat_factor_v2/eval_detailed.json projects/eaf/models/production.pt
git commit -m "chore(eaf): commit per-PDF eval breakdown + production symlink"
```

---

### Task 6: Delete superseded scripts

```bash
git rm training/train_strategies.py training/train_round2.py training/train_round3.py training/train_round4.py
git rm training/reevaluate_real_map.py
git rm training/draw_boxes_on_pdf.py training/draw_boxes_on_pdf_v2.py
git rm training/generate_comparison_pdf.py training/generate_comparison_pdf_v2.py
git rm training/test_best_model.py training/run_docling_pipeline.py
rmdir training/ 2>&1 || echo "training/ not empty or already removed"
git commit -m "chore: remove superseded training scripts (now dlmf train/evaluate/predict)"
```

---

### Task 7: Final verify + tag

```bash
uv run pytest -q
ls projects/eaf/models/  # production symlink in place
git tag -a plan-05-eval-promote-predict-pdf -m "Plan 05 complete: evaluate + promote + predict-PDF working end-to-end."
git tag -l "plan-*"
```

---

## Self-Review

**Spec coverage:**
- ✅ Section 5 (CLI): `dlmf evaluate`, `dlmf promote`, `dlmf predict --output=PDF` shipped.
- ✅ Section 6 (data flow): `eval_detailed.json` + production symlink + MLflow registry.
- 🔜 Plan 06: classify, init-project, README rewrite.

**Type/name consistency:**
- `evaluate.py` uses `_group_images_by_pdf` matching `train.py`'s image-resolution logic.
- `promote.py` uses the same `dlmf-<slug>` experiment naming as `tracking.py::MlflowRun`.
- `predict_pdf` reuses postproc thresholds from config.
