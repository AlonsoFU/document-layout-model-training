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
