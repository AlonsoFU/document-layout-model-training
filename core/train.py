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
    # Pre-populate optional knobs so apply_overrides accepts them as "known" keys.
    cfg.setdefault("training", {}).setdefault("limit", None)
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

    # Build per-task image_ids: ids are project-level. Resolve each COCO file_name
    # to a real path on disk. Project-level COCOs disambiguate same-named images
    # across tasks by suffixing `_N` (e.g. pagina-001_1.png belongs to the second
    # task's pagina-001.png). The on-disk files keep their per-task name unchanged.
    import re as _re
    images_root = project_dir / "data" / "images"
    subdirs = sorted(d for d in images_root.iterdir() if d.is_dir())
    name_to_paths: dict[str, list[Path]] = {}
    for sub in subdirs:
        for png in sorted(sub.glob("*.png")):
            name_to_paths.setdefault(png.name, []).append(png)

    image_path_for: dict[int, Path] = {}
    for img in coco["images"]:
        fname = img["file_name"]
        m = _re.match(r"^(.+)_(\d+)(\.[^.]+)$", fname)
        if m:
            base = m.group(1) + m.group(3)
            occurrence = int(m.group(2))  # 1, 2, ... → second/third occurrence
        else:
            base = fname
            occurrence = 0  # 0 → first occurrence
        candidates = name_to_paths.get(base, [])
        if occurrence < len(candidates):
            image_path_for[img["id"]] = candidates[occurrence]

    valid_ids = sorted(image_path_for.keys())
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
        target = flat_dir / info["file_name"]  # keeps the _N suffix from COCO
        src = image_path_for[iid]
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
