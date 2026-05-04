"""Pre-annotation: run Heron baseline over a project's images, write COCO predictions.

Output goes to `projects/<slug>/cvat/pre_annotations/<timestamp>.json`.
This file is then consumed by `dlmf cvat-push --coco=<file>` to pre-load
predictions into CVAT for human review.

Also provides `predict_pdf()` (Plan 05 Task 3): load the project's production
model, run inference page-by-page on a PDF, and draw boxes using PyMuPDF.
"""
from __future__ import annotations

import datetime as dt
import gc
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from core.lib.config import load_config
from core.lib.model import MODEL_INDEX_TO_LABEL_NAME, load_heron
from core.lib.postproc import (
    apply_thresholds,
    full_page_picture_filter,
    nms_per_category,
)

PROJECTS_ROOT = Path("projects")


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------

def _color_for_label(name: str) -> tuple[float, float, float]:
    """Deterministic per-label RGB color (0..1 floats for PyMuPDF)."""
    h = hashlib.md5(name.encode()).digest()
    return (h[0] / 255.0, h[1] / 255.0, h[2] / 255.0)


# ---------------------------------------------------------------------------
# PDF annotation (Plan 05 — Task 3)
# ---------------------------------------------------------------------------

def predict_pdf(
    project_slug: str,
    pdf_path: str | Path,
    output_path: str | Path,
    threshold: float | None = None,
    limit: int | None = None,
) -> Path:
    """Run inference on a PDF and write an annotated copy.

    Loads the project's production model (``projects/<slug>/models/production.pt``
    symlink if it exists; otherwise falls back to the baseline Heron model).
    Renders each page to PNG via pdftoppm at the project-configured DPI, runs
    inference + post-processing, then draws colored bounding boxes on the
    original PDF page using PyMuPDF (fitz).

    Returns the output path.
    """
    import fitz  # PyMuPDF
    import torch
    from PIL import Image

    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    # --- Load project config ---
    project_dir = PROJECTS_ROOT / project_slug
    cfg = load_config(project_dir / "config.yaml")

    dpi = int(cfg.get("render", {}).get("dpi", 300))
    pp_cfg = cfg.get("postprocess", {})
    thresholds = dict(pp_cfg.get("thresholds", {}))
    default_thr = float(threshold if threshold is not None else thresholds.pop("default", 0.5))
    nms_iou = float(pp_cfg.get("nms_iou", 0.5))
    fullpage_frac = float(pp_cfg.get("full_page_picture_filter", 0.9))

    # --- Load model ---
    production_pt = project_dir / "models" / "production.pt"
    if production_pt.exists():
        print(f"[predict_pdf] loading production model from {production_pt} (LoRA)...")
        import yaml as _yaml
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        from core.lib.model import apply_lora, load_lora_state

        # Find the run this symlink points to so we can read config_resolved.yaml.
        target = production_pt.resolve()
        # Symlink points to best_model.pt inside a run dir.
        run_dir = target.parent

        config_resolved_path = run_dir / "config_resolved.yaml"
        if config_resolved_path.exists():
            config_resolved = _yaml.safe_load(config_resolved_path.read_text())
            tcfg = config_resolved["training"]
            base_model = tcfg["base_model"]
            lora_cfg = tcfg["lora"]
        else:
            # Fallback defaults
            base_model = "docling-project/docling-layout-heron"
            lora_cfg = {"rank": 32, "alpha": 64, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj"]}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = RTDetrImageProcessor.from_pretrained(base_model)
        model = RTDetrV2ForObjectDetection.from_pretrained(base_model)
        model.config.num_denoising = 0
        model.config.eos_coefficient = 0.0001
        for n, p in model.named_parameters():
            if "backbone" in n or "encoder" in n:
                p.requires_grad = False
        apply_lora(
            model,
            rank=int(lora_cfg["rank"]),
            alpha=int(lora_cfg["alpha"]),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            target_substrings=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj"])),
        )
        state = torch.load(target, map_location="cpu")
        load_lora_state(model, state)
        model.to(device).eval()
    else:
        print("[predict_pdf] no production.pt found — falling back to baseline Heron...")
        from core.lib.model import DEFAULT_BASE_MODEL
        model, processor = load_heron(model_name=DEFAULT_BASE_MODEL, device="auto")
        device = str(next(model.parameters()).device)

    # --- Render PDF pages to PNGs in a temp dir ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = str(Path(tmp_dir) / "pagina")
        cmd = ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), prefix]
        print(f"[predict_pdf] rendering {pdf_path.name} at {dpi} DPI...")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        pngs = sorted(Path(tmp_dir).glob("pagina-*.png"))
        if limit is not None:
            pngs = pngs[:limit]

        if not pngs:
            raise FileNotFoundError(f"pdftoppm produced no PNGs for {pdf_path}")

        print(f"[predict_pdf] {len(pngs)} page(s) to annotate")

        # Scale from PNG pixel coords → PDF point coords.
        scale = 72.0 / dpi

        # Open the original PDF for annotation.
        pdf_doc = fitz.open(str(pdf_path))

        for page_idx, png_path in enumerate(pngs):
            image = Image.open(png_path).convert("RGB")
            w, h = image.size

            inputs = processor(images=image, return_tensors="pt")
            _dev = torch.device(device)
            inputs = {k: v.to(_dev) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([[h, w]]).to(_dev)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=default_thr
            )[0]

            # Build detection dicts.
            dets = []
            for box, score, lbl in zip(results["boxes"], results["scores"], results["labels"]):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                name = MODEL_INDEX_TO_LABEL_NAME[int(lbl.item())]
                dets.append({"box": (x1, y1, x2, y2), "label": name, "score": float(score.item())})

            dets = apply_thresholds(dets, thresholds, default=default_thr)
            dets = nms_per_category(dets, iou_threshold=nms_iou)
            dets = full_page_picture_filter(dets, page_w=w, page_h=h, min_fraction=fullpage_frac)

            # Draw on the PDF page.
            page = pdf_doc.load_page(page_idx)
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                label = d["label"]
                score = d["score"]
                color = _color_for_label(label)
                rect = fitz.Rect(x1 * scale, y1 * scale, x2 * scale, y2 * scale)
                page.draw_rect(rect, color=color, width=2)
                page.insert_text(
                    (rect.x0, max(rect.y0 - 2, 4)),
                    f"{label} {score:.2f}",
                    fontsize=8,
                    color=color,
                )

            del outputs, inputs, image
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

            print(f"[predict_pdf]   page {page_idx + 1}/{len(pngs)}: {len(dets)} detections")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_doc.save(str(output_path))
        pdf_doc.close()

    print(f"[predict_pdf] annotated PDF saved to {output_path}")
    return output_path


def predict(
    project_slug: str,
    mode: str = "pre-annotate",
    threshold: float | None = None,
    limit: int | None = None,
) -> Path:
    """Run inference and write a COCO file. Returns the output path.

    `mode='pre-annotate'` is the only mode in Plan 03. Plan 05 adds 'visualize'.
    `threshold` overrides the config's `postprocess.thresholds.default` if given.
    `limit` is an integer that caps the number of images processed (smoke testing).
    """
    if mode != "pre-annotate":
        raise NotImplementedError(f"mode={mode!r} not implemented yet")

    project_dir = PROJECTS_ROOT / project_slug
    cfg = load_config(project_dir / "config.yaml")
    labels: list[str] = list(cfg["labels"])
    label_to_cat_id = {name: i + 1 for i, name in enumerate(labels)}

    pp_cfg = cfg.get("postprocess", {})
    thresholds = dict(pp_cfg.get("thresholds", {}))
    default_thr = float(threshold if threshold is not None else thresholds.pop("default", 0.5))
    nms_iou = float(pp_cfg.get("nms_iou", 0.5))
    fullpage_frac = float(pp_cfg.get("full_page_picture_filter", 0.9))

    images_root = project_dir / "data" / "images"
    image_dirs = sorted(d for d in images_root.iterdir() if d.is_dir())
    if not image_dirs:
        raise FileNotFoundError(f"no image directories in {images_root}")

    # Collect all images across all task dirs.
    all_pngs: list[Path] = []
    for d in image_dirs:
        all_pngs.extend(sorted(d.glob("pagina-*.png")))
    if limit is not None:
        all_pngs = all_pngs[:limit]
    if not all_pngs:
        raise FileNotFoundError(f"no PNGs under {images_root}")

    from core.lib.model import DEFAULT_BASE_MODEL

    print(f"[predict] loading Heron model...")
    model, processor = load_heron(model_name=DEFAULT_BASE_MODEL, device="auto")

    import torch
    from PIL import Image

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 1

    for img_idx, png in enumerate(all_pngs, start=1):
        image = Image.open(png).convert("RGB")
        w, h = image.size
        coco_images.append({
            "id": img_idx,
            "width": w,
            "height": h,
            "file_name": png.name,
        })

        inputs = processor(images=image, return_tensors="pt")
        _device = getattr(model, "device", None)
        if isinstance(_device, torch.device):
            inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        if isinstance(_device, torch.device):
            target_sizes = target_sizes.to(_device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=default_thr
        )[0]

        # Translate to detection dicts.
        dets = []
        for box, score, lbl in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            name = MODEL_INDEX_TO_LABEL_NAME[int(lbl.item())]
            dets.append({
                "box": (x1, y1, x2, y2),
                "label": name,
                "score": float(score.item()),
            })

        # Postprocess: per-class threshold (we already passed default_thr to
        # post_process, but apply per-class for the ones above default), NMS,
        # full-page picture filter.
        dets = apply_thresholds(dets, thresholds, default=default_thr)
        dets = nms_per_category(dets, iou_threshold=nms_iou)
        dets = full_page_picture_filter(dets, page_w=w, page_h=h, min_fraction=fullpage_frac)

        # Convert to COCO annotations.
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            cat_id = label_to_cat_id.get(d["label"])
            if cat_id is None:
                continue  # label not in this project's vocabulary
            bw, bh = x2 - x1, y2 - y1
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": cat_id,
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "segmentation": [],
                "score": round(d["score"], 4),
                "attributes": {"occluded": False, "rotation": 0.0},
            })
            ann_id += 1

        # Cleanup per page to keep VRAM use low.
        del outputs, inputs, image
        if (img_idx % 25) == 0:
            print(f"[predict]   {img_idx}/{len(all_pngs)} pages")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    # Build COCO.
    categories = [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(labels)]
    coco = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "description": f"Heron baseline pre-annotations for {project_slug}",
            "date_created": dt.datetime.now().isoformat(),
        },
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    out_dir = project_dir / "cvat" / "pre_annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{ts}.json"
    out_path.write_text(json.dumps(coco))
    print(f"[predict] wrote {out_path}: {len(coco_images)} images, {len(coco_annotations)} annotations")
    return out_path
