"""Pre-annotation: run Heron baseline over a project's images, write COCO predictions.

Output goes to `projects/<slug>/cvat/pre_annotations/<timestamp>.json`.
This file is then consumed by `dlmf cvat-push --coco=<file>` to pre-load
predictions into CVAT for human review.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
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
