"""Sync between CVAT and projects/<slug>/cvat/.

push: creates the CVAT project + tasks (one per PDF), uploads PNGs from
      data/images/<pdf_stem>/, and optionally imports COCO annotations
      filtered per task.

pull: exports the CVAT project as COCO and writes to
      projects/<slug>/cvat/exports/v<N+1>_<YYYY-MM-DD>/instances_default.json.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

from core.lib.config import load_config
from core.lib.cvat import CvatClient

PROJECTS_ROOT = Path("projects")


def _build_client(cfg: dict[str, Any]) -> CvatClient:
    url = cfg.get("cvat", {}).get("url", "http://localhost:8080")
    user = os.environ.get("CVAT_USER", "admin")
    password = os.environ.get("CVAT_PASSWORD", "admin")
    return CvatClient(url, auth=(user, password))


def push(project_slug: str, coco_path: str | None = None) -> None:
    project_dir = PROJECTS_ROOT / project_slug
    cfg = load_config(project_dir / "config.yaml")
    client = _build_client(cfg)

    project_name = cfg["cvat"]["project_name"]
    labels = list(cfg["labels"])

    # Project (idempotent)
    p = client.get_project_by_name(project_name)
    if p is None:
        project_id = client.create_project(project_name, labels)
        print(f"[push] created project '{project_name}' id={project_id}")
    else:
        project_id = int(p["id"])
        print(f"[push] reusing project '{project_name}' id={project_id}")

    # Optional COCO file for pre-labels
    coco = None
    if coco_path:
        with open(coco_path) as f:
            coco = json.load(f)

    # Tasks: one per PDF, named after the PDF stem
    pdf_dir = project_dir / "data" / "pdfs"
    images_root = project_dir / "data" / "images"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {pdf_dir}")

    for pdf in pdfs:
        task_name = pdf.stem
        img_dir = images_root / task_name
        pngs = sorted(img_dir.glob("pagina-*.png")) if img_dir.exists() else []
        if not pngs:
            raise FileNotFoundError(
                f"no PNGs in {img_dir} — run 'dlmf render --project={project_slug}' first"
            )

        # Task (idempotent)
        t = client.get_task_by_name(project_id, task_name)
        if t is None:
            task_id = client.create_task(task_name, project_id=project_id)
            current_size = 0
            print(f"[push] created task '{task_name}' id={task_id}")
        else:
            task_id = int(t["id"])
            current_size = int(t.get("size") or 0)
            print(f"[push] reusing task '{task_name}' id={task_id} size={current_size}")

        # Images
        if current_size == 0:
            buf = _zip_files(pngs)
            print(f"[push] uploading {len(pngs)} images ({buf.getbuffer().nbytes/1e6:.1f} MB)")
            client.upload_data(task_id, buf)
        elif current_size != len(pngs):
            raise RuntimeError(
                f"task '{task_name}' has {current_size} images, expected {len(pngs)}"
            )
        else:
            print(f"[push] images already present ({current_size})")

        # Annotations
        if coco is None:
            continue
        if client.count_task_annotations(task_id) > 0:
            print(f"[push] annotations already present, skip")
            continue
        task_filenames = {p.name for p in pngs}
        # Detect if this task uses the _1 suffix in the COCO (project-level merge artefact)
        suffix_pattern = re.compile(r"^.*_1\.png$")
        task_has_suffix_in_coco = any(
            suffix_pattern.match(img["file_name"])
            and img["file_name"].rsplit("_1", 1)[0] + ".png" in task_filenames
            for img in coco["images"]
        )
        task_coco = _filter_coco_for_task(
            coco, image_filenames=task_filenames, strip_suffix=task_has_suffix_in_coco
        )
        if not task_coco["images"]:
            print(f"[push] no matching COCO images for task '{task_name}', skip")
            continue
        print(f"[push] importing {len(task_coco['annotations'])} annotations")
        ann_buf = _zip_coco(task_coco)
        client.import_annotations(task_id, ann_buf, fmt="COCO 1.0")


def _zip_files(paths: list[Path]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for p in paths:
            zf.write(p, p.name)
    buf.seek(0)
    return buf


def _zip_coco(coco: dict) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("annotations/instances_default.json", json.dumps(coco))
    buf.seek(0)
    return buf


def _filter_coco_for_task(
    coco: dict, image_filenames: set[str], strip_suffix: bool
) -> dict:
    """Slice a project-level COCO to one task; renumber image_ids 1..N.

    If `strip_suffix`, normalize filenames `pagina-NNN_1.png` → `pagina-NNN.png`
    before matching against `image_filenames`.
    """
    def normalize(name: str) -> str:
        if strip_suffix:
            return re.sub(r"_1(\.png)$", r"\1", name)
        return name

    def matches(img: dict) -> bool:
        norm = normalize(img["file_name"])
        if norm not in image_filenames:
            return False
        # When strip_suffix is active, only include images whose filename was
        # actually normalized (i.e. had the _1 suffix). Plain filenames belong
        # to other tasks in the project-level COCO and should be skipped.
        if strip_suffix and norm == img["file_name"]:
            return False
        return True

    selected = [img for img in coco["images"] if matches(img)]
    selected.sort(key=lambda i: int(i["id"]))
    id_map = {}
    new_images = []
    for new_id, img in enumerate(selected, start=1):
        id_map[img["id"]] = new_id
        new_images.append({**img, "id": new_id, "file_name": normalize(img["file_name"])})
    new_anns = []
    for ann in coco["annotations"]:
        if ann["image_id"] not in id_map:
            continue
        new_anns.append({**ann, "image_id": id_map[ann["image_id"]]})
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": new_images,
        "annotations": new_anns,
    }


def pull(project_slug: str, version: str | None = None) -> None:
    """Export CVAT project to COCO and save to projects/<slug>/cvat/exports/v<N>_<YYYY-MM-DD>/."""
    project_dir = PROJECTS_ROOT / project_slug
    cfg = load_config(project_dir / "config.yaml")
    client = _build_client(cfg)

    project_name = cfg["cvat"]["project_name"]
    p = client.get_project_by_name(project_name)
    if p is None:
        raise RuntimeError(f"CVAT project '{project_name}' not found — push first")
    project_id = int(p["id"])

    exports_dir = project_dir / "cvat" / "exports"
    if version is None:
        version = _next_version(exports_dir)
    out_dir = exports_dir / version
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; pass --version=v<N>_<date> explicitly")
    out_dir.mkdir(parents=True)

    print(f"[pull] exporting project {project_id} → {out_dir}")
    zip_bytes = client.export_dataset(project_id, fmt="COCO 1.0", save_images=False)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("instances_default.json"):
                target = out_dir / "instances_default.json"
                target.write_bytes(zf.read(name))
                # Validate
                d = json.loads(target.read_text())
                print(
                    f"[pull] saved {target} — "
                    f"{len(d['images'])} images, "
                    f"{len(d['annotations'])} annotations, "
                    f"{len(d['categories'])} categories"
                )
                return
    raise RuntimeError("export ZIP did not contain annotations/instances_default.json")


def _next_version(exports_dir: Path) -> str:
    """Compute next sequential version label like v3_2026-05-02."""
    if not exports_dir.exists():
        n = 1
    else:
        existing = [d.name for d in exports_dir.iterdir() if d.is_dir()]
        nums = []
        for name in existing:
            m = re.match(r"v(\d+)_", name)
            if m:
                nums.append(int(m.group(1)))
        n = (max(nums) + 1) if nums else 1
    today = dt.date.today().isoformat()
    return f"v{n}_{today}"
