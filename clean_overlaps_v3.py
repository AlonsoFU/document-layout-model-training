"""
Limpia solapamientos en 2 fases:
  1. Eliminar boxes que contienen 2+ boxes adentro (wrappers falsos)
  2. Recortar solapamientos parciales entre los que quedan
"""
import json
import requests

CVAT_URL = "http://localhost:8080"
AUTH = ("admin", "admin")

COCO_FILES = {
    7: "/home/alonso/cvat_docling/coco_output/EAF-089-2025_coco.json",
    8: "/home/alonso/cvat_docling/coco_output/EAF-477-2025_coco.json",
}

LABELS_MAP = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item",
    5: "Page-footer", 6: "Page-header", 7: "Picture", 8: "Section-header",
    9: "Table", 10: "Text", 11: "Title", 12: "Document Index",
    13: "Code", 14: "Checkbox-selected", 15: "Checkbox-unselected",
    16: "Form", 17: "Key-value-region"
}

CONFIDENCE_THRESHOLDS = {
    "Caption": 0.5, "Footnote": 0.5, "Formula": 0.5, "List-item": 0.5,
    "Page-footer": 0.5, "Page-header": 0.5, "Picture": 0.5,
    "Section-header": 0.45, "Table": 0.5, "Text": 0.5, "Title": 0.45,
    "Code": 0.45, "Checkbox-selected": 0.45, "Checkbox-unselected": 0.45,
    "Form": 0.45, "Key-value-region": 0.45, "Document Index": 0.45,
}

UNCROPABLE = {"Table", "Picture", "Formula", "Code", "Form", "Key-value-region"}

# Get CVAT label mapping
resp = requests.get(f"{CVAT_URL}/api/labels?project_id=2", auth=AUTH)
labels_p1 = resp.json()["results"]
resp2 = requests.get(f"{CVAT_URL}/api/labels?project_id=2&page=2", auth=AUTH)
labels_p2 = resp2.json()["results"]
label_name_to_id = {l["name"]: l["id"] for l in labels_p1 + labels_p2}
label_id_to_name = {v: k for k, v in label_name_to_id.items()}


def box_area(p):
    return max(0, p[2] - p[0]) * max(0, p[3] - p[1])


def intersection_area(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def iou(a, b):
    inter = intersection_area(a, b)
    if inter == 0:
        return 0
    return inter / (box_area(a) + box_area(b) - inter)


def containment(inner, outer):
    inner_area = box_area(inner)
    if inner_area <= 0:
        return 0
    return intersection_area(inner, outer) / inner_area


def crop_box(to_crop, fixed):
    x1, y1, x2, y2 = to_crop
    fx1, fy1, fx2, fy2 = fixed

    if x1 >= fx2 or x2 <= fx1 or y1 >= fy2 or y2 <= fy1:
        return to_crop

    original_area = (x2 - x1) * (y2 - y1)
    if original_area <= 0:
        return None

    candidates = []
    if fy2 < y2:
        candidates.append([x1, fy2, x2, y2])
    if fy1 > y1:
        candidates.append([x1, y1, x2, fy1])
    if fx2 < x2:
        candidates.append([fx2, y1, x2, y2])
    if fx1 > x1:
        candidates.append([x1, y1, fx1, y2])

    if not candidates:
        return None

    best = max(candidates, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
    best_area = (best[2] - best[0]) * (best[3] - best[1])

    if best_area < 0.15 * original_area:
        return None

    w = best[2] - best[0]
    h = best[3] - best[1]
    if w > 0 and h > 0 and max(w / h, h / w) > 12:
        return None

    return best


def is_fullpage_picture(shape, page_w=2550, page_h=3300):
    if label_id_to_name.get(shape["label_id"]) != "Picture":
        return False
    return box_area(shape["points"]) > 0.9 * page_w * page_h


def process_task(task_id, coco_file):
    with open(coco_file) as f:
        coco = json.load(f)

    resp = requests.get(f"{CVAT_URL}/api/tasks/{task_id}/data/meta", auth=AUTH)
    frames_meta = resp.json()["frames"]
    fname_to_frame = {f["name"]: i for i, f in enumerate(frames_meta)}
    img_id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}

    shapes = []
    score_map = {}

    for ann in coco["annotations"]:
        fname = img_id_to_fname[ann["image_id"]]
        frame = fname_to_frame.get(fname)
        if frame is None:
            continue
        cat_name = LABELS_MAP.get(ann["category_id"])
        if cat_name is None:
            continue
        cvat_label_id = label_name_to_id.get(cat_name)
        if cvat_label_id is None:
            continue

        score = ann.get("score", 0.5)
        if score < CONFIDENCE_THRESHOLDS.get(cat_name, 0.5):
            continue

        x, y, w, h = ann["bbox"]
        shape = {
            "type": "rectangle",
            "frame": frame,
            "label_id": cvat_label_id,
            "points": [x, y, x + w, y + h],
            "occluded": False,
            "z_order": 0,
            "rotation": 0.0,
            "attributes": [],
            "group": 0,
            "source": "auto",
        }
        shapes.append(shape)
        score_map[id(shape)] = score

    after_conf = len(shapes)
    shapes = [s for s in shapes if not is_fullpage_picture(s)]
    after_pic = len(shapes)

    # =============================================
    # FASE 1: Matar wrappers (boxes con 2+ hijos)
    # =============================================
    by_frame = {}
    for idx, s in enumerate(shapes):
        by_frame.setdefault(s["frame"], []).append(idx)

    wrappers_killed = 0
    wrapper_indices = set()

    for frame, indices in by_frame.items():
        for i in indices:
            pi = shapes[i]["points"]
            # Contar cuántos boxes están contenidos dentro de este
            children = 0
            for j in indices:
                if i == j:
                    continue
                pj = shapes[j]["points"]
                if containment(pj, pi) > 0.7:
                    children += 1
            if children >= 2:
                wrapper_indices.add(i)
                wrappers_killed += 1

    shapes_phase1 = [s for idx, s in enumerate(shapes) if idx not in wrapper_indices]
    # Rebuild score_map references
    score_map_new = {}
    for s in shapes_phase1:
        score_map_new[id(s)] = score_map.get(id(s), 0.5)
    score_map = score_map_new

    # =============================================
    # FASE 2: Recortar solapamientos restantes
    # =============================================
    by_frame = {}
    for idx, s in enumerate(shapes_phase1):
        by_frame.setdefault(s["frame"], []).append(idx)

    to_remove = set()
    crops_done = 0
    duplicates_killed = 0

    for frame, indices in by_frame.items():
        # Sort: uncropable first, then by confidence
        indices.sort(
            key=lambda i: (
                100 if label_id_to_name.get(shapes_phase1[i]["label_id"], "") in UNCROPABLE else 0,
                score_map.get(id(shapes_phase1[i]), 0.5)
            ),
            reverse=True
        )

        for i_pos in range(len(indices)):
            i = indices[i_pos]
            if i in to_remove:
                continue

            for j_pos in range(i_pos + 1, len(indices)):
                j = indices[j_pos]
                if j in to_remove:
                    continue

                pi = shapes_phase1[i]["points"]
                pj = shapes_phase1[j]["points"]

                inter = intersection_area(pi, pj)
                if inter == 0:
                    continue

                iou_val = iou(pi, pj)
                cont_j_in_i = containment(pj, pi)
                cont_i_in_j = containment(pi, pj)

                # Casi idénticos -> eliminar el de menor prioridad
                if iou_val > 0.8:
                    to_remove.add(j)
                    duplicates_killed += 1
                    continue

                # j mayormente dentro de i
                if cont_j_in_i > 0.7:
                    name_i = label_id_to_name.get(shapes_phase1[i]["label_id"], "")
                    name_j = label_id_to_name.get(shapes_phase1[j]["label_id"], "")
                    if name_i in UNCROPABLE and name_j not in UNCROPABLE:
                        to_remove.add(j)
                    elif name_j in UNCROPABLE and name_i not in UNCROPABLE:
                        cropped = crop_box(pi, pj)
                        if cropped:
                            shapes_phase1[i]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(i)
                            break
                    else:
                        cropped = crop_box(pi, pj)
                        if cropped:
                            shapes_phase1[i]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(i)
                            break
                    continue

                # i mayormente dentro de j
                if cont_i_in_j > 0.7:
                    name_i = label_id_to_name.get(shapes_phase1[i]["label_id"], "")
                    name_j = label_id_to_name.get(shapes_phase1[j]["label_id"], "")
                    if name_j in UNCROPABLE and name_i not in UNCROPABLE:
                        to_remove.add(i)
                        break
                    elif name_i in UNCROPABLE and name_j not in UNCROPABLE:
                        cropped = crop_box(pj, pi)
                        if cropped:
                            shapes_phase1[j]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(j)
                    else:
                        cropped = crop_box(pj, pi)
                        if cropped:
                            shapes_phase1[j]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(j)
                    continue

                # Solapamiento parcial (IoU 0.3-0.8) -> recortar el de menor prioridad
                if iou_val > 0.3:
                    name_j = label_id_to_name.get(shapes_phase1[j]["label_id"], "")
                    name_i = label_id_to_name.get(shapes_phase1[i]["label_id"], "")
                    if name_j not in UNCROPABLE:
                        cropped = crop_box(pj, pi)
                        if cropped:
                            shapes_phase1[j]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(j)
                    elif name_i not in UNCROPABLE:
                        cropped = crop_box(pi, pj)
                        if cropped:
                            shapes_phase1[i]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(i)
                            break

    cleaned = [s for idx, s in enumerate(shapes_phase1) if idx not in to_remove]

    original = len(coco["annotations"])
    print(f"\nTask {task_id}:")
    print(f"  Original:                {original}")
    print(f"  After conf filter:       {after_conf}")
    print(f"  After fullpage filter:   {after_pic}")
    print(f"  FASE 1 - Wrappers (2+ hijos): {wrappers_killed} eliminados -> {len(shapes_phase1)}")
    print(f"  FASE 2 - Duplicados:     {duplicates_killed} eliminados")
    print(f"  FASE 2 - Recortados:     {crops_done}")
    print(f"  FASE 2 - Eliminados:     {len(to_remove) - duplicates_killed}")
    print(f"  Final:                   {len(cleaned)}")

    payload = {"version": 0, "tags": [], "shapes": cleaned, "tracks": []}
    resp = requests.put(f"{CVAT_URL}/api/tasks/{task_id}/annotations/", json=payload, auth=AUTH)
    print(f"  Upload: {resp.status_code}")


import os
import time
import zipfile
import tempfile

IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes"
LABELS_LIST = list(set(LABELS_MAP.values()))

# Create new project
resp = requests.post(
    f"{CVAT_URL}/api/projects",
    json={"name": "Docling Heron CLEAN (sin overlaps v3)", "labels": [{"name": n} for n in LABELS_LIST]},
    auth=AUTH,
)
resp.raise_for_status()
new_project_id = resp.json()["id"]
print(f"Nuevo proyecto creado: id={new_project_id}")

# Get new project label mapping
resp = requests.get(f"{CVAT_URL}/api/labels?project_id={new_project_id}", auth=AUTH)
new_labels = resp.json()["results"]
if resp.json()["next"]:
    resp2_new = requests.get(resp.json()["next"], auth=AUTH)
    new_labels += resp2_new.json()["results"]
new_label_name_to_id = {l["name"]: l["id"] for l in new_labels}

TASK_FOLDERS = {7: "EAF-089-2025", 8: "EAF-477-2025"}

for task_id, coco_file in COCO_FILES.items():
    folder_name = TASK_FOLDERS[task_id]

    # Create task in new project
    resp = requests.post(
        f"{CVAT_URL}/api/tasks",
        json={"name": folder_name, "project_id": new_project_id},
        auth=AUTH,
    )
    resp.raise_for_status()
    new_task_id = resp.json()["id"]
    print(f"\nNueva tarea {new_task_id}: {folder_name}")

    # Upload images
    folder_path = os.path.join(IMAGES_DIR, folder_name)
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for fname in image_files:
            zf.write(os.path.join(folder_path, fname), fname)
    with open(zip_path, "rb") as f:
        resp = requests.post(
            f"{CVAT_URL}/api/tasks/{new_task_id}/data",
            files={"client_files[0]": ("images.zip", f, "application/zip")},
            data={"image_quality": 100, "use_zip_chunks": "true"},
            auth=AUTH,
        )
    resp.raise_for_status()
    os.unlink(zip_path)
    while True:
        resp = requests.get(f"{CVAT_URL}/api/tasks/{new_task_id}/status", auth=AUTH)
        if resp.json().get("state") in ("Finished", "Failed"):
            break
        time.sleep(3)
    print(f"  Imágenes: {resp.json().get('state')}")

    # Process and upload clean annotations
    # Override label mapping to use new project's IDs
    label_name_to_id_bak = label_name_to_id.copy()
    label_name_to_id.update(new_label_name_to_id)
    label_id_to_name.update({v: k for k, v in new_label_name_to_id.items()})

    process_task(new_task_id, coco_file)

    # Restore
    label_name_to_id.update(label_name_to_id_bak)
    label_id_to_name.update({v: k for k, v in label_name_to_id_bak.items()})

print(f"\nListo! Proyecto limpio: http://localhost:8080/projects/{new_project_id}")
