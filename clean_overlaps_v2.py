"""
Limpia solapamientos recortando boxes en vez de eliminarlos.
Si un box se solapa con otro, lo recorta por el lado que minimiza
la pérdida de área. Solo elimina si queda demasiado pequeño.
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

# Labels that should NOT be cropped (their geometry matters)
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
    """
    Recorta to_crop para no solaparse con fixed.
    Prueba recortar por cada lado y elige el corte que conserva más área.
    Retorna los nuevos points o None si no queda nada útil.
    """
    x1, y1, x2, y2 = to_crop
    fx1, fy1, fx2, fy2 = fixed

    # No overlap
    if x1 >= fx2 or x2 <= fx1 or y1 >= fy2 or y2 <= fy1:
        return to_crop

    original_area = (x2 - x1) * (y2 - y1)
    if original_area <= 0:
        return None

    candidates = []

    # Crop from top (keep bottom part): move y1 down to fy2
    if fy2 < y2:
        c = [x1, fy2, x2, y2]
        candidates.append(c)

    # Crop from bottom (keep top part): move y2 up to fy1
    if fy1 > y1:
        c = [x1, y1, x2, fy1]
        candidates.append(c)

    # Crop from left (keep right part): move x1 right to fx2
    if fx2 < x2:
        c = [fx2, y1, x2, y2]
        candidates.append(c)

    # Crop from right (keep left part): move x2 left to fx1
    if fx1 > x1:
        c = [x1, y1, fx1, y2]
        candidates.append(c)

    if not candidates:
        return None

    # Pick the crop that preserves the most area
    best = max(candidates, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
    best_area = (best[2] - best[0]) * (best[3] - best[1])

    # If the remaining area is too small (<15% of original), discard
    if best_area < 0.15 * original_area:
        return None

    # If the remaining box is too thin (aspect ratio > 10:1), discard
    w = best[2] - best[0]
    h = best[3] - best[1]
    if w > 0 and h > 0:
        ratio = max(w / h, h / w)
        if ratio > 12:
            return None

    return best


def priority_score(shape, score):
    """Higher = harder to crop/remove."""
    name = label_id_to_name.get(shape["label_id"], "")
    # Uncropable types get bonus priority
    bonus = 100 if name in UNCROPABLE else 0
    return bonus + score


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

    # Remove full-page pictures
    shapes = [s for s in shapes if not is_fullpage_picture(s)]
    after_pic = len(shapes)

    # Process overlaps per frame
    by_frame = {}
    for idx, s in enumerate(shapes):
        by_frame.setdefault(s["frame"], []).append(idx)

    to_remove = set()
    crops_done = 0

    for frame, indices in by_frame.items():
        # Sort by priority (uncropable first, then by confidence)
        indices.sort(key=lambda i: priority_score(shapes[i], score_map.get(id(shapes[i]), 0.5)), reverse=True)

        for i_pos in range(len(indices)):
            i = indices[i_pos]
            if i in to_remove:
                continue

            for j_pos in range(i_pos + 1, len(indices)):
                j = indices[j_pos]
                if j in to_remove:
                    continue

                pi = shapes[i]["points"]
                pj = shapes[j]["points"]

                inter = intersection_area(pi, pj)
                if inter == 0:
                    continue

                iou_val = iou(pi, pj)
                cont_j_in_i = containment(pj, pi)
                cont_i_in_j = containment(pi, pj)

                # Near-identical (IoU > 0.8) -> remove lower priority
                if iou_val > 0.8:
                    to_remove.add(j)  # j has lower priority (sorted)
                    continue

                # j mostly inside i (>70%)
                if cont_j_in_i > 0.7:
                    # j is smaller and inside i
                    name_i = label_id_to_name.get(shapes[i]["label_id"], "")
                    name_j = label_id_to_name.get(shapes[j]["label_id"], "")

                    if name_i in UNCROPABLE and name_j not in UNCROPABLE:
                        # Remove j (the inner one) since i is structural
                        to_remove.add(j)
                    elif name_j in UNCROPABLE and name_i not in UNCROPABLE:
                        # Crop i around j
                        cropped = crop_box(pi, pj)
                        if cropped:
                            shapes[i]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(i)
                            break
                    else:
                        # Both same type priority -> crop the outer one
                        cropped = crop_box(pi, pj)
                        if cropped:
                            shapes[i]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(i)
                            break
                    continue

                # i mostly inside j
                if cont_i_in_j > 0.7:
                    name_i = label_id_to_name.get(shapes[i]["label_id"], "")
                    name_j = label_id_to_name.get(shapes[j]["label_id"], "")

                    if name_j in UNCROPABLE and name_i not in UNCROPABLE:
                        to_remove.add(i)
                        break
                    elif name_i in UNCROPABLE and name_j not in UNCROPABLE:
                        cropped = crop_box(pj, pi)
                        if cropped:
                            shapes[j]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(j)
                    else:
                        cropped = crop_box(pj, pi)
                        if cropped:
                            shapes[j]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(j)
                    continue

                # Partial overlap (IoU 0.3-0.8) -> crop the lower priority one
                if iou_val > 0.3:
                    name_j = label_id_to_name.get(shapes[j]["label_id"], "")
                    if name_j not in UNCROPABLE:
                        cropped = crop_box(pj, pi)
                        if cropped:
                            shapes[j]["points"] = cropped
                            crops_done += 1
                        else:
                            to_remove.add(j)
                    else:
                        name_i = label_id_to_name.get(shapes[i]["label_id"], "")
                        if name_i not in UNCROPABLE:
                            cropped = crop_box(pi, pj)
                            if cropped:
                                shapes[i]["points"] = cropped
                                crops_done += 1
                            else:
                                to_remove.add(i)
                                break
                        # Both uncropable with moderate overlap -> keep both

    cleaned = [s for idx, s in enumerate(shapes) if idx not in to_remove]
    removed = len(shapes) - len(cleaned)

    original = len(coco["annotations"])
    print(f"\nTask {task_id}:")
    print(f"  Original:           {original}")
    print(f"  After conf filter:  {after_conf}")
    print(f"  After page filter:  {after_pic}")
    print(f"  Boxes recortados:   {crops_done}")
    print(f"  Boxes eliminados:   {removed}")
    print(f"  Final:              {len(cleaned)}")

    payload = {"version": 0, "tags": [], "shapes": cleaned, "tracks": []}
    resp = requests.put(f"{CVAT_URL}/api/tasks/{task_id}/annotations/", json=payload, auth=AUTH)
    print(f"  Upload: {resp.status_code}")


for task_id, coco_file in COCO_FILES.items():
    process_task(task_id, coco_file)
