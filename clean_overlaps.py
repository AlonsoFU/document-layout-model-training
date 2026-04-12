"""
Limpia solapamientos en CVAT usando la estrategia de Docling.
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

# Docling confidence thresholds per category
CONFIDENCE_THRESHOLDS = {
    "Caption": 0.5, "Footnote": 0.5, "Formula": 0.5, "List-item": 0.5,
    "Page-footer": 0.5, "Page-header": 0.5, "Picture": 0.5,
    "Section-header": 0.45, "Table": 0.5, "Text": 0.5, "Title": 0.45,
    "Code": 0.45, "Checkbox-selected": 0.45, "Checkbox-unselected": 0.45,
    "Form": 0.45, "Key-value-region": 0.45, "Document Index": 0.45,
}

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
    """Fraction of inner that is inside outer."""
    inner_area = box_area(inner)
    if inner_area <= 0:
        return 0
    return intersection_area(inner, outer) / inner_area


def choose_winner(s1, s2, scores):
    """
    Given two overlapping shapes, decide which to keep.
    Returns the index of the one to REMOVE, or None if both should stay.
    Uses Docling's strategy.
    """
    p1, p2 = s1["points"], s2["points"]
    a1, a2 = box_area(p1), box_area(p2)
    l1 = label_id_to_name.get(s1["label_id"], "")
    l2 = label_id_to_name.get(s2["label_id"], "")
    sc1 = scores.get(id(s1), 0.5)
    sc2 = scores.get(id(s2), 0.5)

    iou_val = iou(p1, p2)
    cont_1_in_2 = containment(p1, p2)  # how much of s1 is inside s2
    cont_2_in_1 = containment(p2, p1)  # how much of s2 is inside s1

    # Not overlapping enough -> both stay
    if iou_val < 0.3 and cont_1_in_2 < 0.7 and cont_2_in_1 < 0.7:
        return None

    # --- Docling special rules ---

    # List-item vs Text with similar area -> keep List-item
    if {l1, l2} == {"List-item", "Text"}:
        ratio = min(a1, a2) / max(a1, a2) if max(a1, a2) > 0 else 0
        if ratio > 0.8:
            return 1 if l1 == "List-item" else 0  # remove the Text one

    # Code containing another -> keep Code
    if l1 == "Code" and cont_2_in_1 > 0.8:
        return 1  # remove s2
    if l2 == "Code" and cont_1_in_2 > 0.8:
        return 0  # remove s1

    # Key-value-region vs Table overlapping >90% -> remove wrapper
    if l1 == "Key-value-region" and l2 == "Table" and iou_val > 0.5:
        return 0  # remove key-value-region
    if l2 == "Key-value-region" and l1 == "Table" and iou_val > 0.5:
        return 1

    # Near-identical geometry (IoU > 0.8) -> keep higher confidence
    if iou_val > 0.8:
        return 1 if sc1 >= sc2 else 0

    # One contains the other (>70%) -> remove the outer (bigger) one
    if cont_2_in_1 > 0.7 and a1 > a2:
        return 0  # remove outer s1
    if cont_1_in_2 > 0.7 and a2 > a1:
        return 1  # remove outer s2

    # Significant overlap (IoU > 0.5) but not contained -> keep higher confidence
    if iou_val > 0.5:
        return 1 if sc1 >= sc2 else 0

    # Moderate overlap -> keep both (let the human decide in CVAT)
    return None


def is_fullpage_picture(shape, page_w=2550, page_h=3300):
    """Pictures covering >90% of page are false positives."""
    if label_id_to_name.get(shape["label_id"]) != "Picture":
        return False
    area = box_area(shape["points"])
    page_area = page_w * page_h
    return area > 0.9 * page_area


for task_id, coco_file in COCO_FILES.items():
    # Load original COCO for scores
    with open(coco_file) as f:
        coco = json.load(f)

    # Get frame mapping
    resp = requests.get(f"{CVAT_URL}/api/tasks/{task_id}/data/meta", auth=AUTH)
    frames_meta = resp.json()["frames"]
    fname_to_frame = {f["name"]: i for i, f in enumerate(frames_meta)}
    img_id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}

    # Build shapes from COCO with scores
    shapes = []
    score_map = {}  # id(shape) -> score

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

        # Step 1: Apply per-category confidence threshold (Docling style)
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

    after_threshold = len(shapes)

    # Step 2: Remove full-page pictures
    shapes = [s for s in shapes if not is_fullpage_picture(s)]
    after_picture = len(shapes)

    # Step 3: Resolve overlaps per frame
    by_frame = {}
    for idx, s in enumerate(shapes):
        by_frame.setdefault(s["frame"], []).append(idx)

    to_remove = set()

    for frame, indices in by_frame.items():
        # Compare all pairs
        for i_pos in range(len(indices)):
            i = indices[i_pos]
            if i in to_remove:
                continue
            for j_pos in range(i_pos + 1, len(indices)):
                j = indices[j_pos]
                if j in to_remove:
                    continue

                result = choose_winner(shapes[i], shapes[j], score_map)
                if result == 0:
                    to_remove.add(i)
                    break  # i removed, no need to check more
                elif result == 1:
                    to_remove.add(j)

    cleaned = [s for idx, s in enumerate(shapes) if idx not in to_remove]

    original = len(coco["annotations"])
    print(f"\nTask {task_id}:")
    print(f"  Original:           {original}")
    print(f"  After conf filter:  {after_threshold} (-{original - after_threshold} low confidence)")
    print(f"  After page filter:  {after_picture} (-{after_threshold - after_picture} full-page pictures)")
    print(f"  After overlap fix:  {len(cleaned)} (-{after_picture - len(cleaned)} overlaps)")

    # Upload
    payload = {"version": 0, "tags": [], "shapes": cleaned, "tracks": []}
    resp = requests.put(f"{CVAT_URL}/api/tasks/{task_id}/annotations/", json=payload, auth=AUTH)
    print(f"  Upload: {resp.status_code}")


if __name__ == "__main__":
    pass
