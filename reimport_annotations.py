"""
Reimport COCO annotations to CVAT tasks using proper upload + poll.
"""
import os
import json
import time
import zipfile
import tempfile
import requests

CVAT_URL = "http://localhost:8080"
AUTH = ("admin", "admin")
COCO_DIR = "/home/alonso/cvat_docling/coco_output"

TASKS = [
    {"task_id": 7, "coco": "EAF-089-2025_coco.json"},
    {"task_id": 8, "coco": "EAF-477-2025_coco.json"},
]


def upload_annotations(task_id, coco_file):
    coco_path = os.path.join(COCO_DIR, coco_file)

    with open(coco_path) as f:
        coco_data = json.load(f)

    # Remove score field
    for ann in coco_data["annotations"]:
        ann.pop("score", None)

    print(f"  {len(coco_data['annotations'])} anotaciones, {len(coco_data['images'])} imágenes")

    # Create zip with correct structure
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "annotations/instances_default.json",
            json.dumps(coco_data),
        )

    # Upload with PUT
    with open(zip_path, "rb") as f:
        resp = requests.put(
            f"{CVAT_URL}/api/tasks/{task_id}/annotations",
            files={"annotation_file": ("annotations.zip", f, "application/zip")},
            data={"format": "COCO 1.0"},
            auth=AUTH,
        )

    os.unlink(zip_path)
    print(f"  Upload response: {resp.status_code}")
    if resp.status_code >= 400:
        print(f"  Error: {resp.text[:500]}")
        return False

    # Poll for import completion - check the rq_id
    rq_id = None
    try:
        rq_data = resp.json()
        rq_id = rq_data.get("rq_id")
        print(f"  rq_id: {rq_id}")
    except:
        pass

    if rq_id:
        while True:
            resp2 = requests.get(
                f"{CVAT_URL}/api/requests/{rq_id}",
                auth=AUTH,
            )
            if resp2.status_code == 200:
                status = resp2.json()
                state = status.get("status", "")
                print(f"  Import status: {state}")
                if state in ("finished", "failed"):
                    if state == "failed":
                        print(f"  Error: {status.get('message', '')}")
                    break
            time.sleep(2)
    else:
        # Fallback: just wait and check
        time.sleep(5)

    # Verify
    resp3 = requests.get(f"{CVAT_URL}/api/tasks/{task_id}/annotations", auth=AUTH)
    data = resp3.json()
    print(f"  Verificación: {len(data.get('shapes', []))} shapes en CVAT")
    return True


def main():
    for task_info in TASKS:
        print(f"\n--- Task {task_info['task_id']} ({task_info['coco']}) ---")
        upload_annotations(task_info["task_id"], task_info["coco"])


if __name__ == "__main__":
    main()
