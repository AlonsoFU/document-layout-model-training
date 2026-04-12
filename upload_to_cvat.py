"""
Crea un nuevo proyecto en CVAT con las predicciones de Docling Heron.
"""
import os
import json
import time
import zipfile
import tempfile
import requests

CVAT_URL = "http://localhost:8080"
AUTH = ("admin", "admin")
IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes"
COCO_DIR = "/home/alonso/cvat_docling/coco_output"

LABELS = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text",
    "Title", "Document Index", "Code", "Checkbox-selected",
    "Checkbox-unselected", "Form", "Key-value-region"
]

TASKS = [
    {"name": "EAF-089-2025", "coco": "EAF-089-2025_coco.json"},
    {"name": "EAF-477-2025", "coco": "EAF-477-2025_coco.json"},
]


def create_project():
    labels_payload = [{"name": name} for name in LABELS]
    resp = requests.post(
        f"{CVAT_URL}/api/projects",
        json={"name": "Docling Heron Predictions", "labels": labels_payload},
        auth=AUTH,
    )
    resp.raise_for_status()
    project = resp.json()
    print(f"Proyecto creado: id={project['id']}, name={project['name']}")
    return project["id"]


def create_task(project_id, task_name):
    resp = requests.post(
        f"{CVAT_URL}/api/tasks",
        json={"name": task_name, "project_id": project_id},
        auth=AUTH,
    )
    resp.raise_for_status()
    task = resp.json()
    print(f"  Tarea creada: id={task['id']}, name={task['name']}")
    return task["id"]


def upload_images(task_id, folder_name):
    folder_path = os.path.join(IMAGES_DIR, folder_name)
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    print(f"  Subiendo {len(image_files)} imágenes...")

    # Create zip of images for faster upload
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for fname in image_files:
            zf.write(os.path.join(folder_path, fname), fname)

    with open(zip_path, "rb") as f:
        resp = requests.post(
            f"{CVAT_URL}/api/tasks/{task_id}/data",
            files={"client_files[0]": ("images.zip", f, "application/zip")},
            data={
                "image_quality": 100,
                "use_zip_chunks": "true",
            },
            auth=AUTH,
        )
    resp.raise_for_status()
    os.unlink(zip_path)

    # Wait for data processing
    while True:
        resp = requests.get(f"{CVAT_URL}/api/tasks/{task_id}/status", auth=AUTH)
        status = resp.json()
        state = status.get("state", "")
        print(f"    Estado: {state}")
        if state in ("Finished", "Failed"):
            break
        time.sleep(3)

    if state == "Failed":
        print(f"    ERROR: {status.get('message', 'unknown')}")
        return False
    return True


def upload_annotations(task_id, coco_file):
    coco_path = os.path.join(COCO_DIR, coco_file)

    # CVAT expects COCO annotations in a zip with instances_default.json
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name

    # Read and modify the COCO file to remove 'score' field (not valid for COCO import)
    with open(coco_path) as f:
        coco_data = json.load(f)

    for ann in coco_data["annotations"]:
        ann.pop("score", None)

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "annotations/instances_default.json",
            json.dumps(coco_data),
        )

    print(f"  Subiendo anotaciones COCO ({len(coco_data['annotations'])} boxes)...")
    with open(zip_path, "rb") as f:
        resp = requests.put(
            f"{CVAT_URL}/api/tasks/{task_id}/annotations",
            files={"annotation_file": ("annotations.zip", f, "application/zip")},
            data={"format": "COCO 1.0"},
            auth=AUTH,
        )

    os.unlink(zip_path)

    if resp.status_code >= 400:
        print(f"    Error subiendo anotaciones: {resp.status_code}")
        print(f"    {resp.text[:500]}")
        return False

    # Poll for completion
    while True:
        resp2 = requests.get(
            f"{CVAT_URL}/api/tasks/{task_id}/annotations",
            params={"action": "import_status"},
            auth=AUTH,
        )
        if resp2.status_code == 200:
            break
        if resp2.status_code == 202:
            print("    Importando anotaciones...")
            time.sleep(2)
            continue
        break

    print("  Anotaciones cargadas!")
    return True


def main():
    project_id = create_project()

    for task_info in TASKS:
        print(f"\n--- {task_info['name']} ---")
        task_id = create_task(project_id, task_info["name"])

        if upload_images(task_id, task_info["name"]):
            upload_annotations(task_id, task_info["coco"])

    print(f"\n¡Listo! Abre CVAT: {CVAT_URL}/projects/{project_id}")


if __name__ == "__main__":
    main()
