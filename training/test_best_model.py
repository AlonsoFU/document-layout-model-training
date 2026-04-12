"""
Prueba el mejor modelo (B_lora) sobre EAF-477 y sube resultados a CVAT.
"""
import os
import json
import gc
import math
import torch
import torchvision
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import requests

MODEL_NAME = "docling-project/docling-layout-heron"
BEST_MODEL_PATH = "/home/alonso/cvat_docling/training/models/B_lora/best_model.pt"
IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes/EAF-477-2025"
THRESHOLD = 0.3
NMS_IOU = 0.5

CVAT_URL = "http://localhost:8080"
AUTH = ("admin", "admin")

LABELS = {
    0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
    4: "Page-footer", 5: "Page-header", 6: "Picture", 7: "Section-header",
    8: "Table", 9: "Text", 10: "Title", 11: "Document Index",
    12: "Code", 13: "Checkbox-selected", 14: "Checkbox-unselected",
    15: "Form", 16: "Key-value-region"
}


class LoRALinear(torch.nn.Module):
    def __init__(self, original, rank=8, alpha=16):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_features = original.in_features
        out_features = original.out_features
        self.lora_A = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora(model, rank=8, alpha=16):
    replaced = 0
    for name, module in model.named_modules():
        if "decoder" in name and isinstance(module, torch.nn.Linear):
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    if p.isdigit():
                        parent = parent[int(p)]
                    else:
                        parent = getattr(parent, p)
                attr = parts[-1]
                setattr(parent, attr, LoRALinear(module, rank, alpha))
                replaced += 1
    return model


def apply_nms(results, iou_threshold=0.5):
    if len(results["scores"]) == 0:
        return results
    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    keep_indices = []
    for label in labels.unique():
        mask = labels == label
        nms_keep = torchvision.ops.nms(boxes[mask], scores[mask], iou_threshold)
        keep_indices.extend(torch.where(mask)[0][nms_keep].tolist())
    keep = torch.tensor(sorted(keep_indices), dtype=torch.long, device=boxes.device)
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Load model with LoRA
    print("Cargando modelo con LoRA...")
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    model = apply_lora(model, rank=8, alpha=16)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device).eval()

    # Also load original model for comparison
    print("Cargando modelo original...")
    original_model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    original_model.to(device).eval()

    # Create CVAT project
    LABELS_LIST = list(set(LABELS.values()))
    resp = requests.post(
        f"{CVAT_URL}/api/projects",
        json={"name": "Comparación: Original vs LoRA fine-tuned", "labels": [{"name": n} for n in LABELS_LIST]},
        auth=AUTH,
    )
    project_id = resp.json()["id"]
    print(f"Proyecto CVAT: {project_id}")

    # Get label mapping
    resp = requests.get(f"{CVAT_URL}/api/labels?project_id={project_id}", auth=AUTH)
    all_labels = resp.json()["results"]
    if resp.json()["next"]:
        resp2 = requests.get(resp.json()["next"], auth=AUTH)
        all_labels += resp2.json()["results"]
    label_name_to_id = {l["name"]: l["id"] for l in all_labels}

    # Process images
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])
    print(f"Procesando {len(image_files)} imágenes...")

    shapes_lora = []
    shapes_original = []

    for img_idx, fname in enumerate(image_files):
        path = os.path.join(IMAGES_DIR, fname)
        image = Image.open(path).convert("RGB")
        w, h = image.size

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        target_sizes = torch.tensor([image.size[::-1]]).to(device)

        # LoRA model
        with torch.no_grad():
            outputs_lora = model(**inputs)
        results_lora = processor.post_process_object_detection(
            outputs_lora, target_sizes=target_sizes, threshold=THRESHOLD
        )[0]
        results_lora = apply_nms(results_lora, NMS_IOU)

        # Original model
        with torch.no_grad():
            outputs_orig = original_model(**inputs)
        results_orig = processor.post_process_object_detection(
            outputs_orig, target_sizes=target_sizes, threshold=THRESHOLD
        )[0]
        results_orig = apply_nms(results_orig, NMS_IOU)

        for results, shapes_list in [(results_lora, shapes_lora), (results_orig, shapes_original)]:
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                lid = label.item()
                cat_name = LABELS.get(lid)
                if cat_name is None:
                    continue
                cvat_label_id = label_name_to_id.get(cat_name)
                if cvat_label_id is None:
                    continue
                x1, y1, x2, y2 = box.tolist()
                shapes_list.append({
                    "type": "rectangle",
                    "frame": img_idx,
                    "label_id": cvat_label_id,
                    "points": [x1, y1, x2, y2],
                    "occluded": False,
                    "z_order": 0,
                    "rotation": 0.0,
                    "attributes": [],
                    "group": 0,
                    "source": "auto",
                })

        del outputs_lora, outputs_orig, inputs
        torch.cuda.empty_cache()
        if (img_idx + 1) % 50 == 0:
            print(f"  {img_idx + 1}/{len(image_files)}...")

    print(f"  LoRA: {len(shapes_lora)} detecciones | Original: {len(shapes_original)} detecciones")

    # Create tasks and upload
    import time, zipfile, tempfile

    for task_name, shapes in [
        ("EAF-477 - Original Heron", shapes_original),
        ("EAF-477 - LoRA Fine-tuned", shapes_lora),
    ]:
        resp = requests.post(
            f"{CVAT_URL}/api/tasks",
            json={"name": task_name, "project_id": project_id},
            auth=AUTH,
        )
        task_id = resp.json()["id"]

        # Upload images
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = tmp.name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for fname in image_files:
                zf.write(os.path.join(IMAGES_DIR, fname), fname)
        with open(zip_path, "rb") as f:
            requests.post(
                f"{CVAT_URL}/api/tasks/{task_id}/data",
                files={"client_files[0]": ("images.zip", f, "application/zip")},
                data={"image_quality": 100, "use_zip_chunks": "true"},
                auth=AUTH,
            )
        os.unlink(zip_path)
        while True:
            resp = requests.get(f"{CVAT_URL}/api/tasks/{task_id}/status", auth=AUTH)
            if resp.json().get("state") in ("Finished", "Failed"):
                break
            time.sleep(3)

        # Upload annotations
        payload = {"version": 0, "tags": [], "shapes": shapes, "tracks": []}
        resp = requests.put(f"{CVAT_URL}/api/tasks/{task_id}/annotations/", json=payload, auth=AUTH)
        print(f"  {task_name}: task {task_id}, {len(shapes)} shapes, upload={resp.status_code}")

    print(f"\nComparación en: http://localhost:8080/projects/{project_id}")


if __name__ == "__main__":
    main()
