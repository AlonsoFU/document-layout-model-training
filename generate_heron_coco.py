"""
Genera anotaciones COCO desde Docling Heron para ambos EAFs
y crea un nuevo proyecto en CVAT con las predicciones pre-cargadas.
"""
import os
import json
import gc
import torch
import torchvision
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

MODEL_NAME = "docling-project/docling-layout-heron"
IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes"
OUTPUT_DIR = "/home/alonso/cvat_docling/coco_output"
THRESHOLD = 0.3
NMS_IOU = 0.5

LABELS = {
    0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
    4: "Page-footer", 5: "Page-header", 6: "Picture", 7: "Section-header",
    8: "Table", 9: "Text", 10: "Title", 11: "Document Index",
    12: "Code", 13: "Checkbox-selected", 14: "Checkbox-unselected",
    15: "Form", 16: "Key-value-region"
}


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

    print("Cargando modelo Docling Heron...")
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    model.to(device).eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    folders = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, f))
    ])

    for folder_name in folders:
        folder_path = os.path.join(IMAGES_DIR, folder_name)
        output_file = os.path.join(OUTPUT_DIR, f"{folder_name}_coco.json")

        print(f"\nProcesando {folder_name}...")
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
        print(f"  {len(image_files)} páginas")

        # Build COCO structure
        categories = [{"id": k + 1, "name": v, "supercategory": ""} for k, v in LABELS.items()]
        images = []
        annotations = []
        ann_id = 1

        for img_idx, fname in enumerate(image_files):
            path = os.path.join(folder_path, fname)
            image = Image.open(path).convert("RGB")
            w, h = image.size

            images.append({
                "id": img_idx + 1,
                "width": w,
                "height": h,
                "file_name": fname,
            })

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=THRESHOLD
            )[0]
            results = apply_nms(results, NMS_IOU)

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                bw, bh = x2 - x1, y2 - y1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_idx + 1,
                    "category_id": label.item() + 1,  # COCO is 1-indexed
                    "segmentation": [],
                    "area": bw * bh,
                    "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    "iscrowd": 0,
                    "attributes": {"occluded": False, "rotation": 0.0},
                    "score": round(score.item(), 4),
                })
                ann_id += 1

            del outputs, inputs, image
            torch.cuda.empty_cache()

            if (img_idx + 1) % 50 == 0:
                print(f"    Procesadas {img_idx + 1}/{len(image_files)}...")
                gc.collect()

        coco_data = {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {
                "contributor": "Docling Heron",
                "date_created": "",
                "description": f"Predicciones de Docling Heron para {folder_name}",
                "url": "", "version": "", "year": ""
            },
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

        with open(output_file, "w") as f:
            json.dump(coco_data, f, indent=2)

        print(f"  Guardado: {output_file}")
        print(f"  {len(images)} imágenes, {len(annotations)} anotaciones")
        gc.collect()

    print("\nListo! Archivos COCO generados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
