"""
Reevalúa todos los modelos con mAP@[0.5:0.95] (estándar COCO/DocLayNet).
"""
import os
import gc
import json
import math
from collections import defaultdict

import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

MODEL_NAME = "docling-project/docling-layout-heron"
COCO_FILE = "/home/alonso/cvat_docling/training/EAF-477-2025_ground_truth.json"
IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes/EAF-477-2025"
OUTPUT_BASE = "/home/alonso/cvat_docling/training/models"
SPLIT_FILE = "/home/alonso/cvat_docling/training/models/data_split.json"

IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


class LoRALinear(torch.nn.Module):
    def __init__(self, original, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = torch.nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, original.out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_lora(model, rank=8, alpha=16, dropout=0.0):
    for name, module in model.named_modules():
        if "decoder" in name and isinstance(module, torch.nn.Linear):
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))
    return model


class DocLayoutDataset(Dataset):
    def __init__(self, coco_data, image_ids, images_dir, processor):
        self.images_dir = images_dir
        self.processor = processor
        self.image_ids = image_ids
        self.id_to_img = {img["id"]: img for img in coco_data["images"]}
        self.img_to_anns = defaultdict(list)
        for ann in coco_data["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.id_to_img[img_id]
        path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(path).convert("RGB")
        anns = self.img_to_anns[img_id]
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            img_w, img_h = img_info["width"], img_info["height"]
            boxes.append([(x + w/2)/img_w, (y + h/2)/img_h, w/img_w, h/img_h])
            labels.append(ann["category_id"] - 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "class_labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
        }
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), target


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    ix1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    return inter / (area1[:, None] + area2 - inter + 1e-6)


def compute_ap_single_class(predictions, targets, cls, iou_threshold):
    all_scores, all_tp = [], []
    n_gt = 0
    for pred, tgt in zip(predictions, targets):
        gt_mask = tgt["labels"] == cls
        gt_boxes = tgt["boxes"][gt_mask]
        n_gt += len(gt_boxes)
        pred_mask = pred["labels"] == cls
        pred_boxes = pred["boxes"][pred_mask]
        pred_scores_cls = pred["scores"][pred_mask]
        if len(pred_boxes) == 0:
            continue
        order = pred_scores_cls.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_scores_cls = pred_scores_cls[order]
        matched = set()
        for i in range(len(pred_boxes)):
            all_scores.append(pred_scores_cls[i].item())
            if len(gt_boxes) == 0:
                all_tp.append(0)
                continue
            ious = box_iou(pred_boxes[i].unsqueeze(0), gt_boxes)[0]
            best_iou, best_idx = ious.max(0)
            if best_iou.item() >= iou_threshold and best_idx.item() not in matched:
                all_tp.append(1)
                matched.add(best_idx.item())
            else:
                all_tp.append(0)
    if n_gt == 0:
        return 0.0
    indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
    tp_sorted = [all_tp[i] for i in indices]
    tp_sum, fp_sum = 0, 0
    precisions, recalls = [], []
    for tp in tp_sorted:
        tp_sum += tp
        fp_sum += 1 - tp
        precisions.append(tp_sum / (tp_sum + fp_sum))
        recalls.append(tp_sum / n_gt)
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p_at_r) / 11.0 if p_at_r else 0.0
    return ap


@torch.no_grad()
def evaluate_full(model, processor, val_dataset, device):
    """Evaluate with mAP@[0.5:0.95]"""
    model.eval()
    all_predictions = []
    all_targets = []

    for idx in range(len(val_dataset)):
        pixel_values, target = val_dataset[idx]
        pixel_values = pixel_values.unsqueeze(0).to(device)
        image_info = val_dataset.id_to_img[val_dataset.image_ids[idx]]
        orig_size = torch.tensor([[image_info["height"], image_info["width"]]]).to(device)

        with torch.amp.autocast("cuda", enabled=False):
            outputs = model(pixel_values=pixel_values.float())
        results = processor.post_process_object_detection(
            outputs, target_sizes=orig_size, threshold=0.3
        )[0]

        if len(results["scores"]) > 0:
            keep_indices = []
            for label in results["labels"].unique():
                mask = results["labels"] == label
                nms_keep = torchvision.ops.nms(results["boxes"][mask], results["scores"][mask], 0.5)
                keep_indices.extend(torch.where(mask)[0][nms_keep].tolist())
            keep = torch.tensor(sorted(keep_indices), dtype=torch.long, device=device)
            pred_boxes = results["boxes"][keep].cpu()
            pred_scores = results["scores"][keep].cpu()
            pred_labels = results["labels"][keep].cpu()
        else:
            pred_boxes = torch.zeros((0, 4))
            pred_scores = torch.zeros(0)
            pred_labels = torch.zeros(0, dtype=torch.long)

        all_predictions.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})

        gt_boxes_abs = []
        for box in target["boxes"]:
            cx, cy, w, h = box.tolist()
            img_w, img_h = image_info["width"], image_info["height"]
            gt_boxes_abs.append([(cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h])

        all_targets.append({
            "boxes": torch.tensor(gt_boxes_abs) if gt_boxes_abs else torch.zeros((0, 4)),
            "labels": target["class_labels"],
        })

    # Compute mAP at each IoU threshold
    all_classes = set()
    for t in all_targets:
        all_classes.update(t["labels"].tolist())

    maps_per_threshold = {}
    for iou_t in IOU_THRESHOLDS:
        aps = []
        for cls in sorted(all_classes):
            ap = compute_ap_single_class(all_predictions, all_targets, cls, iou_t)
            aps.append(ap)
        maps_per_threshold[f"mAP@{iou_t:.2f}"] = sum(aps) / len(aps) if aps else 0.0

    # mAP@[0.5:0.95]
    coco_map = sum(maps_per_threshold.values()) / len(maps_per_threshold)

    return coco_map, maps_per_threshold


def main():
    with open(COCO_FILE) as f:
        coco_data = json.load(f)
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    val_ids = split["val"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    val_ds = DocLayoutDataset(coco_data, val_ids, IMAGES_DIR, processor)

    # Models to evaluate
    models_to_eval = {
        "Original (baseline)": {"path": None, "lora_rank": 0},
        "A_frozen_decoder": {"path": "A_frozen_decoder/best_model.pt", "lora_rank": 0},
        "B_lora_r8": {"path": "B_lora/best_model.pt", "lora_rank": 8, "lora_alpha": 16},
        "E_lora32_cosine_aug": {"path": "E_lora32_cosine_aug/best_model.pt", "lora_rank": 32, "lora_alpha": 64},
        "F_lora32_ema_aug": {"path": "F_lora32_ema_aug/best_model.pt", "lora_rank": 32, "lora_alpha": 64},
    }

    # Check if G exists
    g_path = os.path.join(OUTPUT_BASE, "G_lora32_all/best_model.pt")
    if os.path.exists(g_path):
        models_to_eval["G_lora32_all"] = {"path": "G_lora32_all/best_model.pt", "lora_rank": 32, "lora_alpha": 64}

    # Round 3 models
    extra = {
        "H_lora64_cosine_aug": {"path": "H_lora64_cosine_aug/best_model.pt", "lora_rank": 64, "lora_alpha": 128},
        "I_lora32_lr5e5": {"path": "I_lora32_lr5e5/best_model.pt", "lora_rank": 32, "lora_alpha": 64},
        "J_lora32_all_linears": {"path": "J_lora32_all_linears/best_model.pt", "lora_rank": 32, "lora_alpha": 64, "lora_target": "all"},
        "K_lora32_warmup10": {"path": "K_lora32_warmup10/best_model.pt", "lora_rank": 32, "lora_alpha": 64},
    }
    for name, cfg in extra.items():
        if os.path.exists(os.path.join(OUTPUT_BASE, cfg["path"])):
            models_to_eval[name] = cfg

    results = {}

    for name, config in models_to_eval.items():
        print(f"\nEvaluando: {name}")

        model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)

        if config["lora_rank"] > 0:
            if config.get("lora_target") == "all":
                # LoRA on all decoder linears (except class/bbox heads)
                for name_m, module in model.named_modules():
                    if "decoder" in name_m and isinstance(module, torch.nn.Linear):
                        if "class_embed" in name_m or "bbox_embed" in name_m:
                            continue
                        parts = name_m.split(".")
                        parent = model
                        for p in parts[:-1]:
                            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                        setattr(parent, parts[-1], LoRALinear(module, config["lora_rank"], config.get("lora_alpha", config["lora_rank"] * 2)))
            else:
                apply_lora(model, config["lora_rank"], config.get("lora_alpha", config["lora_rank"] * 2))

        if config["path"]:
            full_path = os.path.join(OUTPUT_BASE, config["path"])
            if not os.path.exists(full_path):
                print(f"  SKIP: {full_path} no existe")
                continue
            model.load_state_dict(torch.load(full_path, map_location=device))

        model.to(device).eval()

        coco_map, per_threshold = evaluate_full(model, processor, val_ds, device)

        results[name] = {
            "mAP@[0.5:0.95]": round(coco_map, 4),
            **{k: round(v, 4) for k, v in per_threshold.items()},
        }

        print(f"  mAP@[0.5:0.95] = {coco_map:.4f}")
        print(f"  mAP@0.50 = {per_threshold['mAP@0.50']:.4f}")
        print(f"  mAP@0.75 = {per_threshold['mAP@0.75']:.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("RESULTADOS REALES - mAP@[0.5:0.95] (estándar COCO/DocLayNet)")
    print(f"{'='*70}")
    print(f"{'Modelo':<30} {'mAP@[.5:.95]':>13} {'mAP@.50':>9} {'mAP@.75':>9}")
    print("-" * 63)
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['mAP@[0.5:0.95]']:>13.4f} {metrics['mAP@0.50']:>9.4f} {metrics['mAP@0.75']:>9.4f}")

    with open(os.path.join(OUTPUT_BASE, "results_real_map.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nGuardado en: {OUTPUT_BASE}/results_real_map.json")


if __name__ == "__main__":
    main()
