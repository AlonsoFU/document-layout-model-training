"""
Round 4: Corregir desbalance Table vs Picture.
Base: Estrategia E (LoRA r=32 + cosine warmup 5ep + aug)

L: E + oversampling 3x pictures
M: E + oversampling 5x pictures
N: E + class-weighted loss (3x weight para pictures)
O: E + oversampling 3x + weighted loss
"""
import os
import gc
import json
import math
import random
from collections import defaultdict, Counter

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageEnhance, ImageFilter
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

MODEL_NAME = "docling-project/docling-layout-heron"
COCO_FILE = "/home/alonso/cvat_docling/training/EAF-477-2025_ground_truth.json"
IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes/EAF-477-2025"
OUTPUT_BASE = "/home/alonso/cvat_docling/training/models"
SPLIT_FILE = "/home/alonso/cvat_docling/training/models/data_split.json"
MAX_EPOCHS = 50
PATIENCE = 10
GRAD_ACCUM = 4
GRAD_CLIP = 0.1

# Picture category_id in COCO (1-based)
PICTURE_CAT_ID = 7
# Minority classes to oversample
MINORITY_CATS = {7}  # Picture

random.seed(42)
torch.manual_seed(42)


class LoRALinear(torch.nn.Module):
    def __init__(self, original, rank=32, alpha=64, dropout=0.05):
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


def apply_lora(model, rank=32, alpha=64):
    replaced = 0
    for name, module in model.named_modules():
        if "decoder" in name and isinstance(module, torch.nn.Linear):
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(module, rank, alpha))
                replaced += 1
    print(f"  LoRA rank={rank} en {replaced} capas")
    return model


class DocumentAugmenter:
    def __call__(self, image, boxes):
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            image = image.rotate(random.uniform(-3, 3), fillcolor=(255, 255, 255))
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return image, boxes


class DocLayoutDataset(Dataset):
    def __init__(self, coco_data, image_ids, images_dir, processor, augmenter=None,
                 oversample_factor=1, minority_cats=None):
        self.images_dir = images_dir
        self.processor = processor
        self.augmenter = augmenter
        self.id_to_img = {img["id"]: img for img in coco_data["images"]}
        self.img_to_anns = defaultdict(list)
        for ann in coco_data["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

        # Build index with oversampling
        self.image_ids = []
        minority_cats = minority_cats or set()

        for img_id in image_ids:
            anns = self.img_to_anns[img_id]
            has_minority = any(a["category_id"] in minority_cats for a in anns)
            if has_minority and oversample_factor > 1:
                self.image_ids.extend([img_id] * oversample_factor)
            else:
                self.image_ids.append(img_id)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.id_to_img[img_id]
        image = Image.open(os.path.join(self.images_dir, img_info["file_name"])).convert("RGB")
        anns = self.img_to_anns[img_id]
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            iw, ih = img_info["width"], img_info["height"]
            boxes.append([(x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih])
            labels.append(ann["category_id"] - 1)  # 0-based for model
        if self.augmenter and boxes:
            image, boxes = self.augmenter(image, boxes)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "class_labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
        }
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), target


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def box_iou(boxes1, boxes2):
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    ix1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    return inter / (a1[:, None] + a2 - inter + 1e-6)


def compute_ap(preds, tgts, cls, iou_t):
    scores, tps = [], []
    ngt = 0
    for p, t in zip(preds, tgts):
        gm = t["labels"] == cls; gb = t["boxes"][gm]; ngt += len(gb)
        pm = p["labels"] == cls; pb = p["boxes"][pm]; ps = p["scores"][pm]
        if len(pb) == 0: continue
        o = ps.argsort(descending=True); pb = pb[o]; ps = ps[o]
        matched = set()
        for i in range(len(pb)):
            scores.append(ps[i].item())
            if len(gb) == 0: tps.append(0); continue
            ious = box_iou(pb[i].unsqueeze(0), gb)[0]
            bi, bx = ious.max(0)
            if bi.item() >= iou_t and bx.item() not in matched:
                tps.append(1); matched.add(bx.item())
            else: tps.append(0)
    if ngt == 0: return 0.0
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ts = [tps[i] for i in idx]; tp = fp = 0; precs = []; recs = []
    for t in ts:
        tp += t; fp += 1 - t; precs.append(tp / (tp + fp)); recs.append(tp / ngt)
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        pr = [p for p, r in zip(precs, recs) if r >= t]
        ap += max(pr) / 11.0 if pr else 0.0
    return ap


@torch.no_grad()
def evaluate(model, processor, val_ds, device):
    model.eval()
    preds, tgts = [], []
    for idx in range(len(val_ds)):
        pv, tgt = val_ds[idx]
        pv = pv.unsqueeze(0).to(device)
        info = val_ds.id_to_img[val_ds.image_ids[idx]]
        sz = torch.tensor([[info["height"], info["width"]]]).to(device)
        with torch.amp.autocast("cuda", enabled=False):
            out = model(pixel_values=pv.float())
        res = processor.post_process_object_detection(out, target_sizes=sz, threshold=0.3)[0]
        if len(res["scores"]) > 0:
            ki = []
            for l in res["labels"].unique():
                m = res["labels"] == l
                k = torchvision.ops.nms(res["boxes"][m], res["scores"][m], 0.5)
                ki.extend(torch.where(m)[0][k].tolist())
            keep = torch.tensor(sorted(ki), dtype=torch.long, device=device)
            preds.append({"boxes": res["boxes"][keep].cpu(), "scores": res["scores"][keep].cpu(), "labels": res["labels"][keep].cpu()})
        else:
            preds.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
        gb = []
        for b in tgt["boxes"]:
            cx, cy, w, h = b.tolist(); iw, ih = info["width"], info["height"]
            gb.append([(cx - w/2)*iw, (cy - h/2)*ih, (cx + w/2)*iw, (cy + h/2)*ih])
        tgts.append({"boxes": torch.tensor(gb) if gb else torch.zeros((0, 4)), "labels": tgt["class_labels"]})

    # mAP@[0.5:0.95]
    all_cls = set()
    for t in tgts: all_cls.update(t["labels"].tolist())
    if not all_cls: return 0.0, {}

    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    per_threshold = {}
    for iou_t in iou_thresholds:
        aps = [compute_ap(preds, tgts, c, iou_t) for c in sorted(all_cls)]
        per_threshold[iou_t] = sum(aps) / len(aps)

    coco_map = sum(per_threshold.values()) / len(per_threshold)

    # Per-class AP@0.5 for monitoring
    per_class = {}
    LABELS = {0: "caption", 1: "footnote", 2: "formula", 3: "list_item", 4: "page_footer",
              5: "page_header", 6: "picture", 7: "section_header", 8: "table", 9: "text",
              10: "title", 11: "document_index"}
    for c in sorted(all_cls):
        ap = compute_ap(preds, tgts, c, 0.5)
        per_class[LABELS.get(c, f"cls_{c}")] = ap

    return coco_map, per_class


def get_lr(epoch, base_lr, warmup=5):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    remaining = MAX_EPOCHS - warmup
    progress = (epoch - warmup) / remaining
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def apply_class_weights_to_loss(model, class_weights):
    """Monkey-patch the model's loss to use class weights."""
    # Store weights for use in forward
    model._class_weights = class_weights


STRATEGIES = {
    "L_oversample_3x": {
        "oversample": 3,
        "class_weights": None,
        "description": "E + oversampling 3x pictures",
    },
    "M_oversample_5x": {
        "oversample": 5,
        "class_weights": None,
        "description": "E + oversampling 5x pictures",
    },
    "N_weighted_loss": {
        "oversample": 1,
        "class_weights": {6: 3.0},  # picture (0-based) gets 3x weight
        "description": "E + class-weighted focal loss (3x pictures)",
    },
    "O_oversample_weighted": {
        "oversample": 3,
        "class_weights": {6: 3.0},
        "description": "E + oversampling 3x + weighted loss",
    },
}


def train_strategy(name, config, coco_data, train_ids, val_ids, processor):
    print(f"\n{'=' * 60}")
    print(f"ESTRATEGIA: {name}")
    print(f"  {config['description']}")
    print(f"{'=' * 60}", flush=True)

    device = torch.device("cuda")
    output_dir = os.path.join(OUTPUT_BASE, name)
    os.makedirs(output_dir, exist_ok=True)

    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    model.config.num_denoising = 0
    model.config.eos_coefficient = 0.0001
    for n, param in model.named_parameters():
        if "backbone" in n or "encoder" in n:
            param.requires_grad = False
    apply_lora(model, rank=32, alpha=64)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {trainable:,} trainable")

    # Class weights for focal loss
    class_weights = config.get("class_weights")
    if class_weights:
        # Modify the model's class weight
        num_classes = model.config.num_labels
        weight_tensor = torch.ones(num_classes, device=device)
        for cls_id, w in class_weights.items():
            weight_tensor[cls_id] = w
        model.config._class_weights = weight_tensor
        print(f"  Class weights: {class_weights}")

    model.to(device)

    augmenter = DocumentAugmenter()
    oversample = config.get("oversample", 1)

    train_ds = DocLayoutDataset(
        coco_data, train_ids, IMAGES_DIR, processor, augmenter=augmenter,
        oversample_factor=oversample, minority_cats=MINORITY_CATS
    )
    val_ds = DocLayoutDataset(coco_data, val_ids, IMAGES_DIR, processor)

    print(f"  Train samples: {len(train_ds)} (oversample={oversample}x)")

    loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)

    best_map = -1
    patience_counter = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        lr = get_lr(epoch, 1e-4)
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        epoch_loss = 0
        opt.zero_grad()

        for step, (pv, targets) in enumerate(loader):
            pv = pv.to(device)
            labs = [{"boxes": t["boxes"].to(device), "class_labels": t["class_labels"].to(device)} for t in targets]
            outputs = model(pixel_values=pv, labels=labs)

            loss = outputs.loss

            # Apply class weights manually to the VFL loss component
            if class_weights and hasattr(outputs, "loss_dict"):
                # Scale loss by average class weight of targets in this batch
                batch_classes = torch.cat([t["class_labels"] for t in labs])
                if len(batch_classes) > 0:
                    weights = torch.tensor([class_weights.get(c.item(), 1.0) for c in batch_classes], device=device)
                    loss = loss * weights.mean()

            loss = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM

        avg_loss = epoch_loss / len(loader)
        val_map, per_class = evaluate(model, processor, val_ds, device)

        history.append({"epoch": epoch, "loss": avg_loss, "mAP_coco": val_map, "per_class": per_class, "lr": lr})

        pic_ap = per_class.get("picture", 0)
        tab_ap = per_class.get("table", 0)
        print(f"  Epoch {epoch+1}/{MAX_EPOCHS} | loss={avg_loss:.4f} | mAP@[.5:.95]={val_map:.4f} | picture={pic_ap:.3f} table={tab_ap:.3f} | lr={lr:.2e}", flush=True)

        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Get best per-class from history
    best_epoch = max(history, key=lambda h: h["mAP_coco"])
    print(f"  RESULTADO: best mAP@[.5:.95] = {best_map:.4f}")
    print(f"  Per-class AP@0.5 (best epoch): {best_epoch['per_class']}")

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return best_map


def main():
    print("Cargando datos...")
    with open(COCO_FILE) as f:
        coco_data = json.load(f)
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    print(f"  Train: {len(split['train'])} | Val: {len(split['val'])}")

    # Show class distribution
    cats = {c["id"]: c["name"] for c in coco_data["categories"]}
    train_anns = [a for a in coco_data["annotations"] if a["image_id"] in split["train"]]
    counts = Counter(cats[a["category_id"]] for a in train_anns)
    print(f"  Class distribution: {dict(counts.most_common())}")

    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)

    results = {"E_lora32_cosine_aug (ref)": 0.8117}

    for name, config in STRATEGIES.items():
        try:
            best_map = train_strategy(name, config, coco_data, split["train"], split["val"], processor)
            results[name] = best_map
        except Exception as e:
            print(f"\n  ERROR en {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"FAILED: {e}"
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("RESULTADOS ROUND 4 - mAP@[0.5:0.95]")
    print(f"{'=' * 60}")
    print(f"{'Estrategia':<35} {'mAP@[.5:.95]':>13}")
    print("-" * 50)
    for name, score in sorted(results.items(), key=lambda x: x[1] if isinstance(x[1], float) else -1, reverse=True):
        if isinstance(score, float):
            print(f"{name:<35} {score:>13.4f}")
        else:
            print(f"{name:<35} {'FAILED':>13}")

    with open(os.path.join(OUTPUT_BASE, "results_round4.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado en: {OUTPUT_BASE}/results_round4.json")


if __name__ == "__main__":
    main()
