"""
Round 3: Más experimentos basados en lo aprendido.
E (LoRA r=32 + cosine + aug) fue el mejor con 0.9562.

H: LoRA r=64 + cosine + aug (más capacidad)
I: LoRA r=32 + cosine + aug + LR 5e-5 (menos oscilación)
J: LoRA r=32 en TODAS las linears del decoder + cosine + aug
K: LoRA r=32 + cosine + aug + warmup 10 epochs
"""
import os
import gc
import json
import math
import random
from collections import defaultdict

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
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


def apply_lora_attn(model, rank=32, alpha=64):
    """LoRA solo en attention (q, k, v, out_proj)."""
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
    print(f"  LoRA rank={rank} en {replaced} capas (attention only)")
    return model


def apply_lora_all(model, rank=32, alpha=64):
    """LoRA en TODAS las capas lineales del decoder (attention + FFN)."""
    replaced = 0
    for name, module in model.named_modules():
        if "decoder" in name and isinstance(module, torch.nn.Linear):
            # Skip the final class/bbox heads
            if "class_embed" in name or "bbox_embed" in name:
                continue
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, rank, alpha))
            replaced += 1
    print(f"  LoRA rank={rank} en {replaced} capas (ALL decoder linears)")
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
    def __init__(self, coco_data, image_ids, images_dir, processor, augmenter=None):
        self.images_dir = images_dir
        self.processor = processor
        self.image_ids = image_ids
        self.augmenter = augmenter
        self.id_to_img = {img["id"]: img for img in coco_data["images"]}
        self.img_to_anns = defaultdict(list)
        for ann in coco_data["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

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
            labels.append(ann["category_id"] - 1)
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
        gm = t["labels"] == cls
        gb = t["boxes"][gm]
        ngt += len(gb)
        pm = p["labels"] == cls
        pb = p["boxes"][pm]
        ps = p["scores"][pm]
        if len(pb) == 0:
            continue
        o = ps.argsort(descending=True)
        pb = pb[o]
        ps = ps[o]
        matched = set()
        for i in range(len(pb)):
            scores.append(ps[i].item())
            if len(gb) == 0:
                tps.append(0)
                continue
            ious = box_iou(pb[i].unsqueeze(0), gb)[0]
            bi, bx = ious.max(0)
            if bi.item() >= iou_t and bx.item() not in matched:
                tps.append(1)
                matched.add(bx.item())
            else:
                tps.append(0)
    if ngt == 0:
        return 0.0
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ts = [tps[i] for i in idx]
    tp = fp = 0
    precs, recs = [], []
    for t in ts:
        tp += t
        fp += 1 - t
        precs.append(tp / (tp + fp))
        recs.append(tp / ngt)
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
            cx, cy, w, h = b.tolist()
            iw, ih = info["width"], info["height"]
            gb.append([(cx - w / 2) * iw, (cy - h / 2) * ih, (cx + w / 2) * iw, (cy + h / 2) * ih])
        tgts.append({"boxes": torch.tensor(gb) if gb else torch.zeros((0, 4)), "labels": tgt["class_labels"]})
    cls = set()
    for t in tgts:
        cls.update(t["labels"].tolist())
    if not cls:
        return 0.0
    aps = [compute_ap(preds, tgts, c, 0.5) for c in sorted(cls)]
    return sum(aps) / len(aps)


def get_lr(epoch, base_lr, warmup_epochs=5):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    remaining = MAX_EPOCHS - warmup_epochs
    progress = (epoch - warmup_epochs) / remaining
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


STRATEGIES = {
    "H_lora64_cosine_aug": {
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_target": "attn",
        "lr": 1e-4,
        "warmup": 5,
        "description": "LoRA r=64 + cosine warmup + aug",
    },
    "I_lora32_lr5e5": {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_target": "attn",
        "lr": 5e-5,
        "warmup": 5,
        "description": "LoRA r=32 + cosine + aug + LR 5e-5 (más estable)",
    },
    "J_lora32_all_linears": {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_target": "all",
        "lr": 1e-4,
        "warmup": 5,
        "description": "LoRA r=32 en TODAS las linears del decoder + cosine + aug",
    },
    "K_lora32_warmup10": {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_target": "attn",
        "lr": 1e-4,
        "warmup": 10,
        "description": "LoRA r=32 + cosine + aug + warmup 10 epochs",
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

    if config["lora_target"] == "all":
        apply_lora_all(model, config["lora_rank"], config["lora_alpha"])
    else:
        apply_lora_attn(model, config["lora_rank"], config["lora_alpha"])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {trainable:,} trainable", flush=True)
    model.to(device)

    augmenter = DocumentAugmenter()
    train_ds = DocLayoutDataset(coco_data, train_ids, IMAGES_DIR, processor, augmenter=augmenter)
    val_ds = DocLayoutDataset(coco_data, val_ids, IMAGES_DIR, processor)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["lr"], weight_decay=1e-4)

    best_map = -1
    patience_counter = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        lr = get_lr(epoch, config["lr"], config["warmup"])
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        epoch_loss = 0
        opt.zero_grad()

        for step, (pv, targets) in enumerate(loader):
            pv = pv.to(device)
            labs = [{"boxes": t["boxes"].to(device), "class_labels": t["class_labels"].to(device)} for t in targets]
            outputs = model(pixel_values=pv, labels=labs)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM

        avg_loss = epoch_loss / len(loader)
        val_map = evaluate(model, processor, val_ds, device)
        history.append({"epoch": epoch, "loss": avg_loss, "mAP": val_map, "lr": lr})
        print(f"  Epoch {epoch + 1}/{MAX_EPOCHS} | loss={avg_loss:.4f} | mAP@0.5={val_map:.4f} | lr={lr:.2e}", flush=True)

        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"  RESULTADO: best mAP@0.5 = {best_map:.4f}")
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

    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    results = {"E_lora32_cosine_aug (ref)": 0.9562}

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
    print("RESULTADOS ROUND 3")
    print(f"{'=' * 60}")
    print(f"{'Estrategia':<35} {'mAP@0.5':>10}")
    print("-" * 47)
    for name, score in sorted(results.items(), key=lambda x: x[1] if isinstance(x[1], float) else -1, reverse=True):
        if isinstance(score, float):
            print(f"{name:<35} {score:>10.4f}")
        else:
            print(f"{name:<35} {str(score)[:40]:>40}")

    with open(os.path.join(OUTPUT_BASE, "results_round3.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado en: {OUTPUT_BASE}/results_round3.json")


if __name__ == "__main__":
    main()
