import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np

from transformers import SegformerForSemanticSegmentation
from src.data_preprocessing.seg.seg_datasets import make_loaders_cs, NUM_CLASSES


# Parse arguements

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-5)

    ap.add_argument("--project", default="segformer_cityscapes")
    ap.add_argument("--name", default="b1_amp_poly")
    ap.add_argument("--model", default="nvidia/segformer-b1-finetuned-ade-512-512")
    ap.add_argument("--data_root", default="D:/Projects/On_Road_Perception/data/cityscapes")
    return ap.parse_args()


# Training and validation functions

def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    loss_sum = 0.0
    n = 0

    pbar = tqdm(loader, ncols=120, desc=f"Epoch {epoch} | AMP=True")

    for imgs, masks in pbar:
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=imgs, labels=masks)
            loss = outputs.loss

        # 🚨 critical safety check
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix({"loss": "skip"})
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        n += 1
        pbar.set_postfix({"loss": loss.item()})

    return loss_sum / max(n, 1)


def sliding_window_inference(model, image, window=512, stride=256):
    """
    image: (1,3,H,W) tensor
    returns: (1,C,H,W) logits
    """
    _, _, H, W = image.shape
    num_classes = model.config.num_labels

    device = image.device
    logits_full = torch.zeros((1, num_classes, H, W), device=device)
    count_map = torch.zeros((1, 1, H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y + window, H)
            x1 = min(x + window, W)
            y0 = max(y1 - window, 0)
            x0 = max(x1 - window, 0)

            patch = image[:, :, y0:y1, x0:x1]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(pixel_values=patch)
                logits = out.logits
                logits = F.interpolate(
                    logits,
                    size=patch.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            logits_full[:, :, y0:y1, x0:x1] += logits
            count_map[:, :, y0:y1, x0:x1] += 1

    return logits_full / count_map


@torch.no_grad()
def validate(model, loader, epoch):
    model.eval()
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    pbar = tqdm(loader, ncols=120, desc=f"Sliding Val {epoch}")

    for imgs, masks in pbar:
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        # imgs: (B,3,H,W) — we process one by one
        for i in range(imgs.size(0)):
            img = imgs[i:i+1]
            gt = masks[i].cpu().numpy()

            logits = sliding_window_inference(
                model,
                img,
                window=512,
                stride=256
            )

            pred = logits.argmax(1).squeeze(0).cpu().numpy()

            valid = (gt >= 0) & (gt < NUM_CLASSES)
            hist = np.bincount(
                NUM_CLASSES * gt[valid] + pred[valid],
                minlength=NUM_CLASSES ** 2
            )
            conf_mat += hist.reshape(NUM_CLASSES, NUM_CLASSES)

    intersection = np.diag(conf_mat)
    union = conf_mat.sum(1) + conf_mat.sum(0) - intersection
    miou = np.mean(intersection / np.maximum(union, 1))

    return 0.0, float(miou)


# Training and Validation loop

def main():
    args = parse_args()
    run_dir = f"runs/{args.project}/{args.name}"
    os.makedirs(run_dir, exist_ok=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).cuda()

    train_loader, val_loader = make_loaders_cs(
        os.path.join(args.data_root, "leftImg8bit/train"),
        os.path.join(args.data_root, "gtFine_trainIds/train"),
        os.path.join(args.data_root, "leftImg8bit/val"),
        os.path.join(args.data_root, "gtFine_trainIds/val"),
        img_size=args.imgsz,
        batch=args.batch,
        workers=args.workers,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    # Polynomial LR decay
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda e: (1 - e / args.epochs) ** 0.9
    )

    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        
        if epoch == 1 or epoch % 3 == 0:
            _, val_miou = validate(model, val_loader, epoch)
        else:
            val_miou = best_miou


        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val mIoU: {val_miou:.3f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), f"{run_dir}/best_cityscapes.pth")
            print(f"Saved best model (mIoU={best_miou:.3f})")

    print("Training completed.")


if __name__ == "__main__":
    main()
