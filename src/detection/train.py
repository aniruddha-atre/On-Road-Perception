import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import SegformerForSemanticSegmentation
from src.data_preprocessing.seg.seg_datasets import make_loaders_cs, NUM_CLASSES


# -----------------------------------------------------------------------------
# FAST CONFUSION MATRIX + MIOU
# -----------------------------------------------------------------------------
def update_confusion_matrix(conf_mat, preds, gts, num_classes=19):
    mask = (gts >= 0) & (gts < num_classes)
    gt = gts[mask].astype(np.int64)
    pd = preds[mask].astype(np.int64)
    idx = num_classes * gt + pd
    binc = np.bincount(idx, minlength=num_classes*num_classes)
    return conf_mat + binc.reshape(num_classes, num_classes)


def compute_miou_from_confmat(conf_mat):
    inter = np.diag(conf_mat)
    gt_area = conf_mat.sum(1)
    pred_area = conf_mat.sum(0)
    union = gt_area + pred_area - inter
    iou = inter / np.maximum(union, 1)
    return float(np.mean(iou))


# -----------------------------------------------------------------------------
# TRAIN LOOP
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()

    total_loss = 0.0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    pbar = tqdm(loader, ncols=120, desc=f"Epoch {epoch}")

    for imgs, masks in pbar:
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        # Forward
        with torch.autocast("cuda"):
            outputs = model(pixel_values=imgs, labels=masks)
            loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())

        # Update confusion matrix for train mIoU
        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        gts = masks.cpu().numpy()

        for i in range(preds.shape[0]):
            conf_mat = update_confusion_matrix(conf_mat, preds[i], gts[i], NUM_CLASSES)

        # tqdm live printing
        pbar.set_postfix({"loss": float(loss.item())})

    epoch_loss = total_loss / len(loader)
    epoch_miou = compute_miou_from_confmat(conf_mat)

    return epoch_loss, epoch_miou


# -----------------------------------------------------------------------------
# VALIDATION LOOP
# -----------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, epoch):
    model.eval()

    total_loss = 0.0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    pbar = tqdm(loader, ncols=120, desc=f"Validate {epoch}")

    for imgs, masks in pbar:
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        with torch.autocast("cuda"):
            outputs = model(pixel_values=imgs, labels=masks)
            loss = outputs.loss

        total_loss += float(loss.item())

        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        gts = masks.cpu().numpy()

        for i in range(preds.shape[0]):
            conf_mat = update_confusion_matrix(conf_mat, preds[i], gts[i], NUM_CLASSES)

        pbar.set_postfix({"loss": float(loss.item())})

    val_loss = total_loss / len(loader)
    val_miou = compute_miou_from_confmat(conf_mat)

    return val_loss, val_miou


# -----------------------------------------------------------------------------
# ARGPARSE
# -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=6e-5)

    ap.add_argument("--project", default="segformer_cityscapes")
    ap.add_argument("--name", default="b1_run")

    return ap.parse_args()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    OUT_DIR = os.path.join("runs", args.project, args.name)
    os.makedirs(OUT_DIR, exist_ok=True)

    CITY_IMG_TRAIN = r"D:/Projects/On_Road_Perception/data/cityscapes/leftImg8bit/train"
    CITY_MASK_TRAIN = r"D:/Projects/On_Road_Perception/data/cityscapes/gtFine_trainIds/train"

    CITY_IMG_VAL = r"D:/Projects/On_Road_Perception/data/cityscapes/leftImg8bit/val"
    CITY_MASK_VAL = r"D:/Projects/On_Road_Perception/data/cityscapes/gtFine_trainIds/val"

    # DATA
    train_loader, val_loader = make_loaders_cs(
        img_root=CITY_IMG_TRAIN,
        mask_root=CITY_MASK_TRAIN,
        val_img_root=CITY_IMG_VAL,
        val_mask_root=CITY_MASK_VAL,
        img_size=args.imgsz,
        batch=args.batch,
        workers=args.workers,
    )

    # MODEL
    MODEL_NAME = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    best_miou = 0.0

    # TRAINING
    for epoch in range(1, args.epochs + 1):

        train_loss, train_miou = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        val_loss, val_miou = validate(model, val_loader, epoch)

        # Clean summary line (no spam)
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.3f} | "
            f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.3f}"
        )

        # Save best
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_miou.pth"))
            print(f"✔ Saved best model (mIoU={best_miou:.3f})")

    print("Training complete!")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
