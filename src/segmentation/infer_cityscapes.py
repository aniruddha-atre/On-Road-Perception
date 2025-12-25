import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from transformers import SegformerForSemanticSegmentation
from src.data_preprocessing.seg.seg_datasets import NUM_CLASSES


# Cityscapes color palette (trainIds)

PALETTE = np.array([
    [128, 64,128],   # road
    [244, 35,232],   # sidewalk
    [ 70, 70, 70],   # building
    [102,102,156],   # wall
    [190,153,153],   # fence
    [153,153,153],   # pole
    [250,170, 30],   # traffic light
    [220,220,  0],   # traffic sign
    [107,142, 35],   # vegetation
    [152,251,152],   # terrain
    [ 70,130,180],   # sky
    [220, 20, 60],   # person
    [255,  0,  0],   # rider
    [  0,  0,142],   # car
    [  0,  0, 70],   # truck
    [  0, 60,100],   # bus
    [  0, 80,100],   # train
    [  0,  0,230],   # motorcycle
    [119, 11, 32],   # bicycle
], dtype=np.uint8)


# Helper Functions

def normalize(img):
    img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (img - mean) / std


@torch.no_grad()
def sliding_window_inference(model, image, window=512, stride=256):
    """
    image: (1,3,H,W)
    returns: (H,W) prediction
    """
    _, _, H, W = image.shape
    device = image.device

    logits_full = torch.zeros((1, NUM_CLASSES, H, W), device=device)
    count_map  = torch.zeros((1, 1, H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y + window, H)
            x1 = min(x + window, W)
            y0 = max(y1 - window, 0)
            x0 = max(x1 - window, 0)

            patch = image[:, :, y0:y1, x0:x1]

            with torch.autocast("cuda", dtype=torch.float16):
                out = model(pixel_values=patch)
                logits = F.interpolate(
                    out.logits,
                    size=patch.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            logits_full[:, :, y0:y1, x0:x1] += logits
            count_map[:, :, y0:y1, x0:x1] += 1

    logits_full /= count_map
    pred = logits_full.argmax(1).squeeze(0)
    return pred.cpu().numpy()


def colorize(mask):
    return PALETTE[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\Projects\On_Road_Perception\runs\segformer_cityscapes\b1_amp_poly\best_cityscapes.pth")
    ap.add_argument("--img_dir", default=r"D:\Projects\On_Road_Perception\data\cityscapes\leftImg8bit\test\bonn")
    ap.add_argument("--out_dir", default=r"D:\Projects\On_Road_Perception\runs\infer_cityscapes\segformer_b1")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).cuda().eval()

    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))

    img_paths = list(Path(args.img_dir).rglob("*_leftImg8bit.png"))

    for img_path in img_paths:
        img = np.array(Image.open(img_path).convert("RGB"))
        img_t = normalize(img).unsqueeze(0).cuda()

        pred = sliding_window_inference(model, img_t)

        color = colorize(pred)
        overlay = (0.6 * img + 0.4 * color).astype(np.uint8)

        base = img_path.stem.replace("_leftImg8bit", "")
        cv2.imwrite(os.path.join(args.out_dir, f"{base}_mask.png"), color[:,:,::-1])
        cv2.imwrite(os.path.join(args.out_dir, f"{base}_overlay.png"), overlay[:,:,::-1])

        print(f"Saved {base}")

    print("Inference done.")


if __name__ == "__main__":
    main()
