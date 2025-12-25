import argparse
import os
import cv2
import time
import torch
import numpy as np
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation
from src.data_preprocessing.seg.seg_datasets import NUM_CLASSES


# Cityscapes color palette (trainIds)

PALETTE = np.array([
    [128, 64,128], [244, 35,232], [70, 70, 70], [102,102,156],
    [190,153,153], [153,153,153], [250,170, 30], [220,220,  0],
    [107,142, 35], [152,251,152], [70,130,180], [220, 20, 60],
    [255,  0,  0], [0,  0,142], [0,  0, 70], [0, 60,100],
    [0, 80,100], [0,  0,230], [119, 11, 32]
], dtype=np.uint8)


# Helper functions

def normalize(img_rgb):
    img = torch.from_numpy(img_rgb.transpose(2,0,1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (img - mean) / std


@torch.no_grad()
def sliding_window_inference(model, image, window=512, stride=256):
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
    return logits_full.argmax(1).squeeze(0).cpu().numpy()


def colorize(mask):
    return PALETTE[mask]


# Parse arguements

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"runs/segformer_cityscapes/b1_amp_poly/best_cityscapes.pth")
    ap.add_argument("--source", default=r"assets/dashcam_vid.mp4")

    ap.add_argument("--save_mask_path",
                    default="runs/segmentation_inference/semantic_mask.mp4")
    ap.add_argument("--save_overlay_path",
                    default="runs/segmentation_inference/overlay.mp4")

    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max_width", type=int, default=0)

    return ap.parse_args()


def main():
    args = parse_args()

    # ---- model ----
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).cuda().eval()

    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"), strict=True)

    # ---- video ----
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {args.source}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = in_fps if args.fps == 0 else args.fps

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    if args.max_width and frame.shape[1] > args.max_width:
        nw = args.max_width
        nh = int(frame.shape[0] * nw / frame.shape[1])
        frame = cv2.resize(frame, (nw, nh))

    h, w = frame.shape[:2]

    os.makedirs(os.path.dirname(args.save_mask_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_overlay_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mask_writer = cv2.VideoWriter(args.save_mask_path, fourcc, out_fps, (w, h))
    overlay_writer = cv2.VideoWriter(args.save_overlay_path, fourcc, out_fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.max_width and frame.shape[1] > args.max_width:
            nw = args.max_width
            nh = int(frame.shape[0] * nw / frame.shape[1])
            frame = cv2.resize(frame, (nw, nh))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = normalize(rgb).unsqueeze(0).cuda()

        pred = sliding_window_inference(
            model, img_t,
            window=args.window,
            stride=args.stride
        )

        # ---- semantic mask video ----
        mask_rgb = colorize(pred)
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        mask_writer.write(mask_bgr)

        # ---- overlay video ----
        overlay = (0.6 * rgb + 0.4 * mask_rgb).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_writer.write(overlay_bgr)

        frame_idx += 1
        if frame_idx % 30 == 0:
            fps = frame_idx / max(time.time() - t0, 1e-6)
            print(f"[{frame_idx:06d}] avg FPS: {fps:.2f}")

    cap.release()
    mask_writer.release()
    overlay_writer.release()

    print("Saved semantic mask video:", args.save_mask_path)
    print("Saved overlay video:", args.save_overlay_path)
    print("Done.")


if __name__ == "__main__":
    main()
