# src/data_preprocessing/kitti_to_yolo.py
import os, glob
from pathlib import Path
from tqdm import tqdm
import cv2
import yaml

KITTI_TO_YOLO = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
}

def parse_kitti_label_line(line):
    parts = line.strip().split()
    if len(parts) < 15:
        return None
    cls = parts[0]
    if cls == "DontCare" or cls not in KITTI_TO_YOLO:
        return None
    l, t, r, b = map(float, parts[4:8])
    return cls, l, t, r, b

def kitti_to_yolo_bbox(l, t, r, b, img_w, img_h):
    x_c = (l + r) / 2.0 / img_w
    y_c = (t + b) / 2.0 / img_h
    w = (r - l) / img_w
    h = (b - t) / img_h
    return x_c, y_c, w, h

def main():
    # --- YOUR ACTUAL KITTI ROOTS ---
    img_train_dir = Path("data/kitti_raw/data_object_image_2/training/image_2")
    lbl_train_dir = Path("data/kitti_raw/data_object_label_2/training/label_2")
    # test images exist here but have NO labels; we ignore them:
    # img_test_dir  = Path("data/kitti_raw/data_object_image_2/testing/image_2")

    out_root = Path("data/kitti_yolo")
    out_img = out_root / "images" / "all"
    out_lbl = out_root / "labels" / "all"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(glob.glob(str(img_train_dir / "*.*")))
    kept, skipped = 0, 0

    for ip in tqdm(image_paths, desc="Converting (train only)"):
        ip = Path(ip)
        base = ip.stem
        label_path = lbl_train_dir / f"{base}.txt"

        if not label_path.exists():
            skipped += 1
            continue  # Safety: shouldn’t happen in training, but guard anyway

        img = cv2.imread(str(ip))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        yolo_lines = []
        with open(label_path, "r") as f:
            for line in f:
                parsed = parse_kitti_label_line(line)
                if parsed is None:
                    continue
                cls, l, t, r, b = parsed
                x_c, y_c, bw, bh = kitti_to_yolo_bbox(l, t, r, b, w, h)
                if bw <= 0 or bh <= 0:
                    continue
                cid = KITTI_TO_YOLO[cls]
                yolo_lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        # Keep original extension to avoid recompression if you prefer; jpg is fine too.
        ext = ip.suffix.lower()
        out_img_path = out_img / f"{base}{ext}"
        cv2.imwrite(str(out_img_path), img)

        out_lbl_path = out_lbl / f"{base}.txt"
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))
        kept += 1

    meta = {
        "source": "KITTI 2D detection (train only)",
        "classes": KITTI_TO_YOLO,
        "counts": {"images_kept": kept, "skipped": skipped}
    }
    with open(out_root / "meta.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    print(f"Done. Kept: {kept}, Skipped: {skipped}")

if __name__ == "__main__":
    main()