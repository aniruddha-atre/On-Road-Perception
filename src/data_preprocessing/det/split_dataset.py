from pathlib import Path
import shutil, random
from tqdm import tqdm

random.seed(42)

def main():
    root = Path("data/kitti_yolo")
    src_img = root / "images" / "all"
    src_lbl = root / "labels" / "all"

    for p in [root / "images" / "train", root / "images" / "val",
              root / "labels" / "train", root / "labels" / "val"]:
        p.mkdir(parents=True, exist_ok=True)

    # Only consider images that have a matching label file
    images = []
    for ip in sorted(src_img.glob("*.*")):
        base = ip.stem
        lp = src_lbl / f"{base}.txt"
        if lp.exists():
            images.append(ip)

    n = len(images)
    split_idx = int(0.8 * n)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    def copy_set(img_list, subset):
        for ip in tqdm(img_list, desc=f"Copying {subset}"):
            base = ip.stem
            lp = src_lbl / f"{base}.txt"
            shutil.copy(ip, root / "images" / subset / ip.name)
            shutil.copy(lp, root / "labels" / subset / lp.name)

    copy_set(train_imgs, "train")
    copy_set(val_imgs, "val")
    print(f"Train images: {len(train_imgs)}  |  Val images: {len(val_imgs)}")

if __name__ == "__main__":
    main()