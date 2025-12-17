from pathlib import Path
import random

def main():
    root = Path("data/kitti_yolo")
    for split in ("train","val"):
        imgs = list((root/"images"/split).glob("*.jpg"))
        lbls = list((root/"labels"/split).glob("*.txt"))
        print(split, "images:", len(imgs), "labels:", len(lbls))
    # Peek a random label
    val_lbls = list((root/"labels"/"val").glob("*.txt"))
    p = random.choice(val_lbls)
    print("Sample label:", p, "->", p.read_text().splitlines()[:5])

if __name__ == "__main__":
    main()