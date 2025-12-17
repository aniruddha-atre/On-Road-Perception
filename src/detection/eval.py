import argparse
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/y8s_kitti_base/weights/best.pt")
    ap.add_argument("--data", default="configs/kitti.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    return ap.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.val(data=args.data, imgsz=args.imgsz, plots=True, save_json=True)

if __name__ == "__main__":
    main()