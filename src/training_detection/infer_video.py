import argparse, os, cv2
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--source", default="demo.mp4")
    ap.add_argument("--save", action="store_true", help="Save annotated video")
    ap.add_argument("--save_path", type=str, default=None, help="Full path to output .mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--display_width", type=int, default=960)
    ap.add_argument("--no_show", action="store_true")
    return ap.parse_args()

def ensure_parent_dir(path_str: str):
    p = Path(path_str).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

def main():
    args = parse_args()
    model = YOLO(args.weights)

    # Decide output path (if saving)
    out_path = None
    if args.save:
        if args.save_path:
            out_path = ensure_parent_dir(args.save_path)
        else:
            out_path = ensure_parent_dir("infer_out.mp4")

    results = model(source=args.source, stream=True, conf=0.25, verbose=False)
    writer = None
    win_name = "On-Road Detection"
    scale = 1.0

    if not args.no_show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    for r in results:
        frame = r.plot()

        if out_path and writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))

        if writer:
            writer.write(frame)

        if not args.no_show:
            h, w = frame.shape[:2]
            disp_w = max(320, args.display_width)
            disp_h = int(h * (disp_w / w))
            disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            cv2.imshow(win_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    if writer: writer.release()
    if not args.no_show: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()