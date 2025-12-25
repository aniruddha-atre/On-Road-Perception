import os
from pathlib import Path
import argparse
import wandb
from ultralytics import YOLO
from ultralytics.utils import SETTINGS


# Parse arguements

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/kitti.yaml", help="Ultralytics data YAML")
    ap.add_argument("--cfg", default="configs/train.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="Pretrained weights (e.g. yolov8s.pt)")
    ap.add_argument("--project", default="on-road-object-detection", help="Ultralytics 'project' (also W&B project by default)")
    ap.add_argument("--name", default="y8s_kitti_base", help="Run name")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    # ap.add_argument("--resume", type=bool)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cache", default="disk")   # 'disk' avoids RAM spikes
    return ap.parse_args()


def main():
    args = parse_args()

    SETTINGS.update({"wandb": True})

    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB_PROJECT", args.project)

    run = wandb.init(
        project=args.project,
        name=args.name,
        config={
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            # "optimizer": args.optimizer,
            "workers": args.workers,
            "cache": args.cache,
            "data_yaml": args.data,
        },
        settings=wandb.Settings(start_method="thread"),
    )

    # Training

    model = YOLO(args.model)

    overrides = {
        "data": args.data,
        "project": args.project,
        "name": args.name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        # "optimizer": args.optimizer,
        "workers": args.workers,
        "cache": args.cache,
        "verbose": False,
        "device": 0,
        # "resume": args.resume
    }
    results = model.train(cfg=args.cfg, **overrides)

    run_dir = Path("runs") / "detect" / args.name / "weights"
    best_path = run_dir / "best.pt"
    if best_path.exists():
        art = wandb.Artifact(name=f"{args.name}-model", type="model", metadata=dict(run_config=run.config))
        art.add_file(str(best_path))
        data_yaml = Path(args.data)
        if data_yaml.exists():
            art.add_file(str(data_yaml))
        cfg_train = Path("configs/train.yaml")
        if cfg_train.exists():
            art.add_file(str(cfg_train))
        wandb.log_artifact(art)
    else:
        wandb.alert(
            title="best.pt not found",
            text=f"Expected at {best_path}. Check run folder naming.",
            level=wandb.AlertLevel.WARN,
        )

    if hasattr(results, "results_dict"):
        wandb.log(results.results_dict)

    run.finish()


if __name__ == "__main__":
    main()
