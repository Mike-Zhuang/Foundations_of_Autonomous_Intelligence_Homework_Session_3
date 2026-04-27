from __future__ import annotations

import argparse


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练跨中目标 YOLO 模型")
    parser.add_argument("--data", required=True, help="data.yaml 路径")
    parser.add_argument("--base-model", default="yolo11n.pt", help="基础模型")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--project", default="yolo/results/train")
    parser.add_argument("--name", default="midpoint")
    return parser.parse_args()


def main() -> int:
    args = parseArgs()

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
