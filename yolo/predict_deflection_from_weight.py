from __future__ import annotations

import argparse
from pathlib import Path

from bridge_ai.bridge_task_models import loadModelBundle, predictStandardDeflectionFromWeight


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="任务二：给定重量，预测激光标准挠度")
    parser.add_argument("--model-path", required=True, help="train_bridge_task_models.py 输出的 model_bundle.pth/json")
    parser.add_argument("--weight-g", type=float, required=True, help="输入重量，单位 g")
    return parser.parse_args()


def main() -> int:
    args = parseArgs()
    bundle = loadModelBundle(Path(args.model_path))
    if bundle.get("bundleType") != "bridge_task_models":
        raise RuntimeError("该模型不是三任务 bridge model bundle。")
    result = predictStandardDeflectionFromWeight(bundle, args.weight_g)
    print("\n================ 任务二：重量 -> 标准挠度 ================")
    print(f"Input weight: {result['inputWeightG']:.3f} g")
    print(f"Predicted standard deflection: {result['predictedStandardDeflectionMm']:.3f} mm")
    print("Model: deflection_from_weight")
    print(f"Recommended candidate: {result['modelType']}")
    print("======================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
