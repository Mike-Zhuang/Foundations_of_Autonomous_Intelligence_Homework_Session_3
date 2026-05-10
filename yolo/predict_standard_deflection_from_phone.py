from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bridge_ai.bridge_task_models import loadModelBundle, predictStandardDeflectionFromPhone


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="任务三：用手机视觉窗口特征预测/校正到激光标准挠度")
    parser.add_argument("--model-path", required=True, help="train_bridge_task_models.py 输出的 model_bundle.pth/json")
    parser.add_argument("--windows-csv", required=True, help="需要校正的 weight_windows_*.csv")
    parser.add_argument("--output-csv", default="", help="输出 CSV；为空则只打印")
    parser.add_argument(
        "--model-choice",
        choices=["auto", "mlp", "monotonic", "ridge"],
        default="auto",
        help="候选模型选择。auto 使用训练时推荐；mlp 强制神经网络；monotonic 强制单调函数逼近器",
    )
    return parser.parse_args()


def main() -> int:
    args = parseArgs()
    bundle = loadModelBundle(Path(args.model_path))
    if bundle.get("bundleType") != "bridge_task_models":
        raise RuntimeError("该模型不是三任务 bridge model bundle。")

    with Path(args.windows_csv).open("r", encoding="utf-8", newline="") as fileObj:
        rows = list(csv.DictReader(fileObj))

    outputRows: list[dict] = []
    print("\n================ 任务三：手机挠度 -> 标准挠度 ================")
    for row in rows:
        result = predictStandardDeflectionFromPhone(bundle, row, modelPreference=args.model_choice)
        out = {
            **row,
            "predictedStandardDeflectionMm": result["predictedStandardDeflectionMm"],
            "predictedPhoneMinusStandardMm": result["predictedPhoneMinusStandardMm"],
            "correctionModelType": result["modelType"],
        }
        outputRows.append(out)
        print(
            f"{row.get('sampleId', '')}: phone={result['phoneDeflectionMm']:.3f}mm "
            f"predStandard={result['predictedStandardDeflectionMm']:.3f}mm "
            f"phone-standard={result['predictedPhoneMinusStandardMm']:.3f}mm",
            flush=True,
        )
    print("=========================================================\n")

    if args.output_csv:
        outputPath = Path(args.output_csv)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        with outputPath.open("w", encoding="utf-8", newline="") as fileObj:
            writer = csv.DictWriter(fileObj, fieldnames=list(outputRows[0].keys()))
            writer.writeheader()
            writer.writerows(outputRows)
        print(f"校正结果已保存: {outputPath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
