from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from bridge_ai.weight_model import loadRows, saveBestModel, trainAndSelect


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练砝码重量模型：小 MLP + 稳健拟合自动择优")
    parser.add_argument("--windows-csv", required=True, help="collect_weight_data.py 生成的 weight_windows_*.csv")
    parser.add_argument("--output-dir", default="", help="模型输出目录，默认自动命名")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def printDatasetSummary(rows: list[dict]) -> None:
    weights = sorted({float(row["weightG"]) for row in rows})
    print("\n================ 重量模型训练 ================", flush=True)
    print(f"窗口样本数: {len(rows)}", flush=True)
    print(f"重量档位(g): {weights}", flush=True)
    print("每个样本:", flush=True)
    for index, row in enumerate(rows, start=1):
        print(
            f"- {index:02d} id={row.get('sampleId', '')} weight={float(row['weightG']):g}g "
            f"deflMean={float(row['deflectionMeanMm']):.3f}mm "
            f"std={float(row['deflectionStdMm']):.3f}mm "
            f"validRate={float(row['validRate']):.3f}",
            flush=True,
        )
    print("============================================\n", flush=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parseArgs()
    rows = loadRows(Path(args.windows_csv))
    if len(rows) < 2:
        raise RuntimeError("至少需要 2 个窗口样本才能训练/验证重量模型。")
    printDatasetSummary(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputDir = Path(args.output_dir) if args.output_dir else Path(f"yolo/results/weight_model_{timestamp}")
    best = trainAndSelect(rows, epochs=args.epochs, lr=args.lr, weightDecay=args.weight_decay)
    paths = saveBestModel(best, outputDir)

    print("\n================ 训练完成 ================", flush=True)
    print(f"最佳模型: {best['modelType']}", flush=True)
    print(
        f"LOOCV MAE={best['metrics']['maeG']:.3f}g "
        f"RMSE={best['metrics']['rmseG']:.3f}g R2={best['metrics']['r2']:.3f}",
        flush=True,
    )
    print("逐样本留一验证误差:", flush=True)
    for item in best["perSample"]:
        print(
            f"- {item['sampleId']}: true={item['trueG']:.3f}g "
            f"pred={item['predG']:.3f}g absErr={item['absErrorG']:.3f}g",
            flush=True,
        )
    print(f"模型已保存: {paths['modelPath']}", flush=True)
    print(f"指标已保存: {paths['metricsPath']}", flush=True)
    print("========================================\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
