from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

from bridge_ai.bridge_task_models import DEFAULT_WINDOW_CSVS, loadWindowRows, saveModelBundle, trainAllTasks


def loadRich() -> dict[str, Any]:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
        from rich.table import Table

        return {
            "available": True,
            "Console": Console,
            "Panel": Panel,
            "Progress": Progress,
            "TextColumn": TextColumn,
            "BarColumn": BarColumn,
            "TimeElapsedColumn": TimeElapsedColumn,
            "TimeRemainingColumn": TimeRemainingColumn,
            "Table": Table,
        }
    except ImportError:
        return {"available": False}


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练三任务桥梁模型：重量反推、重量到标准挠度、手机到标准挠度")
    parser.add_argument(
        "--windows-csv",
        action="append",
        default=[],
        help="weight_windows_*.csv，可重复传入；默认使用最近两组较准数据",
    )
    parser.add_argument("--output-dir", default="", help="输出目录，默认 yolo/results/bridge_models_<时间>")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于让小 MLP 训练尽量可复现")
    parser.add_argument("--plain-output", action="store_true", help="关闭 Rich 彩色表格和进度条，使用普通文本输出")
    return parser.parse_args()


def setTrainingSeed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def printDatasetSummary(rows: list[dict], paths: list[Path], richTools: dict[str, Any] | None = None) -> None:
    weights = [float(row["weightG"]) for row in rows]
    phone = [float(row["deflectionMeanMm"]) for row in rows]
    laser = [float(row["standardDeflectionMm"]) for row in rows]
    if richTools and richTools.get("available"):
        console = richTools["console"]
        table = richTools["Table"](title="三任务桥梁模型训练")
        table.add_column("项目", style="cyan", no_wrap=True)
        table.add_column("值", style="white")
        table.add_row("输入数据", "\n".join(str(path) for path in paths))
        table.add_row("窗口样本数", str(len(rows)))
        table.add_row("重量范围", f"{min(weights):.1f}g ~ {max(weights):.1f}g")
        table.add_row("手机挠度范围", f"{min(phone):.3f}mm ~ {max(phone):.3f}mm")
        table.add_row("激光标准挠度范围", f"{min(laser):.3f}mm ~ {max(laser):.3f}mm")
        table.add_row("任务 A", "手机视觉挠度 -> 重量")
        table.add_row("任务 B", "重量 -> 激光标准挠度")
        table.add_row("任务 C", "手机视觉挠度 -> 激光标准挠度")
        console.print(table)
        console.print(
            richTools["Panel"](
                "样本只有 27 个，所以训练很快是正常现象；关键看 LOOCV 指标，不看训练耗时长短。",
                title="小数据提醒",
                border_style="yellow",
            )
        )
        return

    print("\n================ 三任务桥梁模型训练 ================", flush=True)
    print("输入数据:", flush=True)
    for path in paths:
        print(f"- {path}", flush=True)
    print(f"窗口样本数: {len(rows)}", flush=True)
    print(f"重量范围: {min(weights):.1f}g ~ {max(weights):.1f}g", flush=True)
    print(f"手机挠度范围: {min(phone):.3f}mm ~ {max(phone):.3f}mm", flush=True)
    print(f"激光标准挠度范围: {min(laser):.3f}mm ~ {max(laser):.3f}mm", flush=True)
    print("任务: A 手机挠度->重量；B 重量->激光标准挠度；C 手机挠度->激光标准挠度", flush=True)
    print("==================================================\n", flush=True)


def printTaskMetrics(bundle: dict, richTools: dict[str, Any] | None = None) -> None:
    if richTools and richTools.get("available"):
        console = richTools["console"]
        table = richTools["Table"](title="训练结果总览")
        table.add_column("任务", style="cyan", no_wrap=True)
        table.add_column("目标", style="white")
        table.add_column("候选模型", style="magenta")
        table.add_column("推荐", style="green")
        table.add_column("MAE", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("R2", justify="right")
        table.add_column("MaxErr", justify="right")
        for taskName, task in bundle["tasks"].items():
            recommended = task["recommended"]
            for candidate in task["candidates"]:
                metrics = candidate["metrics"]
                unit = "G" if task["target"] == "weightG" else "Mm"
                table.add_row(
                    taskName,
                    task["target"],
                    candidate["modelType"],
                    "YES" if candidate["modelType"] == recommended else "",
                    f"{metrics.get(f'mae{unit}', float('nan')):.3f}",
                    f"{metrics.get(f'rmse{unit}', float('nan')):.3f}",
                    f"{metrics.get('r2', float('nan')):.3f}",
                    f"{metrics.get(f'maxAbsError{unit}', float('nan')):.3f}",
                )
        console.print(table)

        for taskName, task in bundle["tasks"].items():
            recommended = next(item for item in task["candidates"] if item["modelType"] == task["recommended"])
            unit = "G" if task["target"] == "weightG" else "Mm"
            detailTable = richTools["Table"](title=f"{taskName} 推荐模型逐样本误差：{recommended['modelType']}")
            detailTable.add_column("sampleId", style="cyan")
            detailTable.add_column(f"true{unit}", justify="right")
            detailTable.add_column(f"pred{unit}", justify="right")
            detailTable.add_column(f"absError{unit}", justify="right")
            for item in recommended["perSample"]:
                detailTable.add_row(
                    str(item["sampleId"]),
                    f"{item[f'true{unit}']:.3f}",
                    f"{item[f'pred{unit}']:.3f}",
                    f"{item[f'absError{unit}']:.3f}",
                )
            console.print(detailTable)
        return

    print("\n================ 训练结果 ================", flush=True)
    for taskName, task in bundle["tasks"].items():
        print(f"\n[{taskName}] target={task['target']} recommended={task['recommended']}", flush=True)
        for candidate in task["candidates"]:
            metrics = candidate["metrics"]
            metricText = " ".join(
                f"{key}={value:.3f}" for key, value in metrics.items() if isinstance(value, (int, float))
            )
            print(f"- {candidate['modelType']}: {metricText}", flush=True)
        recommended = next(item for item in task["candidates"] if item["modelType"] == task["recommended"])
        print("  推荐模型逐样本误差:", flush=True)
        for item in recommended["perSample"]:
            print(f"  - {item}", flush=True)
    print("========================================\n", flush=True)


def makeProgressCallback(richTools: dict[str, Any], epochs: int) -> tuple[Any, Any]:
    progress = richTools["Progress"](
        richTools["TextColumn"]("[bold cyan]{task.description}"),
        richTools["BarColumn"](),
        richTools["TextColumn"]("{task.completed}/{task.total}"),
        richTools["TextColumn"]("loss={task.fields[loss]}"),
        richTools["TimeElapsedColumn"](),
        richTools["TimeRemainingColumn"](),
        console=richTools["console"],
    )
    taskIds: dict[str, int] = {}

    def callback(taskName: str, epoch: int, totalEpochs: int, loss: float) -> None:
        if taskName not in taskIds:
            taskIds[taskName] = progress.add_task(taskName, total=totalEpochs, loss="nan")
        progress.update(taskIds[taskName], completed=epoch, loss=f"{loss:.4f}")

    return progress, callback


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parseArgs()
    setTrainingSeed(args.seed)
    richTools = loadRich()
    if args.plain_output:
        richTools = {"available": False}
    elif richTools.get("available"):
        richTools["console"] = richTools["Console"]()
    else:
        print("未检测到 rich，已使用普通文本输出。可运行：conda run -n fai python -m pip install rich", flush=True)

    inputPaths = [Path(item) for item in (args.windows_csv or DEFAULT_WINDOW_CSVS)]
    rows = loadWindowRows(inputPaths)
    if len(rows) < 3:
        raise RuntimeError("三任务训练至少建议 3 个窗口样本。")
    printDatasetSummary(rows, inputPaths, richTools)

    if richTools.get("available"):
        progress, callback = makeProgressCallback(richTools, args.epochs)
        with progress:
            bundle = trainAllTasks(
                rows,
                epochs=args.epochs,
                lr=args.lr,
                weightDecay=args.weight_decay,
                progressCallback=callback,
            )
    else:
        bundle = trainAllTasks(rows, epochs=args.epochs, lr=args.lr, weightDecay=args.weight_decay)
    printTaskMetrics(bundle, richTools)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputDir = Path(args.output_dir) if args.output_dir else Path(f"yolo/results/bridge_models_{timestamp}")
    paths = saveModelBundle(bundle, outputDir)
    if richTools.get("available"):
        table = richTools["Table"](title="模型包已保存")
        table.add_column("文件", style="cyan")
        table.add_column("路径", style="white")
        for key, value in paths.items():
            table.add_row(key, value)
        richTools["console"].print(table)
    else:
        print("模型包已保存:", flush=True)
        for key, value in paths.items():
            print(f"- {key}: {value}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
