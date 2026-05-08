from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from bridge_ai.calibration import CameraCalibration, loadCalibration, runCharucoCalibration, saveCalibration, undistortFrame
from bridge_ai.config import loadLayout
from bridge_ai.deflection import DeflectionEstimator
from bridge_ai.detection import MidpointTargetDetector
from bridge_ai.geometry import ArucoStaticSolver, drawOverlay
from bridge_ai.io_utils import openVideoSource


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集挠度-重量数据并训练双任务模型（回归+分类）")
    parser.add_argument("--source", default="0", help="视频源")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 模型路径")
    parser.add_argument("--layout", default="yolo/artifacts/static_marker_layout.json", help="静态标记布局 json")
    parser.add_argument("--target-class", default="midpoint_marker", help="跨中目标类别名")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--baseline-frames", type=int, default=60)
    parser.add_argument("--calibration-mode", choices=["off", "use", "recalibrate"], default="use")
    parser.add_argument("--calibration-file", default="yolo/artifacts/camera_calibration.npz")
    parser.add_argument("--overlay-level", choices=["minimal", "balanced", "debug"], default="debug")
    parser.add_argument("--dataset-out", default="", help="输出 CSV 路径，默认自动命名")
    parser.add_argument("--auto-train", choices=["true", "false"], default="true")
    parser.add_argument("--min-valid-frames", type=int, default=120, help="每轮采集最少有效帧数")
    parser.add_argument("--capture-seconds", type=float, default=8.0, help="每轮最长采集时长（秒）")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-rmse", type=float, default=4.0)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.45)
    parser.add_argument("--min-used-points", type=int, default=12)
    parser.add_argument(
        "--deflection-scale",
        type=float,
        default=1.0,
        help="现场比例修正系数。例：尺子 100mm、显示 108.5mm，则填 100/108.5=0.922",
    )
    parser.add_argument("--output-dir", default="yolo/results/weight_model")
    return parser.parse_args()


@dataclass
class SampleWindow:
    weightG: float
    deflectionsMm: list[float]
    confidences: list[float]
    rmsePx: list[float]
    inlierRatio: list[float]
    usedPoints: list[int]


def localizeStatus(status: str) -> str:
    if status == "calibrating-baseline":
        return "calibrating-baseline"
    if status.startswith("tracking:"):
        key = status.split(":", 1)[1]
        mapping = {
            "yolo": "tracking-yolo",
            "fallback-aruco": "tracking-aruco",
        }
        return mapping.get(key, key)
    if status.startswith("missing:"):
        key = status.split(":", 1)[1]
        mapping = {
            "no-static-markers": "missing-static-markers",
            "low-homography-quality": "missing-low-homography-quality",
            "yolo-no-target": "missing-yolo-target",
            "fallback-no-target": "missing-fallback-target",
        }
        return mapping.get(key, key)
    return status


def localizeDetectionSource(status: str) -> str:
    mapping = {
        "yolo": "YOLO",
        "fallback-aruco": "Aruco-fallback",
        "yolo-no-target": "YOLO-none",
        "fallback-no-target": "Fallback-none",
    }
    return mapping.get(status, status)


def printCollectorGuide(args: argparse.Namespace) -> None:
    print("\n================ 采集与训练启动说明 ================")
    print(f"视频源: {args.source}")
    print(f"去畸变模式: {args.calibration_mode}")
    print(f"自动训练: {args.auto_train}")
    print(f"最少有效帧: {args.min_valid_frames}")
    print(f"单轮最长时长: {args.capture_seconds}s")
    print("采集时请按:")
    print("- s: 开始采集当前重量")
    print("- e: 提前结束当前重量")
    print("- q: 中止整个程序")
    print("命令行输入重量时:")
    print("- 输入数字表示克(g)，例如 200")
    print("- 输入 done 结束采集并进入训练/导出")
    print("===============================================\n")


def resolveCalibration(args: argparse.Namespace, capture: cv2.VideoCapture) -> CameraCalibration | None:
    calibrationPath = Path(args.calibration_file)
    if args.calibration_mode == "off":
        return None
    if args.calibration_mode == "recalibrate":
        calibration = runCharucoCalibration(capture, targetSamples=30)
        saveCalibration(calibration, calibrationPath)
        print(f"ChArUco 标定完成，RMS={calibration.rms:.4f}，已保存: {calibrationPath}")
        return calibration
    if calibrationPath.exists():
        calibration = loadCalibration(calibrationPath)
        print(f"加载标定参数成功: {calibrationPath} (RMS={calibration.rms:.4f})")
        return calibration
    print(f"未找到标定文件 {calibrationPath}，自动降级为不去畸变。")
    return None


def computeFeatureRow(sample: SampleWindow) -> dict:
    defl = np.asarray(sample.deflectionsMm, dtype=np.float64)
    conf = np.asarray(sample.confidences, dtype=np.float64)
    rmse = np.asarray(sample.rmsePx, dtype=np.float64) if sample.rmsePx else np.asarray([np.nan], dtype=np.float64)
    inlier = np.asarray(sample.inlierRatio, dtype=np.float64) if sample.inlierRatio else np.asarray([np.nan], dtype=np.float64)
    used = np.asarray(sample.usedPoints, dtype=np.float64) if sample.usedPoints else np.asarray([np.nan], dtype=np.float64)

    t = np.arange(defl.size, dtype=np.float64)
    slope = 0.0
    if defl.size >= 2:
        tCentered = t - np.mean(t)
        slope = float(np.sum(tCentered * (defl - np.mean(defl))) / max(np.sum(tCentered * tCentered), 1e-9))

    return {
        "weightG": float(sample.weightG),
        "frameCount": int(defl.size),
        "deflectionMeanMm": float(np.mean(defl)),
        "deflectionStdMm": float(np.std(defl)),
        "deflectionMinMm": float(np.min(defl)),
        "deflectionMaxMm": float(np.max(defl)),
        "deflectionP05Mm": float(np.percentile(defl, 5)),
        "deflectionP50Mm": float(np.percentile(defl, 50)),
        "deflectionP95Mm": float(np.percentile(defl, 95)),
        "deflectionSlopePerFrame": slope,
        "confidenceMean": float(np.mean(conf)) if conf.size > 0 else 0.0,
        "rmseMeanPx": float(np.nanmean(rmse)),
        "rmseP90Px": float(np.nanpercentile(rmse, 90)),
        "inlierMean": float(np.nanmean(inlier)),
        "usedPointsMean": float(np.nanmean(used)),
        "stabilityStdMm": float(np.std(defl)),
        "driftMmPerMin": float(slope * 30.0 * 60.0),
    }


def writeDataset(rows: list[dict], outputCsv: Path) -> None:
    import csv

    outputCsv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("数据集为空，无法写入。")
    fieldnames = list(rows[0].keys())
    with outputCsv.open("w", encoding="utf-8", newline="") as fileObj:
        writer = csv.DictWriter(fileObj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def computeMacroF1(pred: np.ndarray, true: np.ndarray, classCount: int) -> float:
    f1List: list[float] = []
    for classId in range(classCount):
        tp = float(np.sum((pred == classId) & (true == classId)))
        fp = float(np.sum((pred == classId) & (true != classId)))
        fn = float(np.sum((pred != classId) & (true == classId)))
        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        if precision + recall < 1e-9:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1List.append(f1)
    return float(np.mean(f1List))


def trainDualTaskModel(rows: list[dict], args: argparse.Namespace, outputDir: Path) -> dict:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("缺少 torch，无法训练双任务模型。请安装 pytorch。") from exc

    featureNames = [
        "deflectionMeanMm",
        "deflectionStdMm",
        "deflectionMinMm",
        "deflectionMaxMm",
        "deflectionP05Mm",
        "deflectionP50Mm",
        "deflectionP95Mm",
        "deflectionSlopePerFrame",
        "confidenceMean",
        "rmseMeanPx",
        "rmseP90Px",
        "inlierMean",
        "usedPointsMean",
        "stabilityStdMm",
        "driftMmPerMin",
    ]

    x = np.asarray([[float(row[name]) for name in featureNames] for row in rows], dtype=np.float32)
    yReg = np.asarray([float(row["weightG"]) for row in rows], dtype=np.float32)
    uniqueWeights = sorted({float(v) for v in yReg.tolist()})
    labelMap = {weight: idx for idx, weight in enumerate(uniqueWeights)}
    yCls = np.asarray([labelMap[float(v)] for v in yReg.tolist()], dtype=np.int64)

    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    xNorm = (x - mean) / std

    if len(rows) < 6:
        trainIndex = np.arange(len(rows))
        valIndex = np.arange(len(rows))
    else:
        permutation = np.random.permutation(len(rows))
        split = max(1, int(0.8 * len(rows)))
        trainIndex = permutation[:split]
        valIndex = permutation[split:]
        if valIndex.size == 0:
            valIndex = trainIndex

    xTrain = torch.tensor(xNorm[trainIndex], dtype=torch.float32)
    yRegTrain = torch.tensor(yReg[trainIndex], dtype=torch.float32).unsqueeze(1)
    yClsTrain = torch.tensor(yCls[trainIndex], dtype=torch.long)
    xVal = torch.tensor(xNorm[valIndex], dtype=torch.float32)
    yRegVal = torch.tensor(yReg[valIndex], dtype=torch.float32)
    yClsVal = torch.tensor(yCls[valIndex], dtype=torch.long)

    class DualTaskNet(nn.Module):
        def __init__(self, inputDim: int, classCount: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(inputDim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.regHead = nn.Linear(32, 1)
            self.clsHead = nn.Linear(32, classCount)

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.backbone(features)
            return self.regHead(hidden), self.clsHead(hidden)

    model = DualTaskNet(inputDim=xTrain.shape[1], classCount=len(uniqueWeights))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    regLossFn = nn.HuberLoss(delta=1.0)
    clsLossFn = nn.CrossEntropyLoss()

    bestScore = -1e9
    bestState = None
    bestMetrics: dict = {}

    for _ in range(args.epochs):
        model.train()
        permutation = torch.randperm(xTrain.shape[0])
        for start in range(0, xTrain.shape[0], max(args.batch_size, 1)):
            batchIndex = permutation[start : start + max(args.batch_size, 1)]
            xBatch = xTrain[batchIndex]
            yRegBatch = yRegTrain[batchIndex]
            yClsBatch = yClsTrain[batchIndex]

            optimizer.zero_grad()
            regPredTrain, clsPredTrain = model(xBatch)
            regLoss = regLossFn(regPredTrain, yRegBatch)
            clsLoss = clsLossFn(clsPredTrain, yClsBatch)
            loss = 0.7 * regLoss + 0.3 * clsLoss
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            regPredVal, clsPredVal = model(xVal)
            regPredNp = regPredVal.squeeze(1).cpu().numpy()
            clsProb = torch.softmax(clsPredVal, dim=1).cpu().numpy()
            clsPredNp = np.argmax(clsProb, axis=1)
            yRegValNp = yRegVal.cpu().numpy()
            yClsValNp = yClsVal.cpu().numpy()

            mae = float(np.mean(np.abs(regPredNp - yRegValNp)))
            rmse = float(np.sqrt(np.mean((regPredNp - yRegValNp) ** 2)))
            denom = float(np.sum((yRegValNp - np.mean(yRegValNp)) ** 2))
            r2 = float(1.0 - np.sum((regPredNp - yRegValNp) ** 2) / max(denom, 1e-9))
            accuracy = float(np.mean(clsPredNp == yClsValNp))
            macroF1 = computeMacroF1(clsPredNp, yClsValNp, len(uniqueWeights))
            stabilityStd = float(np.mean([float(row["stabilityStdMm"]) for row in rows]))

            nMae = mae / max(np.std(yRegValNp), 1e-6)
            nStd = stabilityStd / max(np.max(np.abs(yRegValNp)), 1.0)
            score = 0.45 * (1.0 - nMae) + 0.25 * macroF1 + 0.30 * (1.0 - nStd)
            if score > bestScore:
                bestScore = score
                bestState = model.state_dict()
                bestMetrics = {
                    "maeG": mae,
                    "rmseG": rmse,
                    "r2": r2,
                    "accuracy": accuracy,
                    "macroF1": macroF1,
                    "stabilityStdMm": stabilityStd,
                    "compositeScore": float(score),
                }

    assert bestState is not None
    outputDir.mkdir(parents=True, exist_ok=True)
    modelPath = outputDir / "best.pth"
    torch.save(
        {
            "state_dict": bestState,
            "feature_names": featureNames,
            "x_mean": mean.tolist(),
            "x_std": std.tolist(),
            "label_map": {str(weight): idx for weight, idx in labelMap.items()},
        },
        str(modelPath),
    )

    (outputDir / "label_map.json").write_text(
        json.dumps({str(weight): idx for weight, idx in labelMap.items()}, indent=2),
        encoding="utf-8",
    )
    (outputDir / "metrics.json").write_text(json.dumps(bestMetrics, indent=2), encoding="utf-8")
    bestMetrics["modelPath"] = str(modelPath)
    return bestMetrics


def parseWeightInput(text: str) -> Optional[float]:
    raw = text.strip().lower()
    if raw in {"done", "q", "quit", "exit"}:
        return None
    return float(raw)


def main() -> int:
    print(
        "\n这个入口已经不再作为砝码重量训练主流程使用。\n"
        "现在请使用三段式脚本：\n"
        "1) 采集数据: python yolo/collect_weight_data.py\n"
        "2) 训练模型: python yolo/train_weight_model.py --windows-csv <weight_windows.csv>\n"
        "3) 实时预测: python yolo/predict_weight_realtime.py --model-path <best_weight_model.pth>\n"
        "这样做是为了避免采集、训练、预测混在一起，导致实验数据无法复用。\n",
        flush=True,
    )
    return 0

    args = parseArgs()
    printCollectorGuide(args)
    layoutPath = Path(args.layout)
    layout = loadLayout(layoutPath if layoutPath.exists() else None)

    solver = ArucoStaticSolver(layout.dictionaryName)
    detector = MidpointTargetDetector(
        modelPath=args.model,
        confThreshold=args.conf,
        imageSize=args.imgsz,
        targetClassName=args.target_class,
    )
    estimator = DeflectionEstimator(baselineFrames=args.baseline_frames, deflectionScale=args.deflection_scale)
    capture = openVideoSource(args.source, preferAvfoundation=True)
    calibration = resolveCalibration(args, capture)

    windowName = "砝码重量采集器"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = 30.0

    rows: list[dict] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputCsv = Path(args.dataset_out) if args.dataset_out else Path(f"yolo/results/weight_dataset_{timestamp}.csv")
    outputDir = Path(args.output_dir) / timestamp

    try:
        while True:
            rawInput = input("请输入本轮砝码重量(g)，输入 done 结束采集: ").strip()
            if rawInput.lower() in {"done", "q", "quit", "exit"}:
                break
            try:
                weightG = parseWeightInput(rawInput)
                assert weightG is not None
            except Exception:
                print("输入格式错误，请输入数字（单位 g）或 done。")
                continue

            print(f"准备采集重量 {weightG:.3f} g。请在窗口中按 s 开始，按 e 提前结束。")
            collecting = False
            frameCount = 0
            validCount = 0
            sample = SampleWindow(weightG=weightG, deflectionsMm=[], confidences=[], rmsePx=[], inlierRatio=[], usedPoints=[])
            historyMm: deque[float] = deque(maxlen=120)

            while True:
                ok, frameRaw = capture.read()
                if not ok:
                    raise RuntimeError("视频流中断，采集终止。")

                frame = undistortFrame(frameRaw, calibration) if calibration is not None else frameRaw
                homographyResult = solver.solveHomography(frame, layout)
                detection = detector.detect(frame)

                isLowQuality = (
                    homographyResult.homography is not None
                    and (
                        homographyResult.usedPointCount < args.min_used_points
                        or (homographyResult.reprojectionRmsePx is not None and homographyResult.reprojectionRmsePx > args.max_rmse)
                        or (homographyResult.inlierRatio is not None and homographyResult.inlierRatio < args.min_inlier_ratio)
                    )
                )
                worldPoint = None
                if homographyResult.homography is not None and detection.centerPixel is not None and not isLowQuality:
                    worldPoint = solver.pixelToWorld(homographyResult.homography, detection.centerPixel)

                statusHint = detection.status
                if homographyResult.homography is None:
                    statusHint = "no-static-markers"
                elif isLowQuality:
                    statusHint = "low-homography-quality"

                frameCount += 1
                state = estimator.update(
                    worldYmm=(worldPoint[1] if worldPoint is not None else None),
                    timeSec=frameCount / fps,
                    confidence=detection.confidence,
                    statusHint=statusHint,
                )
                overlay = drawOverlay(frame, homographyResult, detection.centerPixel)

                if state.filteredMm is not None:
                    historyMm.append(float(state.filteredMm))

                if collecting and state.filteredMm is not None and state.status.startswith("tracking"):
                    sample.deflectionsMm.append(float(state.filteredMm))
                    sample.confidences.append(float(detection.confidence))
                    if homographyResult.reprojectionRmsePx is not None:
                        sample.rmsePx.append(float(homographyResult.reprojectionRmsePx))
                    if homographyResult.inlierRatio is not None:
                        sample.inlierRatio.append(float(homographyResult.inlierRatio))
                    sample.usedPoints.append(int(homographyResult.usedPointCount))
                    validCount += 1

                cv2.putText(
                    overlay,
                    f"Weight(g): {weightG:.3f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    f"Status: {localizeStatus(state.status)}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    f"Valid Frames: {validCount}/{args.min_valid_frames}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA,
                )
                helpText = "Press S to start, E to end, Q to abort"
                cv2.putText(
                    overlay,
                    helpText,
                    (20, overlay.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )

                if args.overlay_level == "debug":
                    panel = np.zeros((overlay.shape[0], 360, 3), dtype=np.uint8)
                    panel[:] = (24, 24, 24)
                    lines = [
                        f"Undistort: {'ON' if calibration else 'OFF'}",
                        f"Detect src: {localizeDetectionSource(detection.status)}",
                        f"Detect conf: {detection.confidence:.3f}",
                        f"Used points: {homographyResult.usedPointCount}",
                        f"Reproj RMSE(px): {homographyResult.reprojectionRmsePx}",
                        f"Inlier ratio: {homographyResult.inlierRatio}",
                        f"Baseline(mm): {state.baselineMm}",
                        f"Deflection(mm): {state.filteredMm}",
                    ]
                    y = 30
                    for line in lines:
                        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
                        y += 24
                    overlay = np.hstack([overlay, panel])

                cv2.imshow(windowName, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise RuntimeError("用户中止采集流程。")
                if key == ord("s"):
                    collecting = True
                    frameCount = 0
                    validCount = 0
                    sample.deflectionsMm.clear()
                    sample.confidences.clear()
                    sample.rmsePx.clear()
                    sample.inlierRatio.clear()
                    sample.usedPoints.clear()
                    print("开始采集...")
                if key == ord("e") and collecting:
                    print("手动结束采集。")
                    break
                if collecting and frameCount / fps >= args.capture_seconds:
                    print("达到最大采集时长，自动结束。")
                    break
                if collecting and validCount >= args.min_valid_frames:
                    print("达到最少有效帧要求，自动结束。")
                    break

            if len(sample.deflectionsMm) < max(10, args.min_valid_frames // 3):
                print("本轮有效数据过少，已丢弃，请重采。")
                continue
            row = computeFeatureRow(sample)
            rows.append(row)
            print(f"已保存本轮样本，当前样本数: {len(rows)}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError("未采集到任何有效样本，流程结束。")

    writeDataset(rows, outputCsv)
    metadata = {
        "timestamp": timestamp,
        "sampleCount": len(rows),
        "weights": sorted({float(row["weightG"]) for row in rows}),
        "datasetCsv": str(outputCsv),
    }
    outputCsv.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"数据集已写入: {outputCsv}")

    if args.auto_train == "false":
        print("已按参数要求跳过自动训练。")
        return 0

    metrics = trainDualTaskModel(rows, args, outputDir)
    print("训练完成，最佳指标：")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
