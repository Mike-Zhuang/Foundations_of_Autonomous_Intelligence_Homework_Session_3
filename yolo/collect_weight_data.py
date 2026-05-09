from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from bridge_ai.calibration import CameraCalibration, loadCalibration, runCharucoCalibration, saveCalibration
from bridge_ai.config import loadLayout
from bridge_ai.detection import MidpointTargetDetector
from bridge_ai.geometry import drawOverlay
from bridge_ai.io_utils import openVideoSource
from bridge_ai.realtime_measurement import MeasurementConfig, MeasurementFrame, RealtimeDeflectionMeasurer
from bridge_ai.weight_features import WeightSampleFrame, computeQuality, computeWindowFeatures


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="砝码重量数据采集：只采集并保存，不训练")
    parser.add_argument("--source", default="0", help="视频源")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 模型路径")
    parser.add_argument("--layout", default="yolo/artifacts/static_marker_layout.json", help="静态标记布局 json")
    parser.add_argument("--target-class", default="midpoint_marker")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--baseline-frames", type=int, default=60)
    parser.add_argument("--calibration-mode", choices=["off", "use", "recalibrate"], default="use")
    parser.add_argument("--calibration-file", default="yolo/artifacts/camera_calibration.npz")
    parser.add_argument("--overlay-level", choices=["minimal", "balanced", "debug"], default="debug")
    parser.add_argument("--measurement-method", choices=["target-local-scale", "static-compensated-pnp", "target-pnp", "homography"], default="target-local-scale")
    parser.add_argument("--local-scale-mode", choices=["baseline", "current", "average"], default="baseline")
    parser.add_argument("--target-marker-size", type=float, default=50.0)
    parser.add_argument("--filter-profile", choices=["stable", "normal", "fast"], default="stable")
    parser.add_argument("--deadband-mm", type=float, default=0.2)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--deflection-scale", type=float, default=1.0)
    parser.add_argument("--min-used-points", type=int, default=12)
    parser.add_argument("--max-rmse", type=float, default=4.0)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.45)
    parser.add_argument(
        "--target-only-fallback",
        choices=["true", "false"],
        default="true",
        help="静态点不可用时，是否降级为仅用 ID42 + 图像竖直方向继续输出数据",
    )
    parser.add_argument(
        "--static-pose-correction",
        choices=["true", "false"],
        default="true",
        help="有相机标定时，是否用静态标 PnP 姿态修正测量方向",
    )
    parser.add_argument("--static-assist-min-points", type=int, default=4, help="静态姿态辅助所需最少角点数，4 表示 1 个静态标即可辅助")
    parser.add_argument("--capture-seconds", type=float, default=8.0)
    parser.add_argument("--min-valid-frames", type=int, default=80)
    parser.add_argument("--collection-smooth-window", type=int, default=5, help="每轮采集独立滑动滤波窗口帧数，不使用跨轮 deadband")
    parser.add_argument("--output-dir", default="yolo/results")
    return parser.parse_args()


def resolveCalibration(args: argparse.Namespace, capture: cv2.VideoCapture) -> CameraCalibration | None:
    calibrationPath = Path(args.calibration_file)
    if args.calibration_mode == "off":
        return None
    if args.calibration_mode == "recalibrate":
        calibration = runCharucoCalibration(capture, targetSamples=30)
        saveCalibration(calibration, calibrationPath)
        print(f"ChArUco 标定完成，RMS={calibration.rms:.4f}，已保存: {calibrationPath}", flush=True)
        return calibration
    if calibrationPath.exists():
        calibration = loadCalibration(calibrationPath)
        print(f"加载标定参数成功: {calibrationPath} (RMS={calibration.rms:.4f})", flush=True)
        return calibration
    print(f"未找到标定文件 {calibrationPath}，自动降级为不去畸变。", flush=True)
    return None


def printGuide(args: argparse.Namespace) -> None:
    print("\n================ 砝码重量数据采集 ================", flush=True)
    print("阶段 1：空载固定桥和相机，在视频窗口按 s 锁定基线。", flush=True)
    print("阶段 2：命令行输入重量(g)，放稳砝码/小车后，在视频窗口按 s 开始采集。", flush=True)
    print("阶段 3：程序会一直采到足够有效帧；采集完成后，命令行输入激光测距仪标准挠度(mm)。", flush=True)
    print("阶段 4：换下一个重量继续；输入 done 结束并保存数据。", flush=True)
    print("窗口按键：s 开始当前阶段，e 提前结束当前采集，q 中止。", flush=True)
    print(f"测量方法: {args.measurement_method}, local-scale-mode: {args.local_scale_mode}", flush=True)
    print(
        f"最短采集时长: {args.capture_seconds}s, 最少有效帧: {args.min_valid_frames}, "
        f"采集滑动滤波窗口: {args.collection_smooth_window}帧",
        flush=True,
    )
    print("================================================\n", flush=True)


def makeMeasurementConfig(args: argparse.Namespace) -> MeasurementConfig:
    return MeasurementConfig(
        baselineFrames=args.baseline_frames,
        filterProfile=args.filter_profile,
        deadbandMm=args.deadband_mm,
        smoothWindow=args.smooth_window,
        localScaleMode=args.local_scale_mode,
        deflectionScale=args.deflection_scale,
        measurementMethod=args.measurement_method,
        targetMarkerSizeMm=args.target_marker_size,
        minUsedPoints=args.min_used_points,
        maxRmse=args.max_rmse,
        minInlierRatio=args.min_inlier_ratio,
        targetOnlyFallback=args.target_only_fallback == "true",
        staticPoseCorrection=args.static_pose_correction == "true",
        staticAssistMinPoints=args.static_assist_min_points,
    )


def putText(frame: np.ndarray, text: str, y: int, color: tuple[int, int, int] = (235, 235, 235)) -> None:
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def drawDeflectionSparkline(frame: np.ndarray, values: list[float], topLeft: tuple[int, int], size: tuple[int, int]) -> None:
    x0, y0 = topLeft
    width, height = size
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (90, 90, 90), 1)
    if len(values) < 2:
        return

    arr = np.asarray(values, dtype=np.float32)
    vMin = float(np.min(arr))
    vMax = float(np.max(arr))
    if abs(vMax - vMin) < 1e-6:
        vMax = vMin + 1e-6

    points: list[tuple[int, int]] = []
    for index, value in enumerate(arr):
        x = x0 + int(index * (width - 1) / max(len(arr) - 1, 1))
        y = y0 + int((vMax - float(value)) / (vMax - vMin) * (height - 1))
        points.append((x, y))

    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], (0, 255, 120), 2, cv2.LINE_AA)


def renderFrame(
    measurement: MeasurementFrame,
    phase: str,
    infoLines: list[str],
    overlayLevel: str,
    historyMm: list[float],
) -> np.ndarray:
    overlay = drawOverlay(measurement.frame, measurement.homographyResult, measurement.detection.centerPixel)
    putText(overlay, f"Phase: {phase}", 65, (80, 220, 255))
    putText(overlay, f"Status: {measurement.state.status}", 95, (255, 255, 0))
    putText(overlay, f"Deflection(mm): {measurement.state.filteredMm}", 125, (0, 255, 120))
    y = 155
    for line in infoLines:
        putText(overlay, line, y)
        y += 28

    if overlayLevel != "debug":
        chartWidth = min(260, max(160, overlay.shape[1] // 4))
        drawDeflectionSparkline(
            overlay,
            historyMm,
            topLeft=(overlay.shape[1] - chartWidth - 20, overlay.shape[0] - 115),
            size=(chartWidth, 90),
        )
        return overlay

    panel = np.zeros((overlay.shape[0], 380, 3), dtype=np.uint8)
    panel[:] = (24, 24, 24)
    lines = [
        f"Measure: {measurement.measurementMethod}",
        f"Detect: {measurement.detection.status}",
        f"Conf: {measurement.detection.confidence:.3f}",
        f"Static IDs: {measurement.homographyResult.foundMarkerIds}",
        f"Used points: {measurement.homographyResult.usedPointCount}",
        f"RMSE(px): {measurement.homographyResult.reprojectionRmsePx}",
        f"Inlier: {measurement.homographyResult.inlierRatio}",
        f"Raw(mm): {measurement.state.rawMm}",
        f"Filtered(mm): {measurement.state.filteredMm}",
        f"Target pos(px): {measurement.targetPositionPx}",
        f"Target px/mm: {measurement.targetPxPerMm}",
        f"Static axis: {measurement.staticAxisSource}",
        f"Plane tilt(deg): {measurement.staticPoseInfo.planeTiltDeg if measurement.staticPoseInfo else None}",
        f"Static roll(deg): {measurement.staticPoseInfo.rollDeg if measurement.staticPoseInfo else None}",
    ]
    y = 30
    for line in lines:
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
        y += 24
    cv2.putText(panel, "Deflection history (mm)", (16, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1, cv2.LINE_AA)
    drawDeflectionSparkline(panel, historyMm, topLeft=(16, y + 18), size=(348, 110))
    return np.hstack([overlay, panel])


def parseWeightInput(text: str) -> float | None:
    raw = text.strip().lower()
    if raw in {"done", "q", "quit", "exit"}:
        return None
    return float(raw)


def readStandardDeflectionMm() -> float:
    while True:
        raw = input("请输入本轮激光测距仪标准挠度(mm)，例如 -12.35: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("输入格式错误：请输入数字，单位 mm。", flush=True)


def getMeasurementRawMm(measurement: MeasurementFrame) -> float | None:
    rawMm = measurement.state.rawMm
    if rawMm is None or not np.isfinite(rawMm):
        return None
    return float(rawMm)


def updateCollectionFilter(rawMm: float | None, rawWindow: deque[float], windowSize: int) -> float | None:
    if rawMm is None:
        return None
    rawWindow.append(float(rawMm))
    while len(rawWindow) > max(1, windowSize):
        rawWindow.popleft()
    values = np.asarray(rawWindow, dtype=np.float64)
    if values.size == 1:
        return float(values[0])
    medianValue = float(np.median(values))
    meanValue = float(np.mean(values))
    return float(0.7 * medianValue + 0.3 * meanValue)


def writeCsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"没有数据可写入: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fileObj:
        writer = csv.DictWriter(fileObj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def waitForBaseline(capture: cv2.VideoCapture, measurer: RealtimeDeflectionMeasurer, fps: float, windowName: str, overlayLevel: str) -> None:
    print("请保持空载静止，然后在视频窗口按 s 开始基线。", flush=True)
    lastTime = time.time()
    started = False
    historyMm: deque[float] = deque(maxlen=180)
    while True:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("视频流中断，无法完成基线。")
        now = time.time()
        measurement = measurer.process(frame, dtSec=min(now - lastTime, 1.0 / max(fps, 1.0)))
        lastTime = now
        if measurement.state.filteredMm is not None:
            historyMm.append(float(measurement.state.filteredMm))
        info = ["Press S to start empty baseline", "Keep bridge unloaded and still"]
        render = renderFrame(measurement, "baseline", info, overlayLevel, list(historyMm))
        cv2.imshow(windowName, render)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise RuntimeError("用户中止基线流程。")
        if key == ord("s") and not started:
            measurer.startBaseline()
            started = True
            print("已开始空载基线，请保持静止。", flush=True)
        if started and measurement.state.baselineMm is not None:
            print("基线完成。现在可以输入重量并开始采集。", flush=True)
            return


def buildRawRow(
    sampleId: str,
    weightG: float,
    measurement: MeasurementFrame,
    isCollecting: bool,
    collectionFilteredMm: float | None = None,
    standardDeflectionMm: float | None = None,
) -> dict:
    stateDict = asdict(measurement.state)
    return {
        "sampleId": sampleId,
        "weightG": weightG,
        "standardDeflectionMm": standardDeflectionMm,
        "collecting": int(isCollecting),
        "collectionFilteredMm": collectionFilteredMm,
        "measurementMethod": measurement.measurementMethod,
        "detectStatus": measurement.detection.status,
        "usedPointCount": measurement.homographyResult.usedPointCount,
        "rmsePx": measurement.homographyResult.reprojectionRmsePx,
        "inlierRatio": measurement.homographyResult.inlierRatio,
        "staticAxisSource": measurement.staticAxisSource,
        "staticPlaneTiltDeg": measurement.staticPoseInfo.planeTiltDeg if measurement.staticPoseInfo else None,
        "staticRollDeg": measurement.staticPoseInfo.rollDeg if measurement.staticPoseInfo else None,
        "staticPoseRmsePx": measurement.staticPoseInfo.reprojectionRmsePx if measurement.staticPoseInfo else None,
        "quality": computeQuality(
            measurement.homographyResult.reprojectionRmsePx,
            measurement.homographyResult.inlierRatio,
            measurement.homographyResult.usedPointCount,
        ),
        **stateDict,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parseArgs()
    printGuide(args)

    layoutPath = Path(args.layout)
    layout = loadLayout(layoutPath if layoutPath.exists() else None)
    detector = MidpointTargetDetector(
        modelPath=args.model,
        confThreshold=args.conf,
        imageSize=args.imgsz,
        targetClassName=args.target_class,
    )

    capture = openVideoSource(args.source, preferAvfoundation=True)
    calibration = resolveCalibration(args, capture)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = 30.0

    measurer = RealtimeDeflectionMeasurer(layout, detector, makeMeasurementConfig(args), calibration)
    windowName = "Weight Data Collector"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputDir = Path(args.output_dir)
    rawPath = outputDir / f"weight_raw_{timestamp}.csv"
    windowsPath = outputDir / f"weight_windows_{timestamp}.csv"
    metadataPath = outputDir / f"weight_dataset_{timestamp}.metadata.json"

    rawRows: list[dict] = []
    windowRows: list[dict] = []
    sampleIndex = 0

    try:
        waitForBaseline(capture, measurer, fps, windowName, args.overlay_level)
        while True:
            rawInput = input("请输入本轮重量(g)，或输入 done 结束: ").strip()
            try:
                weightG = parseWeightInput(rawInput)
            except ValueError:
                print("输入格式错误：请输入数字重量，例如 200，或输入 done。", flush=True)
                continue
            if weightG is None:
                break

            sampleIndex += 1
            sampleId = f"sample_{sampleIndex:03d}_{weightG:g}g"
            print(f"当前重量 {weightG:g} g。放稳后在视频窗口按 s 开始采集。", flush=True)
            collecting = False
            sampleFrames: list[WeightSampleFrame] = []
            sampleRawRows: list[dict] = []
            validFrames = 0
            startTime: float | None = None
            lastTime = time.time()
            historyMm: deque[float] = deque(maxlen=180)
            previewRawWindow: deque[float] = deque(maxlen=max(1, args.collection_smooth_window))
            collectionRawWindow: deque[float] = deque(maxlen=max(1, args.collection_smooth_window))
            waitingForValidNoticePrinted = False

            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("视频流中断，采集终止。")
                now = time.time()
                measurement = measurer.process(frame, dtSec=min(now - lastTime, 1.0 / max(fps, 1.0)))
                lastTime = now

                rawMm = getMeasurementRawMm(measurement)
                previewFilteredMm = updateCollectionFilter(rawMm, previewRawWindow, args.collection_smooth_window)
                collectionFilteredMm: float | None = None
                isTracking = measurement.state.status.startswith("tracking")
                isValid = False
                if collecting:
                    collectionFilteredMm = updateCollectionFilter(rawMm, collectionRawWindow, args.collection_smooth_window)
                    isValid = collectionFilteredMm is not None and isTracking
                    sampleRawRows.append(
                        buildRawRow(
                            sampleId,
                            weightG,
                            measurement,
                            isCollecting=True,
                            collectionFilteredMm=collectionFilteredMm,
                        )
                    )
                    sampleFrames.append(
                        WeightSampleFrame(
                            timeSec=0.0 if startTime is None else now - startTime,
                            deflectionMm=collectionFilteredMm if collectionFilteredMm is not None else float("nan"),
                            confidence=float(measurement.detection.confidence),
                            quality=computeQuality(
                                measurement.homographyResult.reprojectionRmsePx,
                                measurement.homographyResult.inlierRatio,
                                measurement.homographyResult.usedPointCount,
                            ),
                            isValid=isValid,
                        )
                    )
                    if isValid:
                        validFrames += 1
                    if collectionFilteredMm is not None:
                        historyMm.append(float(collectionFilteredMm))
                elif previewFilteredMm is not None:
                    historyMm.append(float(previewFilteredMm))

                elapsed = 0.0 if startTime is None else now - startTime
                enoughTime = elapsed >= args.capture_seconds
                enoughValid = validFrames >= args.min_valid_frames
                info = [
                    f"Weight(g): {weightG:g}",
                    f"Collecting: {'YES' if collecting else 'NO'}",
                    f"Valid: {validFrames}/{args.min_valid_frames}",
                    f"Elapsed(s): {elapsed:.1f}/{args.capture_seconds:.1f}+",
                    f"Window(mm): {collectionFilteredMm if collecting else previewFilteredMm}",
                    f"Window filter: {args.collection_smooth_window} frames, per-round reset",
                    "Wait stable green curve, then press S",
                    "Keys: S start, E finish-if-ready, Q abort",
                ]
                cv2.imshow(windowName, renderFrame(measurement, "collect", info, args.overlay_level, list(historyMm)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise RuntimeError("用户中止采集流程。")
                if key == ord("s") and not collecting:
                    collecting = True
                    startTime = time.time()
                    sampleFrames.clear()
                    sampleRawRows.clear()
                    collectionRawWindow.clear()
                    validFrames = 0
                    waitingForValidNoticePrinted = False
                    print("开始采集当前重量。本轮采集滑动滤波器已清空。", flush=True)
                if key == ord("e") and collecting:
                    if enoughValid:
                        print("手动结束当前采集。", flush=True)
                        break
                    print(f"有效帧还不够，继续采集: {validFrames}/{args.min_valid_frames}", flush=True)
                if collecting and enoughTime and not enoughValid and not waitingForValidNoticePrinted:
                    print(
                        f"已达到最短采集时长，但有效帧还不够，将继续采到 {args.min_valid_frames} 帧。",
                        flush=True,
                    )
                    waitingForValidNoticePrinted = True
                if collecting and enoughTime and enoughValid:
                    print("达到最短采集时长且有效帧足够，自动结束当前采集。", flush=True)
                    break
                if collecting and enoughValid and args.capture_seconds <= 0:
                    print("达到最少有效帧，自动结束当前采集。", flush=True)
                    break

            if validFrames < args.min_valid_frames:
                print(
                    f"本轮没有达到最少有效帧，未保存。有效帧: {validFrames}/{args.min_valid_frames}。"
                    "请检查识别状态后重采。",
                    flush=True,
                )
                continue
            standardDeflectionMm = readStandardDeflectionMm()
            for row in sampleRawRows:
                row["standardDeflectionMm"] = standardDeflectionMm
            rawRows.extend(sampleRawRows)
            windowRow = computeWindowFeatures(
                sampleFrames,
                weightG=weightG,
                standardDeflectionMm=standardDeflectionMm,
                sampleId=sampleId,
            )
            windowRows.append(windowRow)
            print(
                "本轮已保存到内存："
                f"重量={weightG:g}g，手机平均挠度={windowRow['deflectionMeanMm']:.3f}mm，"
                f"标准挠度={standardDeflectionMm:.3f}mm，"
                f"差值={windowRow['phoneMinusStandardMm']:.3f}mm。"
                f"当前有效样本窗口数: {len(windowRows)}",
                flush=True,
            )
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if not windowRows:
        raise RuntimeError("没有采集到有效窗口，未写入数据集。")

    writeCsv(rawPath, rawRows)
    writeCsv(windowsPath, windowRows)
    metadata = {
        "timestamp": timestamp,
        "rawCsv": str(rawPath),
        "windowsCsv": str(windowsPath),
        "sampleCount": len(windowRows),
        "weightsG": sorted({float(row["weightG"]) for row in windowRows}),
        "hasStandardDeflectionMm": True,
        "args": vars(args),
    }
    metadataPath.parent.mkdir(parents=True, exist_ok=True)
    metadataPath.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"逐帧数据已保存: {rawPath}", flush=True)
    print(f"窗口特征已保存: {windowsPath}", flush=True)
    print(f"元数据已保存: {metadataPath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
