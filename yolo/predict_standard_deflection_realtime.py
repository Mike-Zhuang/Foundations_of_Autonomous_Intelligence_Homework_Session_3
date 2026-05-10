from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from bridge_ai.bridge_task_models import loadModelBundle, predictStandardDeflectionFromPhone
from bridge_ai.calibration import CameraCalibration
from bridge_ai.config import loadLayout
from bridge_ai.detection import MidpointTargetDetector
from bridge_ai.geometry import drawOverlay
from bridge_ai.io_utils import openVideoSource
from bridge_ai.realtime_measurement import MeasurementConfig, MeasurementFrame, RealtimeDeflectionMeasurer
from bridge_ai.weight_features import WeightSampleFrame, computeQuality, computeWindowFeatures
from predict_weight_realtime import (
    drawSparkline,
    formatMaybe,
    getMeasurementRawMm,
    makeMeasurementConfig,
    makeSampleFrame,
    resolveCalibration,
    updateShortFilter,
)


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="任务 C：实时手机视觉挠度 -> 我们预测的激光标准挠度")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model-path", required=True, help="train_bridge_task_models.py 输出的 model_bundle.pth/json")
    parser.add_argument(
        "--model-choice",
        choices=["auto", "mlp", "monotonic", "ridge"],
        default="auto",
        help="候选模型选择。auto 使用训练时推荐；mlp 强制神经网络；monotonic 强制单调函数逼近器",
    )
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--layout", default="yolo/artifacts/static_marker_layout.json")
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
    parser.add_argument("--target-only-fallback", choices=["true", "false"], default="true")
    parser.add_argument("--static-pose-correction", choices=["true", "false"], default="true")
    parser.add_argument("--static-assist-min-points", type=int, default=4)
    parser.add_argument("--live-window-frames", type=int, default=5, help="实时任务 C 使用的短滑动窗口帧数")
    parser.add_argument("--predict-seconds", type=float, default=6.0, help="按 s 后最终窗口的最长采集秒数")
    parser.add_argument("--min-valid-frames", type=int, default=80, help="按 s 后最终窗口的最少有效帧数")
    parser.add_argument("--zero-deflection-threshold-mm", type=float, default=0.3, help="小于该挠度认为未加载")
    return parser.parse_args()


def printGuide(args: argparse.Namespace, bundle: dict) -> None:
    print("\n================ 任务 C 实时标准挠度预测 ================", flush=True)
    print("它显示的是“我们预测的激光等效标准挠度”，不是老师隐藏的真实答案。", flush=True)
    print("阶段 1：空载固定桥和相机，在视频窗口按 s 锁定基线。", flush=True)
    print("阶段 2：放上载荷，窗口实时显示手机挠度、预测标准挠度、二者差值。", flush=True)
    print("阶段 3：按 s 采集稳定窗口并输出最终标准挠度预测；按 q 退出。", flush=True)
    task = bundle["tasks"]["laser_from_phone"]
    print(f"模型: {args.model_path}", flush=True)
    print(f"任务: laser_from_phone, 推荐候选: {task['recommended']}", flush=True)
    print(f"实际候选选择: {args.model_choice}", flush=True)
    print("=====================================================\n", flush=True)


def predictStandardFromFrames(bundle: dict, frames: list[WeightSampleFrame], modelChoice: str = "auto") -> tuple[dict | None, dict | None]:
    validCount = sum(1 for frame in frames if frame.isValid)
    if validCount < 5:
        return None, None
    featureRow = computeWindowFeatures(frames, sampleId="standard_predict_window")
    return predictStandardDeflectionFromPhone(bundle, featureRow, modelPreference=modelChoice), featureRow


def applyDeflectionGuard(prediction: dict | None, featureRow: dict | None, thresholdMm: float) -> dict | None:
    if prediction is None:
        return None
    guarded = dict(prediction)
    phoneMm = float(featureRow.get("deflectionMeanMm", guarded.get("phoneDeflectionMm", 0.0))) if featureRow else 0.0
    if abs(phoneMm) < thresholdMm:
        guarded["phoneDeflectionMm"] = phoneMm
        guarded["predictedStandardDeflectionMm"] = 0.0
        guarded["predictedPhoneMinusStandardMm"] = phoneMm
        guarded["guardReason"] = "zero-deflection"
    return guarded


def drawStandardPrediction(
    measurement: MeasurementFrame,
    livePrediction: dict | None,
    finalPrediction: dict | None,
    featureRow: dict | None,
    collecting: bool,
    validCount: int,
    args: argparse.Namespace,
    phoneHistoryMm: list[float],
    standardHistoryMm: list[float],
) -> np.ndarray:
    overlay = drawOverlay(measurement.frame, measurement.homographyResult, measurement.detection.centerPixel)
    phoneText = "Phone deflection(mm): waiting"
    standardText = "Pred standard(mm): waiting"
    diffText = "Phone-standard(mm): waiting"
    if livePrediction is not None:
        suffix = " (no load)" if livePrediction.get("guardReason") == "zero-deflection" else ""
        phoneText = f"Phone deflection(mm): {livePrediction['phoneDeflectionMm']:.3f}"
        standardText = f"Pred standard(mm): {livePrediction['predictedStandardDeflectionMm']:.3f}{suffix}"
        diffText = f"Phone-standard(mm): {livePrediction['predictedPhoneMinusStandardMm']:.3f}"

    lines = [
        phoneText,
        standardText,
        diffText,
        f"Status: {measurement.state.status}",
        f"Collecting: {'YES' if collecting else 'NO'}",
        f"Valid: {validCount}/{args.min_valid_frames}",
        "Keys: S final prediction, Q quit",
    ]
    y = 65
    for line in lines:
        cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2, cv2.LINE_AA)
        y += 30

    if finalPrediction is not None:
        cv2.putText(
            overlay,
            f"FINAL standard: {finalPrediction['predictedStandardDeflectionMm']:.3f} mm",
            (20, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )

    if args.overlay_level != "debug":
        return overlay

    panel = np.zeros((overlay.shape[0], 400, 3), dtype=np.uint8)
    panel[:] = (24, 24, 24)
    debugLines = [
        "Task C: phone -> predicted standard",
        "Teacher laser answer is not read here",
        f"Measure: {measurement.measurementMethod}",
        f"Detect: {measurement.detection.status}",
        f"Conf: {measurement.detection.confidence:.3f}",
        f"Used points: {measurement.homographyResult.usedPointCount}",
        f"RMSE(px): {measurement.homographyResult.reprojectionRmsePx}",
        f"Inlier: {measurement.homographyResult.inlierRatio}",
        f"Raw(mm): {formatMaybe(measurement.state.rawMm)}",
        f"Filtered(mm): {formatMaybe(measurement.state.filteredMm)}",
        f"Live window(fr): {args.live_window_frames}",
        f"Feature std(mm): {featureRow.get('deflectionStdMm') if featureRow else None}",
        f"Feature drift(mm/min): {featureRow.get('driftMmPerMin') if featureRow else None}",
    ]
    y = 30
    for line in debugLines:
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        y += 24
    cv2.putText(panel, "Phone deflection (mm)", (16, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 160), 1, cv2.LINE_AA)
    drawSparkline(panel, phoneHistoryMm, (16, y + 18), (368, 95), (0, 255, 120))
    y += 132
    cv2.putText(panel, "Pred standard (mm)", (16, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 220, 255), 1, cv2.LINE_AA)
    drawSparkline(panel, standardHistoryMm, (16, y + 18), (368, 95), (255, 180, 60))
    return np.hstack([overlay, panel])


def waitForBaseline(
    capture: cv2.VideoCapture,
    measurer: RealtimeDeflectionMeasurer,
    fps: float,
    windowName: str,
    args: argparse.Namespace,
) -> None:
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
        render = drawStandardPrediction(measurement, None, None, None, False, 0, args, list(historyMm), [])
        cv2.putText(render, "Empty bridge: press S to start baseline", (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow(windowName, render)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise RuntimeError("用户中止基线流程。")
        if key == ord("s") and not started:
            measurer.startBaseline()
            started = True
            print("已开始空载基线，请保持静止。", flush=True)
        if started and measurement.state.baselineMm is not None:
            print("基线完成。可以放上载荷。", flush=True)
            return


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parseArgs()
    bundle = loadModelBundle(Path(args.model_path))
    if bundle.get("bundleType") != "bridge_task_models":
        raise RuntimeError("该模型不是三任务 bridge model bundle。")
    printGuide(args, bundle)

    layoutPath = Path(args.layout)
    layout = loadLayout(layoutPath if layoutPath.exists() else None)
    detector = MidpointTargetDetector(
        modelPath=args.yolo_model,
        confThreshold=args.conf,
        imageSize=args.imgsz,
        targetClassName=args.target_class,
    )
    capture = openVideoSource(args.source, preferAvfoundation=True)
    calibration: CameraCalibration | None = resolveCalibration(args, capture)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = 30.0

    config: MeasurementConfig = makeMeasurementConfig(args)
    measurer = RealtimeDeflectionMeasurer(layout, detector, config, calibration)
    windowName = "Standard Deflection Predictor"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    try:
        waitForBaseline(capture, measurer, fps, windowName, args)
        rollingFrames: deque[WeightSampleFrame] = deque(maxlen=max(args.live_window_frames, 5))
        liveRawWindow: deque[float] = deque(maxlen=max(args.live_window_frames, 1))
        finalRawWindow: deque[float] = deque(maxlen=max(args.live_window_frames, 1))
        phoneHistoryMm: deque[float] = deque(maxlen=180)
        standardHistoryMm: deque[float] = deque(maxlen=180)
        finalFrames: list[WeightSampleFrame] = []
        collecting = False
        finalStart: float | None = None
        finalPrediction = None
        lastFeatureRow = None
        lastTime = time.time()

        while True:
            ok, frame = capture.read()
            if not ok:
                print("视频流结束。", flush=True)
                break
            now = time.time()
            measurement = measurer.process(frame, dtSec=min(now - lastTime, 1.0 / max(fps, 1.0)))
            lastTime = now

            liveFilteredMm = updateShortFilter(getMeasurementRawMm(measurement), liveRawWindow, args.live_window_frames)
            rollingFrames.append(makeSampleFrame(measurement, localTime=now, deflectionMm=liveFilteredMm))
            livePrediction, featureRow = predictStandardFromFrames(bundle, list(rollingFrames), args.model_choice)
            livePrediction = applyDeflectionGuard(livePrediction, featureRow, args.zero_deflection_threshold_mm)
            if featureRow is not None:
                lastFeatureRow = featureRow
            if liveFilteredMm is not None:
                phoneHistoryMm.append(float(liveFilteredMm))
            if livePrediction is not None:
                standardHistoryMm.append(float(livePrediction["predictedStandardDeflectionMm"]))

            validFinal = 0
            if collecting:
                localTime = 0.0 if finalStart is None else now - finalStart
                finalFilteredMm = updateShortFilter(getMeasurementRawMm(measurement), finalRawWindow, args.live_window_frames)
                finalFrames.append(makeSampleFrame(measurement, localTime=localTime, deflectionMm=finalFilteredMm))
                validFinal = sum(1 for item in finalFrames if item.isValid)
                if localTime >= args.predict_seconds or validFinal >= args.min_valid_frames:
                    finalPrediction, lastFeatureRow = predictStandardFromFrames(bundle, finalFrames, args.model_choice)
                    finalPrediction = applyDeflectionGuard(finalPrediction, lastFeatureRow, args.zero_deflection_threshold_mm)
                    if finalPrediction is not None:
                        print(
                            f"最终标准挠度预测: {finalPrediction['predictedStandardDeflectionMm']:.3f} mm, "
                            f"手机-预测标准: {finalPrediction['predictedPhoneMinusStandardMm']:.3f} mm",
                            flush=True,
                        )
                    collecting = False

            render = drawStandardPrediction(
                measurement,
                livePrediction,
                finalPrediction,
                lastFeatureRow,
                collecting,
                validFinal,
                args,
                list(phoneHistoryMm),
                list(standardHistoryMm),
            )
            cv2.imshow(windowName, render)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and not collecting:
                finalFrames.clear()
                finalRawWindow.clear()
                finalStart = time.time()
                finalPrediction = None
                collecting = True
                print("开始采集任务 C 稳定预测窗口...", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
