from __future__ import annotations

import argparse
from collections import deque
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
from bridge_ai.weight_model import loadModel, predictWeight


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时挠度检测 + 砝码重量模型预测")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model-path", required=True, help="train_weight_model.py 输出的 best_weight_model.pth/json")
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
    parser.add_argument("--min-used-points", type=int, default=16)
    parser.add_argument("--max-rmse", type=float, default=2.6)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.65)
    parser.add_argument("--predict-seconds", type=float, default=6.0)
    parser.add_argument("--min-valid-frames", type=int, default=90)
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
    )


def printGuide(args: argparse.Namespace, modelPayload: dict) -> None:
    print("\n================ 实时重量预测 ================", flush=True)
    print("阶段 1：空载固定桥和相机，在视频窗口按 s 锁定基线。", flush=True)
    print("阶段 2：放上未知重量，等读数稳定。窗口会连续显示实时估计。", flush=True)
    print("阶段 3：按 s 采集稳定窗口并输出最终预测；按 q 退出。", flush=True)
    print(f"模型: {args.model_path}", flush=True)
    print(f"模型类型: {modelPayload.get('modelType')}", flush=True)
    print(f"训练档位(g): {modelPayload.get('weightsG')}", flush=True)
    print("============================================\n", flush=True)


def makeSampleFrame(measurement: MeasurementFrame, localTime: float) -> WeightSampleFrame:
    isValid = measurement.state.filteredMm is not None and measurement.state.status.startswith("tracking")
    return WeightSampleFrame(
        timeSec=localTime,
        deflectionMm=float(measurement.state.filteredMm) if measurement.state.filteredMm is not None else float("nan"),
        confidence=float(measurement.detection.confidence),
        quality=computeQuality(
            measurement.homographyResult.reprojectionRmsePx,
            measurement.homographyResult.inlierRatio,
            measurement.homographyResult.usedPointCount,
        ),
        isValid=isValid,
    )


def predictFromFrames(modelPayload: dict, frames: list[WeightSampleFrame]) -> tuple[dict | None, dict | None]:
    validCount = sum(1 for frame in frames if frame.isValid)
    if validCount < 5:
        return None, None
    featureRow = computeWindowFeatures(frames, sampleId="predict_window")
    return predictWeight(modelPayload, featureRow), featureRow


def drawPrediction(
    measurement: MeasurementFrame,
    livePrediction: dict | None,
    finalPrediction: dict | None,
    featureRow: dict | None,
    collecting: bool,
    validCount: int,
    args: argparse.Namespace,
) -> np.ndarray:
    overlay = drawOverlay(measurement.frame, measurement.homographyResult, measurement.detection.centerPixel)
    lines = [
        f"Deflection(mm): {measurement.state.filteredMm}",
        f"Status: {measurement.state.status}",
        f"Live weight(g): {livePrediction['predictedWeightG']:.2f}" if livePrediction else "Live weight(g): nan",
        f"Nearest(g): {livePrediction['nearestWeightG']:.2f}" if livePrediction else "Nearest(g): nan",
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
            f"FINAL: {finalPrediction['predictedWeightG']:.2f} g  nearest {finalPrediction['nearestWeightG']:.2f} g",
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
        f"Measure: {measurement.measurementMethod}",
        f"Detect: {measurement.detection.status}",
        f"Conf: {measurement.detection.confidence:.3f}",
        f"Used points: {measurement.homographyResult.usedPointCount}",
        f"RMSE(px): {measurement.homographyResult.reprojectionRmsePx}",
        f"Inlier: {measurement.homographyResult.inlierRatio}",
        f"Raw(mm): {measurement.state.rawMm}",
        f"Filtered(mm): {measurement.state.filteredMm}",
        f"Feature std(mm): {featureRow.get('deflectionStdMm') if featureRow else None}",
        f"Feature drift(mm/min): {featureRow.get('driftMmPerMin') if featureRow else None}",
    ]
    y = 30
    for line in debugLines:
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        y += 24
    return np.hstack([overlay, panel])


def waitForBaseline(capture: cv2.VideoCapture, measurer: RealtimeDeflectionMeasurer, fps: float, windowName: str) -> None:
    print("请保持空载静止，然后在视频窗口按 s 开始基线。", flush=True)
    lastTime = time.time()
    started = False
    while True:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("视频流中断，无法完成基线。")
        now = time.time()
        measurement = measurer.process(frame, dtSec=min(now - lastTime, 1.0 / max(fps, 1.0)))
        lastTime = now
        render = drawOverlay(measurement.frame, measurement.homographyResult, measurement.detection.centerPixel)
        cv2.putText(render, "Empty bridge: press S to start baseline", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(render, f"Status: {measurement.state.status}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(windowName, render)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise RuntimeError("用户中止基线流程。")
        if key == ord("s") and not started:
            measurer.startBaseline()
            started = True
            print("已开始空载基线，请保持静止。", flush=True)
        if started and measurement.state.baselineMm is not None:
            print("基线完成。可以放上未知重量。", flush=True)
            return


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parseArgs()
    modelPayload = loadModel(Path(args.model_path))
    printGuide(args, modelPayload)

    layoutPath = Path(args.layout)
    layout = loadLayout(layoutPath if layoutPath.exists() else None)
    detector = MidpointTargetDetector(
        modelPath=args.yolo_model,
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
    windowName = "Weight Predictor"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    try:
        waitForBaseline(capture, measurer, fps, windowName)
        rollingFrames: deque[WeightSampleFrame] = deque(maxlen=max(int(args.predict_seconds * fps), 10))
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

            rollingFrames.append(makeSampleFrame(measurement, localTime=now))
            livePrediction, featureRow = predictFromFrames(modelPayload, list(rollingFrames))
            if featureRow is not None:
                lastFeatureRow = featureRow

            validFinal = 0
            if collecting:
                localTime = 0.0 if finalStart is None else now - finalStart
                finalFrames.append(makeSampleFrame(measurement, localTime=localTime))
                validFinal = sum(1 for item in finalFrames if item.isValid)
                if localTime >= args.predict_seconds or validFinal >= args.min_valid_frames:
                    finalPrediction, lastFeatureRow = predictFromFrames(modelPayload, finalFrames)
                    if finalPrediction is not None:
                        print(
                            f"最终预测: {finalPrediction['predictedWeightG']:.3f} g, "
                            f"最近档位: {finalPrediction['nearestWeightG']:.3f} g",
                            flush=True,
                        )
                    collecting = False
            render = drawPrediction(
                measurement,
                livePrediction,
                finalPrediction,
                lastFeatureRow,
                collecting,
                validFinal,
                args,
            )
            cv2.imshow(windowName, render)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and not collecting:
                finalFrames.clear()
                finalStart = time.time()
                finalPrediction = None
                collecting = True
                print("开始采集稳定预测窗口...", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
