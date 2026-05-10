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
from bridge_ai.bridge_task_models import predictWeightFromPhone
from bridge_ai.weight_model import loadModel, predictWeight


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时挠度检测 + 砝码重量模型预测")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model-path", required=True, help="best_weight_model.pth/json 或三任务 model_bundle.pth/json")
    parser.add_argument("--task", choices=["weight-from-phone"], default="weight-from-phone", help="三任务模型包中的预测任务")
    parser.add_argument(
        "--model-choice",
        choices=["auto", "mlp", "monotonic", "ridge", "chained"],
        default="auto",
        help="三任务模型包候选选择。auto 使用训练时推荐；mlp 强制神经网络；monotonic 强制单调函数逼近器",
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
    parser.add_argument("--predict-seconds", type=float, default=6.0)
    parser.add_argument("--min-valid-frames", type=int, default=80)
    parser.add_argument("--live-window-frames", type=int, default=5, help="实时重量预测使用的短滑动窗口帧数")
    parser.add_argument("--weight-smooth-window", type=int, default=3, help="实时重量显示的轻量平滑帧数")
    parser.add_argument("--zero-deflection-threshold-mm", type=float, default=0.3, help="小于该挠度认为未加载，避免模型外推负重量")
    parser.add_argument("--min-output-weight-g", type=float, default=0.0, help="预测重量显示/输出下限")
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
        targetOnlyFallback=args.target_only_fallback == "true",
        staticPoseCorrection=args.static_pose_correction == "true",
        staticAssistMinPoints=args.static_assist_min_points,
    )


def printGuide(args: argparse.Namespace, modelPayload: dict) -> None:
    print("\n================ 实时重量预测 ================", flush=True)
    print("阶段 1：空载固定桥和相机，在视频窗口按 s 锁定基线。", flush=True)
    print("阶段 2：放上未知重量，等读数稳定。窗口会连续显示实时估计。", flush=True)
    print("阶段 3：按 s 采集稳定窗口并输出最终预测；按 q 退出。", flush=True)
    print(f"模型: {args.model_path}", flush=True)
    if modelPayload.get("bundleType") == "bridge_task_models":
        taskPayload = modelPayload["tasks"]["weight_from_phone"]
        print("模型包: bridge_task_models", flush=True)
        print(f"任务: {args.task}, 推荐候选: {taskPayload['recommended']}", flush=True)
        print(f"实际候选选择: {args.model_choice}", flush=True)
    else:
        print(f"模型类型: {modelPayload.get('modelType')}", flush=True)
        print(f"训练档位(g): {modelPayload.get('weightsG')}", flush=True)
    print("============================================\n", flush=True)


def formatMaybe(value: object, digits: int = 3) -> str:
    if value is None:
        return "nan"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def getMeasurementRawMm(measurement: MeasurementFrame) -> float | None:
    rawMm = measurement.state.rawMm
    if rawMm is None or not np.isfinite(rawMm):
        return None
    return float(rawMm)


def updateShortFilter(rawMm: float | None, rawWindow: deque[float], windowSize: int) -> float | None:
    if rawMm is None:
        return None
    rawWindow.append(float(rawMm))
    while len(rawWindow) > max(1, windowSize):
        rawWindow.popleft()
    values = np.asarray(rawWindow, dtype=np.float64)
    if values.size == 1:
        return float(values[0])
    return float(0.7 * np.median(values) + 0.3 * np.mean(values))


def makeSampleFrame(measurement: MeasurementFrame, localTime: float, deflectionMm: float | None = None) -> WeightSampleFrame:
    selectedDeflection = deflectionMm if deflectionMm is not None else measurement.state.filteredMm
    isValid = selectedDeflection is not None and measurement.state.status.startswith("tracking")
    return WeightSampleFrame(
        timeSec=localTime,
        deflectionMm=float(selectedDeflection) if selectedDeflection is not None else float("nan"),
        confidence=float(measurement.detection.confidence),
        quality=computeQuality(
            measurement.homographyResult.reprojectionRmsePx,
            measurement.homographyResult.inlierRatio,
            measurement.homographyResult.usedPointCount,
        ),
        isValid=isValid,
    )


def predictFromFrames(modelPayload: dict, frames: list[WeightSampleFrame], modelChoice: str = "auto") -> tuple[dict | None, dict | None]:
    validCount = sum(1 for frame in frames if frame.isValid)
    if validCount < 5:
        return None, None
    featureRow = computeWindowFeatures(frames, sampleId="predict_window")
    if modelPayload.get("bundleType") == "bridge_task_models":
        return predictWeightFromPhone(modelPayload, featureRow, modelPreference=modelChoice), featureRow
    return predictWeight(modelPayload, featureRow), featureRow


def knownWeights(modelPayload: dict) -> list[float]:
    if modelPayload.get("bundleType") == "bridge_task_models":
        task = modelPayload.get("tasks", {}).get("weight_from_phone", {})
        for candidate in task.get("candidates", []):
            if candidate.get("modelType") == task.get("recommended") and candidate.get("weightsG"):
                return [float(value) for value in candidate["weightsG"]]
        for candidate in task.get("candidates", []):
            if candidate.get("weightsG"):
                return [float(value) for value in candidate["weightsG"]]
        return []
    return [float(value) for value in modelPayload.get("weightsG", [])]


def nearestKnownWeight(weightG: float, weights: list[float]) -> tuple[float, int]:
    if not weights:
        return float(weightG), -1
    arr = np.asarray(weights, dtype=np.float64)
    index = int(np.argmin(np.abs(arr - weightG)))
    return float(arr[index]), index


def applyWeightGuards(
    prediction: dict | None,
    featureRow: dict | None,
    args: argparse.Namespace,
    weights: list[float],
) -> dict | None:
    if prediction is None:
        return None
    guarded = dict(prediction)
    deflectionMean = None if featureRow is None else float(featureRow.get("deflectionMeanMm", float("nan")))
    if deflectionMean is not None and np.isfinite(deflectionMean) and abs(deflectionMean) < args.zero_deflection_threshold_mm:
        guarded["predictedWeightG"] = 0.0
        guarded["nearestWeightG"] = 0.0
        guarded["nearestIndex"] = -1
        guarded["guardReason"] = "zero-deflection"
        return guarded

    predWeight = float(guarded.get("predictedWeightG", 0.0))
    if predWeight < args.min_output_weight_g:
        predWeight = float(args.min_output_weight_g)
        guarded["guardReason"] = "min-output-clamp"
    guarded["predictedWeightG"] = predWeight
    nearestWeight, nearestIndex = nearestKnownWeight(predWeight, weights)
    guarded["nearestWeightG"] = nearestWeight
    guarded["nearestIndex"] = nearestIndex
    return guarded


def smoothWeightPrediction(prediction: dict | None, weightWindow: deque[float], weights: list[float]) -> dict | None:
    if prediction is None:
        return None
    predWeight = float(prediction["predictedWeightG"])
    weightWindow.append(predWeight)
    values = np.asarray(weightWindow, dtype=np.float64)
    smoothedWeight = float(0.7 * np.median(values) + 0.3 * np.mean(values))
    smoothed = dict(prediction)
    smoothed["rawPredictedWeightG"] = predWeight
    smoothed["predictedWeightG"] = smoothedWeight
    nearestWeight, nearestIndex = nearestKnownWeight(smoothedWeight, weights)
    if smoothed.get("guardReason") == "zero-deflection":
        nearestWeight, nearestIndex = 0.0, -1
    smoothed["nearestWeightG"] = nearestWeight
    smoothed["nearestIndex"] = nearestIndex
    return smoothed


def drawSparkline(
    panel: np.ndarray,
    values: list[float],
    topLeft: tuple[int, int],
    size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0 = topLeft
    width, height = size
    cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + height), (90, 90, 90), 1)
    if len(values) < 2:
        return
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return
    vMin = float(np.min(arr))
    vMax = float(np.max(arr))
    if abs(vMax - vMin) < 1e-6:
        vMax = vMin + 1e-6
    points: list[tuple[int, int]] = []
    for index, value in enumerate(arr):
        x = x0 + int(index * (width - 1) / max(arr.size - 1, 1))
        y = y0 + int((vMax - float(value)) / (vMax - vMin) * (height - 1))
        points.append((x, y))
    for index in range(1, len(points)):
        cv2.line(panel, points[index - 1], points[index], color, 2, cv2.LINE_AA)


def drawPrediction(
    measurement: MeasurementFrame,
    livePrediction: dict | None,
    finalPrediction: dict | None,
    featureRow: dict | None,
    collecting: bool,
    validCount: int,
    args: argparse.Namespace,
    deflectionHistoryMm: list[float],
    weightHistoryG: list[float],
) -> np.ndarray:
    overlay = drawOverlay(measurement.frame, measurement.homographyResult, measurement.detection.centerPixel)
    liveText = "Live weight(g): waiting"
    nearestText = "Nearest(g): waiting"
    if livePrediction is not None:
        guardReason = livePrediction.get("guardReason")
        suffix = " (no load)" if guardReason == "zero-deflection" else ""
        liveText = f"Live weight(g): {livePrediction['predictedWeightG']:.2f}{suffix}"
        nearestText = f"Nearest(g): {livePrediction['nearestWeightG']:.2f}"
    lines = [
        f"Deflection(mm): {formatMaybe(measurement.state.filteredMm)}",
        f"Status: {measurement.state.status}",
        liveText,
        nearestText,
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
        f"Panel: ready from startup",
        f"Measure: {measurement.measurementMethod}",
        f"Detect: {measurement.detection.status}",
        f"Conf: {measurement.detection.confidence:.3f}",
        f"Used points: {measurement.homographyResult.usedPointCount}",
        f"RMSE(px): {measurement.homographyResult.reprojectionRmsePx}",
        f"Inlier: {measurement.homographyResult.inlierRatio}",
        f"Static axis: {measurement.staticAxisSource}",
        f"Plane tilt(deg): {measurement.staticPoseInfo.planeTiltDeg if measurement.staticPoseInfo else None}",
        f"Static roll(deg): {measurement.staticPoseInfo.rollDeg if measurement.staticPoseInfo else None}",
        f"Raw(mm): {formatMaybe(measurement.state.rawMm)}",
        f"Filtered(mm): {formatMaybe(measurement.state.filteredMm)}",
        f"Live window(fr): {args.live_window_frames}",
        f"Weight smooth(fr): {args.weight_smooth_window}",
        f"Zero gate(mm): {args.zero_deflection_threshold_mm}",
        f"Feature std(mm): {featureRow.get('deflectionStdMm') if featureRow else None}",
        f"Feature drift(mm/min): {featureRow.get('driftMmPerMin') if featureRow else None}",
    ]
    y = 30
    for line in debugLines:
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        y += 24
    cv2.putText(panel, "Deflection curve (mm)", (16, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 160), 1, cv2.LINE_AA)
    drawSparkline(panel, deflectionHistoryMm, (16, y + 18), (368, 95), (0, 255, 120))
    y += 132
    cv2.putText(panel, "Weight curve (g)", (16, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 220, 255), 1, cv2.LINE_AA)
    drawSparkline(panel, weightHistoryG, (16, y + 18), (368, 95), (255, 180, 60))
    return np.hstack([overlay, panel])


def waitForBaseline(capture: cv2.VideoCapture, measurer: RealtimeDeflectionMeasurer, fps: float, windowName: str, args: argparse.Namespace) -> None:
    print("请保持空载静止，然后在视频窗口按 s 开始基线。", flush=True)
    lastTime = time.time()
    started = False
    deflectionHistoryMm: deque[float] = deque(maxlen=180)
    while True:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("视频流中断，无法完成基线。")
        now = time.time()
        measurement = measurer.process(frame, dtSec=min(now - lastTime, 1.0 / max(fps, 1.0)))
        lastTime = now
        if measurement.state.filteredMm is not None:
            deflectionHistoryMm.append(float(measurement.state.filteredMm))
        render = drawPrediction(
            measurement,
            livePrediction=None,
            finalPrediction=None,
            featureRow=None,
            collecting=False,
            validCount=0,
            args=args,
            deflectionHistoryMm=list(deflectionHistoryMm),
            weightHistoryG=[],
        )
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
        waitForBaseline(capture, measurer, fps, windowName, args)
        weights = knownWeights(modelPayload)
        rollingFrames: deque[WeightSampleFrame] = deque(maxlen=max(args.live_window_frames, 5))
        liveRawWindow: deque[float] = deque(maxlen=max(args.live_window_frames, 1))
        finalRawWindow: deque[float] = deque(maxlen=max(args.live_window_frames, 1))
        weightSmoothWindow: deque[float] = deque(maxlen=max(args.weight_smooth_window, 1))
        deflectionHistoryMm: deque[float] = deque(maxlen=180)
        weightHistoryG: deque[float] = deque(maxlen=180)
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
            livePrediction, featureRow = predictFromFrames(modelPayload, list(rollingFrames), args.model_choice)
            livePrediction = applyWeightGuards(livePrediction, featureRow, args, weights)
            livePrediction = smoothWeightPrediction(livePrediction, weightSmoothWindow, weights)
            if featureRow is not None:
                lastFeatureRow = featureRow
            if liveFilteredMm is not None:
                deflectionHistoryMm.append(float(liveFilteredMm))
            if livePrediction is not None:
                weightHistoryG.append(float(livePrediction["predictedWeightG"]))

            validFinal = 0
            if collecting:
                localTime = 0.0 if finalStart is None else now - finalStart
                finalFilteredMm = updateShortFilter(getMeasurementRawMm(measurement), finalRawWindow, args.live_window_frames)
                finalFrames.append(makeSampleFrame(measurement, localTime=localTime, deflectionMm=finalFilteredMm))
                validFinal = sum(1 for item in finalFrames if item.isValid)
                if localTime >= args.predict_seconds or validFinal >= args.min_valid_frames:
                    finalPrediction, lastFeatureRow = predictFromFrames(modelPayload, finalFrames, args.model_choice)
                    finalPrediction = applyWeightGuards(finalPrediction, lastFeatureRow, args, weights)
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
                list(deflectionHistoryMm),
                list(weightHistoryG),
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
                print("开始采集稳定预测窗口...", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
