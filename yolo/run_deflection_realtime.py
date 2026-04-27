from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from bridge_ai.calibration import CameraCalibration, loadCalibration, runCharucoCalibration, saveCalibration, undistortFrame
from bridge_ai.config import loadLayout
from bridge_ai.deflection import DeflectionEstimator
from bridge_ai.detection import MidpointTargetDetector
from bridge_ai.geometry import ArucoStaticSolver, drawOverlay
from bridge_ai.io_utils import CsvWriter, createVideoWriter, openVideoSource


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="桥梁跨中挠度实时测量")
    parser.add_argument("--source", default="0", help="视频源。连续互通相机通常是 0 或 1")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 模型路径")
    parser.add_argument("--layout", default="yolo/artifacts/static_marker_layout.json", help="静态标记布局 json")
    parser.add_argument("--target-class", default="midpoint_marker", help="跨中目标类别名")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸")
    parser.add_argument("--baseline-frames", type=int, default=60, help="基线标定帧数")
    parser.add_argument("--output-csv", default="", help="输出 CSV 路径，空字符串表示自动命名")
    parser.add_argument("--save-video", default="", help="输出叠加视频路径，空字符串表示不保存")
    parser.add_argument("--calibration-mode", choices=["off", "use", "recalibrate"], default="use")
    parser.add_argument("--calibration-file", default="yolo/artifacts/camera_calibration.npz")
    parser.add_argument("--overlay-level", choices=["minimal", "balanced", "debug"], default="debug")
    parser.add_argument("--min-used-points", type=int, default=16)
    parser.add_argument("--max-rmse", type=float, default=2.6)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.65)
    return parser.parse_args()


def formatNumber(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "nan"
    return f"{value:.{digits}f}"


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


def drawHistorySparkline(panel: np.ndarray, values: list[float], topLeft: tuple[int, int], size: tuple[int, int]) -> None:
    x0, y0 = topLeft
    width, height = size
    cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + height), (90, 90, 90), 1)
    if len(values) < 2:
        return

    arr = np.asarray(values, dtype=np.float32)
    vMin = float(np.min(arr))
    vMax = float(np.max(arr))
    if abs(vMax - vMin) < 1e-6:
        vMax = vMin + 1e-6

    points: list[tuple[int, int]] = []
    for idx, value in enumerate(arr):
        x = x0 + int(idx * (width - 1) / max(len(arr) - 1, 1))
        y = y0 + int((vMax - float(value)) / (vMax - vMin) * (height - 1))
        points.append((x, y))

    for idx in range(1, len(points)):
        cv2.line(panel, points[idx - 1], points[idx], (0, 255, 150), 1, cv2.LINE_AA)


def attachDebugPanel(frame: np.ndarray, lines: list[str], historyMm: list[float]) -> np.ndarray:
    panelWidth = 360
    base = frame.copy()
    h, w = base.shape[:2]
    panel = np.zeros((h, panelWidth, 3), dtype=np.uint8)
    panel[:] = (24, 24, 24)

    y = 28
    for line in lines:
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        y += 24

    cv2.putText(panel, "Deflection History (mm)", (16, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    drawHistorySparkline(panel, historyMm, topLeft=(16, y + 20), size=(panelWidth - 32, 120))
    return np.hstack([base, panel])


def main() -> int:
    args = parseArgs()

    layoutPath = Path(args.layout)
    layout = loadLayout(layoutPath if layoutPath.exists() else None)

    solver = ArucoStaticSolver(layout.dictionaryName)
    detector = MidpointTargetDetector(
        modelPath=args.model,
        confThreshold=args.conf,
        imageSize=args.imgsz,
        targetClassName=args.target_class,
    )
    estimator = DeflectionEstimator(baselineFrames=args.baseline_frames)

    capture = openVideoSource(args.source, preferAvfoundation=True)
    calibration = resolveCalibration(args, capture)

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = 30.0

    firstOk, firstFrameRaw = capture.read()
    if not firstOk or firstFrameRaw is None:
        capture.release()
        raise RuntimeError(
            "视频源已打开但无法读取首帧。请检查："
            "1) --source 索引是否正确（可尝试 0/1/2）；"
            "2) macOS 是否已给当前终端/解释器授予摄像头权限；"
            "3) 是否有其他应用占用了摄像头。"
        )

    firstFrame = undistortFrame(firstFrameRaw, calibration) if calibration is not None else firstFrameRaw
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csvPath = Path(args.output_csv) if args.output_csv else Path(f"yolo/results/realtime_{timestamp}.csv")
    csvWriter = CsvWriter(csvPath)

    frameHeight, frameWidth = firstFrame.shape[:2]
    videoPath = Path(args.save_video) if args.save_video else None
    videoWriter = createVideoWriter(videoPath, frameWidth + (360 if args.overlay_level == "debug" else 0), frameHeight, fps)

    windowName = "Bridge Deflection Realtime"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    startTime = time.time()
    historyMm: deque[float] = deque(maxlen=180)

    try:
        frame = firstFrame
        while True:
            if frame is None:
                ok, frameRaw = capture.read()
                if not ok:
                    print("读取视频帧失败，实时测量结束。")
                    break
                frame = undistortFrame(frameRaw, calibration) if calibration is not None else frameRaw

            elapsed = time.time() - startTime
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

            state = estimator.update(
                worldYmm=(worldPoint[1] if worldPoint is not None else None),
                timeSec=elapsed,
                confidence=detection.confidence,
                statusHint=statusHint,
            )
            csvWriter.write(state)
            if state.filteredMm is not None:
                historyMm.append(float(state.filteredMm))

            overlay = drawOverlay(frame, homographyResult, detection.centerPixel)
            if worldPoint is not None and args.overlay_level in ("balanced", "debug"):
                cv2.putText(
                    overlay,
                    f"World(mm): ({worldPoint[0]:.1f}, {worldPoint[1]:.1f})",
                    (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (200, 240, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                overlay,
                f"Deflection(cm): {formatNumber(state.deflectionCm, 3)}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 220, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"Baseline(mm): {formatNumber(state.baselineMm, 2)}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"Status: {state.status}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                "Press q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            if args.overlay_level == "debug":
                lines = [
                    f"Undistort: {'ON' if calibration is not None else 'OFF'}",
                    f"Calibration RMS: {formatNumber(calibration.rms if calibration else None, 4)}",
                    f"Found IDs: {homographyResult.foundMarkerIds}",
                    f"Used points: {homographyResult.usedPointCount}",
                    f"H RMSE(px): {formatNumber(homographyResult.reprojectionRmsePx, 3)}",
                    f"Inlier ratio: {formatNumber(homographyResult.inlierRatio, 3)}",
                    f"Detect src: {detection.status}",
                    f"Detect conf: {detection.confidence:.3f}",
                    f"Target pixel: {detection.centerPixel}",
                ]
                renderFrame = attachDebugPanel(overlay, lines, list(historyMm))
            else:
                renderFrame = overlay

            if videoWriter is not None:
                videoWriter.write(renderFrame)

            cv2.imshow(windowName, renderFrame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            frame = None
    finally:
        csvWriter.close()
        capture.release()
        if videoWriter is not None:
            videoWriter.release()
        cv2.destroyAllWindows()

    print(f"CSV saved: {csvPath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
