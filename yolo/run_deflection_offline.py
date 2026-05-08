from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from bridge_ai.calibration import loadCalibration, undistortFrame
from bridge_ai.config import loadLayout
from bridge_ai.deflection import DeflectionEstimator
from bridge_ai.detection import MidpointTargetDetector
from bridge_ai.geometry import ArucoStaticSolver, drawOverlay
from bridge_ai.io_utils import CsvWriter, createVideoWriter, openVideoSource


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="桥梁跨中挠度离线分析")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 模型路径")
    parser.add_argument("--layout", default="yolo/artifacts/static_marker_layout.json", help="静态标记布局 json")
    parser.add_argument("--target-class", default="midpoint_marker", help="跨中目标类别名")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--baseline-frames", type=int, default=60)
    parser.add_argument("--output-csv", default="", help="输出 CSV 路径")
    parser.add_argument("--output-summary", default="", help="输出 summary json 路径")
    parser.add_argument("--output-video", default="", help="输出叠加视频路径")
    parser.add_argument("--preview", action="store_true", help="是否显示预览窗口")
    parser.add_argument("--overlay-level", choices=["minimal", "balanced", "debug"], default="debug")
    parser.add_argument("--calibration-mode", choices=["off", "use"], default="use")
    parser.add_argument("--calibration-file", default="yolo/artifacts/camera_calibration.npz")
    parser.add_argument("--min-used-points", type=int, default=12)
    parser.add_argument("--max-rmse", type=float, default=4.0)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.45)
    parser.add_argument(
        "--deflection-scale",
        type=float,
        default=1.0,
        help="赛前固定比例修正系数。例：赛前验证真值 100mm、显示 108.5mm，则填 100/108.5=0.922",
    )
    return parser.parse_args()


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


def computeSummary(filteredMm: list[float]) -> dict:
    if not filteredMm:
        return {
            "validFrameCount": 0,
            "maxDeflectionMm": None,
            "minDeflectionMm": None,
            "meanDeflectionMm": None,
            "stdDeflectionMm": None,
        }

    arr = np.asarray(filteredMm, dtype=np.float64)
    return {
        "validFrameCount": int(arr.size),
        "maxDeflectionMm": float(np.max(arr)),
        "minDeflectionMm": float(np.min(arr)),
        "meanDeflectionMm": float(np.mean(arr)),
        "stdDeflectionMm": float(np.std(arr)),
    }


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
    estimator = DeflectionEstimator(baselineFrames=args.baseline_frames, deflectionScale=args.deflection_scale)

    calibration = None
    if args.calibration_mode == "use":
        calibrationPath = Path(args.calibration_file)
        if calibrationPath.exists():
            calibration = loadCalibration(calibrationPath)
            print(f"加载标定参数成功: {calibrationPath} (RMS={calibration.rms:.4f})")
        else:
            print(f"未找到标定文件 {calibrationPath}，自动降级为不去畸变。")

    capture = openVideoSource(args.video, preferAvfoundation=False)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = 30.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputCsv = Path(args.output_csv) if args.output_csv else Path(f"yolo/results/offline_{timestamp}.csv")
    outputSummary = Path(args.output_summary) if args.output_summary else Path(f"yolo/results/offline_{timestamp}_summary.json")
    outputVideo = Path(args.output_video) if args.output_video else None

    csvWriter = CsvWriter(outputCsv)
    frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panelWidth = 360 if args.overlay_level == "debug" else 0
    videoWriter = createVideoWriter(outputVideo, frameWidth + panelWidth, frameHeight, fps)

    frameIndex = 0
    filteredSeries: list[float] = []

    try:
        while True:
            ok, frameRaw = capture.read()
            if not ok:
                break
            frame = undistortFrame(frameRaw, calibration) if calibration is not None else frameRaw

            timeSec = frameIndex / fps
            frameIndex += 1

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
                timeSec=timeSec,
                confidence=detection.confidence,
                statusHint=statusHint,
            )
            csvWriter.write(state)

            if state.filteredMm is not None and state.status.startswith("tracking"):
                filteredSeries.append(state.filteredMm)

            overlay = drawOverlay(frame, homographyResult, detection.centerPixel)
            cv2.putText(
                overlay,
                f"Deflection(cm): {state.deflectionCm if state.deflectionCm is not None else float('nan'):.3f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 220, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"Status: {localizeStatus(state.status)}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if worldPoint is not None and args.overlay_level in ("balanced", "debug"):
                cv2.putText(
                    overlay,
                    f"World(mm): ({worldPoint[0]:.1f}, {worldPoint[1]:.1f})",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (200, 240, 255),
                    2,
                    cv2.LINE_AA,
                )

            renderFrame = overlay
            if args.overlay_level == "debug":
                panel = np.zeros((overlay.shape[0], panelWidth, 3), dtype=np.uint8)
                panel[:] = (24, 24, 24)
                lines = [
                    f"Undistort: {'ON' if calibration is not None else 'OFF'}",
                    f"Used points: {homographyResult.usedPointCount}",
                    f"Reproj RMSE(px): {homographyResult.reprojectionRmsePx}",
                    f"Inlier ratio: {homographyResult.inlierRatio}",
                    f"Detect src: {localizeDetectionSource(detection.status)}",
                    f"Detect conf: {detection.confidence:.3f}",
                    f"Scale: {args.deflection_scale:.5f}",
                    f"Raw(mm): {state.rawMm}",
                    f"Raw0(mm): {state.unscaledRawMm}",
                    f"Filtered(mm): {state.filteredMm}",
                ]
                y = 30
                for line in lines:
                    cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
                    y += 24
                renderFrame = np.hstack([overlay, panel])

            if videoWriter is not None:
                videoWriter.write(renderFrame)

            if args.preview:
                cv2.imshow("Bridge Deflection Offline", renderFrame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        csvWriter.close()
        capture.release()
        if videoWriter is not None:
            videoWriter.release()
        cv2.destroyAllWindows()

    summary = computeSummary(filteredSeries)
    outputSummary.parent.mkdir(parents=True, exist_ok=True)
    outputSummary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"CSV saved: {outputCsv}")
    print(f"Summary saved: {outputSummary}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
