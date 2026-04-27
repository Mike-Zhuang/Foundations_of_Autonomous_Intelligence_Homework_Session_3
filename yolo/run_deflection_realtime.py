from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2

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
    return parser.parse_args()


def formatNumber(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "nan"
    return f"{value:.{digits}f}"


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
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1.0:
        fps = 30.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csvPath = Path(args.output_csv) if args.output_csv else Path(f"yolo/results/realtime_{timestamp}.csv")
    csvWriter = CsvWriter(csvPath)

    frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoPath = Path(args.save_video) if args.save_video else None
    videoWriter = createVideoWriter(videoPath, frameWidth, frameHeight, fps)

    windowName = "Bridge Deflection Realtime"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    startTime = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            elapsed = time.time() - startTime
            homographyResult = solver.solveHomography(frame, layout)
            detection = detector.detect(frame)

            worldPoint = None
            if homographyResult.homography is not None and detection.centerPixel is not None:
                worldPoint = solver.pixelToWorld(homographyResult.homography, detection.centerPixel)

            statusHint = detection.status
            if homographyResult.homography is None:
                statusHint = "no-static-markers"

            state = estimator.update(
                worldYmm=(worldPoint[1] if worldPoint is not None else None),
                timeSec=elapsed,
                confidence=detection.confidence,
                statusHint=statusHint,
            )
            csvWriter.write(state)

            overlay = drawOverlay(frame, homographyResult, detection.centerPixel)
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

            if videoWriter is not None:
                videoWriter.write(overlay)

            cv2.imshow(windowName, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
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
