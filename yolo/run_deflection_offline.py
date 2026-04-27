from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

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
    return parser.parse_args()


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
    estimator = DeflectionEstimator(baselineFrames=args.baseline_frames)

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
    videoWriter = createVideoWriter(outputVideo, frameWidth, frameHeight, fps)

    frameIndex = 0
    filteredSeries: list[float] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timeSec = frameIndex / fps
            frameIndex += 1

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
                f"Status: {state.status}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if videoWriter is not None:
                videoWriter.write(overlay)

            if args.preview:
                cv2.imshow("Bridge Deflection Offline", overlay)
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
