from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionResult:
    centerPixel: Optional[Tuple[float, float]]
    confidence: float
    status: str


class MidpointTargetDetector:
    def __init__(
        self,
        modelPath: str,
        confThreshold: float = 0.25,
        imageSize: int = 640,
        targetClassName: str = "midpoint_marker",
        fallbackArucoId: int = 42,
    ) -> None:
        self.confThreshold = confThreshold
        self.imageSize = imageSize
        self.targetClassName = targetClassName
        self.fallbackArucoId = fallbackArucoId

        self.yoloModel = None
        modelFile = Path(modelPath)
        if modelFile.exists():
            try:
                from ultralytics import YOLO

                self.yoloModel = YOLO(str(modelFile))
            except Exception:
                self.yoloModel = None

        fallbackDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        fallbackParams = cv2.aruco.DetectorParameters()
        self.fallbackArucoDetector = cv2.aruco.ArucoDetector(fallbackDict, fallbackParams)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self.yoloModel is not None:
            yoloResult = self._detectWithYolo(frame)
            if yoloResult.centerPixel is not None:
                return yoloResult

        return self._detectWithArucoFallback(frame)

    def _detectWithYolo(self, frame: np.ndarray) -> DetectionResult:
        prediction = self.yoloModel.predict(
            frame,
            conf=self.confThreshold,
            imgsz=self.imageSize,
            verbose=False,
        )[0]

        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return DetectionResult(centerPixel=None, confidence=0.0, status="yolo-no-target")

        names = prediction.names
        bestScore = -1.0
        bestCenter = None

        xyxy = boxes.xyxy.cpu().numpy()
        clsArray = boxes.cls.cpu().numpy().astype(int)
        confArray = boxes.conf.cpu().numpy()

        for box, classId, score in zip(xyxy, clsArray, confArray):
            className = names.get(classId, str(classId)) if isinstance(names, dict) else str(classId)
            if self.targetClassName and className != self.targetClassName:
                continue
            if score > bestScore:
                x1, y1, x2, y2 = box.tolist()
                bestCenter = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                bestScore = float(score)

        if bestCenter is None:
            # 没有命中目标类别时，退化为最高置信度框，避免空输出。
            bestIndex = int(np.argmax(confArray))
            x1, y1, x2, y2 = xyxy[bestIndex].tolist()
            bestCenter = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
            bestScore = float(confArray[bestIndex])
            return DetectionResult(centerPixel=bestCenter, confidence=bestScore, status="yolo-fallback-top1")

        return DetectionResult(centerPixel=bestCenter, confidence=bestScore, status="yolo")

    def _detectWithArucoFallback(self, frame: np.ndarray) -> DetectionResult:
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.fallbackArucoDetector.detectMarkers(grayFrame)

        if ids is None:
            return DetectionResult(centerPixel=None, confidence=0.0, status="fallback-no-target")

        for markerCorners, markerIdRaw in zip(corners, ids.flatten()):
            markerId = int(markerIdRaw)
            if markerId != self.fallbackArucoId:
                continue
            points = markerCorners.reshape(4, 2)
            center = points.mean(axis=0)
            return DetectionResult(
                centerPixel=(float(center[0]), float(center[1])),
                confidence=0.6,
                status="fallback-aruco",
            )

        return DetectionResult(centerPixel=None, confidence=0.0, status="fallback-no-target")
