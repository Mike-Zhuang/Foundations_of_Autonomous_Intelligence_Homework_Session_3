from __future__ import annotations

from dataclasses import dataclass
import math
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

        arucoResult = self._detectWithArucoFallback(frame)
        if arucoResult.centerPixel is not None:
            return arucoResult

        return self._detectWithCircleFallback(frame)

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

    def _detectWithCircleFallback(self, frame: np.ndarray) -> DetectionResult:
        """检测同心圆标记（黑色外圆 + 白色内圆）"""
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)
        _, binaryInv = cv2.threshold(blurFrame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(binaryInv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            return DetectionResult(centerPixel=None, confidence=0.0, status="fallback-no-target")

        hierarchy = hierarchy[0]
        frameArea = float(frame.shape[0] * frame.shape[1])
        minArea = frameArea * 0.0002
        maxArea = frameArea * 0.2

        bestScore = -1.0
        bestCenter: Optional[Tuple[float, float]] = None

        for contourIndex, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < minArea or area > maxArea:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue

            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.65:
                continue

            childIndex = hierarchy[contourIndex][2]
            if childIndex < 0:
                continue

            # 选择面积最大的内轮廓作为白色内圆边界
            innerBestIndex = -1
            innerBestArea = 0.0
            currentChild = childIndex
            while currentChild >= 0:
                childArea = cv2.contourArea(contours[currentChild])
                if childArea > innerBestArea:
                    innerBestArea = childArea
                    innerBestIndex = currentChild
                currentChild = hierarchy[currentChild][0]

            if innerBestIndex < 0 or innerBestArea <= 1.0:
                continue

            if innerBestArea / area < 0.03 or innerBestArea / area > 0.6:
                continue

            (outerX, outerY), outerRadius = cv2.minEnclosingCircle(contour)
            (innerX, innerY), innerRadius = cv2.minEnclosingCircle(contours[innerBestIndex])
            if outerRadius <= 1.0 or innerRadius <= 0.5:
                continue

            radiusRatio = innerRadius / outerRadius
            if radiusRatio < 0.18 or radiusRatio > 0.58:
                continue

            centerDistance = math.hypot(outerX - innerX, outerY - innerY)
            concentricPenalty = centerDistance / max(outerRadius, 1e-6)
            if concentricPenalty > 0.25:
                continue

            # 分数兼顾圆度与同心性，取最优候选
            score = circularity * (1.0 - concentricPenalty)
            if score > bestScore:
                bestScore = score
                bestCenter = ((outerX + innerX) * 0.5, (outerY + innerY) * 0.5)

        if bestCenter is None:
            return DetectionResult(centerPixel=None, confidence=0.0, status="fallback-no-target")

        refinedCenter = self._refineCenterSubPixel(grayFrame, bestCenter)
        confidence = float(min(max(bestScore, 0.0), 1.0))
        return DetectionResult(centerPixel=refinedCenter, confidence=confidence, status="fallback-circle")

    def _refineCenterSubPixel(self, grayFrame: np.ndarray, center: Tuple[float, float]) -> Tuple[float, float]:
        centerArray = np.asarray([[center]], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 25, 0.01)
        try:
            cv2.cornerSubPix(grayFrame, centerArray, (7, 7), (-1, -1), criteria)
            x, y = centerArray[0, 0]
            return float(x), float(y)
        except cv2.error:
            return center
