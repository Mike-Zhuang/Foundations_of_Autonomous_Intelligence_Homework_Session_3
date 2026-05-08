from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from bridge_ai.aruco_utils import buildPrecisionArucoDetectorParameters
from bridge_ai.config import MarkerLayout


@dataclass
class HomographyResult:
    homography: Optional[np.ndarray]
    foundMarkerIds: list[int]
    usedPointCount: int
    reprojectionRmsePx: Optional[float]
    inlierRatio: Optional[float]
    imagePoints: list[Tuple[float, float]]
    worldPoints: list[Tuple[float, float]]


@dataclass
class StaticPoseInfo:
    rvec: np.ndarray
    tvec: np.ndarray
    usedPointCount: int
    reprojectionRmsePx: Optional[float]
    planeTiltDeg: float
    xTiltDeg: float
    yTiltDeg: float
    rollDeg: float


class ArucoStaticSolver:
    def __init__(self, dictionaryName: str) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionaryName))
        params = buildPrecisionArucoDetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def solveHomography(self, frame: np.ndarray, layout: MarkerLayout) -> HomographyResult:
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(grayFrame)

        if ids is None or len(ids) == 0:
            return HomographyResult(
                homography=None,
                foundMarkerIds=[],
                usedPointCount=0,
                reprojectionRmsePx=None,
                inlierRatio=None,
                imagePoints=[],
                worldPoints=[],
            )

        imagePoints: list[Tuple[float, float]] = []
        worldPoints: list[Tuple[float, float]] = []
        foundIds: list[int] = []

        for markerCorners, markerIdRaw in zip(corners, ids.flatten()):
            markerId = int(markerIdRaw)
            markerDef = layout.markers.get(markerId)
            if markerDef is None:
                continue

            foundIds.append(markerId)
            pixelCorners = markerCorners.reshape(4, 2)
            worldCorners = markerDef.worldCornersMm()

            for pixelCorner, worldCorner in zip(pixelCorners, worldCorners):
                imagePoints.append((float(pixelCorner[0]), float(pixelCorner[1])))
                worldPoints.append((float(worldCorner[0]), float(worldCorner[1])))

        if len(imagePoints) < 4:
            return HomographyResult(
                homography=None,
                foundMarkerIds=foundIds,
                usedPointCount=len(imagePoints),
                reprojectionRmsePx=None,
                inlierRatio=None,
                imagePoints=imagePoints,
                worldPoints=worldPoints,
            )

        imageArray = np.asarray(imagePoints, dtype=np.float32)
        worldArray = np.asarray(worldPoints, dtype=np.float32)

        if len(imagePoints) == 4:
            homography, mask = cv2.findHomography(imageArray, worldArray, 0)
        else:
            homography, mask = cv2.findHomography(imageArray, worldArray, cv2.RANSAC, ransacReprojThreshold=3.5)
        if homography is None:
            return HomographyResult(
                homography=None,
                foundMarkerIds=foundIds,
                usedPointCount=len(imagePoints),
                reprojectionRmsePx=None,
                inlierRatio=None,
                imagePoints=imagePoints,
                worldPoints=worldPoints,
            )

        reprojection = cv2.perspectiveTransform(imageArray.reshape(-1, 1, 2), homography).reshape(-1, 2)
        error = reprojection - worldArray
        rmse = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
        inlierRatio = None
        if mask is not None and mask.size > 0:
            inlierRatio = float(np.mean(mask.astype(np.float32)))

        return HomographyResult(
            homography=homography,
            foundMarkerIds=foundIds,
            usedPointCount=len(imagePoints),
            reprojectionRmsePx=rmse,
            inlierRatio=inlierRatio,
            imagePoints=imagePoints,
            worldPoints=worldPoints,
        )

    @staticmethod
    def pixelToWorld(homography: np.ndarray, pixelPoint: Tuple[float, float]) -> Tuple[float, float]:
        src = np.asarray([[[pixelPoint[0], pixelPoint[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, homography)
        xWorld, yWorld = dst[0, 0]
        return float(xWorld), float(yWorld)


def drawOverlay(frame: np.ndarray, homographyResult: HomographyResult, targetPixel: Optional[Tuple[float, float]]) -> np.ndarray:
    overlay = frame.copy()
    statusColor = (0, 200, 0) if homographyResult.homography is not None else (0, 0, 255)
    cv2.putText(
        overlay,
        f"Static IDs: {homographyResult.foundMarkerIds}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        statusColor,
        2,
        cv2.LINE_AA,
    )

    if targetPixel is not None:
        cv2.circle(overlay, (int(targetPixel[0]), int(targetPixel[1])), 8, (255, 255, 0), 2)

    for imagePoint in homographyResult.imagePoints:
        cv2.circle(overlay, (int(imagePoint[0]), int(imagePoint[1])), 3, (0, 180, 255), -1)

    return overlay


def estimateMarkerPoseTvec(
    markerCorners: np.ndarray,
    markerSizeMm: float,
    cameraMatrix: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    halfSize = markerSizeMm * 0.5
    objectPoints = np.asarray(
        [
            (-halfSize, -halfSize, 0.0),
            (halfSize, -halfSize, 0.0),
            (halfSize, halfSize, 0.0),
            (-halfSize, halfSize, 0.0),
        ],
        dtype=np.float32,
    )
    imagePoints = np.asarray(markerCorners, dtype=np.float32).reshape(4, 2)
    distCoeffs = np.zeros((5, 1), dtype=np.float64)
    ok, _, tvec = cv2.solvePnP(
        objectPoints,
        imagePoints,
        cameraMatrix,
        distCoeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    tx, ty, tz = tvec.reshape(3).tolist()
    return float(tx), float(ty), float(tz)


def estimateStaticBoardPose(
    homographyResult: HomographyResult,
    cameraMatrix: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    poseInfo = estimateStaticPoseInfo(homographyResult, cameraMatrix)
    if poseInfo is None:
        return None
    return poseInfo.rvec, poseInfo.tvec


def estimateStaticPoseInfo(
    homographyResult: HomographyResult,
    cameraMatrix: np.ndarray,
    minPointCount: int = 4,
) -> Optional[StaticPoseInfo]:
    if len(homographyResult.imagePoints) < minPointCount or len(homographyResult.worldPoints) < minPointCount:
        return None

    objectPoints = np.asarray(
        [(xWorld, yWorld, 0.0) for xWorld, yWorld in homographyResult.worldPoints],
        dtype=np.float32,
    )
    imagePoints = np.asarray(homographyResult.imagePoints, dtype=np.float32)
    distCoeffs = np.zeros((5, 1), dtype=np.float64)

    ok = False
    rvec = None
    tvec = None
    if len(imagePoints) >= 8 and hasattr(cv2, "solvePnPRansac"):
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints,
            imagePoints,
            cameraMatrix,
            distCoeffs,
            iterationsCount=100,
            reprojectionError=5.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok and inliers is not None and len(inliers) < minPointCount:
            ok = False

    if not ok:
        flags = cv2.SOLVEPNP_ITERATIVE
        if len(imagePoints) == 4 and hasattr(cv2, "SOLVEPNP_IPPE"):
            flags = cv2.SOLVEPNP_IPPE
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints,
            imagePoints,
            cameraMatrix,
            distCoeffs,
            flags=flags,
        )

    if not ok:
        return None

    projected, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
    projected2d = projected.reshape(-1, 2)
    error = projected2d - imagePoints.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))

    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    normalNorm = float(np.linalg.norm(normal))
    if normalNorm < 1e-9:
        return None
    normal = normal / normalNorm
    absNz = float(np.clip(abs(normal[2]), 0.0, 1.0))
    planeTiltDeg = float(np.degrees(np.arccos(absNz)))
    xTiltDeg = float(np.degrees(np.arctan2(normal[0], max(abs(normal[2]), 1e-9))))
    yTiltDeg = float(np.degrees(np.arctan2(-normal[1], max(abs(normal[2]), 1e-9))))

    axisResult = estimateStaticPosePixelAxis(
        StaticPoseInfo(
            rvec=rvec,
            tvec=tvec,
            usedPointCount=len(imagePoints),
            reprojectionRmsePx=rmse,
            planeTiltDeg=planeTiltDeg,
            xTiltDeg=xTiltDeg,
            yTiltDeg=yTiltDeg,
            rollDeg=0.0,
        ),
        cameraMatrix,
    )
    rollDeg = 0.0
    if axisResult is not None:
        _, axisUnit = axisResult
        # axisUnit 是静态平面 Y 轴在图像上的方向；这里给出它相对屏幕竖直向下的夹角。
        rollDeg = float(np.degrees(np.arctan2(axisUnit[0], axisUnit[1])))

    return StaticPoseInfo(
        rvec=rvec,
        tvec=tvec,
        usedPointCount=len(imagePoints),
        reprojectionRmsePx=rmse,
        planeTiltDeg=planeTiltDeg,
        xTiltDeg=xTiltDeg,
        yTiltDeg=yTiltDeg,
        rollDeg=rollDeg,
    )


def estimateStaticPosePixelAxis(
    poseInfo: StaticPoseInfo,
    cameraMatrix: np.ndarray,
    originWorldMm: Tuple[float, float] = (105.0, 148.0),
    yStepMm: float = 40.0,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    objectPoints = np.asarray(
        [
            (originWorldMm[0], originWorldMm[1], 0.0),
            (originWorldMm[0], originWorldMm[1] + yStepMm, 0.0),
        ],
        dtype=np.float32,
    )
    imagePoints, _ = cv2.projectPoints(
        objectPoints,
        poseInfo.rvec,
        poseInfo.tvec,
        cameraMatrix,
        np.zeros((5, 1), dtype=np.float64),
    )
    points = imagePoints.reshape(2, 2)
    axis = points[1] - points[0]
    axisNorm = float(np.linalg.norm(axis))
    if axisNorm < 1e-6:
        return None
    axisUnit = axis / axisNorm
    return (
        (float(points[0][0]), float(points[0][1])),
        (float(axisUnit[0]), float(axisUnit[1])),
    )


def cameraPointToStaticWorld(
    cameraPointMm: Tuple[float, float, float],
    staticBoardPose: Tuple[np.ndarray, np.ndarray],
) -> Tuple[float, float, float]:
    rvec, tvec = staticBoardPose
    rotation, _ = cv2.Rodrigues(rvec)
    cameraPoint = np.asarray(cameraPointMm, dtype=np.float64).reshape(3, 1)
    worldPoint = rotation.T @ (cameraPoint - tvec.reshape(3, 1))
    xWorld, yWorld, zWorld = worldPoint.reshape(3).tolist()
    return float(xWorld), float(yWorld), float(zWorld)


def estimateStaticPixelAxis(
    homography: np.ndarray,
    originWorldMm: Tuple[float, float] = (105.0, 148.0),
    yStepMm: float = 40.0,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    try:
        inverseHomography = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None

    worldPoints = np.asarray(
        [
            [[originWorldMm[0], originWorldMm[1]]],
            [[originWorldMm[0], originWorldMm[1] + yStepMm]],
        ],
        dtype=np.float32,
    )
    pixelPoints = cv2.perspectiveTransform(worldPoints, inverseHomography).reshape(2, 2)
    originPixel = pixelPoints[0]
    yPixel = pixelPoints[1]
    axis = yPixel - originPixel
    axisNorm = float(np.linalg.norm(axis))
    if axisNorm < 1e-6:
        return None

    axisUnit = axis / axisNorm
    return (
        (float(originPixel[0]), float(originPixel[1])),
        (float(axisUnit[0]), float(axisUnit[1])),
    )


def estimateTargetLocalCoordinateY(
    markerCorners: np.ndarray,
    markerSizeMm: float,
    originPixel: Tuple[float, float],
    axisUnit: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    points = np.asarray(markerCorners, dtype=np.float32).reshape(4, 2)
    centerPixel = points.mean(axis=0)
    origin = np.asarray(originPixel, dtype=np.float32)
    axis = np.asarray(axisUnit, dtype=np.float32)
    axisNorm = float(np.linalg.norm(axis))
    if axisNorm < 1e-6:
        return None
    axis = axis / axisNorm

    # 用 ID42 四角建立“图像像素 -> 标记真实毫米坐标”的局部透视映射。
    # 然后沿静态 Y 方向做有限差分，得到目标中心附近该方向的真实局部比例尺。
    # 这样比直接取某条边的像素长度更稳，能处理 ID42 有旋转或轻微透视的情况。
    halfSize = markerSizeMm * 0.5
    markerWorld = np.asarray(
        [
            [-halfSize, -halfSize],
            [halfSize, -halfSize],
            [halfSize, halfSize],
            [-halfSize, halfSize],
        ],
        dtype=np.float32,
    )
    imageToMarker = cv2.getPerspectiveTransform(points.astype(np.float32), markerWorld)
    stepPx = 10.0
    probePixels = np.asarray(
        [
            [[float(centerPixel[0]), float(centerPixel[1])]],
            [[float(centerPixel[0] + axis[0] * stepPx), float(centerPixel[1] + axis[1] * stepPx)]],
        ],
        dtype=np.float32,
    )
    probeMarker = cv2.perspectiveTransform(probePixels, imageToMarker).reshape(2, 2)
    mmPerStep = float(np.linalg.norm(probeMarker[1] - probeMarker[0]))
    if mmPerStep < 1e-6:
        return None

    pxPerMm = stepPx / mmPerStep
    positionPx = float(np.dot(centerPixel - origin, axis))
    return positionPx, pxPerMm
