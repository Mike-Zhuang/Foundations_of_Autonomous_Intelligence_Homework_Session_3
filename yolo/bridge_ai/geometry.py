from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

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


class ArucoStaticSolver:
    def __init__(self, dictionaryName: str) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionaryName))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
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

        if len(imagePoints) < 8:
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

        homography, mask = cv2.findHomography(imageArray, worldArray, cv2.RANSAC, ransacReprojThreshold=2.5)
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
