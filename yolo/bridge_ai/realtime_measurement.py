from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from bridge_ai.calibration import CameraCalibration, undistortFrame
from bridge_ai.config import MarkerLayout
from bridge_ai.deflection import DeflectionEstimator, DeflectionState, PixelScaleDeflectionEstimator
from bridge_ai.detection import DetectionResult, MidpointTargetDetector
from bridge_ai.geometry import (
    ArucoStaticSolver,
    HomographyResult,
    StaticPoseInfo,
    cameraPointToStaticWorld,
    estimateMarkerPoseTvec,
    estimateStaticBoardPose,
    estimateStaticPixelAxis,
    estimateStaticPoseInfo,
    estimateStaticPosePixelAxis,
    estimateTargetLocalCoordinateY,
)


@dataclass
class MeasurementConfig:
    baselineFrames: int = 60
    filterProfile: str = "stable"
    deadbandMm: float = 0.2
    smoothWindow: int = 9
    localScaleMode: str = "baseline"
    deflectionScale: float = 1.0
    measurementMethod: str = "target-local-scale"
    targetMarkerSizeMm: float = 50.0
    minUsedPoints: int = 12
    maxRmse: float = 4.0
    minInlierRatio: float = 0.45
    targetOnlyFallback: bool = True
    staticPoseCorrection: bool = True
    staticAssistMinPoints: int = 4


@dataclass
class MeasurementFrame:
    frame: np.ndarray
    homographyResult: HomographyResult
    detection: DetectionResult
    state: DeflectionState
    isLowQuality: bool
    measurementMethod: str
    worldPoint: Optional[Tuple[float, float]]
    targetPositionPx: Optional[float]
    targetPxPerMm: Optional[float]
    poseTvec: Optional[Tuple[float, float, float]]
    compensatedWorldPoint: Optional[Tuple[float, float, float]]
    staticPoseInfo: Optional[StaticPoseInfo] = None
    staticAxisSource: str = "none"


class RealtimeDeflectionMeasurer:
    def __init__(
        self,
        layout: MarkerLayout,
        detector: MidpointTargetDetector,
        config: MeasurementConfig,
        calibration: CameraCalibration | None = None,
    ) -> None:
        self.layout = layout
        self.detector = detector
        self.config = config
        self.calibration = calibration
        self.solver = ArucoStaticSolver(layout.dictionaryName)
        self.estimator: DeflectionEstimator | PixelScaleDeflectionEstimator | None = None
        self.elapsedSec = 0.0
        self.started = False

    def startBaseline(self) -> None:
        estimatorClass = (
            PixelScaleDeflectionEstimator if self.config.measurementMethod == "target-local-scale" else DeflectionEstimator
        )
        self.estimator = estimatorClass(
            baselineFrames=self.config.baselineFrames,
            filterProfile=self.config.filterProfile,
            deadbandMm=self.config.deadbandMm,
            smoothWindow=self.config.smoothWindow,
            deflectionScale=self.config.deflectionScale,
            localScaleMode=self.config.localScaleMode,
        )
        self.elapsedSec = 0.0
        self.started = True

    def process(self, frameRaw: np.ndarray, dtSec: float) -> MeasurementFrame:
        frame = undistortFrame(frameRaw, self.calibration) if self.calibration is not None else frameRaw
        homographyResult = self.solver.solveHomography(frame, self.layout)
        detection = self.detector.detect(frame)
        isLowQuality = self._isLowQuality(homographyResult)

        worldPoint = None
        poseTvec = None
        staticBoardPose = None
        staticPoseInfo = None
        staticAxisSource = "none"
        compensatedWorldPoint = None
        targetPositionPx = None
        targetPxPerMm = None
        targetOnlyPositionPx = None
        targetOnlyPxPerMm = None
        measurementYmm = None
        measurementMethod = "none"
        targetOnlyReason = None

        if homographyResult.homography is not None and detection.centerPixel is not None and not isLowQuality:
            worldPoint = self.solver.pixelToWorld(homographyResult.homography, detection.centerPixel)

        if (
            self.config.staticPoseCorrection
            and self.calibration is not None
            and homographyResult.usedPointCount >= self.config.staticAssistMinPoints
        ):
            staticPoseInfo = estimateStaticPoseInfo(
                homographyResult,
                self.calibration.cameraMatrix,
                minPointCount=self.config.staticAssistMinPoints,
            )
            if staticPoseInfo is not None:
                staticBoardPose = (staticPoseInfo.rvec, staticPoseInfo.tvec)
        elif self.calibration is not None and homographyResult.usedPointCount >= self.config.minUsedPoints and not isLowQuality:
            staticBoardPose = estimateStaticBoardPose(homographyResult, self.calibration.cameraMatrix)

        if (
            self.config.measurementMethod == "target-local-scale"
            and detection.markerCorners is not None
        ):
            staticPixelAxis = None
            if staticPoseInfo is not None and self.calibration is not None:
                staticPixelAxis = estimateStaticPosePixelAxis(staticPoseInfo, self.calibration.cameraMatrix)
                if staticPixelAxis is not None:
                    staticAxisSource = "pose"
            if staticPixelAxis is None and homographyResult.homography is not None:
                staticPixelAxis = estimateStaticPixelAxis(homographyResult.homography)
                staticAxisSource = "homography" if staticPixelAxis is not None else "none"
            if staticPixelAxis is not None:
                originPixel, axisUnit = staticPixelAxis
                localScaleResult = estimateTargetLocalCoordinateY(
                    markerCorners=detection.markerCorners,
                    markerSizeMm=self.config.targetMarkerSizeMm,
                    originPixel=originPixel,
                    axisUnit=axisUnit,
                )
                if localScaleResult is not None:
                    targetPositionPx, targetPxPerMm = localScaleResult
                    measurementMethod = "target-local-scale"

        if (
            self.config.measurementMethod == "target-local-scale"
            and self.config.targetOnlyFallback
            and detection.markerCorners is not None
        ):
            localScaleResult = estimateTargetLocalCoordinateY(
                markerCorners=detection.markerCorners,
                markerSizeMm=self.config.targetMarkerSizeMm,
                originPixel=(0.0, 0.0),
                axisUnit=(0.0, 1.0),
            )
            if localScaleResult is not None:
                targetOnlyPositionPx, targetOnlyPxPerMm = localScaleResult
                if targetPositionPx is None:
                    targetPositionPx, targetPxPerMm = localScaleResult
                    measurementMethod = "target-only-image-y"
                    if homographyResult.homography is None:
                        targetOnlyReason = "no-static-markers"
                    elif isLowQuality:
                        targetOnlyReason = "low-homography-quality"
                    else:
                        targetOnlyReason = "static-axis-unavailable"

        if (
            self.config.measurementMethod in ("static-compensated-pnp", "target-pnp")
            and self.calibration is not None
            and detection.markerCorners is not None
        ):
            poseTvec = estimateMarkerPoseTvec(
                markerCorners=detection.markerCorners,
                markerSizeMm=self.config.targetMarkerSizeMm,
                cameraMatrix=self.calibration.cameraMatrix,
            )
            if poseTvec is not None:
                if self.config.measurementMethod == "static-compensated-pnp" and staticBoardPose is not None:
                    compensatedWorldPoint = cameraPointToStaticWorld(poseTvec, staticBoardPose)
                    measurementYmm = compensatedWorldPoint[1]
                    measurementMethod = "static-compensated-pnp"
                else:
                    measurementYmm = poseTvec[1]
                    measurementMethod = "target-pnp"

        if measurementYmm is None and worldPoint is not None and measurementMethod == "none":
            measurementYmm = worldPoint[1]
            measurementMethod = "homography"

        self.elapsedSec += max(0.0, dtSec)
        state = self._updateState(
            measurementYmm=measurementYmm,
            targetPositionPx=targetPositionPx,
            targetPxPerMm=targetPxPerMm,
            targetOnlyPositionPx=targetOnlyPositionPx,
            targetOnlyPxPerMm=targetOnlyPxPerMm,
            timeSec=self.elapsedSec,
            confidence=detection.confidence,
            detectionStatus=detection.status,
            homographyResult=homographyResult,
            isLowQuality=isLowQuality,
            measurementMethod=measurementMethod,
            targetOnlyReason=targetOnlyReason,
            staticAxisSource=staticAxisSource,
        )

        return MeasurementFrame(
            frame=frame,
            homographyResult=homographyResult,
            detection=detection,
            state=state,
            isLowQuality=isLowQuality,
            measurementMethod=measurementMethod,
            worldPoint=worldPoint,
            targetPositionPx=targetPositionPx,
            targetPxPerMm=targetPxPerMm,
            poseTvec=poseTvec,
            compensatedWorldPoint=compensatedWorldPoint,
            staticPoseInfo=staticPoseInfo,
            staticAxisSource=staticAxisSource,
        )

    def _isLowQuality(self, homographyResult: HomographyResult) -> bool:
        if homographyResult.homography is None:
            return False
        return (
            homographyResult.usedPointCount < self.config.minUsedPoints
            or (
                homographyResult.reprojectionRmsePx is not None
                and homographyResult.reprojectionRmsePx > self.config.maxRmse
            )
            or (
                homographyResult.inlierRatio is not None
                and homographyResult.inlierRatio < self.config.minInlierRatio
            )
        )

    def _updateState(
        self,
        measurementYmm: Optional[float],
        targetPositionPx: Optional[float],
        targetPxPerMm: Optional[float],
        targetOnlyPositionPx: Optional[float],
        targetOnlyPxPerMm: Optional[float],
        timeSec: float,
        confidence: float,
        detectionStatus: str,
        homographyResult: HomographyResult,
        isLowQuality: bool,
        measurementMethod: str,
        targetOnlyReason: Optional[str],
        staticAxisSource: str,
    ) -> DeflectionState:
        if self.estimator is None:
            return DeflectionState(
                timeSec=timeSec,
                rawMm=None,
                unscaledRawMm=None,
                filteredMm=None,
                baselineMm=None,
                deflectionCm=None,
                confidence=confidence,
                status="waiting-start",
                measurementPositionPx=targetPositionPx,
                measurementPxPerMm=targetPxPerMm,
            )

        statusHint = detectionStatus
        if self.config.measurementMethod in ("target-local-scale", "homography"):
            if measurementMethod == "target-only-image-y":
                statusHint = f"target-only-{targetOnlyReason or 'static-unavailable'}"
            elif measurementMethod == "target-local-scale" and staticAxisSource == "pose" and isLowQuality:
                statusHint = "static-pose-low-quality"
            elif measurementMethod == "target-local-scale" and staticAxisSource == "pose":
                statusHint = "static-pose"
            elif measurementMethod == "target-local-scale" and staticAxisSource == "homography" and isLowQuality:
                statusHint = "static-homography-low-quality"
            elif homographyResult.homography is None:
                statusHint = "no-static-markers"
            elif isLowQuality:
                statusHint = "low-homography-quality"

        if isinstance(self.estimator, PixelScaleDeflectionEstimator) and self.config.measurementMethod == "target-local-scale":
            return self.estimator.updatePixelScaleWithFallback(
                references={
                    "static": (targetPositionPx if measurementMethod == "target-local-scale" else None, targetPxPerMm if measurementMethod == "target-local-scale" else None),
                    "image": (targetOnlyPositionPx, targetOnlyPxPerMm),
                },
                preferredKey="static",
                fallbackKey="image",
                timeSec=timeSec,
                confidence=confidence,
                statusHint=statusHint,
                fallbackStatusHint=f"target-only-{targetOnlyReason or 'static-unavailable'}",
            )

        return self.estimator.update(
            worldYmm=measurementYmm,
            timeSec=timeSec,
            confidence=confidence,
            statusHint=statusHint,
        )
