from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Optional


@dataclass
class DeflectionState:
    timeSec: float
    rawMm: Optional[float]
    unscaledRawMm: Optional[float]
    filteredMm: Optional[float]
    baselineMm: Optional[float]
    deflectionCm: Optional[float]
    confidence: float
    status: str
    measurementPositionPx: Optional[float] = None
    measurementPxPerMm: Optional[float] = None


class ScalarKalmanFilter:
    def __init__(self, processVar: float = 0.08, measureVar: float = 1.5) -> None:
        self.processVar = processVar
        self.measureVar = measureVar
        self.estimate = 0.0
        self.errorCov = 1.0
        self.isReady = False

    def update(self, measurement: float) -> float:
        if not self.isReady:
            self.estimate = measurement
            self.errorCov = 1.0
            self.isReady = True
            return self.estimate

        self.errorCov += self.processVar
        kalmanGain = self.errorCov / (self.errorCov + self.measureVar)
        self.estimate = self.estimate + kalmanGain * (measurement - self.estimate)
        self.errorCov = (1.0 - kalmanGain) * self.errorCov
        return self.estimate


class DeflectionEstimator:
    def __init__(
        self,
        baselineFrames: int = 60,
        filterProfile: str = "normal",
        deadbandMm: float = 0.0,
        smoothWindow: int = 1,
        deflectionScale: float = 1.0,
        localScaleMode: str = "baseline",
    ) -> None:
        self.baselineFrames = baselineFrames
        self.baselineSamples: list[float] = []
        self.baselineMm: Optional[float] = None
        self.filterProfile = filterProfile
        self.deadbandMm = max(0.0, deadbandMm)
        self.smoothWindow = max(1, smoothWindow)
        self.smoothSamples: list[float] = []
        self.lastStableMm: Optional[float] = None
        self.deflectionScale = deflectionScale
        self.localScaleMode = localScaleMode
        self.filter = self._buildFilter(filterProfile)

    @staticmethod
    def _buildFilter(filterProfile: str) -> ScalarKalmanFilter:
        if filterProfile == "stable":
            return ScalarKalmanFilter(processVar=0.015, measureVar=6.0)
        if filterProfile == "fast":
            return ScalarKalmanFilter(processVar=0.16, measureVar=0.9)
        return ScalarKalmanFilter(processVar=0.08, measureVar=1.5)

    def _filterRawMm(self, rawMm: float) -> float:
        kalmanMm = float(self.filter.update(rawMm))
        self.smoothSamples.append(kalmanMm)
        if len(self.smoothSamples) > self.smoothWindow:
            self.smoothSamples.pop(0)

        if self.filterProfile == "stable" and len(self.smoothSamples) >= 3:
            windowMedian = float(median(self.smoothSamples))
            windowMean = float(mean(self.smoothSamples))
            candidateMm = 0.65 * windowMedian + 0.35 * windowMean
        else:
            candidateMm = kalmanMm

        if self.deadbandMm > 0.0:
            if self.lastStableMm is None:
                filteredMm = candidateMm
                self.lastStableMm = filteredMm
            elif abs(candidateMm - self.lastStableMm) <= self.deadbandMm:
                filteredMm = self.lastStableMm
            else:
                filteredMm = candidateMm
                self.lastStableMm = filteredMm
        else:
            filteredMm = candidateMm
            self.lastStableMm = filteredMm

        return filteredMm

    def update(self, worldYmm: Optional[float], timeSec: float, confidence: float, statusHint: str) -> DeflectionState:
        if worldYmm is None:
            return DeflectionState(
                timeSec=timeSec,
                rawMm=None,
                unscaledRawMm=None,
                filteredMm=None,
                baselineMm=self.baselineMm,
                deflectionCm=None,
                confidence=confidence,
                status=f"missing:{statusHint}",
            )

        if self.baselineMm is None:
            self.baselineSamples.append(worldYmm)
            if len(self.baselineSamples) >= self.baselineFrames:
                self.baselineMm = float(median(self.baselineSamples))
            return DeflectionState(
                timeSec=timeSec,
                rawMm=0.0,
                unscaledRawMm=0.0,
                filteredMm=0.0,
                baselineMm=self.baselineMm,
                deflectionCm=0.0,
                confidence=confidence,
                status="calibrating-baseline",
            )

        unscaledRawMm = float(worldYmm - self.baselineMm)
        rawMm = float(unscaledRawMm * self.deflectionScale)
        filteredMm = self._filterRawMm(rawMm)

        return DeflectionState(
            timeSec=timeSec,
            rawMm=rawMm,
            unscaledRawMm=unscaledRawMm,
            filteredMm=filteredMm,
            baselineMm=self.baselineMm,
            deflectionCm=filteredMm / 10.0,
            confidence=confidence,
            status=f"tracking:{statusHint}",
        )


class PixelScaleDeflectionEstimator(DeflectionEstimator):
    def __init__(
        self,
        baselineFrames: int = 60,
        filterProfile: str = "normal",
        deadbandMm: float = 0.0,
        smoothWindow: int = 1,
        deflectionScale: float = 1.0,
        localScaleMode: str = "baseline",
    ) -> None:
        super().__init__(
            baselineFrames=baselineFrames,
            filterProfile=filterProfile,
            deadbandMm=deadbandMm,
            smoothWindow=smoothWindow,
            deflectionScale=deflectionScale,
            localScaleMode=localScaleMode,
        )
        self.baselinePixelSamples: list[float] = []
        self.scaleSamples: list[float] = []
        self.baselinePixel: Optional[float] = None
        self.baselinePxPerMm: Optional[float] = None

    def updatePixelScale(
        self,
        positionPx: Optional[float],
        pxPerMm: Optional[float],
        timeSec: float,
        confidence: float,
        statusHint: str,
    ) -> DeflectionState:
        if positionPx is None or pxPerMm is None or pxPerMm <= 1e-6:
            return DeflectionState(
                timeSec=timeSec,
                rawMm=None,
                unscaledRawMm=None,
                filteredMm=None,
                baselineMm=self.baselineMm,
                deflectionCm=None,
                confidence=confidence,
                status=f"missing:{statusHint}",
                measurementPositionPx=positionPx,
                measurementPxPerMm=pxPerMm,
            )

        if self.baselinePixel is None or self.baselinePxPerMm is None:
            self.baselinePixelSamples.append(float(positionPx))
            self.scaleSamples.append(float(pxPerMm))
            if len(self.baselinePixelSamples) >= self.baselineFrames:
                self.baselinePixel = float(median(self.baselinePixelSamples))
                self.baselinePxPerMm = float(median(self.scaleSamples))
                self.baselineMm = self.baselinePixel / self.baselinePxPerMm
            return DeflectionState(
                timeSec=timeSec,
                rawMm=0.0,
                unscaledRawMm=0.0,
                filteredMm=0.0,
                baselineMm=self.baselineMm,
                deflectionCm=0.0,
                confidence=confidence,
                status="calibrating-baseline",
                measurementPositionPx=float(positionPx),
                measurementPxPerMm=float(pxPerMm),
            )

        # 关键点：只把“当前中心 - 基线中心”的像素差换成毫米。
        # 不再把当前局部比例尺作用到整段绝对坐标，避免稳定倍率偏差。
        if self.localScaleMode == "current":
            scalePxPerMm = float(pxPerMm)
        elif self.localScaleMode == "average":
            scalePxPerMm = float((pxPerMm + self.baselinePxPerMm) * 0.5)
        else:
            scalePxPerMm = float(self.baselinePxPerMm)

        unscaledRawMm = float((positionPx - self.baselinePixel) / scalePxPerMm)
        rawMm = float(unscaledRawMm * self.deflectionScale)
        filteredMm = self._filterRawMm(rawMm)

        return DeflectionState(
            timeSec=timeSec,
            rawMm=rawMm,
            unscaledRawMm=unscaledRawMm,
            filteredMm=filteredMm,
            baselineMm=self.baselineMm,
            deflectionCm=filteredMm / 10.0,
            confidence=confidence,
            status=f"tracking:{statusHint}",
            measurementPositionPx=float(positionPx),
            measurementPxPerMm=float(pxPerMm),
        )
