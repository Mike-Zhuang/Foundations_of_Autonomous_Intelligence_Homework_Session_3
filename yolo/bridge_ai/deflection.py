from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Optional


@dataclass
class DeflectionState:
    timeSec: float
    rawMm: Optional[float]
    filteredMm: Optional[float]
    baselineMm: Optional[float]
    deflectionCm: Optional[float]
    confidence: float
    status: str


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
    def __init__(self, baselineFrames: int = 60) -> None:
        self.baselineFrames = baselineFrames
        self.baselineSamples: list[float] = []
        self.baselineMm: Optional[float] = None
        self.filter = ScalarKalmanFilter()

    def update(self, worldYmm: Optional[float], timeSec: float, confidence: float, statusHint: str) -> DeflectionState:
        if worldYmm is None:
            return DeflectionState(
                timeSec=timeSec,
                rawMm=None,
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
                filteredMm=0.0,
                baselineMm=self.baselineMm,
                deflectionCm=0.0,
                confidence=confidence,
                status="calibrating-baseline",
            )

        rawMm = float(worldYmm - self.baselineMm)
        filteredMm = float(self.filter.update(rawMm))

        return DeflectionState(
            timeSec=timeSec,
            rawMm=rawMm,
            filteredMm=filteredMm,
            baselineMm=self.baselineMm,
            deflectionCm=filteredMm / 10.0,
            confidence=confidence,
            status=f"tracking:{statusHint}",
        )
