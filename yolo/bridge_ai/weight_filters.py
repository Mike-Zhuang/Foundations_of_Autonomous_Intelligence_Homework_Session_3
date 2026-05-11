from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class WeightFilter(ABC):
    @abstractmethod
    def update(self, rawWeightG: float, confidence: float | None = None, quality: float | None = None) -> float:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class DefaultFilter(WeightFilter):
    def __init__(self, windowSize: int = 3) -> None:
        self._window: deque[float] = deque(maxlen=max(windowSize, 1))

    @property
    def name(self) -> str:
        return "default"

    def update(self, rawWeightG: float, confidence: float | None = None, quality: float | None = None) -> float:
        self._window.append(rawWeightG)
        values = np.asarray(self._window, dtype=np.float64)
        return float(0.7 * np.median(values) + 0.3 * np.mean(values))

    def reset(self) -> None:
        self._window.clear()


class ConfEmaFilter(WeightFilter):
    def __init__(self, alpha: float = 0.3, minConfidence: float = 0.1) -> None:
        self._alpha = alpha
        self._minConf = minConfidence
        self._ema: float | None = None

    @property
    def name(self) -> str:
        return "conf_ema"

    def update(self, rawWeightG: float, confidence: float | None = None, quality: float | None = None) -> float:
        conf = max(float(confidence or 0.5), self._minConf)
        if self._ema is None:
            self._ema = rawWeightG
        else:
            effectiveAlpha = self._alpha * conf
            self._ema = (1.0 - effectiveAlpha) * self._ema + effectiveAlpha * rawWeightG
        return float(self._ema)

    def reset(self) -> None:
        self._ema = None


class KalmanFilter1D(WeightFilter):
    def __init__(self, processNoise: float = 1.0, baseMeasNoise: float = 5.0) -> None:
        self._q = processNoise
        self._r0 = baseMeasNoise
        self._x: float | None = None
        self._p: float = 1.0

    @property
    def name(self) -> str:
        return "kalman"

    def update(self, rawWeightG: float, confidence: float | None = None, quality: float | None = None) -> float:
        if self._x is None:
            self._x = rawWeightG
            return float(self._x)
        self._p += self._q
        conf = float(confidence or 0.5)
        r = self._r0 / max(conf, 0.05)
        k = self._p / (self._p + r)
        self._x = self._x + k * (rawWeightG - self._x)
        self._p = (1.0 - k) * self._p
        return float(self._x)

    def reset(self) -> None:
        self._x = None
        self._p = 1.0


class AdaptivePeakFilter(WeightFilter):
    def __init__(self, windowSize: int = 7, qualityThreshold: float = 0.3) -> None:
        self._windowSize = max(windowSize, 3)
        self._qThresh = qualityThreshold
        self._buffer: deque[tuple[float, float]] = deque(maxlen=self._windowSize)

    @property
    def name(self) -> str:
        return "adaptive_peak"

    def update(self, rawWeightG: float, confidence: float | None = None, quality: float | None = None) -> float:
        q = float(quality or 0.5)
        self._buffer.append((rawWeightG, q))
        highQuality = [w for w, qv in self._buffer if qv >= self._qThresh]
        if not highQuality:
            highQuality = [w for w, _ in self._buffer]
        arr = np.asarray(highQuality, dtype=np.float64)
        peak = float(np.max(arr))
        tolerance = float(np.std(arr)) if arr.size > 1 else 1.0
        neighborhood = arr[arr >= peak - tolerance]
        return float(np.mean(neighborhood)) if neighborhood.size > 0 else float(np.mean(arr))

    def reset(self) -> None:
        self._buffer.clear()


FILTER_REGISTRY: dict[str, type[WeightFilter]] = {
    "default": DefaultFilter,
    "conf_ema": ConfEmaFilter,
    "kalman": KalmanFilter1D,
    "adaptive_peak": AdaptivePeakFilter,
}

AVAILABLE_FILTER_NAMES: list[str] = list(FILTER_REGISTRY.keys())


def createFilter(method: str, windowSize: int = 3) -> WeightFilter:
    cls = FILTER_REGISTRY.get(method)
    if cls is None:
        raise ValueError(f"Unknown filter method '{method}'. Available: {AVAILABLE_FILTER_NAMES}")
    if method == "default":
        return cls(windowSize=windowSize)
    if method == "adaptive_peak":
        return cls(windowSize=max(windowSize * 2, 7))
    return cls()


def smoothWeightWithFilter(
    prediction: dict | None,
    weightFilter: WeightFilter,
    weights: list[float],
) -> dict | None:
    if prediction is None:
        return None
    rawWeight = float(prediction["predictedWeightG"])
    confidence = None
    probs = prediction.get("classProbabilities")
    if probs and isinstance(probs, list) and len(probs) > 0:
        confidence = float(max(probs))
    quality = prediction.get("qualityMean")

    smoothedWeight = weightFilter.update(rawWeight, confidence=confidence, quality=quality)
    smoothed = dict(prediction)
    smoothed["rawPredictedWeightG"] = rawWeight
    smoothed["predictedWeightG"] = smoothedWeight
    smoothed["smoothMethod"] = weightFilter.name

    from bridge_ai.weight_model import nearestClass
    nearestWeight, nearestIndex = nearestClass(smoothedWeight, weights)
    if smoothed.get("guardReason") == "zero-deflection":
        nearestWeight, nearestIndex = 0.0, -1
    smoothed["nearestWeightG"] = nearestWeight
    smoothed["nearestIndex"] = nearestIndex
    return smoothed
