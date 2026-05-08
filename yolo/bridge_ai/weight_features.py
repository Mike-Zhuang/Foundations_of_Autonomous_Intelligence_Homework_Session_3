from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FEATURE_NAMES = [
    "deflectionMeanMm",
    "deflectionP50Mm",
    "deflectionP05Mm",
    "deflectionP95Mm",
    "deflectionStdMm",
    "deflectionRangeMm",
    "deflectionSlopeMmPerSec",
    "driftMmPerMin",
    "validRate",
    "confidenceMean",
    "qualityMean",
]


@dataclass
class WeightSampleFrame:
    timeSec: float
    deflectionMm: float
    confidence: float
    quality: float
    isValid: bool


def computeQuality(rmsePx: float | None, inlierRatio: float | None, usedPoints: int | None) -> float:
    rmseScore = 1.0
    if rmsePx is not None and np.isfinite(rmsePx):
        rmseScore = float(np.clip(1.0 - rmsePx / 4.0, 0.0, 1.0))

    inlierScore = 0.0
    if inlierRatio is not None and np.isfinite(inlierRatio):
        inlierScore = float(np.clip(inlierRatio, 0.0, 1.0))

    pointScore = 0.0
    if usedPoints is not None:
        pointScore = float(np.clip(usedPoints / 20.0, 0.0, 1.0))

    return float(0.45 * rmseScore + 0.35 * inlierScore + 0.20 * pointScore)


def computeWindowFeatures(
    frames: Iterable[WeightSampleFrame],
    weightG: float | None = None,
    standardDeflectionMm: float | None = None,
    sampleId: str | None = None,
) -> dict:
    frameList = list(frames)
    validFrames = [frame for frame in frameList if frame.isValid and np.isfinite(frame.deflectionMm)]
    if not validFrames:
        raise ValueError("窗口内没有有效挠度帧，无法计算特征。")

    deflections = np.asarray([frame.deflectionMm for frame in validFrames], dtype=np.float64)
    times = np.asarray([frame.timeSec for frame in validFrames], dtype=np.float64)
    confidences = np.asarray([frame.confidence for frame in validFrames], dtype=np.float64)
    qualities = np.asarray([frame.quality for frame in validFrames], dtype=np.float64)

    slope = 0.0
    if deflections.size >= 2 and float(np.max(times) - np.min(times)) > 1e-6:
        tCentered = times - float(np.mean(times))
        slope = float(np.sum(tCentered * (deflections - np.mean(deflections))) / max(np.sum(tCentered * tCentered), 1e-9))

    row = {
        "sampleId": sampleId or "",
        "frameCount": int(len(frameList)),
        "validFrameCount": int(len(validFrames)),
        "deflectionMeanMm": float(np.mean(deflections)),
        "deflectionP50Mm": float(np.percentile(deflections, 50)),
        "deflectionP05Mm": float(np.percentile(deflections, 5)),
        "deflectionP95Mm": float(np.percentile(deflections, 95)),
        "deflectionStdMm": float(np.std(deflections)),
        "deflectionRangeMm": float(np.max(deflections) - np.min(deflections)),
        "deflectionSlopeMmPerSec": slope,
        "driftMmPerMin": float(slope * 60.0),
        "validRate": float(len(validFrames) / max(len(frameList), 1)),
        "confidenceMean": float(np.mean(confidences)) if confidences.size > 0 else 0.0,
        "qualityMean": float(np.mean(qualities)) if qualities.size > 0 else 0.0,
    }
    if weightG is not None:
        row["weightG"] = float(weightG)
    if standardDeflectionMm is not None:
        row["standardDeflectionMm"] = float(standardDeflectionMm)
        row["phoneMinusStandardMm"] = float(row["deflectionMeanMm"] - standardDeflectionMm)
    return row


def featureVector(row: dict) -> np.ndarray:
    return np.asarray([float(row[name]) for name in FEATURE_NAMES], dtype=np.float32)
