from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Optional, TextIO

import cv2

from bridge_ai.deflection import DeflectionState


def openVideoSource(source: str, preferAvfoundation: bool = True) -> cv2.VideoCapture:
    if source.isdigit():
        index = int(source)
        if preferAvfoundation:
            capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    return capture


class CsvWriter:
    def __init__(self, outputPath: Path) -> None:
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        self.file: TextIO = outputPath.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "timeSec",
                "rawMm",
                "unscaledRawMm",
                "filteredMm",
                "baselineMm",
                "deflectionCm",
                "confidence",
                "status",
                "measurementPositionPx",
                "measurementPxPerMm",
            ],
        )
        self.writer.writeheader()

    def write(self, state: DeflectionState) -> None:
        self.writer.writerow(asdict(state))

    def close(self) -> None:
        self.file.close()


def createVideoWriter(path: Optional[Path], frameWidth: int, frameHeight: int, fps: float) -> Optional[cv2.VideoWriter]:
    if path is None:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (frameWidth, frameHeight))
    return writer
