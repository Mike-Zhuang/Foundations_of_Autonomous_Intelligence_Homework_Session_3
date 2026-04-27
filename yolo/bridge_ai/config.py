from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple

Point2D = Tuple[float, float]


@dataclass
class MarkerDefinition:
    markerId: int
    topLeftMm: Point2D
    sizeMm: float

    def worldCornersMm(self) -> List[Point2D]:
        x0, y0 = self.topLeftMm
        s = self.sizeMm
        return [
            (x0, y0),
            (x0 + s, y0),
            (x0 + s, y0 + s),
            (x0, y0 + s),
        ]


@dataclass
class MarkerLayout:
    dictionaryName: str
    markerSizeMm: float
    markers: Dict[int, MarkerDefinition]

    def toDict(self) -> dict:
        return {
            "dictionaryName": self.dictionaryName,
            "markerSizeMm": self.markerSizeMm,
            "markers": [
                {
                    "markerId": marker.markerId,
                    "topLeftMm": [marker.topLeftMm[0], marker.topLeftMm[1]],
                    "sizeMm": marker.sizeMm,
                }
                for marker in self.markers.values()
            ],
        }

    @staticmethod
    def fromDict(payload: dict) -> "MarkerLayout":
        markers: Dict[int, MarkerDefinition] = {}
        for rawMarker in payload["markers"]:
            marker = MarkerDefinition(
                markerId=int(rawMarker["markerId"]),
                topLeftMm=(float(rawMarker["topLeftMm"][0]), float(rawMarker["topLeftMm"][1])),
                sizeMm=float(rawMarker.get("sizeMm", payload["markerSizeMm"])),
            )
            markers[marker.markerId] = marker

        return MarkerLayout(
            dictionaryName=str(payload["dictionaryName"]),
            markerSizeMm=float(payload["markerSizeMm"]),
            markers=markers,
        )


def buildDefaultStaticLayout() -> MarkerLayout:
    markerSizeMm = 40.0
    # 五点布局：上排3个 + 下排2个，减少共线退化风险。
    markerCoords = {
        10: (0.0, 0.0),
        11: (60.0, 0.0),
        12: (120.0, 0.0),
        13: (30.0, 60.0),
        14: (90.0, 60.0),
    }

    markers = {
        markerId: MarkerDefinition(markerId=markerId, topLeftMm=coord, sizeMm=markerSizeMm)
        for markerId, coord in markerCoords.items()
    }

    return MarkerLayout(
        dictionaryName="DICT_5X5_250",
        markerSizeMm=markerSizeMm,
        markers=markers,
    )


def saveLayout(layout: MarkerLayout, outputPath: Path) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text(json.dumps(layout.toDict(), indent=2), encoding="utf-8")


def loadLayout(layoutPath: Path | None) -> MarkerLayout:
    if layoutPath is None:
        return buildDefaultStaticLayout()

    payload = json.loads(layoutPath.read_text(encoding="utf-8"))
    return MarkerLayout.fromDict(payload)
