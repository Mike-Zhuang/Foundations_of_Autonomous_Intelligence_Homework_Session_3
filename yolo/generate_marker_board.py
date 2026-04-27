from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bridge_ai.config import buildDefaultStaticLayout, saveLayout


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 A4 可打印 ArUco 标记板")
    parser.add_argument("--output-dir", default="yolo/artifacts", help="输出目录")
    parser.add_argument("--dpi", type=int, default=300, help="输出分辨率")
    parser.add_argument("--margin-mm", type=float, default=15.0, help="纸张边距")
    parser.add_argument("--midpoint-id", type=int, default=42, help="跨中回退 ArUco ID")
    return parser.parse_args()


def mmToPx(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


def makeCircleMarker(sizePx: int) -> np.ndarray:
    marker = np.full((sizePx, sizePx), 255, dtype=np.uint8)
    center = sizePx // 2
    radius = int(sizePx * 0.35)
    cv2.circle(marker, (center, center), radius, 0, thickness=-1)
    cv2.circle(marker, (center, center), radius // 3, 255, thickness=-1)
    return marker


def main() -> int:
    args = parseArgs()
    outputDir = Path(args.output_dir)
    outputDir.mkdir(parents=True, exist_ok=True)

    layout = buildDefaultStaticLayout()
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, layout.dictionaryName))

    # A4 纵向尺寸：210mm x 297mm
    pageWidthPx = mmToPx(210.0, args.dpi)
    pageHeightPx = mmToPx(297.0, args.dpi)
    canvas = np.full((pageHeightPx, pageWidthPx), 255, dtype=np.uint8)

    marginPx = mmToPx(args.margin_mm, args.dpi)

    for markerDef in layout.markers.values():
        markerSizePx = mmToPx(markerDef.sizeMm, args.dpi)
        markerImg = cv2.aruco.generateImageMarker(dictionary, markerDef.markerId, markerSizePx)

        xPx = marginPx + mmToPx(markerDef.topLeftMm[0], args.dpi)
        yPx = marginPx + mmToPx(markerDef.topLeftMm[1], args.dpi)
        canvas[yPx : yPx + markerSizePx, xPx : xPx + markerSizePx] = markerImg

        cv2.putText(
            canvas,
            f"ID {markerDef.markerId}",
            (xPx, yPx + markerSizePx + mmToPx(5, args.dpi)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            0,
            2,
            cv2.LINE_AA,
        )

    boardPng = outputDir / "static_marker_board_a4_300dpi.png"
    cv2.imwrite(str(boardPng), canvas)

    layoutJson = outputDir / "static_marker_layout.json"
    saveLayout(layout, layoutJson)

    midpointAruco = cv2.aruco.generateImageMarker(dictionary, args.midpoint_id, mmToPx(50.0, args.dpi))
    cv2.imwrite(str(outputDir / "midpoint_fallback_aruco_id42.png"), midpointAruco)

    circleMarker = makeCircleMarker(mmToPx(50.0, args.dpi))
    cv2.imwrite(str(outputDir / "midpoint_circle_marker_50mm.png"), circleMarker)

    notesPath = outputDir / "marker_print_notes.txt"
    notesPath.write_text(
        """
打印参数（必须 100% 原始比例，不可缩放）:
- 静态标记字典: DICT_5X5_250
- 静态标记ID: 10, 11, 12, 13, 14
- 静态标记边长: 40 mm
- 布局: 上排3个 + 下排2个
- 跨中回退标记: ArUco ID 42（可选）
- 跨中主目标建议: midpoint_circle_marker_50mm.png 中央圆点（直径约12mm）
""".strip()
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved board image: {boardPng}")
    print(f"Saved layout json: {layoutJson}")
    print(f"Saved marker notes: {notesPath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
