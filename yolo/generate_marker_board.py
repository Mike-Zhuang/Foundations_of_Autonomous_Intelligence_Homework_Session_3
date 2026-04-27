from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bridge_ai.config import buildDefaultStaticLayout, saveLayout


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成可单独打印的 ArUco 标记")
    parser.add_argument("--output-dir", default="yolo/artifacts", help="输出目录")
    parser.add_argument("--dpi", type=int, default=300, help="输出分辨率")
    parser.add_argument("--midpoint-id", type=int, default=42, help="跨中回退 ArUco ID")
    parser.add_argument("--charuco-squares-x", type=int, default=5)
    parser.add_argument("--charuco-squares-y", type=int, default=7)
    parser.add_argument("--charuco-square-mm", type=float, default=24.0)
    parser.add_argument("--charuco-marker-mm", type=float, default=18.0)
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


def mmToName(mm: float) -> str:
    if float(mm).is_integer():
        return f"{int(mm)}"
    return f"{mm:.1f}".replace(".", "p")


def main() -> int:
    args = parseArgs()
    outputDir = Path(args.output_dir)
    outputDir.mkdir(parents=True, exist_ok=True)

    layout = buildDefaultStaticLayout()
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, layout.dictionaryName))

    staticMarkerFiles: list[str] = []
    for markerDef in sorted(layout.markers.values(), key=lambda item: item.markerId):
        markerSizePx = mmToPx(markerDef.sizeMm, args.dpi)
        markerImg = cv2.aruco.generateImageMarker(dictionary, markerDef.markerId, markerSizePx)

        markerName = f"static_marker_id{markerDef.markerId}_{mmToName(markerDef.sizeMm)}mm.png"
        cv2.imwrite(str(outputDir / markerName), markerImg)
        staticMarkerFiles.append(markerName)

    layoutJson = outputDir / "static_marker_layout.json"
    saveLayout(layout, layoutJson)

    midpointSizeMm = 50.0
    midpointAruco = cv2.aruco.generateImageMarker(dictionary, args.midpoint_id, mmToPx(midpointSizeMm, args.dpi))
    fallbackName = f"midpoint_fallback_aruco_id{args.midpoint_id}_{mmToName(midpointSizeMm)}mm.png"
    cv2.imwrite(str(outputDir / fallbackName), midpointAruco)

    circleName = f"midpoint_circle_marker_{mmToName(midpointSizeMm)}mm.png"
    circleMarker = makeCircleMarker(mmToPx(midpointSizeMm, args.dpi))
    cv2.imwrite(str(outputDir / circleName), circleMarker)

    charucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    charucoBoard = cv2.aruco.CharucoBoard(
        (args.charuco_squares_x, args.charuco_squares_y),
        args.charuco_square_mm,
        args.charuco_marker_mm,
        charucoDict,
    )
    charucoWidthPx = mmToPx(args.charuco_squares_x * args.charuco_square_mm, args.dpi)
    charucoHeightPx = mmToPx(args.charuco_squares_y * args.charuco_square_mm, args.dpi)
    charucoImage = charucoBoard.generateImage((charucoWidthPx, charucoHeightPx))
    charucoName = (
        f"charuco_{args.charuco_squares_x}x{args.charuco_squares_y}"
        f"_{mmToName(args.charuco_square_mm)}mm_{mmToName(args.charuco_marker_mm)}mm.png"
    )
    cv2.imwrite(str(outputDir / charucoName), charucoImage)

    notesPath = outputDir / "marker_print_notes.txt"
    notesPath.write_text(
        """
打印参数（必须 100% 原始比例，不可缩放）:
- 静态标记字典: DICT_5X5_250
- 静态标记ID: 10, 11, 12, 13, 14
- 静态标记边长: 40 mm
- 生成方式: 每个静态标记独立文件，按需单独打印
- 命名规则: static_marker_id{ID}_{SIZE}mm.png
- 跨中回退标记: ArUco ID 42（可选）
- 跨中主目标建议: midpoint_circle_marker_50mm.png 中央圆点（直径约12mm）
- ChArUco 标定板: 5x7, square=24mm, marker=18mm
""".strip()
        + "\n",
        encoding="utf-8",
    )

    print("Saved static markers:")
    for markerName in staticMarkerFiles:
        print(f"  - {outputDir / markerName}")
    print(f"Saved fallback marker: {outputDir / fallbackName}")
    print(f"Saved circle marker: {outputDir / circleName}")
    print(f"Saved ChArUco board: {outputDir / charucoName}")
    print(f"Saved layout json: {layoutJson}")
    print(f"Saved marker notes: {notesPath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
