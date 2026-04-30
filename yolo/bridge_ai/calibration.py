from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    cameraMatrix: np.ndarray
    distCoeffs: np.ndarray
    rms: float
    imageWidth: int
    imageHeight: int


def saveCalibration(calibration: CameraCalibration, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        cameraMatrix=calibration.cameraMatrix,
        distCoeffs=calibration.distCoeffs,
        rms=np.asarray([calibration.rms], dtype=np.float64),
        imageWidth=np.asarray([calibration.imageWidth], dtype=np.int32),
        imageHeight=np.asarray([calibration.imageHeight], dtype=np.int32),
    )


def loadCalibration(path: Path) -> CameraCalibration:
    payload = np.load(str(path))
    return CameraCalibration(
        cameraMatrix=np.asarray(payload["cameraMatrix"], dtype=np.float64),
        distCoeffs=np.asarray(payload["distCoeffs"], dtype=np.float64),
        rms=float(np.asarray(payload["rms"]).reshape(-1)[0]),
        imageWidth=int(np.asarray(payload["imageWidth"]).reshape(-1)[0]),
        imageHeight=int(np.asarray(payload["imageHeight"]).reshape(-1)[0]),
    )


def undistortFrame(frame: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    return cv2.undistort(frame, calibration.cameraMatrix, calibration.distCoeffs)


def runCharucoCalibration(
    capture: cv2.VideoCapture,
    windowName: str = "ChArUco Calibration",
    targetSamples: int = 30,
    minCornersPerSample: int = 4,
    squaresX: int = 5,
    squaresY: int = 7,
    squareLengthMm: float = 24.0,
    markerLengthMm: float = 18.0,
) -> CameraCalibration:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLengthMm, markerLengthMm, dictionary)
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
    charucoDetector = cv2.aruco.CharucoDetector(board) if hasattr(cv2.aruco, "CharucoDetector") else None

    allCharucoCorners: list[np.ndarray] = []
    allCharucoIds: list[np.ndarray] = []
    imageSize: tuple[int, int] | None = None

    print("进入 ChArUco 标定模式：")
    print("- 仅在视频窗口内按键（不要在命令行输入 c）")
    print("- 按 s 开始采样")
    print("- 按 c 抓取当前帧")
    print("- 按 a 切换自动采样")
    print("- 按 q 取消")
    print(f"- 目标采样数: {targetSamples}")
    print(f"- 单帧最少角点数: {minCornersPerSample}")

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    autoCapture = True
    samplingStarted = False
    frameCounter = 0
    lastCaptureFrame = -1000

    # 预热相机，避免首次启动时自动曝光/对焦尚未稳定导致角点检测失败。
    for _ in range(20):
        ok, _ = capture.read()
        if not ok:
            break
    def detectCharuco(
        grayFrame: np.ndarray, corners: tuple[np.ndarray, ...], ids: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        # 新版 OpenCV: 优先走 CharucoDetector（兼容无 interpolateCornersCharuco 的环境）
        if charucoDetector is not None:
            charucoCorners, charucoIds, _, _ = charucoDetector.detectBoard(grayFrame)
            return charucoCorners, charucoIds

        # 旧版 OpenCV: 走 interpolateCornersCharuco
        if hasattr(cv2.aruco, "interpolateCornersCharuco") and ids is not None and len(ids) > 0:
            charucoCorners, charucoIds, _ = cv2.aruco.interpolateCornersCharuco(corners, ids, grayFrame, board)
            return charucoCorners, charucoIds
        return None, None

    while len(allCharucoIds) < targetSamples:
        frameCounter += 1
        ok, frame = capture.read()
        if not ok:
            break

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(grayFrame)
        showFrame = frame.copy()

        charucoCorners = None
        charucoIds = None
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(showFrame, corners, ids)
            charucoCorners, charucoIds = detectCharuco(grayFrame, corners, ids)
            if charucoIds is not None and len(charucoIds) > 0:
                cv2.aruco.drawDetectedCornersCharuco(showFrame, charucoCorners, charucoIds)
        else:
            charucoCorners, charucoIds = detectCharuco(grayFrame, corners, ids)

        arucoCount = int(len(ids)) if ids is not None else 0
        charucoCount = int(len(charucoIds)) if charucoIds is not None else 0

        cv2.putText(
            showFrame,
            f"Samples: {len(allCharucoIds)}/{targetSamples}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            showFrame,
            f"Aruco: {arucoCount}  Charuco: {charucoCount}  Auto: {'ON' if autoCapture else 'OFF'}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            showFrame,
            "Window keys: S start, C capture, A auto, Q quit",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if not samplingStarted:
            cv2.putText(
                showFrame,
                "Press S to start calibration sampling",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 220, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(windowName, showFrame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            raise RuntimeError("用户取消了 ChArUco 标定")
        if key == ord("s"):
            samplingStarted = True
            print("已开始采样。请缓慢改变标定板姿态。")
        if not samplingStarted:
            continue
        if key == ord("a"):
            autoCapture = not autoCapture
            print(f"自动采样已{'开启' if autoCapture else '关闭'}")

        requestCapture = key == ord("c")
        if autoCapture and charucoCount >= minCornersPerSample and (frameCounter - lastCaptureFrame) >= 8:
            requestCapture = True

        if not requestCapture:
            continue

        if ids is None or len(ids) == 0:
            print("当前帧未检测到 ArUco，跳过")
            continue

        if charucoIds is None or len(charucoIds) < minCornersPerSample:
            print(
                f"当前帧有效 ChArUco 角点不足（{charucoCount} < {minCornersPerSample}），跳过。"
                "请让标定板占画面更大、角度别太斜、避免反光。"
            )
            continue

        allCharucoCorners.append(charucoCorners)
        allCharucoIds.append(charucoIds)
        imageSize = (grayFrame.shape[1], grayFrame.shape[0])
        lastCaptureFrame = frameCounter
        print(f"已抓取标定样本: {len(allCharucoIds)}/{targetSamples}")

    cv2.destroyWindow(windowName)

    minSamplesRequired = max(8, targetSamples // 3)
    if imageSize is None or len(allCharucoIds) < minSamplesRequired:
        raise RuntimeError(
            "ChArUco 标定样本不足，无法完成标定。"
            f"当前样本数={len(allCharucoIds)}，最低要求={minSamplesRequired}。"
            "请让标定板在画面中更大、减少反光，或切到自动采样模式（按 a）。"
        )

    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        calibrationResult = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=allCharucoCorners,
            charucoIds=allCharucoIds,
            board=board,
            imageSize=imageSize,
            cameraMatrix=None,
            distCoeffs=None,
        )
        rms, cameraMatrix, distCoeffs, _, _ = calibrationResult
    else:
        # 新版 API 兼容路径：把 ChArUco 角点映射到 3D 棋盘点，再用 calibrateCamera。
        chessboardPoints = board.getChessboardCorners()
        objectPoints: list[np.ndarray] = []
        imagePoints: list[np.ndarray] = []
        for charucoCorners, charucoIds in zip(allCharucoCorners, allCharucoIds):
            idsFlat = charucoIds.reshape(-1).astype(np.int32)
            obj = chessboardPoints[idsFlat].reshape(-1, 1, 3).astype(np.float32)
            img = charucoCorners.reshape(-1, 1, 2).astype(np.float32)
            objectPoints.append(obj)
            imagePoints.append(img)

        rms, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            imageSize=imageSize,
            cameraMatrix=None,
            distCoeffs=None,
        )
    return CameraCalibration(
        cameraMatrix=np.asarray(cameraMatrix, dtype=np.float64),
        distCoeffs=np.asarray(distCoeffs, dtype=np.float64),
        rms=float(rms),
        imageWidth=int(imageSize[0]),
        imageHeight=int(imageSize[1]),
    )
