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
    squaresX: int = 5,
    squaresY: int = 7,
    squareLengthMm: float = 24.0,
    markerLengthMm: float = 18.0,
) -> CameraCalibration:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLengthMm, markerLengthMm, dictionary)
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

    allCharucoCorners: list[np.ndarray] = []
    allCharucoIds: list[np.ndarray] = []
    imageSize: tuple[int, int] | None = None

    print("进入 ChArUco 标定模式：")
    print("- 按 c 抓取当前帧")
    print("- 按 q 取消")
    print(f"- 目标采样数: {targetSamples}")

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    while len(allCharucoIds) < targetSamples:
        ok, frame = capture.read()
        if not ok:
            break

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(grayFrame)
        showFrame = frame.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(showFrame, corners, ids)
            charucoCorners, charucoIds, _ = cv2.aruco.interpolateCornersCharuco(corners, ids, grayFrame, board)
            if charucoIds is not None and len(charucoIds) > 0:
                cv2.aruco.drawDetectedCornersCharuco(showFrame, charucoCorners, charucoIds)

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
            "Press C to capture, Q to cancel",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(windowName, showFrame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            raise RuntimeError("用户取消了 ChArUco 标定")
        if key != ord("c"):
            continue

        if ids is None or len(ids) == 0:
            print("当前帧未检测到 ArUco，跳过")
            continue

        charucoCorners, charucoIds, _ = cv2.aruco.interpolateCornersCharuco(corners, ids, grayFrame, board)
        if charucoIds is None or len(charucoIds) < 4:
            print("当前帧有效 ChArUco 角点不足（<4），跳过")
            continue

        allCharucoCorners.append(charucoCorners)
        allCharucoIds.append(charucoIds)
        imageSize = (grayFrame.shape[1], grayFrame.shape[0])
        print(f"已抓取标定样本: {len(allCharucoIds)}/{targetSamples}")

    cv2.destroyWindow(windowName)

    if imageSize is None or len(allCharucoIds) < max(10, targetSamples // 2):
        raise RuntimeError("ChArUco 标定样本不足，无法完成标定")

    calibrationResult = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=allCharucoCorners,
        charucoIds=allCharucoIds,
        board=board,
        imageSize=imageSize,
        cameraMatrix=None,
        distCoeffs=None,
    )
    rms, cameraMatrix, distCoeffs, _, _ = calibrationResult
    return CameraCalibration(
        cameraMatrix=np.asarray(cameraMatrix, dtype=np.float64),
        distCoeffs=np.asarray(distCoeffs, dtype=np.float64),
        rms=float(rms),
        imageWidth=int(imageSize[0]),
        imageHeight=int(imageSize[1]),
    )
