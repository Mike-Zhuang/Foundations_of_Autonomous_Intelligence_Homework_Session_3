from __future__ import annotations

import cv2


def buildPrecisionArucoDetectorParameters() -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()

    # 精度优先：先使用 OpenCV 自带 SUBPIX 角点精修。
    # 这里不启用 APRILTAG，避免引入版本差异；后续可以用同一接口做 A/B 测试。
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 80
    params.cornerRefinementMinAccuracy = 0.001

    # 透视矫正后对每个 cell 更密采样，提升远距离/透视情况下的 bit extraction 稳定性。
    params.perspectiveRemovePixelPerCell = 10
    params.perspectiveRemoveIgnoredMarginPerCell = 0.18

    # 保守纠错：降低错 ID 的概率，宁可少识别一点，也不要把错误标记参与几何计算。
    params.errorCorrectionRate = 0.3
    params.maxErroneousBitsInBorderRate = 0.15

    # 保留稍宽松的候选轮廓搜索，让小一些/远一些的标记仍有机会进入精修阶段。
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.015
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.04

    return params
