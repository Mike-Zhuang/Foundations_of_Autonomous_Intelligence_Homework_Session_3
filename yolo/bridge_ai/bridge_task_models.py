from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bridge_ai.weight_features import FEATURE_NAMES


DEFAULT_WINDOW_CSVS = [
    "yolo/results/weight_windows_20260508_191017.csv",
    "yolo/results/weight_windows_20260508_194555.csv",
]
WEIGHT_FEATURE_NAMES = ["weightG", "weightSquared", "sqrtWeightG", "log1pWeightG"]
RIDGE_LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def loadWindowRows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, float]] = set()
    required = {"weightG", "deflectionMeanMm", "standardDeflectionMm"}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as fileObj:
            for row in csv.DictReader(fileObj):
                missing = sorted(required - set(row.keys()))
                if missing:
                    raise ValueError(f"{path} 缺少字段: {missing}")
                key = (str(row.get("sampleId", "")), float(row["weightG"]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda item: (float(item["weightG"]), str(item.get("sampleId", ""))))
    return rows


def saveJson(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def regressionMetrics(pred: np.ndarray, target: np.ndarray, unit: str) -> dict:
    err = pred - target
    denom = float(np.sum((target - np.mean(target)) ** 2))
    return {
        f"mae{unit}": float(np.mean(np.abs(err))),
        f"rmse{unit}": float(np.sqrt(np.mean(err * err))),
        f"maxAbsError{unit}": float(np.max(np.abs(err))),
        "r2": float(1.0 - np.sum(err * err) / max(denom, 1e-9)),
    }


def nearestClass(predWeight: float, weights: list[float]) -> tuple[float, int]:
    arr = np.asarray(weights, dtype=np.float64)
    index = int(np.argmin(np.abs(arr - predWeight)))
    return float(arr[index]), index


def phoneFeatureMatrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float64)


def weightFeatureMatrixFromRows(rows: list[dict]) -> np.ndarray:
    return weightFeatureMatrix(np.asarray([float(row["weightG"]) for row in rows], dtype=np.float64))


def weightFeatureMatrix(weightsG: np.ndarray) -> np.ndarray:
    weights = np.asarray(weightsG, dtype=np.float64).reshape(-1)
    return np.column_stack(
        [
            weights,
            weights * weights,
            np.sqrt(np.maximum(weights, 0.0)),
            np.log1p(np.maximum(weights, 0.0)),
        ]
    )


def targetArray(rows: list[dict], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype=np.float64)


def polynomialDesign(x: np.ndarray, degree: int) -> np.ndarray:
    if degree <= 1:
        return np.hstack([np.ones((x.shape[0], 1), dtype=np.float64), x])
    return np.hstack([np.ones((x.shape[0], 1), dtype=np.float64), x, x * x])


def ridgeFit(design: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    eye = np.eye(design.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + lam * eye) @ design.T @ y


def trainRidgeCandidate(
    rows: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    unit: str,
    featureNames: list[str],
) -> dict:
    candidates: list[dict] = []
    for degree in (1, 2):
        for lam in RIDGE_LAMBDAS:
            looPred: list[float] = []
            for holdout in range(len(rows)):
                trainMask = np.ones(len(rows), dtype=bool)
                trainMask[holdout] = False
                coef = ridgeFit(polynomialDesign(x[trainMask], degree), y[trainMask], lam)
                pred = polynomialDesign(x[[holdout]], degree) @ coef
                looPred.append(float(pred[0]))
            predArr = np.asarray(looPred, dtype=np.float64)
            candidates.append(
                {
                    "modelType": "ridge",
                    "degree": degree,
                    "lambda": lam,
                    "looPredictions": looPred,
                    "metrics": regressionMetrics(predArr, y, unit),
                }
            )
    candidates.sort(key=lambda item: (item["metrics"][f"mae{unit}"], item["metrics"][f"rmse{unit}"]))
    best = candidates[0]
    coef = ridgeFit(polynomialDesign(x, int(best["degree"])), y, float(best["lambda"]))
    return {
        "modelType": "ridge",
        "degree": int(best["degree"]),
        "lambda": float(best["lambda"]),
        "coef": coef.tolist(),
        "featureNames": featureNames,
        "metrics": best["metrics"],
        "perSample": perSample(rows, y, np.asarray(best["looPredictions"], dtype=np.float64), unit),
    }


def isotonicIncreasing(y: np.ndarray) -> np.ndarray:
    values = [float(value) for value in y.tolist()]
    weights = [1.0 for _ in values]
    starts = list(range(len(values)))
    ends = list(range(len(values)))
    index = 0
    while index < len(values) - 1:
        if values[index] <= values[index + 1]:
            index += 1
            continue
        totalWeight = weights[index] + weights[index + 1]
        mergedValue = (values[index] * weights[index] + values[index + 1] * weights[index + 1]) / totalWeight
        values[index] = mergedValue
        weights[index] = totalWeight
        ends[index] = ends[index + 1]
        del values[index + 1]
        del weights[index + 1]
        del starts[index + 1]
        del ends[index + 1]
        if index > 0:
            index -= 1

    fitted = np.zeros(len(y), dtype=np.float64)
    for value, start, end in zip(values, starts, ends):
        fitted[start : end + 1] = value
    return fitted


def aggregateSortedPoints(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xSorted = np.asarray(x[order], dtype=np.float64)
    ySorted = np.asarray(y[order], dtype=np.float64)
    uniqueX: list[float] = []
    avgY: list[float] = []
    index = 0
    while index < len(xSorted):
        currentX = float(xSorted[index])
        same = np.isclose(xSorted, currentX, rtol=0.0, atol=1e-9)
        same[:index] = False
        sameIndices = np.where(same)[0]
        uniqueX.append(currentX)
        avgY.append(float(np.mean(ySorted[sameIndices])))
        index = int(sameIndices[-1]) + 1
    return np.asarray(uniqueX, dtype=np.float64), np.asarray(avgY, dtype=np.float64)


def smoothMonotonicAnchors(y: np.ndarray, passes: int = 2) -> np.ndarray:
    smoothed = np.asarray(y, dtype=np.float64).copy()
    if len(smoothed) < 3:
        return smoothed
    for _ in range(passes):
        current = smoothed.copy()
        current[0] = 0.75 * smoothed[0] + 0.25 * smoothed[1]
        current[-1] = 0.75 * smoothed[-1] + 0.25 * smoothed[-2]
        current[1:-1] = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
        smoothed = np.maximum.accumulate(current)
    return smoothed


def fitMonotonicPiecewise(x: np.ndarray, y: np.ndarray, smoothingPasses: int = 2) -> dict:
    anchorsX, rawY = aggregateSortedPoints(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    anchorsY = smoothMonotonicAnchors(isotonicIncreasing(rawY), passes=smoothingPasses)
    return {"anchorsX": anchorsX.tolist(), "anchorsY": anchorsY.tolist(), "smoothingPasses": smoothingPasses}


def predictMonotonicAnchors(model: dict, xValue: float) -> float:
    anchorsX = np.asarray(model["anchorsX"], dtype=np.float64)
    anchorsY = np.asarray(model["anchorsY"], dtype=np.float64)
    if len(anchorsX) == 1:
        return float(anchorsY[0])
    if xValue < anchorsX[0]:
        denom = max(float(anchorsX[1] - anchorsX[0]), 1e-9)
        slope = float(anchorsY[1] - anchorsY[0]) / denom
        return float(anchorsY[0] + slope * (xValue - anchorsX[0]))
    if xValue > anchorsX[-1]:
        denom = max(float(anchorsX[-1] - anchorsX[-2]), 1e-9)
        slope = float(anchorsY[-1] - anchorsY[-2]) / denom
        return float(anchorsY[-1] + slope * (xValue - anchorsX[-1]))
    return float(np.interp(xValue, anchorsX, anchorsY))


def trainMonotonicPiecewiseCandidate(
    rows: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    unit: str,
    inputName: str,
    modelType: str,
) -> dict:
    x1d = np.asarray(x, dtype=np.float64).reshape(-1)
    looPred: list[float] = []
    for holdout in range(len(rows)):
        trainMask = np.ones(len(rows), dtype=bool)
        trainMask[holdout] = False
        model = fitMonotonicPiecewise(x1d[trainMask], y[trainMask])
        looPred.append(predictMonotonicAnchors(model, float(x1d[holdout])))
    predArr = np.asarray(looPred, dtype=np.float64)
    fullModel = fitMonotonicPiecewise(x1d, y)
    return {
        "modelType": modelType,
        "inputName": inputName,
        "anchorsX": fullModel["anchorsX"],
        "anchorsY": fullModel["anchorsY"],
        "metrics": regressionMetrics(predArr, y, unit),
        "perSample": perSample(rows, y, predArr, unit),
    }


def trainChainedMonotonicWeightCandidate(rows: list[dict]) -> dict:
    phoneMm = targetArray(rows, "deflectionMeanMm")
    standardMm = targetArray(rows, "standardDeflectionMm")
    weightG = targetArray(rows, "weightG")
    looPred: list[float] = []
    for holdout in range(len(rows)):
        trainMask = np.ones(len(rows), dtype=bool)
        trainMask[holdout] = False
        phoneToStandard = fitMonotonicPiecewise(phoneMm[trainMask], standardMm[trainMask])
        standardToWeight = fitMonotonicPiecewise(standardMm[trainMask], weightG[trainMask])
        predictedStandard = predictMonotonicAnchors(phoneToStandard, float(phoneMm[holdout]))
        looPred.append(predictMonotonicAnchors(standardToWeight, predictedStandard))
    predArr = np.asarray(looPred, dtype=np.float64)
    return {
        "modelType": "chained_monotonic",
        "chain": "deflectionMeanMm -> standardDeflectionMm -> weightG",
        "phoneToStandard": fitMonotonicPiecewise(phoneMm, standardMm),
        "standardToWeight": fitMonotonicPiecewise(standardMm, weightG),
        "weightsG": sorted({float(value) for value in weightG.tolist()}),
        "metrics": regressionMetrics(predArr, weightG, "G"),
        "perSample": perSample(rows, weightG, predArr, "G"),
    }


def perSample(rows: list[dict], truth: np.ndarray, pred: np.ndarray, unit: str) -> list[dict]:
    return [
        {
            "sampleId": str(rows[index].get("sampleId", index)),
            f"true{unit}": float(truth[index]),
            f"pred{unit}": float(pred[index]),
            f"absError{unit}": float(abs(pred[index] - truth[index])),
        }
        for index in range(len(rows))
    ]


def normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x - mean) / std, mean, std


def trainRegressionMlpCandidate(
    rows: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    unit: str,
    featureNames: list[str],
    epochs: int,
    lr: float,
    weightDecay: float,
    taskName: str,
    progressCallback: Callable[[str, int, int, float], None] | None = None,
) -> dict | None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    xNorm, mean, std = normalize(x)
    yMean = float(np.mean(y))
    yStd = float(np.std(y))
    if yStd < 1e-6:
        yStd = 1.0
    yNorm = (y - yMean) / yStd
    looPred: list[float] = []

    class RegressionNet(nn.Module):
        def __init__(self, inputDim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(inputDim, 32),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, features: Any) -> Any:
            return self.net(features)

    for holdout in range(len(rows)):
        trainMask = np.ones(len(rows), dtype=bool)
        trainMask[holdout] = False
        model = RegressionNet(xNorm.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weightDecay)
        lossFn = nn.HuberLoss(delta=1.0)
        xTrain = torch.tensor(xNorm[trainMask], dtype=torch.float32)
        yTrain = torch.tensor(yNorm[trainMask], dtype=torch.float32).unsqueeze(1)
        bestLoss = float("inf")
        stale = 0
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = lossFn(model(xTrain), yTrain)
            loss.backward()
            optimizer.step()
            currentLoss = float(loss.detach().cpu().item())
            if currentLoss + 1e-7 < bestLoss:
                bestLoss = currentLoss
                stale = 0
            else:
                stale += 1
            if stale >= 100:
                break
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(xNorm[[holdout]], dtype=torch.float32))
            predNorm = float(pred.reshape(-1)[0].cpu().item())
            looPred.append(predNorm * yStd + yMean)

    fullModel = RegressionNet(xNorm.shape[1])
    optimizer = torch.optim.AdamW(fullModel.parameters(), lr=lr, weight_decay=weightDecay)
    lossFn = nn.HuberLoss(delta=1.0)
    xAll = torch.tensor(xNorm, dtype=torch.float32)
    yAll = torch.tensor(yNorm, dtype=torch.float32).unsqueeze(1)
    for epoch in range(1, epochs + 1):
        fullModel.train()
        optimizer.zero_grad()
        loss = lossFn(fullModel(xAll), yAll)
        loss.backward()
        optimizer.step()
        lossValue = float(loss.detach().cpu().item())
        if progressCallback is not None:
            progressCallback(taskName, epoch, epochs, lossValue)
        elif epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print(f"[{taskName} MLP] epoch={epoch:04d}/{epochs} loss={lossValue:.4f}", flush=True)

    predArr = np.asarray(looPred, dtype=np.float64)
    return {
        "modelType": "mlp_regression",
        "architecture": "input-32-16-1",
        "featureNames": featureNames,
        "xMean": mean.tolist(),
        "xStd": std.tolist(),
        "yMean": yMean,
        "yStd": yStd,
        "stateDict": fullModel.state_dict(),
        "metrics": regressionMetrics(predArr, y, unit),
        "perSample": perSample(rows, y, predArr, unit),
    }


def trainWeightMlpCandidate(
    rows: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    weightDecay: float,
    progressCallback: Callable[[str, int, int, float], None] | None = None,
) -> dict | None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    xNorm, mean, std = normalize(x)
    yMean = float(np.mean(y))
    yStd = float(np.std(y))
    if yStd < 1e-6:
        yStd = 1.0
    yNorm = (y - yMean) / yStd
    weights = sorted({float(value) for value in y.tolist()})
    labelMap = {str(weight): index for index, weight in enumerate(weights)}
    labels = np.asarray([labelMap[str(float(value))] for value in y.tolist()], dtype=np.int64)
    looPred: list[float] = []
    looCls: list[int] = []

    class WeightNet(nn.Module):
        def __init__(self, inputDim: int, classCount: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(inputDim, 32),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
            )
            self.regHead = nn.Linear(8, 1)
            self.clsHead = nn.Linear(8, classCount)

        def forward(self, features: Any) -> tuple[Any, Any]:
            hidden = self.backbone(features)
            return self.regHead(hidden), self.clsHead(hidden)

    for holdout in range(len(rows)):
        trainMask = np.ones(len(rows), dtype=bool)
        trainMask[holdout] = False
        model = WeightNet(xNorm.shape[1], len(weights))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weightDecay)
        regLossFn = nn.HuberLoss(delta=1.0)
        clsLossFn = nn.CrossEntropyLoss()
        xTrain = torch.tensor(xNorm[trainMask], dtype=torch.float32)
        yTrain = torch.tensor(yNorm[trainMask], dtype=torch.float32).unsqueeze(1)
        clsTrain = torch.tensor(labels[trainMask], dtype=torch.long)
        bestLoss = float("inf")
        stale = 0
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            regPred, clsPred = model(xTrain)
            loss = 0.8 * regLossFn(regPred, yTrain) + 0.2 * clsLossFn(clsPred, clsTrain)
            loss.backward()
            optimizer.step()
            currentLoss = float(loss.detach().cpu().item())
            if currentLoss + 1e-7 < bestLoss:
                bestLoss = currentLoss
                stale = 0
            else:
                stale += 1
            if stale >= 100:
                break
        model.eval()
        with torch.no_grad():
            regPred, clsPred = model(torch.tensor(xNorm[[holdout]], dtype=torch.float32))
            predNorm = float(regPred.reshape(-1)[0].cpu().item())
            looPred.append(predNorm * yStd + yMean)
            looCls.append(int(torch.argmax(clsPred, dim=1).cpu().item()))

    fullModel = WeightNet(xNorm.shape[1], len(weights))
    optimizer = torch.optim.AdamW(fullModel.parameters(), lr=lr, weight_decay=weightDecay)
    regLossFn = nn.HuberLoss(delta=1.0)
    clsLossFn = nn.CrossEntropyLoss()
    xAll = torch.tensor(xNorm, dtype=torch.float32)
    yAll = torch.tensor(yNorm, dtype=torch.float32).unsqueeze(1)
    clsAll = torch.tensor(labels, dtype=torch.long)
    for epoch in range(1, epochs + 1):
        fullModel.train()
        optimizer.zero_grad()
        regPred, clsPred = fullModel(xAll)
        loss = 0.8 * regLossFn(regPred, yAll) + 0.2 * clsLossFn(clsPred, clsAll)
        loss.backward()
        optimizer.step()
        lossValue = float(loss.detach().cpu().item())
        if progressCallback is not None:
            progressCallback("weight_from_phone", epoch, epochs, lossValue)
        elif epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print(f"[weight_from_phone MLP] epoch={epoch:04d}/{epochs} loss={lossValue:.4f}", flush=True)

    predArr = np.asarray(looPred, dtype=np.float64)
    metrics = regressionMetrics(predArr, y, "G")
    metrics["classificationAccuracy"] = float(np.mean(np.asarray(looCls, dtype=np.int64) == labels))
    return {
        "modelType": "mlp_weight",
        "architecture": "input-32-16-8-regression-classification",
        "featureNames": FEATURE_NAMES,
        "xMean": mean.tolist(),
        "xStd": std.tolist(),
        "yMean": yMean,
        "yStd": yStd,
        "stateDict": fullModel.state_dict(),
        "labelMap": labelMap,
        "weightsG": weights,
        "metrics": metrics,
        "perSample": perSample(rows, y, predArr, "G"),
    }


def candidateSelectionScore(candidate: dict, unit: str) -> float:
    metrics = candidate["metrics"]
    mae = float(metrics[f"mae{unit}"])
    rmse = float(metrics[f"rmse{unit}"])
    maxAbs = float(metrics[f"maxAbsError{unit}"])
    score = mae + 0.25 * rmse + 0.10 * maxAbs
    metrics["selectionScore"] = score
    return score


def selectCandidate(candidates: list[dict], unit: str) -> dict:
    candidates.sort(key=lambda item: (candidateSelectionScore(item, unit), item["metrics"][f"mae{unit}"]))
    return candidates[0]


def trainWeightFromPhone(
    rows: list[dict],
    epochs: int = 800,
    lr: float = 1e-3,
    weightDecay: float = 1e-3,
    progressCallback: Callable[[str, int, int, float], None] | None = None,
) -> dict:
    x = phoneFeatureMatrix(rows)
    y = targetArray(rows, "weightG")
    ridge = trainRidgeCandidate(rows, x, y, "G", FEATURE_NAMES)
    ridge["weightsG"] = sorted({float(value) for value in y.tolist()})
    monotonic = trainMonotonicPiecewiseCandidate(
        rows,
        targetArray(rows, "deflectionMeanMm"),
        y,
        "G",
        "deflectionMeanMm",
        "monotonic_phone_weight",
    )
    monotonic["weightsG"] = ridge["weightsG"]
    chained = trainChainedMonotonicWeightCandidate(rows)
    mlp = trainWeightMlpCandidate(rows, x, y, epochs, lr, weightDecay, progressCallback)
    candidates = [ridge, monotonic, chained] + ([mlp] if mlp is not None else [])
    recommended = selectCandidate(candidates, "G")
    return {"taskName": "weight_from_phone", "target": "weightG", "recommended": recommended["modelType"], "candidates": candidates}


def trainDeflectionFromWeight(
    rows: list[dict],
    epochs: int = 800,
    lr: float = 1e-3,
    weightDecay: float = 1e-3,
    progressCallback: Callable[[str, int, int, float], None] | None = None,
) -> dict:
    x = weightFeatureMatrixFromRows(rows)
    y = targetArray(rows, "standardDeflectionMm")
    ridge = trainRidgeCandidate(rows, x, y, "Mm", WEIGHT_FEATURE_NAMES)
    monotonic = trainMonotonicPiecewiseCandidate(
        rows,
        targetArray(rows, "weightG"),
        y,
        "Mm",
        "weightG",
        "monotonic_weight_deflection",
    )
    mlp = trainRegressionMlpCandidate(
        rows,
        x,
        y,
        "Mm",
        WEIGHT_FEATURE_NAMES,
        epochs,
        lr,
        weightDecay,
        "deflection_from_weight",
        progressCallback,
    )
    candidates = [ridge, monotonic] + ([mlp] if mlp is not None else [])
    recommended = selectCandidate(candidates, "Mm")
    return {"taskName": "deflection_from_weight", "target": "standardDeflectionMm", "recommended": recommended["modelType"], "candidates": candidates}


def trainLaserFromPhone(
    rows: list[dict],
    epochs: int = 800,
    lr: float = 1e-3,
    weightDecay: float = 1e-3,
    progressCallback: Callable[[str, int, int, float], None] | None = None,
) -> dict:
    x = phoneFeatureMatrix(rows)
    y = targetArray(rows, "standardDeflectionMm")
    ridge = trainRidgeCandidate(rows, x, y, "Mm", FEATURE_NAMES)
    monotonic = trainMonotonicPiecewiseCandidate(
        rows,
        targetArray(rows, "deflectionMeanMm"),
        y,
        "Mm",
        "deflectionMeanMm",
        "monotonic_phone_deflection",
    )
    mlp = trainRegressionMlpCandidate(
        rows,
        x,
        y,
        "Mm",
        FEATURE_NAMES,
        epochs,
        lr,
        weightDecay,
        "laser_from_phone",
        progressCallback,
    )
    candidates = [ridge, monotonic] + ([mlp] if mlp is not None else [])
    recommended = selectCandidate(candidates, "Mm")
    return {"taskName": "laser_from_phone", "target": "standardDeflectionMm", "recommended": recommended["modelType"], "candidates": candidates}


def trainAllTasks(
    rows: list[dict],
    epochs: int,
    lr: float,
    weightDecay: float,
    progressCallback: Callable[[str, int, int, float], None] | None = None,
) -> dict:
    return {
        "bundleType": "bridge_task_models",
        "version": 1,
        "rowCount": len(rows),
        "sourceSampleIds": [str(row.get("sampleId", "")) for row in rows],
        "weightRangeG": [float(min(targetArray(rows, "weightG"))), float(max(targetArray(rows, "weightG")))],
        "tasks": {
            "weight_from_phone": trainWeightFromPhone(rows, epochs, lr, weightDecay, progressCallback),
            "deflection_from_weight": trainDeflectionFromWeight(rows, epochs, lr, weightDecay, progressCallback),
            "laser_from_phone": trainLaserFromPhone(rows, epochs, lr, weightDecay, progressCallback),
        },
    }


def stripStateDicts(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: stripStateDicts(item) for key, item in value.items() if key != "stateDict"}
    if isinstance(value, list):
        return [stripStateDicts(item) for item in value]
    return value


def metricsSummary(bundle: dict) -> dict:
    return {
        taskName: {
            "target": taskPayload["target"],
            "recommended": taskPayload["recommended"],
            "candidates": [
                {
                    "modelType": candidate["modelType"],
                    "metrics": candidate["metrics"],
                    "perSample": candidate["perSample"],
                }
                for candidate in taskPayload["candidates"]
            ],
        }
        for taskName, taskPayload in bundle["tasks"].items()
    }


def saveModelBundle(bundle: dict, outputDir: Path) -> dict:
    outputDir.mkdir(parents=True, exist_ok=True)
    jsonPath = outputDir / "model_bundle.json"
    pthPath = outputDir / "model_bundle.pth"
    metricsPath = outputDir / "metrics.json"
    featurePath = outputDir / "feature_config.json"
    saveJson(stripStateDicts(bundle), jsonPath)
    saveJson(metricsSummary(bundle), metricsPath)
    saveJson(
        {
            "phoneFeatureNames": FEATURE_NAMES,
            "weightFeatureNames": WEIGHT_FEATURE_NAMES,
            "defaultWindowCsvs": DEFAULT_WINDOW_CSVS,
        },
        featurePath,
    )
    try:
        import torch

        torch.save(bundle, str(pthPath))
        modelPath = pthPath
    except ImportError:
        modelPath = jsonPath
    return {
        "modelPath": str(modelPath),
        "jsonPath": str(jsonPath),
        "metricsPath": str(metricsPath),
        "featureConfigPath": str(featurePath),
    }


def loadModelBundle(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import torch

        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        import torch

        return torch.load(str(path), map_location="cpu")


def recommendedCandidate(taskPayload: dict) -> dict:
    for candidate in taskPayload["candidates"]:
        if candidate["modelType"] == taskPayload["recommended"] and (
            not candidate["modelType"].startswith("mlp") or "stateDict" in candidate
        ):
            return candidate
    for candidate in taskPayload["candidates"]:
        if not candidate["modelType"].startswith("mlp") or "stateDict" in candidate:
            return candidate
    return taskPayload["candidates"][0]


def predictRidge(candidate: dict, features: np.ndarray) -> float:
    design = polynomialDesign(features, int(candidate["degree"]))
    return float((design @ np.asarray(candidate["coef"], dtype=np.float64))[0])


def predictRegressionMlp(candidate: dict, features: np.ndarray) -> float:
    import torch
    import torch.nn as nn

    class RegressionNet(nn.Module):
        def __init__(self, inputDim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(inputDim, 32),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, data: Any) -> Any:
            return self.net(data)

    xMean = np.asarray(candidate["xMean"], dtype=np.float64)
    xStd = np.asarray(candidate["xStd"], dtype=np.float64)
    yMean = float(candidate.get("yMean", 0.0))
    yStd = float(candidate.get("yStd", 1.0))
    xNorm = (features - xMean) / xStd
    model = RegressionNet(features.shape[1])
    model.load_state_dict(candidate["stateDict"])
    model.eval()
    with torch.no_grad():
        predNorm = float(model(torch.tensor(xNorm, dtype=torch.float32)).reshape(-1)[0].cpu().item())
        return predNorm * yStd + yMean


def predictWeightMlp(candidate: dict, features: np.ndarray) -> tuple[float, list[float]]:
    import torch
    import torch.nn as nn

    weights = [float(value) for value in candidate["weightsG"]]

    class WeightNet(nn.Module):
        def __init__(self, inputDim: int, classCount: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(inputDim, 32),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
            )
            self.regHead = nn.Linear(8, 1)
            self.clsHead = nn.Linear(8, classCount)

        def forward(self, data: Any) -> tuple[Any, Any]:
            hidden = self.backbone(data)
            return self.regHead(hidden), self.clsHead(hidden)

    xMean = np.asarray(candidate["xMean"], dtype=np.float64)
    xStd = np.asarray(candidate["xStd"], dtype=np.float64)
    yMean = float(candidate.get("yMean", 0.0))
    yStd = float(candidate.get("yStd", 1.0))
    xNorm = (features - xMean) / xStd
    model = WeightNet(features.shape[1], len(weights))
    model.load_state_dict(candidate["stateDict"])
    model.eval()
    with torch.no_grad():
        regPred, clsPred = model(torch.tensor(xNorm, dtype=torch.float32))
        predG = float(regPred.reshape(-1)[0].cpu().item()) * yStd + yMean
        probs = torch.softmax(clsPred, dim=1).reshape(-1).cpu().numpy().tolist()
        return predG, [float(value) for value in probs]


def predictChainedMonotonicWeight(candidate: dict, featureRow: dict) -> float:
    phoneMm = float(featureRow["deflectionMeanMm"])
    predictedStandard = predictMonotonicAnchors(candidate["phoneToStandard"], phoneMm)
    return predictMonotonicAnchors(candidate["standardToWeight"], predictedStandard)


def predictWeightFromPhone(bundle: dict, featureRow: dict) -> dict:
    task = bundle["tasks"]["weight_from_phone"]
    candidate = recommendedCandidate(task)
    features = np.asarray([[float(featureRow[name]) for name in FEATURE_NAMES]], dtype=np.float64)
    if candidate["modelType"] == "mlp_weight":
        predG, probs = predictWeightMlp(candidate, features)
    elif candidate["modelType"] == "monotonic_phone_weight":
        predG = predictMonotonicAnchors(candidate, float(featureRow["deflectionMeanMm"]))
        probs = []
    elif candidate["modelType"] == "chained_monotonic":
        predG = predictChainedMonotonicWeight(candidate, featureRow)
        probs = []
    else:
        predG = predictRidge(candidate, features)
        probs = []
    weights = [float(value) for value in candidate.get("weightsG", [])]
    if not weights:
        weights = sorted({float(item[f"trueG"]) for item in candidate["perSample"]})
    nearestWeight, nearestIndex = nearestClass(predG, weights)
    return {
        "predictedWeightG": predG,
        "nearestWeightG": nearestWeight,
        "nearestIndex": nearestIndex,
        "modelType": candidate["modelType"],
        "task": "weight_from_phone",
        "classProbabilities": probs,
    }


def predictStandardDeflectionFromWeight(bundle: dict, weightG: float) -> dict:
    task = bundle["tasks"]["deflection_from_weight"]
    candidate = recommendedCandidate(task)
    features = weightFeatureMatrix(np.asarray([float(weightG)], dtype=np.float64))
    if candidate["modelType"] == "mlp_regression":
        predMm = predictRegressionMlp(candidate, features)
    elif candidate["modelType"] == "monotonic_weight_deflection":
        predMm = predictMonotonicAnchors(candidate, float(weightG))
    else:
        predMm = predictRidge(candidate, features)
    return {
        "predictedStandardDeflectionMm": predMm,
        "modelType": candidate["modelType"],
        "task": "deflection_from_weight",
        "inputWeightG": float(weightG),
    }


def predictStandardDeflectionFromPhone(bundle: dict, featureRow: dict) -> dict:
    task = bundle["tasks"]["laser_from_phone"]
    candidate = recommendedCandidate(task)
    features = np.asarray([[float(featureRow[name]) for name in FEATURE_NAMES]], dtype=np.float64)
    if candidate["modelType"] == "mlp_regression":
        predMm = predictRegressionMlp(candidate, features)
    elif candidate["modelType"] == "monotonic_phone_deflection":
        predMm = predictMonotonicAnchors(candidate, float(featureRow["deflectionMeanMm"]))
    else:
        predMm = predictRidge(candidate, features)
    phoneMm = float(featureRow["deflectionMeanMm"])
    return {
        "predictedStandardDeflectionMm": predMm,
        "phoneDeflectionMm": phoneMm,
        "predictedPhoneMinusStandardMm": phoneMm - predMm,
        "modelType": candidate["modelType"],
        "task": "laser_from_phone",
    }
