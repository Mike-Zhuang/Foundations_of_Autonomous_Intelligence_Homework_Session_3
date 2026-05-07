from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bridge_ai.weight_features import FEATURE_NAMES


RIDGE_LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def loadRows(csvPath: Path) -> list[dict]:
    import csv

    with csvPath.open("r", encoding="utf-8", newline="") as fileObj:
        return list(csv.DictReader(fileObj))


def saveJson(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def buildMatrices(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[float], dict[str, int]]:
    if not rows:
        raise ValueError("训练数据为空。")
    x = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    y = np.asarray([float(row["weightG"]) for row in rows], dtype=np.float64)
    weights = sorted({float(value) for value in y.tolist()})
    labelMap = {str(weight): index for index, weight in enumerate(weights)}
    return x, y, weights, labelMap


def polynomialDesign(x: np.ndarray, degree: int) -> np.ndarray:
    main = x[:, [0]]
    if degree <= 1:
        return np.hstack([np.ones((x.shape[0], 1), dtype=np.float64), main])
    return np.hstack([np.ones((x.shape[0], 1), dtype=np.float64), main, main * main])


def ridgeFit(design: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    eye = np.eye(design.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + lam * eye) @ design.T @ y


def evaluateRegression(pred: np.ndarray, y: np.ndarray) -> dict:
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(err * err) / max(denom, 1e-9))
    return {"maeG": mae, "rmseG": rmse, "r2": r2}


def nearestClass(predWeight: float, weights: list[float]) -> tuple[float, int]:
    arr = np.asarray(weights, dtype=np.float64)
    index = int(np.argmin(np.abs(arr - predWeight)))
    return float(arr[index]), index


def trainRidgeModels(rows: list[dict]) -> dict:
    x, y, weights, labelMap = buildMatrices(rows)
    candidates: list[dict] = []

    for degree in (1, 2):
        for lam in RIDGE_LAMBDAS:
            looPred: list[float] = []
            for holdout in range(len(rows)):
                trainMask = np.ones(len(rows), dtype=bool)
                trainMask[holdout] = False
                if np.sum(trainMask) == 0:
                    trainMask[holdout] = True
                designTrain = polynomialDesign(x[trainMask], degree)
                coef = ridgeFit(designTrain, y[trainMask], lam)
                pred = polynomialDesign(x[[holdout]], degree) @ coef
                looPred.append(float(pred[0]))

            looPredArr = np.asarray(looPred, dtype=np.float64)
            metrics = evaluateRegression(looPredArr, y)
            candidates.append(
                {
                    "degree": degree,
                    "lambda": lam,
                    "looPredictions": looPred,
                    "metrics": metrics,
                }
            )

    candidates.sort(key=lambda item: (item["metrics"]["maeG"], item["metrics"]["rmseG"]))
    best = candidates[0]
    coef = ridgeFit(polynomialDesign(x, int(best["degree"])), y, float(best["lambda"]))
    predFull = polynomialDesign(x, int(best["degree"])) @ coef
    return {
        "modelType": "ridge",
        "degree": int(best["degree"]),
        "lambda": float(best["lambda"]),
        "coef": coef.tolist(),
        "featureNames": FEATURE_NAMES,
        "labelMap": labelMap,
        "weightsG": weights,
        "metrics": {
            **best["metrics"],
            "trainMaeG": evaluateRegression(predFull, y)["maeG"],
        },
        "perSample": [
            {
                "sampleId": str(rows[index].get("sampleId", index)),
                "trueG": float(y[index]),
                "predG": float(best["looPredictions"][index]),
                "absErrorG": float(abs(best["looPredictions"][index] - y[index])),
            }
            for index in range(len(rows))
        ],
    }


def trainMlpModel(rows: list[dict], epochs: int, lr: float, weightDecay: float, verboseEvery: int = 25) -> dict:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("缺少 torch，无法训练 MLP。") from exc

    x, y, weights, labelMap = buildMatrices(rows)
    labels = np.asarray([labelMap[str(float(value))] for value in y.tolist()], dtype=np.int64)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    xNorm = (x - mean) / std

    class TinyWeightNet(nn.Module):
        def __init__(self, inputDim: int, classCount: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(inputDim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
            )
            self.regHead = nn.Linear(8, 1)
            self.clsHead = nn.Linear(8, classCount)

        def forward(self, features: Any) -> tuple[Any, Any]:
            hidden = self.backbone(features)
            return self.regHead(hidden), self.clsHead(hidden)

    looPred: list[float] = []
    looCls: list[int] = []
    bestStateFull = None

    for holdout in range(len(rows)):
        trainMask = np.ones(len(rows), dtype=bool)
        trainMask[holdout] = False
        if np.sum(trainMask) == 0:
            trainMask[holdout] = True

        model = TinyWeightNet(inputDim=xNorm.shape[1], classCount=len(weights))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weightDecay)
        regLossFn = nn.HuberLoss(delta=5.0)
        clsLossFn = nn.CrossEntropyLoss()

        xTrain = torch.tensor(xNorm[trainMask], dtype=torch.float32)
        yTrain = torch.tensor(y[trainMask], dtype=torch.float32).unsqueeze(1)
        clsTrain = torch.tensor(labels[trainMask], dtype=torch.long)
        xVal = torch.tensor(xNorm[[holdout]], dtype=torch.float32)

        bestLoss = float("inf")
        patience = 80
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            regPred, clsPred = model(xTrain)
            loss = 0.75 * regLossFn(regPred, yTrain) + 0.25 * clsLossFn(clsPred, clsTrain)
            loss.backward()
            optimizer.step()

            currentLoss = float(loss.detach().cpu().item())
            if currentLoss + 1e-7 < bestLoss:
                bestLoss = currentLoss
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        model.eval()
        with torch.no_grad():
            regPred, clsPred = model(xVal)
            looPred.append(float(regPred.reshape(-1)[0].cpu().item()))
            looCls.append(int(torch.argmax(clsPred, dim=1).cpu().item()))

    fullModel = TinyWeightNet(inputDim=xNorm.shape[1], classCount=len(weights))
    optimizer = torch.optim.AdamW(fullModel.parameters(), lr=lr, weight_decay=weightDecay)
    regLossFn = nn.HuberLoss(delta=5.0)
    clsLossFn = nn.CrossEntropyLoss()
    xAll = torch.tensor(xNorm, dtype=torch.float32)
    yAll = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    clsAll = torch.tensor(labels, dtype=torch.long)
    for epoch in range(1, epochs + 1):
        fullModel.train()
        optimizer.zero_grad()
        regPred, clsPred = fullModel(xAll)
        loss = 0.75 * regLossFn(regPred, yAll) + 0.25 * clsLossFn(clsPred, clsAll)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % max(verboseEvery, 1) == 0 or epoch == epochs:
            print(f"[MLP] epoch={epoch:04d}/{epochs} loss={float(loss.detach().cpu().item()):.4f}", flush=True)
    bestStateFull = fullModel.state_dict()

    looPredArr = np.asarray(looPred, dtype=np.float64)
    metrics = evaluateRegression(looPredArr, y)
    clsAcc = float(np.mean(np.asarray(looCls, dtype=np.int64) == labels))
    metrics["classificationAccuracy"] = clsAcc
    return {
        "modelType": "mlp",
        "featureNames": FEATURE_NAMES,
        "xMean": mean.tolist(),
        "xStd": std.tolist(),
        "stateDict": bestStateFull,
        "labelMap": labelMap,
        "weightsG": weights,
        "metrics": metrics,
        "perSample": [
            {
                "sampleId": str(rows[index].get("sampleId", index)),
                "trueG": float(y[index]),
                "predG": float(looPred[index]),
                "absErrorG": float(abs(looPred[index] - y[index])),
            }
            for index in range(len(rows))
        ],
    }


def trainAndSelect(rows: list[dict], epochs: int, lr: float, weightDecay: float) -> dict:
    print("开始训练稳健拟合模型（线性/二次/岭回归，LOOCV）...", flush=True)
    ridge = trainRidgeModels(rows)
    print(
        f"[Ridge] degree={ridge['degree']} lambda={ridge['lambda']} "
        f"LOOCV MAE={ridge['metrics']['maeG']:.3f}g RMSE={ridge['metrics']['rmseG']:.3f}g",
        flush=True,
    )

    mlp = None
    try:
        print("开始训练小 MLP（LOOCV + 全量重训）...", flush=True)
        mlp = trainMlpModel(rows, epochs=epochs, lr=lr, weightDecay=weightDecay)
        print(
            f"[MLP] LOOCV MAE={mlp['metrics']['maeG']:.3f}g "
            f"RMSE={mlp['metrics']['rmseG']:.3f}g Acc={mlp['metrics']['classificationAccuracy']:.3f}",
            flush=True,
        )
    except RuntimeError as exc:
        print(f"MLP 训练跳过: {exc}", flush=True)

    candidates = [ridge]
    if mlp is not None:
        candidates.append(mlp)
    candidates.sort(key=lambda item: (item["metrics"]["maeG"], item["metrics"]["rmseG"]))
    best = candidates[0]
    best["allCandidates"] = [
        {
            "modelType": item["modelType"],
            "metrics": item["metrics"],
        }
        for item in candidates
    ]
    return best


def saveBestModel(modelPayload: dict, outputDir: Path) -> dict:
    outputDir.mkdir(parents=True, exist_ok=True)
    modelJson = {
        key: value
        for key, value in modelPayload.items()
        if key not in {"stateDict"}
    }
    saveJson(modelJson, outputDir / "best_weight_model.json")
    saveJson(
        {
            "modelType": modelPayload["modelType"],
            "metrics": modelPayload["metrics"],
            "perSample": modelPayload["perSample"],
            "allCandidates": modelPayload.get("allCandidates", []),
        },
        outputDir / "metrics.json",
    )
    saveJson({"featureNames": FEATURE_NAMES}, outputDir / "feature_config.json")
    saveJson(modelPayload["labelMap"], outputDir / "label_map.json")

    modelPath = outputDir / "best_weight_model.pth"
    try:
        import torch

        torch.save(modelPayload, str(modelPath))
    except ImportError:
        modelPath = outputDir / "best_weight_model.json"

    return {
        "modelPath": str(modelPath),
        "jsonPath": str(outputDir / "best_weight_model.json"),
        "metricsPath": str(outputDir / "metrics.json"),
    }


def loadModel(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import torch

        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        import torch

        return torch.load(str(path), map_location="cpu")


def predictWeight(modelPayload: dict, row: dict) -> dict:
    features = np.asarray([[float(row[name]) for name in FEATURE_NAMES]], dtype=np.float64)
    modelType = modelPayload["modelType"]
    weights = [float(value) for value in modelPayload["weightsG"]]

    if modelType == "ridge":
        design = polynomialDesign(features, int(modelPayload["degree"]))
        predG = float((design @ np.asarray(modelPayload["coef"], dtype=np.float64))[0])
    elif modelType == "mlp":
        import torch
        import torch.nn as nn

        class TinyWeightNet(nn.Module):
            def __init__(self, inputDim: int, classCount: int) -> None:
                super().__init__()
                self.backbone = nn.Sequential(nn.Linear(inputDim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
                self.regHead = nn.Linear(8, 1)
                self.clsHead = nn.Linear(8, classCount)

            def forward(self, data: Any) -> tuple[Any, Any]:
                hidden = self.backbone(data)
                return self.regHead(hidden), self.clsHead(hidden)

        xMean = np.asarray(modelPayload["xMean"], dtype=np.float64)
        xStd = np.asarray(modelPayload["xStd"], dtype=np.float64)
        xNorm = (features - xMean) / xStd
        model = TinyWeightNet(inputDim=len(FEATURE_NAMES), classCount=len(weights))
        model.load_state_dict(modelPayload["stateDict"])
        model.eval()
        with torch.no_grad():
            regPred, clsPred = model(torch.tensor(xNorm, dtype=torch.float32))
            predG = float(regPred.reshape(-1)[0].cpu().item())
            clsProb = torch.softmax(clsPred, dim=1).reshape(-1).cpu().numpy()
    else:
        raise ValueError(f"未知模型类型: {modelType}")

    nearestWeight, nearestIndex = nearestClass(predG, weights)
    result = {
        "predictedWeightG": predG,
        "nearestWeightG": nearestWeight,
        "nearestIndex": nearestIndex,
        "modelType": modelType,
    }
    if modelType == "mlp":
        result["classProbabilities"] = [float(value) for value in clsProb.tolist()]
    return result
