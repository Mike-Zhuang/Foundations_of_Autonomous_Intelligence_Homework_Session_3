# 桥梁跨中挠度测量系统（YOLO + ArUco）

本文档是本仓库的唯一标准操作说明。目标是让你从零开始，按本文一步一步执行后，能稳定得到可复现实验结果，不依赖口头补充。

## 1. 目标与硬约束

### 1.1 目标

- 任务：测量桥梁跨中静挠度。
- 输出：实时显示、逐帧 CSV、离线 summary 统计。
- 目标精度：至少 0.1 cm（即 1 mm 级）。

### 1.2 本实现的测量边界（必须理解）

- 该实现使用平面单应矩阵（Homography）做像素到毫米映射。
- 因此，静态标定板与跨中标记应尽量处于同一观测平面。
- 若跨中标记明显偏离该平面（前后景深差大），会引入系统误差。

## 2. 代码结构与职责

- yolo/test.py：最小摄像头 YOLO 连通性测试。
- yolo/generate_marker_board.py：生成静态标定板和跨中标记图。
- yolo/run_deflection_realtime.py：实时测量入口。
- yolo/run_deflection_offline.py：离线复算入口。
- yolo/train_midpoint_yolo.py：砝码重量数据采集 + 双任务模型（回归+分类）训练入口。
- yolo/bridge_ai/config.py：静态标记布局定义与读写。
- yolo/bridge_ai/geometry.py：ArUco 检测 + 单应矩阵解算。
- yolo/bridge_ai/detection.py：跨中目标检测（YOLO 主路径 + ArUco 回退 + 同心圆回退）。
- yolo/bridge_ai/deflection.py：基线标定与卡尔曼滤波。
- yolo/bridge_ai/io_utils.py：视频源、CSV、视频写出工具。
- yolo/bridge_ai/calibration.py：ChArUco 标定、相机内参保存与去畸变。

## 3. 环境准备

## 3.1 Conda 环境

推荐使用环境名 fai：

```bash
conda activate fai
```

若未安装依赖，至少保证以下可导入：

```bash
python -c "import cv2; import ultralytics; print('ok')"
```

## 3.2 macOS + iPhone 连续互通相机

必须满足：

1. iPhone 与 Mac 登录同一 Apple ID。
2. 蓝牙和 Wi-Fi 打开。
3. iPhone 解锁并靠近 Mac。

## 4. 标定标记生成、尺寸、坐标、打印

## 4.1 生成命令

在仓库根目录执行：

```bash
conda run -n fai python yolo/generate_marker_board.py
```

提示：`conda run` 默认可能缓存输出。若希望在运行中实时看到命令行提示，请用：

```bash
conda run --no-capture-output -n fai python ...
```

默认输出目录：yolo/artifacts

产物：

- static_marker_id10_40mm.png
- static_marker_id11_40mm.png
- static_marker_id12_40mm.png
- static_marker_id13_40mm.png
- static_marker_id14_40mm.png
- static_marker_layout.json
- midpoint_fallback_aruco_id42_50mm.png
- midpoint_circle_marker_50mm.png
- charuco_5x7_24mm_18mm.png
- marker_print_notes.txt

静态标记命名规则：

- static*marker_id{ID}*{SIZE}mm.png
- 示例：static_marker_id10_40mm.png

## 4.2 打印硬参数（必须）

- 打印缩放：100% 原尺寸（禁止适应页面、禁止自动缩放）。
- 默认分辨率：300 DPI。
- 每个标记独立打印，方便你单独调节每个标记最终纸面尺寸。

## 4.3 静态标定点位（5 个 ArUco）

字典固定：DICT_5X5_250。

单个静态标记边长：40 mm。

以下坐标均为毫米（mm），用于几何映射坐标系（写入 static_marker_layout.json）。
坐标系定义：A4 竖版，原点在左上角，纸面尺寸 210×297 mm。

| ID  | 标记左上角 (x,y) | 边长 | 标记中心 (x,y) |
| --- | ---------------: | ---: | -------------: |
| 10  |           (8, 8) |   40 |       (28, 28) |
| 11  |         (162, 8) |   40 |      (182, 28) |
| 12  |       (162, 249) |   40 |     (182, 269) |
| 13  |         (8, 249) |   40 |      (28, 269) |
| 14  |       (85, 128)  |   40 |     (105, 148) |

说明：

- 该布局采用“四角 + 中心”，几何基线更长，能更好抑制透视误差与拟合不稳定。
- 贴附时必须保证每个静态标记的打印边长确实为 40 mm，且 A4 打印为 100% 原尺寸。

## 4.4 跨中目标标记（主目标）

文件：midpoint_circle_marker_50mm.png

- 外框尺寸：50 mm × 50 mm。
- 黑色外圆直径：约 35 mm。
- 白色内圆直径：约 12 mm（脚本按像素整数近似）。

## 4.5 跨中回退标记（备用）

文件：midpoint_fallback_aruco_id42_50mm.png

- 字典：DICT_5X5_250。
- 标记 ID：42。
- 边长：50 mm。

当 YOLO 模型缺失、加载失败或未检出目标时，系统会自动尝试该回退标记。

## 4.6 打印后实物验收（必须执行）

使用游标卡尺或钢尺逐项核验：

1. 任一静态标记边长应为 40.0 mm。
2. 圆形主目标外框边长应为 50.0 mm。
3. 以 ID10 和 ID11 为例，其左上角水平间距应为 154.0 mm。

建议容差：±0.3 mm。若超差，重新打印，直到满足。

## 5. 标定点位贴附位置（无歧义版本）

## 5.1 静态标定板贴在哪里

必须同时满足：

1. 贴在绝对静止、不随桥体运动的刚性背景上。
2. 贴在跨中标记附近，尽量同一测量平面。
3. 全程完整入镜，不能被小车、手、支架遮挡。

推荐：贴在桥梁侧面外侧同平面的固定背板上，距离跨中标记中心不超过 300 mm。

## 5.2 跨中标记贴在哪里

桥梁跨径 L = 1200 mm 时：

- 跨中位置定义：距左支座 600 mm（或距右支座 600 mm）。
- 贴附点：桥底或桥侧的跨中截面中心附近。

要求：

1. 标记面尽量正对相机（减小透视畸变）。
2. 标记四周不要有高对比纹理干扰。
3. 使用哑光材质，避免反光。

## 5.3 方向与符号约定

- 系统用 worldYmm 做挠度估计。
- 计算公式：deflectionMm = worldYmm - baselineMm。
- 在常规“图像向下为正”的安装方式下，下挠通常为正值。

第一次实验请手动做 2~3 mm 已知上下位移，确认符号方向后再正式采集。

## 6. 相机安装与拍摄参数

## 6.1 安装要求

1. 相机固定在三脚架上，不手持。
2. 视角尽量侧视，避免过大俯仰角。
3. 画面中同时完整包含：5 个静态标记 + 跨中标记。

## 6.2 建议参数

- 分辨率：1920×1080 或更高。
- 帧率：30 fps（建议）或 60 fps。
- 曝光：尽量固定，避免自动曝光频繁跳变。
- 对焦：锁定在桥梁与标定板平面。

## 6.3 去畸变标定（ChArUco）

实时/离线/训练脚本都支持：

- `--calibration-mode off`：关闭去畸变。
- `--calibration-mode use`：加载历史标定参数（默认）。
- `--calibration-mode recalibrate`：现场重标定并写入标定文件。

默认标定文件：`yolo/artifacts/camera_calibration.npz`

推荐 ChArUco 板参数：

- squaresX = 5
- squaresY = 7
- squareLength = 24 mm
- markerLength = 18 mm

重标定时键盘操作：

- `c` 抓取当前姿态样本（建议 25~40 帧）
- `q` 取消标定

## 6.4 什么时候要重标定（最重要）

按下面规则判断即可：

1. **第一次做实验**：必须重标定（`--calibration-mode recalibrate`）。
2. **相机位置变了**（高度、角度、左右位置变化）：重标定。
3. **相机参数变了**（分辨率、变焦、对焦策略明显变化）：重标定。
4. **换了设备**（换手机/换摄像头）：重标定。
5. **仅重复同机位同参数实验**：直接复用（`--calibration-mode use`）。

一句话版：

- 不确定就先重标定一次；
- 确定机位和参数都没变就用 `use`。

## 6.5 ChArUco 尺寸不要弄错

`squareLength=24mm` 与 `markerLength=18mm` 说的是**单个格子内部尺寸**，不是整张图尺寸。

5x7 板整板尺寸应为：

- 宽 = `5 × 24 = 120 mm`
- 高 = `7 × 24 = 168 mm`

打印时请确保：

- 100% 原始尺寸；
- 禁止“适应页面/自动缩放”；
- 打印后用尺量：任意方格边长约 24mm、黑块边长约 18mm。

## 7. 首次连通性检查

先确认 iPhone 连续互通索引：

```bash
conda run -n fai python yolo/test.py --source 0
```

若不是 iPhone 画面，依次尝试：

```bash
conda run -n fai python yolo/test.py --source 1
conda run -n fai python yolo/test.py --source 2
```

找到正确索引后，记录下来用于实时脚本。

## 8. 实时测量标准流程

## 8.1 启动命令

```bash
conda run --no-capture-output -n fai python yolo/run_deflection_realtime.py \
  --source 0 \
  --model yolov8n.pt \
  --layout yolo/artifacts/static_marker_layout.json \
  --target-class midpoint_marker \
  --conf 0.25 \
  --imgsz 640 \
  --baseline-frames 60 \
  --calibration-mode use \
  --calibration-file yolo/artifacts/camera_calibration.npz \
  --overlay-level debug
```

### 8.1.1 首次实验推荐命令（会重标定）

```bash
conda run --no-capture-output -n fai python yolo/run_deflection_realtime.py \
  --source 0 \
  --calibration-mode recalibrate \
  --overlay-level debug
```

### 8.1.2 后续重复实验推荐命令（复用标定）

```bash
conda run --no-capture-output -n fai python yolo/run_deflection_realtime.py \
  --source 0 \
  --calibration-mode use \
  --calibration-file yolo/artifacts/camera_calibration.npz \
  --overlay-level balanced
```

## 8.2 基线阶段（非常关键）

- 前 baseline-frames 帧处于 calibrating-baseline 状态。
- 此阶段必须无载荷、无外力、无触碰。
- 基线值 baselineMm 取这些帧 worldYmm 的中位数。

## 8.3 实时结果解释

界面显示：

- Deflection(cm)：滤波后的挠度值。
- Baseline(mm)：基线 worldY。
- Status：当前状态。

状态含义：

- calibrating-baseline：正在标定基线。
- tracking:yolo：YOLO 检测并跟踪成功。
- tracking:yolo-fallback-top1：YOLO 未命中目标类，使用最高置信框。
- tracking:fallback-aruco：YOLO 未用上，使用 ID42 回退标记。
- tracking:fallback-circle：YOLO 与 ArUco 回退都未命中时，启用同心圆回退检测。
- missing:no-static-markers：静态标定点不足，无法解算。
- missing:low-homography-quality：静态点已识别，但几何质量不达标（RMSE/内点比/点数门控失败）。
- missing:\*：目标点缺失，当前帧无有效挠度。

窗口显示已汉化：你会看到“状态: 缺失：几何质量不足...”这类中文提示；CSV 里仍保留英文状态码，便于后处理与统计。

退出按键：q。

默认 CSV 输出到 yolo/results/realtime\_时间戳.csv。

`--overlay-level` 说明：

- `minimal`：仅保留核心值，遮挡最少。
- `balanced`：核心值 + 世界坐标。
- `debug`：右侧调试面板，显示检测来源、置信度、几何质量、短历史曲线。

## 9. 离线复算标准流程

## 9.1 启动命令

```bash
conda run --no-capture-output -n fai python yolo/run_deflection_offline.py \
  --video path/to/your_video.mp4 \
  --model yolov8n.pt \
  --layout yolo/artifacts/static_marker_layout.json \
  --target-class midpoint_marker \
  --conf 0.25 \
  --imgsz 640 \
  --baseline-frames 60 \
  --calibration-mode use \
  --calibration-file yolo/artifacts/camera_calibration.npz \
  --overlay-level debug
```

## 9.2 输出内容

- CSV：逐帧结果。
- summary JSON：
  - validFrameCount
  - maxDeflectionMm
  - minDeflectionMm
  - meanDeflectionMm
  - stdDeflectionMm

## 10. CSV 字段定义（统一口径）

CSV 字段固定为：

- timeSec：时间戳（秒）。
- rawMm：基线差原始值（mm）。
- filteredMm：卡尔曼滤波后（mm）。
- baselineMm：基线 worldY（mm）。
- deflectionCm：filteredMm / 10（cm）。
- confidence：检测置信度。
- status：状态字符串。

## 11. 砝码重量数据采集与模型训练

`yolo/train_midpoint_yolo.py` 已重构为“采集 + 训练”一体化入口，不再用于 YOLO 检测模型训练。

## 11.1 采集流程（交互式）

1. 启动脚本后，先按 `--calibration-mode` 执行去畸变流程（复用或重标定）。
2. 命令行输入当前砝码重量（单位 g）。
3. 在视频窗口按 `s` 开始采集，按 `e` 提前结束，或达到 `--min-valid-frames` 自动结束。
4. 更换砝码后继续输入下一重量；输入 `done` 结束全部采集。

避免混乱的小建议：

- 每个重量建议采集 2~3 轮；
- 每轮开始前等画面状态稳定到 `tracking:*` 再按 `s`；
- 若出现 `missing:low-homography-quality`，先调整机位/光照再重采，不要硬采。

## 11.2 训练命令（自动采集 + 自动训练）

```bash
conda run --no-capture-output -n fai python yolo/train_midpoint_yolo.py \
  --source 0 \
  --layout yolo/artifacts/static_marker_layout.json \
  --model yolov8n.pt \
  --calibration-mode use \
  --calibration-file yolo/artifacts/camera_calibration.npz \
  --min-valid-frames 120 \
  --capture-seconds 8 \
  --auto-train true \
  --epochs 220 \
  --batch-size 16 \
  --lr 1e-3
```

仅采集不训练（后续手动训练）：

```bash
conda run --no-capture-output -n fai python yolo/train_midpoint_yolo.py --auto-train false
```

## 11.3 特征与模型

- 采集输出：`weight_dataset_*.csv` + `*.metadata.json`
- 特征：挠度统计量（mean/std/p05/p50/p95 等）+ 质量指标（confidence/rmse/inlier/usedPoints）+ 稳定性指标
- 模型：前馈全连接双任务网络（共享 backbone，回归头 + 分类头）
- 损失：`0.7 * Huber + 0.3 * CrossEntropy`
- 输出：`best.pth`、`label_map.json`、`metrics.json`

## 12. 精度验收流程（建议按此提交实验结果）

## 12.1 已知位移台阶法

1. 空载采集 10 秒（生成基线）。
2. 施加已知位移台阶：1 mm、2 mm、5 mm、10 mm。
3. 每个台阶保持 5~10 秒。
4. 卸载回零，重复 3 次。

## 12.2 验收指标

- 单次误差：|测量值 - 真值|。
- 重复性：同台阶 3 次标准差。
- 目标建议：
  - 平均绝对误差 <= 1 mm
  - 重复性标准差 <= 1 mm

## 13. 常见问题与唯一处理方式

## 13.1 看不到静态标记

检查顺序（必须按顺序）：

1. 打印是否 100% 原比例。
2. 实测边长是否 40 mm。
3. 字典是否 DICT_5X5_250。
4. ID 是否 10~14。
5. 画面是否完整包含至少 2 个标记。

## 13.2 挠度输出为 nan 或大量 missing

1. 确认状态是否 missing:no-static-markers。
2. 增大标记在画面中的像素尺寸（靠近相机或放大打印）。
3. 降低反光并增强均匀照明。

## 13.3 YOLO 无法稳定识别

1. 使用你自己的训练权重替换 yolov8n.pt。
2. 临时贴上 ID42 回退标记验证链路。
3. 检查 --target-class 是否与 data.yaml 的类别名一致。

## 13.4 输出符号与物理方向相反

1. 先做 2~3 mm 人工已知位移测试。
2. 若符号反向，在结果分析阶段统一取负号，不要混改代码和数据口径。

## 14. 一次完整实验最短执行清单

1. 生成并打印标记。
2. 实测尺寸合格。
3. 按第 5 节贴附静态板与跨中标记。
4. 用 yolo/test.py 找到 iPhone 正确 source 索引。
5. 运行实时脚本完成基线 + 实验加载。
6. 保存 CSV（和可选视频）。
7. 用离线脚本复算并导出 summary。
8. 按第 12 节做精度验收并记录误差。

## 15. 一页式快速流程（建议直接照做）

1. 生成并打印标记（含 ChArUco 板）。
2. 相机固定好后，先跑一次 `recalibrate`。
3. 再跑实时脚本观察状态和调试面板，确认识别稳定。
4. 进入采集训练脚本，按“输入重量 -> 按 s 采集 -> done 结束”循环。
5. 若 `--auto-train true`，程序自动训练并输出 `best.pth` 与 `metrics.json`。

---

如果你严格按照本 README 执行，实验中每个关键环节（尺寸、位置、参数、数据格式、输出口径）都已经明确，不需要再依赖口头解释。
