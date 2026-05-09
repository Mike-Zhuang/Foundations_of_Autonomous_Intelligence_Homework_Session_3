# 桥梁跨中挠度测量系统（YOLO + ArUco）

本文档是本仓库的唯一标准操作说明。目标是让你从零开始，按本文一步一步执行后，能稳定得到可复现实验结果，不依赖口头补充。

## 1. 目标与硬约束

### 1.1 目标

- 任务：测量桥梁跨中静挠度。
- 输出：实时显示、逐帧 CSV、离线 summary 统计。
- 目标精度：至少 0.1 cm（即 1 mm 级）。

### 1.2 本实现的测量边界（必须理解）

- 当前推荐方法是 `target-local-scale`。
- ID42 目标标记负责主测量：程序用它真实的 50mm 边长估计目标点附近的局部像素/mm。
- 5 个静态 ArUco 标记负责提供不动参考方向和相机晃动补偿，不再把跨中目标强行投到静态板平面上。
- ChArUco 只用于相机内参与畸变标定；若使用 `target-local-scale`，它不是绝对必要，但仍建议保留以减少边缘畸变。
- ArUco/ChArUco 检测均使用 OpenCV，当前配置为 SUBPIX 精度优先角点精修；没有引入 AprilTag 检测器，也不依赖额外权重。

## 2. 代码结构与职责

- yolo/test.py：最小摄像头 YOLO 连通性测试。
- yolo/generate_marker_board.py：生成静态标定板和跨中标记图。
- yolo/run_deflection_realtime.py：实时测量入口。
- yolo/run_deflection_offline.py：离线复算入口。
- yolo/collect_weight_data.py：砝码重量数据采集入口。
- yolo/train_bridge_task_models.py：三任务桥梁模型训练入口（推荐）。
- yolo/train_weight_model.py：旧版单任务重量模型训练入口。
- yolo/predict_weight_realtime.py：实时重量预测入口。
- yolo/predict_deflection_from_weight.py：任务 B，给定重量预测激光标准挠度。
- yolo/predict_standard_deflection_from_phone.py：任务 C，手机挠度校正为激光标准挠度。
- yolo/train_midpoint_yolo.py：旧兼容入口，仅提示迁移到三段式流程。
- yolo/bridge_ai/config.py：静态标记布局定义与读写。
- yolo/bridge_ai/aruco_utils.py：OpenCV ArUco 精度优先检测参数。
- yolo/bridge_ai/geometry.py：ArUco 检测 + 单应矩阵解算。
- yolo/bridge_ai/detection.py：跨中目标检测（YOLO 主路径 + ArUco ID42 回退）。
- yolo/bridge_ai/deflection.py：基线标定与卡尔曼滤波。
- yolo/bridge_ai/io_utils.py：视频源、CSV、视频写出工具。
- yolo/bridge_ai/calibration.py：ChArUco 标定、相机内参保存与去畸变。
- yolo/bridge_ai/bridge_task_models.py：三任务模型训练、保存、加载和预测。

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
- midpoint_circle_marker_50mm.png（已弃用，仅保留文件）
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
| 14  |        (85, 128) |   40 |     (105, 148) |

说明：

- 该布局采用“四角 + 中心”，几何基线更长，能更好抑制透视误差与拟合不稳定。
- 贴附时必须保证每个静态标记的打印边长确实为 40 mm，且 A4 打印为 100% 原尺寸。

## 4.4 跨中目标标记（当前主目标）

文件：midpoint_fallback_aruco_id42_50mm.png

- 字典：DICT_5X5_250
- 标记 ID：42
- 边长：50 mm
- 当前版本始终使用该标记作为跨中目标回退检测主方案。

## 4.5 跨中回退标记（备用）

同心圆文件 `midpoint_circle_marker_50mm.png` 目前不参与检测流程，仅作为历史产物保留。

## 4.6 打印后实物验收（必须执行）

使用游标卡尺或钢尺逐项核验：

1. 任一静态标记边长应为 40.0 mm。
2. ID42 目标标记边长应为 50.0 mm。
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
- `s` 开始采样（未按前不会采样）
- `a` 开启/关闭自动采样（检测到足够角点时自动抓取）
- `q` 取消标定

注意：按键必须在**视频窗口**里按，不要在命令行里输入 `c`。  
若窗口提示“有效 ChArUco 角点不足”，优先做这三件事：

1. 让标定板在画面中更大（至少占画面高度 40% 左右）。
2. 降低反光，保证棋盘黑白对比明显。
3. 角度不要过斜（先正对，再逐步增加倾斜角）。

标定板不需要充满全屏，建议占画面高度约 **35%~70%**。  
关键是：同一轮标定里要覆盖多个距离与角度，而不是只在一个位置采样。

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
  --start-mode manual \
  --filter-profile stable \
  --deadband-mm 0.2 \
  --smooth-window 9 \
  --local-scale-mode baseline \
  --target-only-fallback true \
  --static-pose-correction true \
  --static-assist-min-points 4 \
  --deflection-scale 1.0 \
  --measurement-method target-local-scale \
  --calibration-mode use \
  --calibration-file yolo/artifacts/camera_calibration.npz \
  --overlay-level debug
```

### 8.1.1 首次实验推荐命令（会重标定）

```bash
conda run --no-capture-output -n fai python yolo/run_deflection_realtime.py \
  --source 0 \
  --start-mode manual \
  --filter-profile stable \
  --deadband-mm 0.2 \
  --smooth-window 9 \
  --local-scale-mode baseline \
  --target-only-fallback true \
  --static-pose-correction true \
  --static-assist-min-points 4 \
  --deflection-scale 1.0 \
  --measurement-method target-local-scale \
  --calibration-mode recalibrate \
  --overlay-level debug
```

### 8.1.2 后续重复实验推荐命令（复用标定）

```bash
conda run --no-capture-output -n fai python yolo/run_deflection_realtime.py \
  --source 0 \
  --start-mode manual \
  --filter-profile stable \
  --deadband-mm 0.2 \
  --smooth-window 9 \
  --local-scale-mode baseline \
  --target-only-fallback true \
  --static-pose-correction true \
  --static-assist-min-points 4 \
  --deflection-scale 1.0 \
  --measurement-method target-local-scale \
  --calibration-mode use \
  --calibration-file yolo/artifacts/camera_calibration.npz \
  --overlay-level debug
```

## 8.2 基线阶段（非常关键）

- `--start-mode manual` 时，先按 `s` 才会进入基线阶段。
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
- waiting-start：等待按 `s` 开始基线（仅 manual 模式）。
- tracking:yolo：YOLO 检测并跟踪成功。
- tracking:fallback-aruco：YOLO 未用上，使用 ID42 回退标记。
- tracking:static-pose：使用静态标 PnP 姿态辅助，静态平面方向已参与测量方向计算。
- tracking:static-pose-low-quality：静态点几何质量一般，但仍使用静态姿态方向辅助；ID42 自身 50mm 仍是主尺度来源。
- tracking:static-homography-low-quality：没有可用 PnP 姿态时，用较低质量 homography 的静态方向辅助。
- tracking:target-only-no-static-markers：静态 5 点不足，但已降级为仅用 ID42 + 图像竖直方向继续输出挠度。
- tracking:target-only-low-homography-quality：静态点质量不足，但已降级为仅用 ID42 + 图像竖直方向继续输出挠度。
- missing:no-static-markers：静态标定点不足且没有启用/无法进入 target-only 降级。
- missing:low-homography-quality：静态点质量不足且没有启用/无法进入 target-only 降级。
- missing:\*：目标点缺失，当前帧无有效挠度。

退出按键：q。

默认 CSV 输出到 yolo/results/realtime\_时间戳.csv。

`--overlay-level` 说明：

- `minimal`：仅保留核心值，遮挡最少。
- `balanced`：核心值 + 世界坐标（**不显示右侧灰色调试面板**）。
- `debug`：显示右侧灰色调试面板，包含检测来源、置信度、几何质量、短历史曲线。
- `start-mode`：`manual` 需按 `s` 开始基线，`auto` 启动即开始。
- `filter-profile`：推荐 `stable`，静止读数更稳；`normal/fast` 响应更快但波动更明显。
- `deadband-mm`：静止死区阈值，默认 `0.2mm`。小于该幅度的短时抖动会被保持为上一稳定值。
- `smooth-window`：平滑窗口帧数，默认 `9`。数值越大越稳，但响应越慢。
- `local-scale-mode`：推荐 `baseline`。基线阶段锁定 ID42 的局部比例尺，后续只测相对像素位移，避免桥受力后 ID42 轻微转动导致比例尺漂移。
- `target-only-fallback`：默认 `true`。静态点缺失或质量不足时，仍然用 ID42 自身比例尺和图像竖直方向输出数据；界面会提示 target-only 状态。
- `static-pose-correction`：默认 `true`。有 ChArUco 相机标定时，用静态 40mm 标的角点做 PnP，估计静态平面在相机中的姿态，并把静态平面的竖直方向投影到图像里参与 ID42 挠度方向计算。
- `static-assist-min-points`：默认 `4`。只要识别到 1 个静态标的 4 个角点，就可以尝试提供静态姿态/方向辅助；识别到更多静态标时会更稳。
- `deflection-scale`：赛前固定比例修正系数，默认 `1.0`。比赛现场不要用真值临时调整它。
- `measurement-method`：推荐 `target-local-scale`。它在基线阶段记录 ID42 中心投影位置，tracking 阶段只把“当前中心 - 基线中心”的像素差换算成毫米；ID42 的 50mm 真实边长用于计算目标附近局部比例，5 个静态点提供参考方向。
- 静态点质量门控默认已放宽：`min-used-points=12`、`max-rmse=4.0`、`min-inlier-ratio=0.45`。如果静态点没过门控，默认会进入 target-only 降级并继续输出数据。

重要算法说明：

- `homography` 方法假设跨中目标与 5 个静态标记在同一平面；若目标贴在桥上、静态标记贴在后方背板上，两者不共面，真实 40mm 位移可能只被算成几 mm。
- `target-local-scale` 方法不需要比赛现场已知位移；它只依赖赛前已经确认准确的 ID42=50mm 和静态点=40mm。
- 现在 5 个静态点是“软增强”：能识别时用于估计静态平面的投影方向、相机相对静态平面的 tilt/roll，并把这个方向带入 ID42 局部尺度计算；识别不好时不会直接中断主测量。
- debug 面板中的 `Static axis` 为 `pose` 表示使用了 PnP 姿态辅助；`homography` 表示使用了静态点平面映射方向；`none` 表示当前帧只能用 ID42 图像竖直方向降级。
- debug 面板中的 `Plane tilt(deg)` 是静态平面法线相对相机光轴的倾斜角，`Static roll(deg)` 是静态平面竖直方向相对屏幕竖直方向的夹角。它们用于诊断机位是否过斜，不是额外需要手算的参数。
- 当前 OpenCV ArUco 参数偏向精度：`CORNER_REFINE_SUBPIX`、`cornerRefinementWinSize=7`、`cornerRefinementMaxIterations=80`、`cornerRefinementMinAccuracy=0.001`、`perspectiveRemovePixelPerCell=10`、`perspectiveRemoveIgnoredMarginPerCell=0.18`、`errorCorrectionRate=0.3`。这会牺牲一点速度，换取更稳定的角点和更保守的 ID 解码。
- 该方法不会把局部比例尺作用到整张画面的绝对坐标上，因此比上一版更不容易出现 `5.0cm -> 5.3cm` 这种固定倍率偏差。
- 默认 `local-scale-mode=baseline` 会使用基线阶段的 ID42 比例尺；如果你确认 ID42 只平移、不转动，这通常比使用当前帧比例尺更稳。
- `target-pnp` 方法只使用 ID42 与相机内参，适合相机完全固定且 ChArUco 标定非常可靠的情况。
- `static-compensated-pnp` 会使用相机内参、ID42 和 5 个静态点做 3D 位姿解算；若 ChArUco 标定或相机焦距状态变化，它可能出现稳定比例误差。

静态标和 ID42 是否必须在同一竖直平面：

- 最理想：5 个静态 40mm 标和 ID42 主标在同一个竖直平面，或者至少互相平行且深度差很小。这样静态姿态方向、ID42 局部比例尺、实际挠度方向三者一致，精度最高。
- 可以接受：静态标在旁边作为方向参考，ID42 贴在真实测点上；此时 ID42 自己的 50mm 边长仍提供局部尺度，静态标主要提供“竖直方向/相机倾斜”参考。
- 不推荐：静态标在后方很远的背板上，ID42 在前方桥面上，且两者平面夹角明显不同。这样静态方向参考可能和 ID42 实际运动方向不一致，只能作为弱辅助。
- 不能做到共面时，至少让静态 A4 板与 ID42 主标尽量平行，并且让两者都完整、清晰地出现在画面中。

固定比例误差判断：

- 如果 ID42 不动时只是小幅跳动，这是滤波/角点稳定性问题。
- 如果尺子真值 `-5.0cm` 稳定显示约 `-5.3~-5.4cm`，真值 `-10.0cm` 稳定显示约 `-10.85cm`，这不是滤波问题，而是尺度倍率问题。
- 比赛不能用现场真值校正；若要使用 `--deflection-scale`，只能在赛前用同一相机、同一机位策略、同一套打印件固定确定一次，然后比赛时保持不变。
- 现在优先尝试 `--measurement-method target-local-scale`，因为它直接使用 ID42 自身的 50mm 边长做局部尺度，理论上比纯 PnP 更不容易受相机内参倍率影响。

方块边缘是否要和手机画面平行：

- 不需要严格平行，ArUco 可以有旋转。
- 但不要过度倾斜、不要反光、不要太靠画面边缘。
- ID42 最好尽量正对相机，并且加载前后不要发生明显扭转；如果桥面弯曲导致标记自己大幅转角，任意视觉方法都会变差。

说明：OpenCV 默认 `cv2.putText` 字体在很多环境下不支持中文，窗口里可能显示 `????`。  
因此当前版本将**窗口叠加文字**统一为英文短语（避免乱码），命令行提示与 README 保持中文说明。

`manual` 模式下的基线提示：

- 按 `s` 后会打印：`已锁定并开始基线标定。`
- 当基线完成时会打印：`基线标定完成，可开始加载/施加载荷。`
- 窗口内也会短暂显示：`Baseline done. You may load now.`

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
- rawMm：经过 `deflection-scale` 修正后的基线差原始值（mm）。
- unscaledRawMm：未经过 `deflection-scale` 修正的原始几何基线差（mm）。
- filteredMm：滤波 + 短窗口平滑 + 静止死区后的显示值（mm）。
- baselineMm：基线 worldY（mm）。
- deflectionCm：filteredMm / 10（cm）。
- confidence：检测置信度。
- status：状态字符串。

滤波说明：

- `rawMm` 用于追溯修正后的真实输入，不受死区冻结影响。
- `unscaledRawMm` 用于排查尺度倍率问题。
- `filteredMm` 用于实时显示和 `deflectionCm`，默认偏稳。
- 如果想看响应更快，可把 `--filter-profile stable` 改成 `normal` 或 `fast`。

## 11. 砝码重量数据采集、训练、预测

重量工作流已经拆成三段：**采集数据**、**训练模型**、**实时预测**。  
旧入口 `yolo/train_midpoint_yolo.py` 只保留迁移提示，不再作为主流程使用。

三段脚本都复用当前挠度检测主算法：`target-local-scale + local-scale-mode baseline`。  
这样不会影响第 8 节的实时挠度检测，也不会再走旧的 homography 采集逻辑。

## 11.1 第一步：采集重量数据

启动命令：

```bash
conda run --no-capture-output -n fai python yolo/collect_weight_data.py \
  --source 0 \
  --measurement-method target-local-scale \
  --local-scale-mode baseline \
  --target-only-fallback true \
  --static-pose-correction true \
  --static-assist-min-points 4 \
  --target-marker-size 50.0 \
  --calibration-mode use \
  --calibration-file yolo/artifacts/camera_calibration.npz \
  --capture-seconds 8 \
  --min-valid-frames 80 \
  --collection-smooth-window 5 \
  --overlay-level debug
```

操作顺序：

1. 启动后先保持**空载**，在视频窗口按 `s` 锁定基线。
2. 基线完成后，命令行会提示输入重量，单位是 g，例如 `200`。
3. 放稳当前砝码或小车后，先看窗口里的绿色挠度曲线；曲线平稳后，在视频窗口按 `s` 开始采集。
4. `--capture-seconds` 是最短采集时长；如果时间到了但有效帧不足，程序会继续采到 `--min-valid-frames`，不会直接丢弃。
5. 有效帧足够后自动结束；也可以按 `e` 提前结束，但只有有效帧足够时才会真正结束。
6. 命令行会提示输入激光测距仪的标准挠度，单位是 mm，例如 `-12.35`。
7. 换下一个重量，继续在命令行输入重量。
8. 所有重量采完后，命令行输入 `done` 保存数据并退出。

窗口按键：

- `s`：开始当前阶段。基线阶段表示开始空载基线；采集阶段表示开始采集当前重量。
- `e`：提前结束当前重量采集；如果有效帧不足，程序会提示并继续采集。
- `q`：中止整个采集流程。

采集输出：

- `yolo/results/weight_raw_<时间>.csv`：逐帧手机挠度、重量、标准挠度、检测状态、质量指标。
- `yolo/results/weight_windows_<时间>.csv`：每轮重量窗口的聚合特征，至少包含重量、手机平均挠度、激光标准挠度。
- `yolo/results/weight_dataset_<时间>.metadata.json`：采集参数、重量档位、文件路径。

关键字段：

- `weightG`：本轮砝码/小车重量，单位 g。
- `deflectionMeanMm`：手机视觉测得的本轮平均挠度，单位 mm。
- `standardDeflectionMm`：激光测距仪测得的本轮标准挠度，单位 mm。
- `phoneMinusStandardMm`：`deflectionMeanMm - standardDeflectionMm`，用于评估手机测量误差。
- `staticAxisSource`：逐帧静态辅助来源，常见值是 `pose`、`homography`、`none`。
- `staticPlaneTiltDeg` / `staticRollDeg`：逐帧静态平面姿态诊断值，用来回看机位是否过斜。

说明：`standardDeflectionMm` 会保存下来用于误差分析、赛后校准或后续改进拟合方法；当前重量模型默认仍然只用手机视觉挠度特征预测重量，避免比赛预测时依赖激光测距仪。

采集滤波说明：

- `collect_weight_data.py` 的训练特征不直接使用实时显示用的长状态滤波值，而是使用 `collectionFilteredMm`。
- `collectionFilteredMm` 是每一轮按 `s` 后独立清空的 5 帧滑动滤波结果，默认由 `--collection-smooth-window 5` 控制。
- 这个采集滤波没有跨轮 deadband，所以换小重量造成的零点几毫米变化不会被上一轮结果“粘住”。
- `rawMm` 和实时 `filteredMm` 仍然会保存在逐帧 CSV 中，用于追溯；真正用于窗口特征训练的是本轮独立的 `collectionFilteredMm`。

建议：

- 每个重量至少采 2 轮；机会少时优先保证每轮稳定，不要追求太多重量档。
- 每轮采集前等 `Deflection(mm)` 稳定再按 `s`。
- 激光标准挠度请在同一轮采集完成后立即输入，避免换砝码后混淆。
- 若 debug 面板里 `Detect`、`Used points`、`RMSE` 明显异常，本轮不要采。

## 11.2 第二步：训练三任务模型（推荐）

现在比赛至少准备三个任务，所以推荐使用新的三任务训练入口：

- 任务 A：手机视觉挠度 `->` 车重。
- 任务 B：给定重量 `->` 激光标准挠度。
- 任务 C：手机视觉挠度 `->` 激光标准挠度，用于校正或比较手机测量和激光测距仪的差异。

默认训练集就是最近两组较准数据：

- `yolo/results/weight_windows_20260508_191017.csv`
- `yolo/results/weight_windows_20260508_194555.csv`

训练命令：

```bash
conda run --no-capture-output -n fai python yolo/train_bridge_task_models.py \
  --windows-csv yolo/results/weight_windows_20260508_191017.csv \
  --windows-csv yolo/results/weight_windows_20260508_194555.csv \
  --output-dir yolo/results/bridge_models_final \
  --epochs 800 \
  --lr 1e-3 \
  --seed 42
```

如果不写 `--windows-csv`，脚本也会默认使用上面两组数据。

训练界面：

- 默认使用 `rich` 彩色表格和进度条显示训练过程。
- 如果新环境缺少 `rich`，先安装：

```bash
conda run -n fai python -m pip install rich
```

- 如果你想回到纯文本输出，可以加：

```bash
--plain-output
```

训练逻辑：

- 三个任务分别训练候选模型；每个任务都会同时训练小 MLP、线性/二次/岭回归，以及带物理先验的单调分段模型。
- 任务 A 还会训练链式候选：`手机挠度 -> 激光标准挠度 -> 重量`，用于利用 Task C 的校正能力反推重量。
- 单调分段模型会做轻微平滑，不把每个训练点当作必须命中的硬锚点，避免小数据下过拟合。
- 数据很少，所以用留一验证（LOOCV）评估每个样本的预测误差。
- 最终按 `MAE + 0.25*RMSE + 0.10*MaxAbsError` 的稳健分数自动推荐模型；这样不会只为了平均误差好看而牺牲最坏情况。
- 三个任务都会保留所有候选模型结果，不会只保存一个黑盒结果。

为什么训练很快：

- 当前默认训练集只有 27 个窗口样本，不是几千张图片，也不是训练 YOLO。
- MLP 只有几层全连接网络，输入最多 11 个特征，所以 800 epoch 几秒到几十秒完成都正常。
- 判断模型是否靠谱主要看 `metrics.json` 和命令行里的 LOOCV MAE/RMSE/R2，不看训练时间长不长。

输出文件：

- `model_bundle.pth`：完整三任务模型包，包含神经网络权重，实时预测优先加载这个文件。
- `model_bundle.json`：去掉神经网络权重后的摘要，方便人工查看。
- `metrics.json`：三个任务的 MAE、RMSE、R2、最大误差和逐样本误差。
- `feature_config.json`：手机视觉特征、重量工程特征和默认训练数据。

旧脚本 `yolo/train_weight_model.py` 仍然可用，但它只负责“手机视觉挠度 `->` 重量”这一个旧任务。最终比赛准备建议优先用 `train_bridge_task_models.py`。

## 11.3 任务 A：实时测挠度并预测车重

三任务模型包可以直接给实时重量预测脚本使用：

```bash
conda run --no-capture-output -n fai python yolo/predict_weight_realtime.py \
  --source 0 \
  --model-path yolo/results/bridge_models_final/model_bundle.pth \
  --task weight-from-phone \
  --measurement-method target-local-scale \
  --local-scale-mode baseline \
  --target-only-fallback true \
  --static-pose-correction true \
  --static-assist-min-points 4 \
  --calibration-mode use \
  --calibration-file yolo/artifacts/camera_calibration.npz \
  --overlay-level debug
```

操作顺序：

1. 启动后先保持空载，在视频窗口按 `s` 锁定基线。
2. 放上未知重量，等待挠度读数和绿色曲线稳定。
3. 窗口会连续显示实时重量估计，你可以观察稳定值。
4. 想要最终输出时按 `s`，程序会采集一个稳定窗口并在命令行输出最终重量。
5. 按 `q` 退出。

预测输出：

- `Live weight(g)`：滚动窗口实时估计。
- `Nearest(g)`：最接近训练过的重量档位。
- `FINAL`：按 `s` 后稳定窗口的最终预测。
- debug 面板会显示挠度标准差、漂移等稳定性信息。

## 11.4 任务 B：给定重量预测激光标准挠度

如果比赛不给手机测量机会，只给重量，就用任务 B：

```bash
conda run --no-capture-output -n fai python yolo/predict_deflection_from_weight.py \
  --model-path yolo/results/bridge_models_final/model_bundle.pth \
  --weight-g 4300
```

输出含义：

- `Predicted standard deflection`：预测的激光标准挠度，单位 mm。
- `Model`：任务名，固定为 `deflection_from_weight`。
- `Recommended candidate`：当前模型包推荐使用的候选模型，例如 `ridge`、`monotonic_weight_deflection` 或 `mlp_regression`。

## 11.5 任务 C：手机挠度校正为激光标准挠度

如果现场同时有手机视觉和激光测距仪，或者需要比较两者误差，就用任务 C。它不会重新检测视频，只读取已有的窗口特征 CSV：

```bash
conda run --no-capture-output -n fai python yolo/predict_standard_deflection_from_phone.py \
  --model-path yolo/results/bridge_models_final/model_bundle.pth \
  --windows-csv yolo/results/weight_windows_20260508_194555.csv \
  --output-csv yolo/results/phone_laser_correction.csv
```

输出含义：

- `phone`：手机视觉测得的平均挠度，来自 `deflectionMeanMm`。
- `predStandard`：模型预测的激光标准挠度。
- `phone-standard`：手机视觉挠度减去预测标准挠度，可用于看手机测量偏差。

## 11.6 比赛推荐流程

1. 先用第 8 节确认挠度检测稳定、误差可接受。
2. 空载启动 `collect_weight_data.py`，按 `s` 锁定基线。
3. 按重量档逐轮采集，同时输入激光标准挠度，保存 `weight_raw` 和 `weight_windows`。
4. 用 `train_bridge_task_models.py` 训练三任务模型，检查 `metrics.json` 中三个任务的留一验证误差。
5. 任务 A 用 `predict_weight_realtime.py`，先空载按 `s` 锁定基线，再放未知重量。
6. 任务 B 用 `predict_deflection_from_weight.py`，直接输入重量预测标准挠度。
7. 任务 C 用 `predict_standard_deflection_from_phone.py`，把手机视觉窗口数据校正到激光标准挠度。

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

1. 如果状态是 `tracking:target-only-*`，说明已经降级成功，数据仍然会输出；只是没有静态 5 点的参考方向/相机运动补偿。
2. 如果状态是 `missing:no-static-markers` 或 `missing:low-homography-quality`，先确认 ID42 是否完整入镜，因为 target-only 降级也需要 ID42。
3. 增大标记在画面中的像素尺寸（靠近相机或放大打印）。
4. 降低反光并增强均匀照明。
5. 临时救场可继续放宽，例如加 `--max-rmse 5.0 --min-inlier-ratio 0.35`，但会降低相机运动补偿和参考方向稳定性。

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
4. 运行 `collect_weight_data.py`，先空载按 `s` 锁基线，再按“输入重量 -> 按 s 采集 -> done 结束”循环。
5. 运行 `train_weight_model.py`，用 `weight_windows_*.csv` 训练并输出 `best_weight_model.pth`。
6. 运行 `predict_weight_realtime.py`，先空载按 `s` 锁基线，再放未知重量并按 `s` 输出最终预测。

---

如果你严格按照本 README 执行，实验中每个关键环节（尺寸、位置、参数、数据格式、输出口径）都已经明确，不需要再依赖口头解释。
