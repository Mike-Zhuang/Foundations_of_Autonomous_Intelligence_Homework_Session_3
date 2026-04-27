# 桥梁跨中挠度测量 Guide（YOLO + ArUco）

本项目用于课程实验：在给定荷载（小车位于跨中附近）条件下，通过图像识别测量跨中静挠度，目标精度至少 0.1 cm。

## 1. 方案概览

核心算法为四段式：
1. YOLO 检测跨中目标（`midpoint_marker`）。
2. 静态 ArUco 多标记解算每帧单应矩阵（抗相机抖动）。
3. 像素点映射到毫米坐标，得到跨中竖向位移。
4. 基线对比 + 卡尔曼滤波，输出稳定挠度曲线（cm）。

你已确认的实验参数：
- 跨径：1200 mm
- 摄像头：iPhone 连续互通相机（macOS 系统摄像头）
- 输出：实时显示 + 离线复算 + CSV

## 2. 文件结构

- `yolo/generate_marker_board.py`：生成可打印静态标记板和布局参数。
- `yolo/run_deflection_realtime.py`：实时测量入口。
- `yolo/run_deflection_offline.py`：离线复算入口。
- `yolo/train_midpoint_yolo.py`：YOLO 训练入口。
- `yolo/bridge_ai/config.py`：标记布局配置。
- `yolo/bridge_ai/geometry.py`：ArUco 识别与几何反算。
- `yolo/bridge_ai/detection.py`：YOLO 检测与 ArUco 回退检测。
- `yolo/bridge_ai/deflection.py`：基线标定与滤波。
- `yolo/bridge_ai/io_utils.py`：视频源、CSV、视频输出。

## 3. 标记打印与粘贴

### 3.1 生成标记图

在仓库根目录执行：

```bash
conda run -n fai python yolo/generate_marker_board.py
```

生成目录：`yolo/artifacts/`
- `static_marker_board_a4_300dpi.png`
- `static_marker_layout.json`
- `midpoint_fallback_aruco_id42.png`
- `midpoint_circle_marker_50mm.png`
- `marker_print_notes.txt`

### 3.2 打印参数（必须）

- 打印比例：100%（禁止“适应页面”）
- 静态字典：`DICT_5X5_250`
- 静态 ID：`10, 11, 12, 13, 14`
- 静态边长：`40 mm`
- 跨中回退标记：`ArUco ID 42`

### 3.3 粘贴建议

- 静态板贴在固定背景（不随桥梁运动）。
- 跨中标记贴在桥底跨中位置，尽量避免反光。
- 允许额外贴毫米刻度尺用于外部精度校验。

## 4. iPhone 连续互通相机设置

1. iPhone 与 Mac 使用同一个 Apple ID，且蓝牙、Wi-Fi 打开。
2. 在 Mac 上打开目标脚本后，视频源通常是 `0` 或 `1`。
3. 先运行 `yolo/test.py --source 0` 试通，再切主程序。

如果画面不是 iPhone，相机索引改为 `--source 1` 或 `--source 2`。

## 5. 实时测量

```bash
conda run -n fai python yolo/run_deflection_realtime.py \
  --source 0 \
  --model yolov8n.pt \
  --target-class midpoint_marker \
  --baseline-frames 60
```

说明：
- 启动后前 60 帧用于空载基线标定。
- 放车前请保持“无载荷”状态。
- 按 `q` 退出。
- 结果 CSV 默认保存在 `yolo/results/`。

## 6. 离线复算

```bash
conda run -n fai python yolo/run_deflection_offline.py \
  --video path/to/your_video.mp4 \
  --model yolov8n.pt \
  --target-class midpoint_marker
```

输出：
- CSV：逐帧结果
- summary json：最大/最小/均值/标准差

## 7. YOLO 训练（高精度模式）

建议采集 200-500 帧，标注类别至少包含 `midpoint_marker`。

```bash
conda run -n fai python yolo/train_midpoint_yolo.py \
  --data path/to/data.yaml \
  --base-model yolo11n.pt \
  --epochs 120 \
  --imgsz 960
```

训练完成后，将最佳权重路径替换 `--model` 参数用于实时和离线脚本。

## 8. 精度验收流程（建议）

1. 先做空载基线采集。
2. 用已知位移（如 1mm、2mm、5mm、10mm）做阶梯试验。
3. 检查 `deflectionCm` 误差是否在目标范围内。
4. 重复三次，评估稳定性和重复性。

## 9. 常见问题

1. 识别不到静态标记：
- 检查光照与反光。
- 检查打印是否 100% 比例。
- 检查 `static_marker_layout.json` 是否与打印板一致。

2. YOLO 无输出：
- 先用训练好的权重。
- 或在早期先启用 `ArUco ID 42` 回退目标追踪验证流程。

3. 挠度抖动较大：
- 固定相机，关闭自动曝光跳变。
- 增加 `--baseline-frames`，并确保基线阶段无载荷。

## 10. 参考链接

- OpenCV ArUco 官方说明：
  https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- ArUco 在线生成器（可选）：
  https://chev.me/arucogen/
