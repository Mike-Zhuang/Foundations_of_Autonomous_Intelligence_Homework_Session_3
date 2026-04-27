"""简单的 YOLO 摄像头识别程序。

用途：
- 默认打开第一个摄像头，适合测试 Mac 和 iPhone 连续互通相机
- 使用 `yolov8n.pt` 做实时目标识别
- 按 `q` 退出
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2


def parseArgs() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="简单的 YOLO 摄像头识别程序")
	parser.add_argument("--source", default=0, help="摄像头编号、视频路径或 RTSP 地址，默认使用 0")
	parser.add_argument("--model", default="yolov8n.pt", help="YOLO 权重文件，默认使用 yolov8n.pt")
	parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值，默认 0.35")
	parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸，默认 640")
	return parser.parse_args()


def openCamera(source: str | int) -> cv2.VideoCapture:
	if isinstance(source, str) and source.isdigit():
		source = int(source)

	capture = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
	if not capture.isOpened():
		raise RuntimeError(f"无法打开摄像头或视频源: {source}")
	return capture


def main() -> int:
	args = parseArgs()

	try:
		from ultralytics import YOLO
	except ImportError as exc:
		print("缺少 ultralytics，请先在当前环境中安装：conda install -n fai -c conda-forge ultralytics", file=sys.stderr)
		raise SystemExit(1) from exc

	model = YOLO(args.model)
	capture = openCamera(args.source)

	windowName = "YOLO Camera Test"
	cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
	lastTime = time.time()

	try:
		while True:
			success, frame = capture.read()
			if not success:
				print("读取摄像头失败，程序退出。", file=sys.stderr)
				break

			result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
			annotatedFrame = result.plot()

			currentTime = time.time()
			fps = 1.0 / max(currentTime - lastTime, 1e-6)
			lastTime = currentTime

			cv2.putText(
				annotatedFrame,
				f"FPS: {fps:.1f}",
				(20, 40),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.0,
				(0, 255, 0),
				2,
				cv2.LINE_AA,
			)
			cv2.putText(
				annotatedFrame,
				"Press Q to quit",
				(20, 80),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(255, 255, 255),
				2,
				cv2.LINE_AA,
			)

			cv2.imshow(windowName, annotatedFrame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		capture.release()
		cv2.destroyAllWindows()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
