import argparse
from pathlib import Path

import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--interval-seconds", type=float, default=2.0)
args = parser.parse_args()

video_path = Path(args.video)
output_dir = Path(args.output)

if not video_path.is_file():
    raise SystemExit(f"找不到视频：{video_path}")

if args.interval_seconds <= 0:
    raise SystemExit("--interval-seconds 必须大于 0")

output_dir.mkdir(parents=True, exist_ok=True)

capture = cv2.VideoCapture(str(video_path))
if not capture.isOpened():
    raise SystemExit(f"无法打开视频：{video_path}")

fps = capture.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise SystemExit("无法读取视频帧率")

frame_interval = max(1, round(fps * args.interval_seconds))
frame_number = 0
saved = 0
image_size = None

while True:
    ok, frame = capture.read()
    if not ok:
        break

    if image_size is None:
        image_size = (frame.shape[1], frame.shape[0])

    if frame_number % frame_interval == 0:
        destination = output_dir / f"frame_{saved:04d}.jpg"
        if not cv2.imwrite(str(destination), frame):
            raise SystemExit(f"无法写入图片：{destination}")
        saved += 1

    frame_number += 1

capture.release()

print(f"视频帧率：{fps:.3f}")
print(f"原始分辨率：{image_size[0]}x{image_size[1]}")
print(f"已保存 {saved} 张图片到：{output_dir}")
