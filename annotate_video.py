#!/usr/bin/env python3
"""
Script để annotate video với bounding boxes từ JSON và lưu video đã annotate
"""

import cv2
import json
import os
import argparse
from pathlib import Path
import numpy as np


def load_annotations(json_path):
    """Load annotations từ file JSON"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def annotate_video(video_path, annotations, output_video_path):
    """
    Annotate video với bounding boxes và lưu video

    Args:
        video_path: Đường dẫn đến video input
        annotations: List các annotation cho video này
        output_video_path: Đường dẫn đến video output
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    # Lấy thông số video gốc
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tạo VideoWriter để lưu video output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Tạo dict để lookup nhanh các bboxes theo frame
    frame_annotations = {}
    if annotations and len(annotations) > 0:
        for detection in annotations:
            if "bboxes" in detection:
                for bbox in detection["bboxes"]:
                    frame_num = bbox["frame"]
                    if frame_num not in frame_annotations:
                        frame_annotations[frame_num] = []
                    frame_annotations[frame_num].append(bbox)

    frame_count = 0
    annotated_count = 0

    print(f"Processing video: {video_path}")
    print(
        f"Total frames: {total_frames}, Frames with annotations: {len(frame_annotations)}"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Kiểm tra xem frame này có annotation không
        if frame_count in frame_annotations:
            # Vẽ bounding boxes lên frame
            for bbox in frame_annotations[frame_count]:
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

                # Vẽ rectangle (màu đỏ)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Vẽ text thông tin frame
                text = f"Frame {frame_count}"
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            annotated_count += 1

        # Ghi frame vào video output (dù có annotation hay không)
        out.write(frame)

        frame_count += 1

        # Hiển thị progress mỗi 1000 frames
        if frame_count % 1000 == 0:
            print(f"Đã xử lý {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(
        f"Hoàn thành video {video_path}: Video annotate đã lưu tại {output_video_path}"
    )
    print(f"Số frames có annotation: {annotated_count}/{total_frames}")


def main():
    parser = argparse.ArgumentParser(description="Annotate videos with bounding boxes")
    parser.add_argument(
        "--json_path", default="output1.json", help="Path to JSON annotation file"
    )
    parser.add_argument(
        "--video_dir",
        default="../data/public_test/public_test/samples",
        help="Directory containing videos",
    )
    parser.add_argument(
        "--output_dir",
        default="./annotated_videos",
        help="Output directory for annotated videos",
    )

    args = parser.parse_args()

    # Load annotations
    print(f"Loading annotations from {args.json_path}")
    annotations = load_annotations(args.json_path)

    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all video IDs
    video_ids = [item["video_id"] for item in annotations]
    print(f"Found {len(video_ids)} videos: {video_ids}")

    # Process each video
    for item in annotations:
        video_id = item["video_id"]
        detections = item["detections"]

        video_path = os.path.join(args.video_dir, video_id, "drone_video.mp4")
        output_video_path = os.path.join(args.output_dir, f"{video_id}_annotated.mp4")

        if os.path.exists(video_path):
            print(f"\n--- Processing {video_id} ---")
            annotate_video(video_path, detections, output_video_path)
        else:
            print(f"Video not found: {video_path}")

    print("\n=== HOÀN THÀNH ===")
    print(f"Các video đã annotate được lưu trong: {args.output_dir}")


if __name__ == "__main__":
    main()
