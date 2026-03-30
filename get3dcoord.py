#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import time
from datetime import datetime

import cv2
import rospy
import torch
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO

bridge = CvBridge()

ENGINE_MODEL_PATH = "/home/jetson/ultralytics_robot/best.engine"
CAMERA_INFO_PATH = "/home/jetson/yolov5/ost.yaml"
WINDOW_NAME = "YOLOv11 Detection"
SAVE_ROOT_DIR = os.path.join(os.getcwd(), "saved_frames")
SAVE_FRAME_COUNT = 10

model = YOLO(ENGINE_MODEL_PATH, task="detect")
names = model.names
img_size = 640

depth_image = None
camera_info = None

capture_session_dir = None
capture_csv_path = None
capture_remaining = 0
capture_frame_index = 0


def load_camera_info(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def format_xyz(xyz):
    if xyz is None:
        return "N/A"
    return f"X:{xyz[0]:.3f} Y:{xyz[1]:.3f} Z:{xyz[2]:.3f} m"


def pixel_to_3d(u, v, depth_img, cam_info_dict):
    fx = cam_info_dict["camera_matrix"]["data"][0]
    fy = cam_info_dict["camera_matrix"]["data"][4]
    cx = cam_info_dict["camera_matrix"]["data"][2]
    cy = cam_info_dict["camera_matrix"]["data"][5]

    u_i, v_i = int(round(u)), int(round(v))
    if v_i >= depth_img.shape[0] or u_i >= depth_img.shape[1] or v_i < 0 or u_i < 0:
        return None

    depth_value = depth_img[v_i, u_i]
    if depth_value == 0:
        return None

    depth_m = float(depth_value) / 1000.0
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return (x, y, z)


def draw_info_panel(frame, valve_xyz, infer_ms, max_fps):
    lines = [
        f"Valve Center 3D: {format_xyz(valve_xyz)}",
        f"Inference: {infer_ms:.1f} ms",
        f"Theoretical Max FPS: {max_fps:.2f}",
        f"Press SPACE to save next {SAVE_FRAME_COUNT} frames",
    ]

    line_h = 28
    panel_h = 18 + len(lines) * line_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (700, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for idx, text in enumerate(lines):
        y = 35 + idx * line_h
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def get_next_capture_dir():
    os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
    existing_ids = []

    for name in os.listdir(SAVE_ROOT_DIR):
        path = os.path.join(SAVE_ROOT_DIR, name)
        if os.path.isdir(path) and name.isdigit():
            existing_ids.append(int(name))

    next_id = max(existing_ids, default=0) + 1
    capture_dir = os.path.join(SAVE_ROOT_DIR, str(next_id))
    os.makedirs(capture_dir, exist_ok=False)
    return capture_dir


def start_capture_session():
    global capture_session_dir, capture_csv_path, capture_remaining, capture_frame_index

    capture_session_dir = get_next_capture_dir()
    capture_csv_path = os.path.join(capture_session_dir, "metadata.csv")
    capture_remaining = SAVE_FRAME_COUNT
    capture_frame_index = 0

    with open(capture_csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame_index",
                "image_name",
                "timestamp",
                "valve_x_m",
                "valve_y_m",
                "valve_z_m",
                "inference_ms",
                "theoretical_max_fps",
            ]
        )

    rospy.loginfo(f"Started capture session: {capture_session_dir}")


def finish_capture_session():
    global capture_session_dir, capture_csv_path, capture_remaining, capture_frame_index

    rospy.loginfo("Capture session completed.")
    capture_session_dir = None
    capture_csv_path = None
    capture_remaining = 0
    capture_frame_index = 0


def save_capture_frame(frame, valve_xyz, infer_ms, max_fps):
    global capture_remaining, capture_frame_index

    if capture_remaining <= 0 or capture_session_dir is None or capture_csv_path is None:
        return

    capture_frame_index += 1
    image_name = f"frame_{capture_frame_index:02d}.jpg"
    image_path = os.path.join(capture_session_dir, image_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    cv2.imwrite(image_path, frame)

    if valve_xyz is None:
        valve_x, valve_y, valve_z = "", "", ""
    else:
        valve_x, valve_y, valve_z = valve_xyz

    with open(capture_csv_path, "a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                capture_frame_index,
                image_name,
                timestamp,
                valve_x,
                valve_y,
                valve_z,
                infer_ms,
                max_fps,
            ]
        )

    capture_remaining -= 1
    rospy.loginfo(
        f"Saved frame {capture_frame_index}/{SAVE_FRAME_COUNT}: {image_path}"
    )

    if capture_remaining == 0:
        finish_capture_session()


def depth_callback(msg):
    global depth_image
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    except Exception as exc:
        rospy.logwarn_throttle(5, f"Depth image parse failed: {exc}")


def image_callback(msg):
    global depth_image, camera_info, capture_remaining

    if depth_image is None or camera_info is None:
        rospy.logwarn_throttle(5, "Waiting for depth image and camera intrinsics...")
        return

    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")

        infer_start = time.perf_counter()
        results = model(
            frame,
            imgsz=img_size,
            conf=0.65,
            iou=0.45,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        max_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0

        valve_xyz = None

        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                det = torch.cat(
                    (
                        boxes.xyxy,
                        boxes.conf.view(-1, 1),
                        boxes.cls.view(-1, 1),
                    ),
                    dim=1,
                )

                valve_det = det[det[:, 5] == 0]
                if len(valve_det) > 0:
                    best_idx = torch.argmax(valve_det[:, 4]).item()
                    x1, y1, x2, y2, conf, cls = valve_det[best_idx].tolist()
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    cls = int(cls)

                    valve_center_x = (x1 + x2) / 2.0
                    valve_center_y = (y1 + y2) / 2.0
                    valve_xyz = pixel_to_3d(
                        valve_center_x, valve_center_y, depth_image, camera_info
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(
                        frame,
                        (int(round(valve_center_x)), int(round(valve_center_y))),
                        5,
                        (0, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        f"{names[cls]} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    rospy.loginfo_throttle(2, "Valve not detected.")

        draw_info_panel(frame, valve_xyz, infer_ms, max_fps)

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        if capture_remaining > 0:
            save_capture_frame(frame, valve_xyz, infer_ms, max_fps)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            if capture_remaining > 0:
                rospy.logwarn("Capture session is already in progress.")
            else:
                start_capture_session()

    except Exception as exc:
        rospy.logerr(f"Image processing failed: {exc}")


def main():
    global camera_info

    rospy.init_node("ros_yolo_detect_node", anonymous=True)
    camera_info = load_camera_info(CAMERA_INFO_PATH)

    rospy.Subscriber(
        "/camera/color/image_raw",
        Image,
        image_callback,
        queue_size=1,
        buff_size=2**24,
    )

    rospy.Subscriber(
        "/camera/depth/image_raw",
        Image,
        depth_callback,
        queue_size=1,
        buff_size=2**24,
    )

    rospy.loginfo("YOLO valve 3D coordinate node started.")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
