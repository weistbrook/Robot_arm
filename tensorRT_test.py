#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import torch
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

bridge = CvBridge()

ENGINE_MODEL_PATH = "/home/jetson/ultralytics_robot/best.engine"
model = YOLO(ENGINE_MODEL_PATH, task="detect")

names = model.names
img_size = 640

fps = 0.0
last_end_time = None


def image_callback(msg):
    global fps, last_end_time

    try:
        callback_start = time.perf_counter()

        frame = bridge.imgmsg_to_cv2(msg, "bgr8")

        infer_start = time.perf_counter()
        results = model(
            frame,
            imgsz=img_size,
            conf=0.65,
            iou=0.45,
            verbose=False
        )
        infer_end = time.perf_counter()

        infer_ms = (infer_end - infer_start) * 1000.0

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes

            if boxes is not None and len(boxes) > 0:
                det = torch.cat(
                    (
                        boxes.xyxy,
                        boxes.conf.view(-1, 1),
                        boxes.cls.view(-1, 1)
                    ),
                    dim=1
                )

                for *xyxy, conf, cls in det.tolist():
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls = int(cls)
                    label = f"{names[cls]} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

        callback_end = time.perf_counter()

        # 统计完整处理FPS：以上一帧结束到这一帧结束的时间差为准
        if last_end_time is not None:
            dt = callback_end - last_end_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps
        last_end_time = callback_end

        total_ms = (callback_end - callback_start) * 1000.0

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Infer: {infer_ms:.1f} ms",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Total: {total_ms:.1f} ms",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.imshow("YOLO Detection", frame)
        cv2.waitKey(1)

    except Exception as e:
        rospy.logerr(f"图像处理失败: {e}")


def main():
    rospy.init_node("ros_yolo_detect_node", anonymous=True)

    rospy.Subscriber(
        "/camera/color/image_raw",
        Image,
        image_callback,
        queue_size=1,
        buff_size=2**24
    )

    rospy.loginfo("YOLO 目标检测节点已启动")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()