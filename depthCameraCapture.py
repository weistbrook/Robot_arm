import os

import rospy
import numpy as np
import yaml
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from datetime import datetime

bridge = CvBridge()
depth_image = None
clicked_point = None
clicked_distance = None
INVALID_DEPTH_VALUE = 9999.0
WINDOW_NAME = "Depth Image"
SAVE_DIR = "depth_frames"

def save_current_frame(frame):
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(SAVE_DIR, f"frame_{ts}.jpg")
    cv2.imwrite(save_path, frame)
    rospy.loginfo(f"Saved  Depthframe to: {save_path}")
def load_camera_info(yaml_file):
    with open(yaml_file, "r") as file:
        cam_info = yaml.safe_load(file)
    rospy.loginfo(f"Loaded camera info: {cam_info}")
    return cam_info


def pixel_to_3d(u, v, depth_img, cam_info_dict):
    if v >= depth_img.shape[0] or u >= depth_img.shape[1] or v < 0 or u < 0:
        return INVALID_DEPTH_VALUE

    d = depth_img[int(v), int(u)]
    if d == 0:
        return INVALID_DEPTH_VALUE

    d = d / 1000.0
    return d


def depth_callback(msg):
    global depth_image
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    except Exception as e:
        rospy.logwarn_throttle(5, f"Failed to parse depth image: {e}")


def build_depth_display(depth_img):
    depth = depth_img.astype(np.float32)
    valid_mask = depth > 0

    if not np.any(valid_mask):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    valid_depth = depth[valid_mask]
    min_depth = np.min(valid_depth)
    max_depth = np.max(valid_depth)

    normalized = np.zeros_like(depth, dtype=np.uint8)
    if max_depth > min_depth:
        normalized[valid_mask] = np.clip(
            (depth[valid_mask] - min_depth) * 255.0 / (max_depth - min_depth),
            0,
            255,
        ).astype(np.uint8)

    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def on_mouse_click(event, x, y, flags, param):
    global clicked_point, clicked_distance

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if depth_image is None:
        return

    clicked_point = (x, y)
    clicked_distance = pixel_to_3d(x, y, depth_image, None)


def main():
    global camera_info

    rospy.init_node("capture_node", anonymous=True)
    camera_info = load_camera_info("/home/jetson/yolov5/ost.yaml")

    rospy.Subscriber(
        "/camera/depth/image_raw",
        Image,
        depth_callback,
        queue_size=1,
        buff_size=2**24,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if depth_image is not None:
            depth_vis = build_depth_display(depth_image)
            if clicked_point is not None and clicked_distance is not None:
                x, y = clicked_point
                cv2.circle(depth_vis, (x, y), 5, (255, 255, 255), 2)
                text = f"({x}, {y}) depth: {clicked_distance:.3f} m"
                cv2.putText(
                    depth_vis,
                    text,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                save_current_frame(depth_vis)

        rate.sleep()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
