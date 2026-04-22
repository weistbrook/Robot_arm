#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 导入必要的库
import rospy                  # ROS Python库，用于节点创建和消息处理
import cv2                    # OpenCV库，用于图像处理
import torch                  # PyTorch库，用于加载和运行YOLOv5模型
import numpy as np            # NumPy库，用于数值计算
import yaml                   # YAML库，用于加载相机内参配置文件
import os
from sensor_msgs.msg import Image, CameraInfo  # ROS图像和相机信息消息类型
from cv_bridge import CvBridge  # 用于ROS图像消息与OpenCV图像的转换 
from robot_controller import RobotController  # 机器人控制类（用户自定义）
import sys                     # 系统库，用于路径配置
# 添加YOLOv5代码路径（根据实际安装位置修改）
sys.path.append('/home/jetson/yolov5')
from scipy.spatial.transform import Rotation as R  # 用于旋转计算（预留）
from ultralytics import YOLO
import dev_angle  # 自定义模块，用于目标角度计算（如阀门角度检测）
from datetime import datetime  # 日期时间库（预留）
import time  # 时间库，用于控制频率和延时
import re
from threading import Thread, Lock  # 多线程相关库：线程类、锁（保证线程安全）
import queue  # 队列库，用于线程间通信（传递机器人控制指令）
from utility import init_robot_controller, prevent_stop, move_axis, keep_parallel, load_camera_info, judge_proper, pixel_to_3d # 自定义模块，包含机器人控制相关函数（如保持平行）

# =========================
# 机器人连接与YOLO模型初始化
# =========================
# 创建机器人控制器实例（用于发送控制指令）
controller = RobotController()
# 发送机器人登录与上电指令（放在顶层初始化，确保启动时完成基础配置）
init_robot_controller(controller)  # 包含登录、上电、线程启动、速度设置等步骤
time.sleep(1)  # 等待机器人初始化完成

# 创建CvBridge实例（用于ROS与OpenCV图像转换）
bridge = CvBridge()

prevent_stop(controller)  # 确保机器人处于可控制状态，避免进入停止状态
# 使用 Ultralytics YOLO 加载 ENGINE 模型（可通过环境变量覆盖路径）
ENGINE_MODEL_PATH = "/home/jetson/ultralytics_robot/best.engine"
model = YOLO(ENGINE_MODEL_PATH, task="detect")
names = model.names           # 保持变量名不变，后面代码照用
img_size = 640                # 输入尺寸和原来一致

# 全局变量：缓存最新数据（避免频繁读取话题导致的延迟）
depth_image = None   # 存储最近收到的深度图像（OpenCV格式）
camera_info = None   # 存储相机内参（从YAML文件加载的字典）

# ========== 新增：后台动作执行通道（解耦检测与控制，避免阻塞） ==========
# 动作队列：用于主线程（检测）向动作线程（控制）传递指令，仅保留最新指令
action_q = queue.Queue(maxsize=1)   # maxsize=1：队列满时，新指令会覆盖旧指令
control_lock = Lock()               # 控制锁：确保机器人指令串行执行，避免冲突


def make_action(kind, **kwargs):
    """
    生成标准化的动作指令字典
    参数：
        kind: 动作类型（如"far_move"、"check_and_spin"）
        **kwargs: 动作参数（如移动距离、旋转角度等）
    返回：
        包含动作信息的字典
    """
    return {"kind": kind, "args": kwargs, "t": time.time()}  # "t"为时间戳（调试用）



def _execute_move_command(lx, ly, lz, description="移动"):
    """
    执行线性偏移移动命令的通用函数
    参数：
        lx, ly, lz: 线性偏移（单位：米）
        description: 动作描述（用于日志）
    """
    keep_parallel(controller)  # 确保机器人末端保持平行状态，避免移动时姿态不稳定
    prevent_stop(controller)
    command = f"Move.LOffset {{{lx:.3f},{ly:.3f},{lz:.3f},0,0,0}}"
    resp = controller.send_command(command)
    rospy.loginfo(f"[动作线程] 执行{description}: {command} -> 响应: {resp}")
    rospy.sleep(3.0)
    return resp


def _execute_axis_rotation(axis, angle, description="旋转"):
    """
    执行轴旋转命令的通用函数
    参数：
        axis: 轴编号
        angle: 旋转角度（单位：度）
        description: 动作描述（用于日志）
    """
    keep_parallel(controller)  # 确保机器人末端保持平行状态，避免旋转时姿态不稳定
    prevent_stop(controller)
    command = f"Move.Axis {axis},{angle:.3f}"
    resp = controller.send_command(command)
    rospy.loginfo(f"[动作线程] 执行{description}: {command} -> 响应: {resp}")
    rospy.sleep(3.0)
    return resp


def action_worker():
    """
    动作执行线程：专门处理机器人控制指令，避免阻塞主线程的检测流程
    从action_q队列中获取指令并执行，确保控制逻辑异步运行
    """
    rospy.loginfo("动作执行线程已启动")
    
    # 动作类型与描述的映射
    move_descriptions = {
        "far_move": "远距位移",
        "no_ahead_check": "不前进调整位移",
        "small_move": "小物体微调位移",
        "small_no_head": "小物体微调位移",
    }
    
    while not rospy.is_shutdown():
        try:
            act = action_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if act is None:
            continue
        
        with control_lock:
            kind = act["kind"]
            args = act["args"]
            try:
                if kind == "check_and_spin":
                    # 若角度偏差过大，先旋转校正
                    if not args["is_proper"]:
                        _execute_axis_rotation(6, args['offset_deg'], "旋转校正")
                    
                    # 旋转校正后，小幅移动靠近目标
                    _execute_move_command(args['lx'], args['ly'], args['lz'], "近距靠近")
                
                elif kind in ["far_move", "no_ahead_check", "small_move", "small_no_head"]:
                    # 统一处理所有线性移动动作
                    desc = move_descriptions.get(kind, "移动")
                    _execute_move_command(args['lx'], args['ly'], args['lz'], desc)
                
                else:
                    rospy.logwarn(f"[动作线程] 未知动作类型: {kind}")

            except Exception as e:
                rospy.logerr(f"[动作线程] 执行动作失败: {e}")



def depth_callback(msg):
    """
    深度图话题回调函数：仅缓存深度图，避免在回调中做耗时处理（防止消息丢失）
    参数：
        msg: ROS深度图像消息（sensor_msgs/Image）
    """
    global depth_image
    try:
        # 将ROS消息转换为OpenCV格式（passthrough保持原始编码，通常为16位深度值，单位毫米）
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    except Exception as e:
        # 每5秒最多打印一次警告（避免日志刷屏）
        rospy.logwarn_throttle(5, f"深度图解析失败: {e}")


# ========== 图像回调：核心处理函数（检测+决策） ==========
def image_callback(msg):
    """
    彩色图像话题回调函数：执行目标检测、3D坐标计算、生成控制指令
    为避免阻塞，仅处理关键逻辑，耗时操作（如显示）尽量轻量化
    参数：
        msg: ROS彩色图像消息（sensor_msgs/Image）
    """
    # 引用全局变量（深度图、相机内参、显示模式、推理时间戳）
    global depth_image, camera_info

    # 检查依赖数据是否就绪（深度图和相机内参必须存在）
    if depth_image is None or camera_info is None:
        rospy.logwarn_throttle(5, "等待深度图和相机内参加载完成...")
        return

    try:
        # 将ROS彩色图像消息转换为OpenCV BGR格式
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")

        # 使用 Ultralytics YOLO 推理（ENGINE）
        results = model(
            frame,           # 直接用原始 BGR 图像
            imgsz=img_size,
            conf=0.65,
            iou=0.45,
            verbose=False,
        )

        # 将 Ultralytics 输出转换为和原来一样结构的 dets（列表，每个元素是 [x1,y1,x2,y2,conf,cls] 的 tensor）
        if not results:
            # 没有结果时，构造一个空 det，后面的 for det in dets 逻辑保持一致
            dets = [torch.empty((0, 6))]
        else:
            res = results[0]
            boxes = res.boxes  # Boxes 对象

            if boxes is None or len(boxes) == 0:
                dets = [torch.empty((0, 6))]
            else:
                # 拼成和原来 det 一样的格式：[x1,y1,x2,y2,conf,cls]
                det = torch.cat(
                    (
                        boxes.xyxy,                   # (N,4)
                        boxes.conf.view(-1, 1),       # (N,1)
                        boxes.cls.view(-1, 1),        # (N,1)
                    ),
                    dim=1,
                )
                dets = [det]

        # 处理检测结果
        for det in dets:
            if det is None or len(det) == 0:
                rospy.loginfo(f"无检测结果")
                continue  # 无检测结果，跳过

            # 将检测框坐标从640x640缩放回原图尺寸
            #det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            # 计算图像中心点（用于判断目标是否居中）
            img_center_x = frame.shape[1] / 2
            img_center_y = frame.shape[0] / 2
            famen_det = det[det[:, 5] == 0]
            small_det = det[det[:, 5] == 1]
            # 选择置信度最高的检测框（优先处理最可信目标）
            #max_conf_idx_famen = torch.argmax(famen_det[:, 4]).item()  # famen目标最大置信度索引
            #max_conf_idx_small = torch.argmax(small_det[:, 4]).item()  # famen目标最大置信度索引
            # 提取框坐标（x1,y1左上角；x2,y2右下角）、置信度、类别
            def get_xyz(det,max_conf_idx):
                x1, y1, x2, y2, conf, cls = det[max_conf_idx].tolist()
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # 转换为整数坐标
                cls = int(cls)  # 类别索引

                # 在图像上绘制检测框和标签（可视化）
                label = f'{names[cls]} {conf:.2f}'  # 标签：类别+置信度
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 标签文本

                # 计算目标中心点像素坐标
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0

                # 计算目标在相机坐标系下的3D坐标
                xyz = pixel_to_3d(x_center, y_center, depth_image, camera_info)
                if cls==0:
                    return xyz,cls,x1,x2,y1,y2
                elif cls==1:
                    return xyz,cls
            # xyz,f_cls,x1,x2,y1,y2 = get_xyz(famen_det,max_conf_idx_famen)
            # xyz_small,s_cls = get_xyz(small_det,max_conf_idx_small) 
            xyz, f_cls, x1, x2, y1, y2 = None, None, None, None, None, None
            xyz_small, s_cls = None, None

            # 处理阀门主体（cls=0）
            if len(famen_det) > 0:
                max_conf_idx_famen = torch.argmax(famen_det[:, 4]).item()
                # 注意：get_xyz的参数要传famen_det，不是det！（下面会说这个bug）
                xyz, f_cls, x1, x2, y1, y2 = get_xyz(famen_det, max_conf_idx_famen)
            else:
                rospy.loginfo_throttle(2, "未检测到阀门主体")

            # 处理小目标（cls=1）
            if len(small_det) > 0:
                max_conf_idx_small = torch.argmax(small_det[:, 4]).item()
                xyz_small, s_cls = get_xyz(small_det, max_conf_idx_small)
            else:
                rospy.loginfo_throttle(2, "未检测到小目标")   

            # 若3D坐标有效且检测到目标（假设cls=0为关注的目标，如阀门）
            if xyz is not None and f_cls == 0 and 1000*xyz[2] > 260:
                Z_mm = 1000.0 * xyz[2]  # Z坐标转换为毫米（判断距离）

                # 远距离处理（>40cm）：大步移动靠近
                if abs(Z_mm) >= 400.0:
                    try:
                        # 生成远距移动指令（坐标转换：米→米，注意与机器人坐标系匹配）
                        act = make_action(
                            "far_move",
                            lx=(1000.0 * xyz[0]),  # xyz[0]为相机X，转换为机器人X偏移（单位：米）
                            ly=Z_mm-350,            # 我这里是想让机械臂末端运动到距离阀门350mm的地方进行微调
                            lz=(-1000.0 * xyz[1]),  # xyz[1]为相机Y，转换为机器人Z偏移（单位：米）
                            
                        )
                        # 若队列满，先清空旧指令（确保执行最新指令）
                        if action_q.full():
                            _ = action_q.get_nowait()
                        action_q.put_nowait(act)  # 放入动作队列
                    except queue.Full:
                        pass  # 队列仍满时忽略（理论上不会发生，因已清空）

                # 近距离处理（<40cm）：角度校正+精细移动
                else:
                    if (abs(1000*xyz[0])>5 or abs(1000*xyz[1])>5):
                        try:
                            act = make_action(
                                "no_ahead_check",
                                lx=(1000.0 * xyz[0]),  # xyz[0]为相机X，转换为机器人X偏移（单位：米）
                                ly=0,
                                lz=(-1000.0 * xyz[1]),  # xyz[1]为相机Y，转换为机器人Z偏移（单位：米）
                            )
                            # 若队列满，先清空旧指令（确保执行最新指令）
                            if action_q.full():
                                _ = action_q.get_nowait()
                            action_q.put_nowait(act)  # 放入动作队列
                        except queue.Full:
                            pass  # 队列仍满时忽略（理论上不会发生，因已清空）

                    elif (abs(1000*xyz[0])<5 and abs(1000*xyz[1])<5):
                        # 裁剪目标ROI区域（用于角度判断）
                        H, W = frame.shape[:2]  # 原图高和宽
                        # 确保ROI在图像范围内（防止越界）
                        x1c, x2c = sorted((max(0, x1), min(W, x2)))
                        y1c, y2c = sorted((max(0, y1), min(H, y2)))
                        # 确保ROI有效（宽高至少为1）
                        if x2c - x1c > 1 and y2c - y1c > 1:
                            roi = frame[y1c:y2c, x1c:x2c]  # 提取ROI
                            # 判断目标是否对正，获取角度偏差
                            is_proper, offset_deg = judge_proper(roi)

                            try:
                                # 生成近距离校正+移动指令
                                act = make_action(
                                    "check_and_spin",
                                    is_proper=is_proper,       # 是否对正
                                    offset_deg=float(offset_deg),  # 角度偏差
                                    lx=(1000.0 * xyz[0]),     # X方向偏移（米）
                                    ly=(1000.0 * xyz[2]-250),     # Y方向偏移（米）
                                    lz=(-1000.0 * xyz[1]),     # Z方向偏移（米）
                                )
                                # 若队列满，清空旧指令
                                if action_q.full():
                                    _ = action_q.get_nowait()
                                action_q.put_nowait(act)  # 放入动作队列
                            except queue.Full:
                                pass  # 忽略队列满的情况
           
            elif xyz_small is not None and s_cls == 1 and 1000*xyz_small[2]<260:
                Z_mm=1000*xyz_small[2]
                if 1000*xyz_small[0]<2 and 1000*xyz_small[1]<2:                   
                    try:
                    # 生成利用小物体微调移动指令（坐标转换：米→米，注意与机器人坐标系匹配）
                        act = make_action(
                            "small_move",
                            lx=(1000.0 * xyz_small[0]),  # xyz[0]为相机X，转换为机器人X偏移（单位：米）
                            ly=Z_mm,            
                            lz=(-1000.0 * xyz_small[1]),  # xyz[1]为相机Y，转换为机器人Z偏移（单位：米）
                            
                        )
                        # 若队列满，先清空旧指令（确保执行最新指令）
                        if action_q.full():
                            _ = action_q.get_nowait()
                        action_q.put_nowait(act)  # 放入动作队列
                    except queue.Full:
                        pass  # 队列仍满时忽略（理论上不会发生，因已清空）
                else:
                    try:
                    # 生成利用小物体微调移动指令（坐标转换：米→米，注意与机器人坐标系匹配）
                        act = make_action(
                            "small_no_head",
                            lx=(1000.0 * xyz_small[0]),  # xyz[0]为相机X，转换为机器人X偏移（单位：米）
                            ly=0,            
                            lz=(-1000.0 * xyz_small[1]),  # xyz[1]为相机Y，转换为机器人Z偏移（单位：米）
                            
                        )
                        # 若队列满，先清空旧指令（确保执行最新指令）
                        if action_q.full():
                            _ = action_q.get_nowait()
                        action_q.put_nowait(act)  # 放入动作队列
                    except queue.Full:
                        pass  # 队列仍满时忽略（理论上不会发生，因已清空）
        
        try:
            cv2.imshow("YOLOv11 Detection", frame)
            # 等待1ms按键输入（不阻塞主线程）
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"显示失败: {e}")

    except Exception as e:
        rospy.logerr(f"图像处理失败: {e}")  # 记录处理错误日志

# ========== 主程序：初始化节点与话题订阅 ==========
def main():
    """主函数：初始化ROS节点、加载配置、启动线程、订阅话题"""
    global camera_info  # 引用全局相机内参变量

    # 加载相机内参（从YAML文件，根据实际路径修改）
    camera_info = load_camera_info("/home/jetson/yolov5/ost.yaml")

    # 初始化ROS节点（名称为"ros_yolov5_node"，允许匿名实例）
    rospy.init_node("ros_yolov11_node", anonymous=True)

    # 启动动作执行线程（后台运行，daemon=True：主线程退出时自动结束）
    worker_th = Thread(target=action_worker, daemon=True)
    worker_th.start()

    # 订阅ROS话题：
    # 1. 彩色图像话题（用于目标检测）
    rospy.Subscriber(
        "/camera/color/image_raw",  # 话题名称（根据实际相机修改）
        Image, 
        image_callback, 
        queue_size=1,  # 队列大小1：只保留最新帧
        buff_size=2**24  # 缓冲区大小：确保大图像能完整接收
    )
    # 2. 深度图像话题（用于获取3D坐标）
    rospy.Subscriber(
        "/camera/depth/image_raw",  # 话题名称（根据实际相机修改）
        Image, 
        depth_callback, 
        queue_size=1, 
        buff_size=2**24
    )

    rospy.spin()  # 阻塞等待ROS节点关闭

if __name__ == "__main__":
    main()  # 启动主函数