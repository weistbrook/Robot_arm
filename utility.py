from robot_controller import RobotController
import re
import dev_angle
#axismap是一个字典，键是轴的名称，值是需要转的角度，转动的格式是Move.Axis [axis], [angle]
def init_robot_controller(controller):
    try:
        # 登录机器人（0为默认用户）
        response = controller.send_command("System.Login 0")
        print(f"机器人登录响应: {response}")
        # 使能机器人动力（1-使能，1-默认参数）
        response = controller.send_command("Robot.PowerEnable 1,1")
        print(f"机器人上电响应: {response}")
        # 启动机器人控制线程
        response = controller.send_command("Thread.Start")
        print(f"机器人线程启动响应: {response}")
        # 设置机器人运动速度（3为中等速度）
        response = controller.send_command("System.Speed 3")
        print(f"机器人速度设置响应: {response}")
    except Exception as e:
        print(f"机器人初始化失败: {e}")

def pixel_to_3d(u, v, depth_img, cam_info_dict):
    """
    根据像素坐标和深度值，计算目标在相机坐标系下的3D坐标
    原理：针孔相机模型，通过内参将像素坐标反投影到3D空间
    参数：
        u, v: 目标像素坐标（图像平面）
        depth_img: 深度图像（OpenCV格式，值为深度，单位毫米）
        cam_info_dict: 相机内参字典（包含焦距、主点等）
    返回：
        3D坐标 tuple (X, Y, Z)（单位：米），若无效则返回None
    """
    # 从内参字典中提取焦距(fx, fy)和主点坐标(cx, cy)
    fx = cam_info_dict['camera_matrix']['data'][0]  # 焦距x
    fy = cam_info_dict['camera_matrix']['data'][4]  # 焦距y
    cx = cam_info_dict['camera_matrix']['data'][2]  # 主点x
    cy = cam_info_dict['camera_matrix']['data'][5]  # 主点y

    # 检查像素坐标是否在图像范围内（防止越界）
    if v >= depth_img.shape[0] or u >= depth_img.shape[1] or v < 0 or u < 0:
        return None

    # 获取深度值（单位：毫米），若为0则表示深度无效
    d = depth_img[int(v), int(u)]
    if d == 0:
        return None  # 深度无效，返回None
    d = d / 1000.0  # 转换为米

    # 计算相机坐标系下的3D坐标（右手坐标系：X右，Y下，Z前）
    X = (u - cx) * d / fx  # X坐标
    Y = (v - cy) * d / fy  # Y坐标
    Z = d                  # Z坐标（深度）
    return (X, Y, Z)

def prevent_stop(controller):
    """
    防止机器人进入停止状态的辅助函数：发送启动和继续指令，确保机器人可响应控制
    用于动作执行前，避免因超时或异常导致机器人无法移动
    """
    try:
        controller.send_command("Thread.Start")       # 启动控制线程
        controller.send_command("System.Speed 3")     # 重置速度
        controller.send_command("Thread.Continue")    # 继续执行（若之前暂停）
        # 获取机器人当前位置（调试用）
        response = controller.send_command("Robot.Where 1")
        print(f"机器人当前位置响应: {response}")
    except Exception as e:
        rospy.logwarn(f"prevent_stop 执行异常: {e}")  # 打印警告日志

def move_axis(controller, axismap):
    if axismap is None:
        return
    for axis, angle in axismap.items():
        if angle > 0:
            response = controller.send_command(f"Move.Axis {axis},{-angle}")
        else: 
            response = controller.send_command(f"Move.Axis {axis},{abs(angle)}")

"""
让机械臂末端的法兰盘与阀门平面保持平行状态，深度相机是装在法兰盘中心的,角度响应的格式为
[1#0 axis1_angle, axis2_angle, axis3_angle, axis4_angle, axis5_angle, axis6_angle],
其中axis1_angle到axis6_angle分别是六个轴的当前角度，单位是度。需要根据后三个轴角度判断机械臂末端法兰盘是否处于平行状态，
如果不平行则进行调整。
"""

def keep_parallel(controller):
    response = controller.send_command("Robot.WhereAngle 1")
    rospy.loginfo(f"机器人角度响应: {response}")
    try:
        s = response.strip("[] \n")
        # 按 “逗号或任意空白” 分割，多个连续分隔符会被当成一个
        parts = re.split(r"[,\s,#]+", s)
        vals = [int(x) for x in parts if x != ""]
        axismap = {}
        # if vals[4] != 0:
        #     rospy.loginfo("机器人未处于平行状态，正在调整...")
        axismap = {f"Axis{i}": vals[i] for i in range(3, 5) if vals[i] != 0}
        move_axis(controller, axismap)
        #response = controller.send_command("Move.JOffset {0, 0, 0, 0, 0, 0}")    
    except Exception as e:
        rospy.logwarn(f"judge_action 解析失败: {repr(response)}, {e}")

# ========== 工具函数：辅助实现核心功能 ==========
def load_camera_info(yaml_file):
    """
    从YAML文件加载相机内参（焦距、主点等）
    参数：
        yaml_file: 相机内参YAML文件路径
    返回：
        包含相机内参的字典（如camera_matrix、distortion_coefficients等）
    """
    with open(yaml_file, 'r') as file:
        cam_info = yaml.safe_load(file)  # 安全加载YAML内容
    rospy.loginfo(f"加载的相机内参: {cam_info}")  # 打印内参信息（调试用）
    return cam_info

def judge_proper(roi):
    """
    判断目标（如阀门）是否对正：通过dev_angle模块计算角度偏差
    参数：
        roi: 目标区域图像（ROI，OpenCV格式）
    返回：
        (is_proper, offset_deg): 元组，is_proper为是否对正（布尔值），offset_deg为角度偏差（度）
    """
    try:
        # 生成目标掩码（如阀门区域掩码）
        mask = dev_angle.valve_mask(roi)
        # 计算目标主方向角度（如阀门十字的角度）
        angle_deg, center = dev_angle.dominant_cross_angle(mask)
        # 计算与正方向的偏差角度
        offset_deg = float(dev_angle.angle_offset_from_upright(angle_deg))
        # 偏差小于10度视为对正
        return (abs(offset_deg) < 10.0), offset_deg
    except Exception as e:
        rospy.logwarn(f"judge_proper 计算失败: {e}")  # 打印警告
        return False, 0.0  # 异常时返回默认值
    
# def judge_action():
#     cmd = "Robot.State 1"
#     while not rospy.is_shutdown():
#         resp = controller.send_command(cmd)
#         try:
#             s = resp.strip("[] \n")
#             # 按 “逗号或任意空白” 分割，多个连续分隔符会被当成一个
#             parts = re.split(r"[,\s,#]+", s)
#             vals = [int(x) for x in parts if x != ""]
#             rospy.loginfo(f"judge_action 解析成功: {repr(resp)}, {e}")
#             state = vals[4]
#         except Exception as e:
#             rospy.logwarn(f"judge_action 解析失败: {repr(resp)}, {e}")
#             break

#         if state != 1:
#             break
#         rospy.sleep(0.1)    