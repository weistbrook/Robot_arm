import json
import os
from PIL import Image
from pathlib import Path

def json_to_yolo(json_path, image_folder, output_folder, label_map):
    """
    单个JSON文件转换为YOLO格式TXT
    :param json_path: JSON文件路径
    :param image_folder: 图像存放文件夹
    :param output_folder: TXT输出文件夹
    :param label_map: 标签到ID的映射
    """
    # 读取JSON数据
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取{json_path}失败: {e}")
        return

    # 获取对应图像路径（从JSON的imagePath字段获取）
    image_name = data.get('imagePath')
    if not image_name:
        print(f"{json_path}缺少imagePath字段，已跳过")
        return
    
    image_path = os.path.join(image_folder, image_name)
    if not os.path.exists(image_path):
        print(f"图像{image_path}不存在，已跳过{json_path}")
        return

    # 获取图像宽高
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"读取图像{image_path}失败: {e}，已跳过{json_path}")
        return

    # 生成输出TXT路径（与JSON同名）
    txt_name = os.path.splitext(os.path.basename(json_path))[0] + '.txt'
    output_txt_path = os.path.join(output_folder, txt_name)

    # 写入YOLO格式内容
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for shape in data.get('shapes', []):
            if shape['shape_type'] != 'rectangle':
                continue  # 只处理矩形标注
            
            label = shape['label']
            if label not in label_map:
                print(f"{json_path}中存在未定义标签'{label}'，已跳过该目标")
                continue
            
            class_id = label_map[label]
            (x1, y1), (x2, y2) = shape['points']
            
            # 计算坐标（确保最小值在前）
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            
            # 归一化计算
            center_x = (xmin + xmax) / (2 * img_width)
            center_y = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # 写入（保留6位小数）
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

def batch_convert(json_folder, image_folder, output_folder, label_map):
    """批量转换文件夹中的所有JSON文件"""
    # 创建输出文件夹（如果不存在）
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 遍历JSON文件夹中的所有.json文件
    for filename in os.listdir(json_folder):
        if filename.lower().endswith('.json'):
            json_path = os.path.join(json_folder, filename)
            print(f"正在处理: {filename}")
            json_to_yolo(json_path, image_folder, output_folder, label_map)
    
    print("批量转换完成！")

if __name__ == "__main__":
    # 配置参数（请根据实际路径修改）
    json_folder = "yellowBlockdataset/images"    # 存放所有JSON文件的文件夹
    image_folder = "yellowBlockdataset/images"       # 存放对应图像的文件夹（与JSON中imagePath对应）
    output_folder = "yellowBlockdataset/yolotxt"    # 输出TXT文件的文件夹
    label_map = {                         # 标签映射（根据你的实际类别修改）
        "block": 1,
        "valve": 0
        # 可添加更多类别，如"person": 2, "car": 3...
    }
    
    # 执行批量转换
    batch_convert(json_folder, image_folder, output_folder, label_map)