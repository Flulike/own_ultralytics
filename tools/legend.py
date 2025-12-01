"""
创建 Ultralytics YOLO 检测框颜色图例
显示不同类别对应的边界框颜色
"""

import cv2
import numpy as np
from ultralytics.utils.plotting import Colors

# carclass
CLASS_NAMES = {
    0: "car",
    1: "light-cargo",
    2: "bus",
    3: "cargo",
    4: "special",
    5: "motorcycle",
    6: "bicycle",
    7: "person",
}

# visdrone
# CLASS_NAMES = {
#     0: "pedestrian",
#     1: "people",
#     2: "bicycle",
#     3: "car",
#     4: "van",
#     5: "truck",
#     6: "tricycle",
#     7: "awning-tricycle",
#     8: "bus",
#     9: "motor"
# }

# 初始化颜色实例
colors = Colors()

# 图例参数
box_width = 150
box_height = 40
padding = 15
line_thickness = 3  # 边框线条粗细

# 选择要显示的类别（7个类别）
selected_classes = list(range(8))

# 计算图像尺寸
num_classes = len(selected_classes)
img_width = (box_width + padding) * num_classes + padding
img_height = box_height + padding * 2

# 创建浅灰色背景（避免与白色框冲突）
img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 200

# 绘制每个类别的颜色框和标签
for idx, class_id in enumerate(selected_classes):
    # 计算位置
    x = padding + idx * (box_width + padding)
    y = padding
    
    # 获取该类别的颜色（BGR格式）
    color = colors(class_id, bgr=True)
    
    # 绘制空心矩形
    cv2.rectangle(img, (x, y), (x + box_width, y + box_height), color, line_thickness)
    
    # 添加类别名称
    class_name = CLASS_NAMES.get(class_id, f"class{class_id}")
    
    # 设置文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    # 获取文本尺寸
    (text_width, text_height), baseline = cv2.getTextSize(
        class_name, font, font_scale, font_thickness
    )
    
    # 计算文本位置（居中）
    text_x = x + (box_width - text_width) // 2
    text_y = y + (box_height + text_height) // 2
    
    # 使用黑色文字（在浅灰背景上清晰可见）
    text_color = (0, 0, 0)
    
    cv2.putText(
        img,
        class_name,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

# 保存图像
output_path = "color_legend.jpg"
cv2.imwrite(output_path, img)
print(f"颜色图例已保存到: {output_path}")
