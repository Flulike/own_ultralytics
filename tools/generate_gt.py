import json
import cv2
import os
from pathlib import Path
import numpy as np

def load_coco_data(json_path):
    """加载COCO格式的标注文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def draw_ground_truth(image_path, annotations, categories, output_path):
    """在图像上绘制ground truth标注框"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return False
    
    # 类别颜色映射
    colors = {
        1: (0, 255, 0),    # 绿色
        2: (255, 0, 0),    # 蓝色
        3: (0, 0, 255),    # 红色
        4: (255, 255, 0),  # 青色
        5: (255, 0, 255),  # 洋红色
    }
    
    # 类别名称映射
    category_names = {cat['id']: cat['name'] for cat in categories}
    
    # 绘制每个标注框
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        category_id = ann['category_id']
        
        # 获取颜色和类别名称
        color = colors.get(category_id, (128, 128, 128))
        category_name = category_names.get(category_id, f"class_{category_id}")
        
        # 绘制边界框
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        
        # 绘制类别标签
        label = f"{category_name}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (int(x), int(y) - label_size[1] - 10), 
                     (int(x) + label_size[0], int(y)), color, -1)
        cv2.putText(image, label, (int(x), int(y) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 保存结果
    cv2.imwrite(output_path, image)
    print(f"Ground truth结果已保存到: {output_path}")
    return True

def generate_ground_truth_for_image(json_path, target_image_name, output_dir):
    """为指定图像生成ground truth可视化结果"""
    # 加载COCO数据
    coco_data = load_coco_data(json_path)
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    # 查找目标图像
    target_image = None
    for img in images:
        if target_image_name in img['file_name']:
            target_image = img
            break
    
    if target_image is None:
        print(f"未找到包含'{target_image_name}'的图像")
        return False
    
    # 获取该图像的所有标注
    image_id = target_image['id']
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
    
    print(f"找到图像: {target_image['file_name']}")
    print(f"标注数量: {len(image_annotations)}")
    
    # 构建图像路径
    image_path = f"data/Fisheye/test/images/{target_image['file_name']}"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出路径
    output_filename = f"gt_{Path(target_image['file_name']).stem}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    
    # 绘制ground truth
    success = draw_ground_truth(image_path, image_annotations, categories, output_path)
    
    if success:
        print(f"成功生成ground truth可视化结果")
        print(f"类别统计:")
        category_count = {}
        for ann in image_annotations:
            cat_id = ann['category_id']
            cat_name = next((cat['name'] for cat in categories if cat['id'] == cat_id), f"class_{cat_id}")
            category_count[cat_name] = category_count.get(cat_name, 0) + 1
        
        for cat_name, count in category_count.items():
            print(f"  {cat_name}: {count}")
    
    return success

if __name__ == "__main__":
    # 配置参数
    json_path = "data/Fisheye/test/test.json"
    target_image_name = "camera1_A_119"  # 根据你的predict.py中的图像名
    output_dir = "results/ground_truth"
    
    # 生成ground truth
    generate_ground_truth_for_image(json_path, target_image_name, output_dir) 