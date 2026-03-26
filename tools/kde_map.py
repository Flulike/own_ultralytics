"""
分析 COCO、VisDrone 和 MLITcctv 数据集
生成组合图：KDE 图（Height Ratio）和 Occlusion 折线图
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde
from collections import defaultdict

# 设置绘图样式
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def load_coco_annotations(json_path):
    """加载 COCO 数据集标注，计算归一化的高度"""
    print(f"Loading COCO annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 创建图像ID到尺寸的映射
    image_sizes = {img['id']: (img['width'], img['height']) for img in data['images']}
    
    height_ratios = []
    for ann in data['annotations']:
        bbox = ann['bbox']  # [x, y, width, height]
        image_id = ann['image_id']
        
        if image_id in image_sizes:
            img_width, img_height = image_sizes[image_id]
            if img_height > 0 and bbox[3] > 0:
                # 计算归一化的高度（相对于图像高度）
                normalized_height = bbox[3] / img_height
                height_ratios.append(normalized_height)
    
    print(f"Loaded {len(height_ratios)} COCO annotations")
    return np.array(height_ratios)


def load_yolo_labels(dataset_root, split_dirs):
    """加载 YOLO 格式数据集标注"""
    print(f"Loading YOLO annotations from {dataset_root}...")
    dataset_root = Path(dataset_root)
    
    height_ratios = []
    
    for split in split_dirs:
        labels_dir = dataset_root / split
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} not found, skipping...")
            continue
        
        label_files = list(labels_dir.glob('*.txt'))
        print(f"Processing {len(label_files)} files from {split}...")
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # class, x_center, y_center, width, height
                            height = float(parts[4])  # YOLO格式中的height已经是归一化的
                            if height > 0 and height <= 1.0:
                                height_ratios.append(height)
            except:
                continue
    
    print(f"Loaded {len(height_ratios)} annotations")
    return np.array(height_ratios)


def load_visdrone_labels(dataset_root):
    """加载 VisDrone 数据集标注 (YOLO 格式)"""
    split_dirs = ['VisDrone2019-DET-train/labels', 'VisDrone2019-DET-val/labels']
    return load_yolo_labels(dataset_root, split_dirs)


def load_mlitcctv_labels(dataset_root):
    """加载 MLITcctv 数据集标注 (YOLO 格式)"""
    split_dirs = ['train/labels', 'valid/labels', 'test/labels']
    return load_yolo_labels(dataset_root, split_dirs)


def load_visdrone_original_annotations(dataset_root):
    """
    从 VisDrone 原始标注文件中读取遮挡信息
    格式: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    occlusion: 0=no, 1=partial, 2=heavy
    """
    print(f"Loading VisDrone original annotations from {dataset_root}...")
    dataset_root = Path(dataset_root)
    
    occlusion_counts = [0, 0, 0]  # [no, partial, heavy]
    
    for split in ['train', 'val']:
        anno_dir = dataset_root / split
        if not anno_dir.exists():
            print(f"Warning: {anno_dir} not found, skipping...")
            continue
        
        txt_files = list(anno_dir.glob('*.txt'))
        print(f"Processing {len(txt_files)} annotation files from {split}...")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 8:
                            occlusion = int(parts[7])  # 最后一个数字是遮挡程度
                            if 0 <= occlusion <= 2:
                                occlusion_counts[occlusion] += 1
            except:
                continue
    
    print(f"Occlusion counts - No: {occlusion_counts[0]}, Partial: {occlusion_counts[1]}, Heavy: {occlusion_counts[2]}")
    return occlusion_counts


def plot_combined_visualization(coco_ratios, visdrone_ratios, 
                                  occlusion_counts, save_path='combined_analysis.png'):
    """绘制组合图：左侧KDE，右侧Occlusion折线图"""
    print("Creating combined visualization...")
    
    # 创建包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ==================== 左图：KDE 对比图（色块填充 + 边界线）====================
    # 限制高度比范围
    coco_ratios = coco_ratios[(coco_ratios > 0) & (coco_ratios <= 1.0)]
    visdrone_ratios = visdrone_ratios[(visdrone_ratios > 0) & (visdrone_ratios <= 1.0)]
    
    # 扩展 x 范围，从负值开始，让曲线完整显示
    x_range = np.linspace(-0.02, 1.0, 1000)
    
    # 计算 KDE
    kde_visdrone = gaussian_kde(visdrone_ratios)
    density_visdrone = kde_visdrone(x_range)
    
    kde_coco = gaussian_kde(coco_ratios)
    density_coco = kde_coco(x_range)
    
    # 绘制填充色块，使用较浅的颜色
    ax1.fill_between(x_range, density_visdrone, alpha=0.4, color='#7FCD91', edgecolor='none')
    ax1.fill_between(x_range, density_coco, alpha=0.4, color='#6B9BD1', edgecolor='none')
    
    # 添加边界线，使用深色，这样即使填充不透明，边界线也能清晰显示
    ax1.plot(x_range, density_visdrone, color='#2D8E4E', linewidth=2.5, label='VisDrone')
    ax1.plot(x_range, density_coco, color='#2E6DA4', linewidth=2.5, label='coco')
    
    ax1.set_xlabel('Height Ratio', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Proportion', fontsize=22, fontweight='bold')
    ax1.set_xlim(-0.02, 1.0)
    ax1.set_ylim(0, None)
    ax1.legend(title='Dataset', fontsize=18, title_fontsize=19, frameon=True, 
               fancybox=True, shadow=True, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(labelsize=14)
    
    # ==================== 右图：Occlusion 折线图 ====================
    # 计算比例
    total_all = sum(occlusion_counts)
    proportions = [count / total_all for count in occlusion_counts]
    
    occlusion_levels = ['no', 'partial', 'heavy']
    
    # 绘制 VisDrone 折线
    ax2.plot(occlusion_levels, proportions, 'o-', 
            color='#F4A300', linewidth=3.5, markersize=12, 
            label='VisDrone', markeredgewidth=2, markeredgecolor='white')
    
    ax2.set_xlabel('Occlusion Level', fontsize=22, fontweight='bold')
    ax2.set_ylabel('Proportion', fontsize=22, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(-0.3, 2.3)
    ax2.legend(loc='upper right', fontsize=18, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.tick_params(labelsize=14)
    
    # 添加垂直虚线
    for i in range(len(occlusion_levels)):
        ax2.axvline(x=i, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    
    # 保存文件，如果是 EPS 格式，设置特殊参数
    if save_path.endswith('.eps'):
        # EPS 格式（不支持透明度）
        plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight', 
                    facecolor='white', transparent=False)
        # 同时保存 PDF 格式（支持透明度）
        pdf_path = save_path.replace('.eps', '.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', 
                    facecolor='white')
        print(f"Combined visualization saved to {save_path} and {pdf_path}")
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Combined visualization saved to {save_path}")
    
    plt.close()
    
    # 打印统计信息
    print(f"\n=== Height Ratio Statistics ===")
    print(f"COCO - Mean: {np.mean(coco_ratios):.3f}, Median: {np.median(coco_ratios):.3f}")
    print(f"VisDrone - Mean: {np.mean(visdrone_ratios):.3f}, Median: {np.median(visdrone_ratios):.3f}")
    
    print(f"\n=== VisDrone Occlusion Proportions (from original annotations) ===")
    print(f"No occlusion: {proportions[0]:.3f} ({proportions[0]*100:.1f}%) - Count: {occlusion_counts[0]}")
    print(f"Partial occlusion: {proportions[1]:.3f} ({proportions[1]*100:.1f}%) - Count: {occlusion_counts[1]}")
    print(f"Heavy occlusion: {proportions[2]:.3f} ({proportions[2]*100:.1f}%) - Count: {occlusion_counts[2]}")


def main():
    # 数据集路径
    coco_train_json = '/mnt/vmlqnap02/home/guo/dataset/coco/annotations/instances_train2017.json'
    coco_val_json = '/mnt/vmlqnap02/home/guo/dataset/coco/annotations/instances_val2017.json'
    visdrone_root = '/mnt/vmlqnap01/datasets/VisDrone_yolo'
    visdrone_anno_root = '/mnt/vmlqnap02/home/guo/dataset/anovis'
    
    # 加载数据
    print("\n=== Loading COCO dataset ===")
    coco_train_ratios = load_coco_annotations(coco_train_json)
    coco_val_ratios = load_coco_annotations(coco_val_json)
    coco_ratios = np.concatenate([coco_train_ratios, coco_val_ratios])
    
    print("\n=== Loading VisDrone dataset ===")
    visdrone_ratios = load_visdrone_labels(visdrone_root)
    
    print("\n=== Loading VisDrone original annotations for occlusion ===")
    occlusion_counts = load_visdrone_original_annotations(visdrone_anno_root)
    
    # 绘制组合图
    print("\n=== Creating combined visualization ===")
    plot_combined_visualization(coco_ratios, visdrone_ratios, 
                                 occlusion_counts, save_path='combined_analysis.eps')
    
    print("\n=== Done! ===")


if __name__ == '__main__':
    main()