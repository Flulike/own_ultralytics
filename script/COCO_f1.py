"""
Author: Huaxiang Zhang
Date: 2025-01-17
Version: 1.0

Description:
This script is a sample file for evaluating and validating COCO formatted annotation and prediction files.
It performs the following tasks:
- Verifies that the image IDs in the annotation and prediction files match.
- Verifies that the categories in the annotation and prediction files are consistent.
- Checks for common issues in the prediction file, such as invalid bounding box formats, missing or invalid scores, and incorrect category IDs.
- Verifies the content of the annotation file, including checking the number of annotations, images, and categories.
- Provides examples of a few annotation and prediction entries to help with debugging.
- Attempts to run the COCO evaluation using pycocotools to evaluate the predicted bounding boxes against the ground truth annotations.

Note:
This script is intended as a simple example to demonstrate how to handle COCO evaluation using Python and pycocotools.
It was generated with the help of ChatGPT.


"""
"""
Edit by Guo
Date: 2025-05-26

Description:
This script is a modified version of the COCO.py script.
It adds the F1 score calculation to the script.


"""

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def check_image_ids(anno_json, pred_json):
    anno = load_json(anno_json)
    pred = load_json(pred_json)
    
    anno_image_ids = set([img['id'] for img in anno['images']])
    pred_image_ids = set([ann['image_id'] for ann in pred])
    
    missing_in_pred = anno_image_ids - pred_image_ids
    missing_in_anno = pred_image_ids - anno_image_ids
    
    print(f"Total images in annotations: {len(anno_image_ids)}")
    print(f"Total images in predictions: {len(pred_image_ids)}")
    print(f"Images in annotations but not in predictions: {len(missing_in_pred)}")
    print(f"Images in predictions but not in annotations: {len(missing_in_anno)}")
    
    if missing_in_pred:
        print("Some images in annotations are missing in predictions:")
        for img_id in list(missing_in_pred)[:10]:  # 打印前10个
            print(img_id)
    
    if missing_in_anno:
        print("Some images in predictions are missing in annotations:")
        for img_id in list(missing_in_anno)[:10]:
            print(img_id)

def check_categories(anno_json, pred_json):
    anno = load_json(anno_json)
    pred = load_json(pred_json)
    
    anno_categories = set([cat['id'] for cat in anno['categories']])
    pred_categories = set([ann['category_id'] for ann in pred])
    
    missing_in_pred = anno_categories - pred_categories
    missing_in_anno = pred_categories - anno_categories
    
    print(f"Total categories in annotations: {len(anno_categories)}")
    print(f"Total categories in predictions: {len(pred_categories)}")
    print(f"Categories in annotations but not in predictions: {len(missing_in_pred)}")
    print(f"Categories in predictions but not in annotations: {len(missing_in_anno)}")
    
    if missing_in_pred:
        print("Some categories in annotations are missing in predictions:")
        for cat_id in list(missing_in_pred):
            print(cat_id)
    
    if missing_in_anno:
        print("Some categories in predictions are missing in annotations:")
        for cat_id in list(missing_in_anno):
            print(cat_id)

def check_predictions(pred_json):
    pred = load_json(pred_json)
    print(f"Total predictions: {len(pred)}")
    
    # 检查边界框格式
    invalid_bbox = [ann for ann in pred if not (isinstance(ann.get('bbox'), list) and len(ann['bbox']) == 4)]
    print(f"Predictions with invalid bbox format: {len(invalid_bbox)}")
    
    # 检查得分是否存在且在合理范围内
    invalid_scores = [ann for ann in pred if 'score' not in ann or not (0 <= ann['score'] <= 1)]
    print(f"Predictions with invalid or missing scores: {len(invalid_scores)}")
    
    # 检查类别ID是否为正整数
    invalid_cat_ids = [ann for ann in pred if not isinstance(ann.get('category_id'), int) or ann['category_id'] <= 0]
    print(f"Predictions with invalid category_ids: {len(invalid_cat_ids)}")

def check_annotations(anno_json):
    anno = load_json(anno_json)
    print(f"Total annotations: {len(anno.get('annotations', []))}")
    print(f"Total images: {len(anno.get('images', []))}")
    print(f"Total categories: {len(anno.get('categories', []))}")

def print_sample_predictions(pred_json, num_samples=5):
    pred = load_json(pred_json)
    print(f"\n打印 {num_samples} 个预测条目的示例:")
    for idx, ann in enumerate(pred[:num_samples]):
        print(f"条目 {idx + 1} 数据类型: {type(ann)}")
        print(json.dumps(ann, indent=2))

def print_sample_annotations(anno_json, num_samples=5):
    anno = load_json(anno_json)
    print(f"\n打印 {num_samples} 个注释条目的示例:")
    for idx, ann in enumerate(anno['annotations'][:num_samples]):
        print(f"条目 {idx + 1} 数据类型: {type(ann)}")
        print(json.dumps(ann, indent=2))

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)
    
    Args:
        box1, box2: [x, y, width, height] format
    
    Returns:
        float: IoU value
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 转换为 [x1, y1, x2, y2] 格式
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # 计算交集
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_f1_score(anno_json, pred_json, iou_threshold=0.5, score_threshold=0.5):
    """
    计算F1 score
    
    Args:
        anno_json: 标注文件路径
        pred_json: 预测文件路径
        iou_threshold: IoU阈值，用于判断是否为正确检测
        score_threshold: 置信度阈值，用于过滤预测结果
    
    Returns:
        dict: 包含precision, recall, f1_score的字典
    """
    print(f"\n计算F1 Score (IoU阈值: {iou_threshold}, 置信度阈值: {score_threshold})...")
    
    anno_data = load_json(anno_json)
    pred_data = load_json(pred_json)
    
    # 过滤低置信度的预测
    filtered_predictions = [pred for pred in pred_data if pred.get('score', 0) >= score_threshold]
    print(f"过滤后的预测数量: {len(filtered_predictions)} (原始: {len(pred_data)})")
    
    # 按图像ID组织数据
    gt_by_image = {}
    for ann in anno_data['annotations']:
        image_id = ann['image_id']
        if image_id not in gt_by_image:
            gt_by_image[image_id] = []
        gt_by_image[image_id].append(ann)
    
    pred_by_image = {}
    for pred in filtered_predictions:
        image_id = pred['image_id']
        if image_id not in pred_by_image:
            pred_by_image[image_id] = []
        pred_by_image[image_id].append(pred)
    
    # 计算每个类别的统计信息
    category_stats = {}
    all_categories = set([cat['id'] for cat in anno_data['categories']])
    
    for cat_id in all_categories:
        category_stats[cat_id] = {
            'tp': 0,  # True Positives
            'fp': 0,  # False Positives
            'fn': 0   # False Negatives
        }
    
    # 遍历所有图像进行匹配
    for image_id in set(list(gt_by_image.keys()) + list(pred_by_image.keys())):
        gt_anns = gt_by_image.get(image_id, [])
        pred_anns = pred_by_image.get(image_id, [])
        
        # 按类别分组
        for cat_id in all_categories:
            gt_cat = [ann for ann in gt_anns if ann['category_id'] == cat_id]
            pred_cat = [pred for pred in pred_anns if pred['category_id'] == cat_id]
            
            # 创建匹配矩阵
            matched_gt = set()
            matched_pred = set()
            
            # 按预测置信度排序
            pred_cat_sorted = sorted(pred_cat, key=lambda x: x.get('score', 0), reverse=True)
            
            for pred_idx, pred in enumerate(pred_cat_sorted):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_cat):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    # True Positive
                    category_stats[cat_id]['tp'] += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                else:
                    # False Positive
                    category_stats[cat_id]['fp'] += 1
            
            # 未匹配的ground truth为False Negatives
            category_stats[cat_id]['fn'] += len(gt_cat) - len(matched_gt)
    
    # 计算每个类别的precision, recall, f1
    category_results = {}
    overall_tp, overall_fp, overall_fn = 0, 0, 0
    
    for cat_id in all_categories:
        tp = category_stats[cat_id]['tp']
        fp = category_stats[cat_id]['fp']
        fn = category_stats[cat_id]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        category_results[cat_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
    
    # 计算总体指标
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # 打印结果
    print("\n=== F1 Score 结果 ===")
    print(f"总体指标:")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1 Score: {overall_f1:.4f}")
    print(f"  TP: {overall_tp}, FP: {overall_fp}, FN: {overall_fn}")
    
    print(f"\n各类别详细指标:")
    for cat_id in sorted(all_categories):
        result = category_results[cat_id]
        print(f"  类别 {cat_id}:")
        print(f"    Precision: {result['precision']:.4f}")
        print(f"    Recall: {result['recall']:.4f}")
        print(f"    F1 Score: {result['f1_score']:.4f}")
        print(f"    TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}")
    
    return {
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'tp': overall_tp,
            'fp': overall_fp,
            'fn': overall_fn
        },
        'categories': category_results
    }

def main(anno_json, pred_json):
    print("检查注释文件内容...")
    check_annotations(anno_json)
    
    print("\n检查图像ID匹配情况...")
    check_image_ids(anno_json, pred_json)
    
    print("\n检查类别ID匹配情况...")
    check_categories(anno_json, pred_json)
    
    print("\n检查预测文件内容...")
    check_predictions(pred_json)
    
    # 打印示例预测条目
    print_sample_predictions(pred_json)
    
    print_sample_annotations(anno_json)
    
    # 尝试进行COCO评估
    print("\n尝试进行COCO评估...")
    try:
        coco_gt = COCO(anno_json)
        coco_dt = coco_gt.loadRes(pred_json)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        print(f"COCO评估过程中出现错误: {e}")
    
    # 计算F1 Score
    print("\n计算F1 Score...")
    try:
        # 使用不同的IoU阈值计算F1 Score
        iou_thresholds = [0.5, 0.75]
        score_threshold = 0.5
        
        for iou_thresh in iou_thresholds:
            f1_results = calculate_f1_score(anno_json, pred_json, 
                                          iou_threshold=iou_thresh, 
                                          score_threshold=score_threshold)
            
        # 也可以尝试不同的置信度阈值
        print(f"\n使用不同置信度阈值 (IoU=0.5):")
        score_thresholds = [0.3, 0.5]
        for score_thresh in score_thresholds:
            print(f"\n--- 置信度阈值: {score_thresh} ---")
            f1_results = calculate_f1_score(anno_json, pred_json, 
                                          iou_threshold=0.5, 
                                          score_threshold=score_thresh)
            overall = f1_results['overall']
            print(f"总体 F1 Score: {overall['f1_score']:.4f}")
            
    except Exception as e:
        print(f"F1 Score计算过程中出现错误: {e}")


if __name__ == '__main__':
    anno_json = '/mnt/vmlqnap01/datasets/Fisheye/test/test.json'
    pred_json = '/home/guo/own_ultralytics/results/ultralytics/yolov11/x/fisheye_vml3_test3/predictions_converted.json'
    main(anno_json, pred_json)
