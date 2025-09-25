#!/usr/bin/env python3
"""
Complete Inner-IoU Analysis
===========================
Single script for comprehensive Inner-IoU analysis including:
1. Comparison with regular IoU
2. GT size effects
3. Ratio comparison (including ratio > 1.0)
"""

import torch
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def calculate_iou_metrics(box1, box2, ratio=0.7, eps=1e-7):
    """Calculate both regular IoU and Inner-IoU for given ratio"""
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    
    # Original box coordinates
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    
    # Scaled box coordinates (inner when ratio<1, outer when ratio>1)
    scaled_b1_x1, scaled_b1_x2 = x1 - w1_*ratio, x1 + w1_*ratio
    scaled_b1_y1, scaled_b1_y2 = y1 - h1_*ratio, y1 + h1_*ratio
    scaled_b2_x1, scaled_b2_x2 = x2 - w2_*ratio, x2 + w2_*ratio
    scaled_b2_y1, scaled_b2_y2 = y2 - h2_*ratio, y2 + h2_*ratio
    
    # Regular IoU
    regular_inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                    (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    regular_union = w1 * h1 + w2 * h2 - regular_inter + eps
    regular_iou = regular_inter / regular_union
    
    # Scaled IoU (Inner/Outer)
    scaled_inter = (torch.min(scaled_b1_x2, scaled_b2_x2) - torch.max(scaled_b1_x1, scaled_b2_x1)).clamp(0) * \
                   (torch.min(scaled_b1_y2, scaled_b2_y2) - torch.max(scaled_b1_y1, scaled_b2_y1)).clamp(0)
    scaled_union = w1*ratio * h1*ratio + w2*ratio * h2*ratio - scaled_inter + eps
    scaled_iou = scaled_inter / scaled_union
    
    return regular_iou.item(), scaled_iou.item()

def analyze_vs_regular_iou():
    """Compare Inner-IoU vs Regular IoU"""
    print("=" * 80)
    print("1. INNER-IOU vs REGULAR IOU COMPARISON")
    print("=" * 80)
    
    test_cases = [
        ([2, 2, 2, 2], [2, 2, 2, 2], "Perfect match"),
        ([2, 2, 2, 2], [2.5, 2, 2, 2], "Small offset"),
        ([2, 2, 2, 2], [3, 2, 2, 2], "Medium offset"),
        ([2, 2, 2, 2], [2, 2, 3, 3], "GT larger"),
        ([2, 2, 1.5, 1.5], [2, 2, 2.5, 2.5], "Size mismatch"),
    ]
    
    ratio = 0.7
    print(f"Using ratio = {ratio}")
    print("Case\t\t\tRegular IoU\tInner-IoU\tDifference")
    print("-" * 65)
    
    for pred, gt, desc in test_cases:
        pred_tensor = torch.tensor([pred])
        gt_tensor = torch.tensor([gt])
        regular_iou, inner_iou = calculate_iou_metrics(pred_tensor, gt_tensor, ratio)
        diff = inner_iou - regular_iou
        print(f"{desc:<20}\t{regular_iou:.4f}\t\t{inner_iou:.4f}\t\t{diff:+.4f}")

def analyze_gt_size_effects():
    """Analyze GT size effects on Inner-IoU"""
    print("\n" + "=" * 80)
    print("2. GROUND TRUTH SIZE EFFECTS")
    print("=" * 80)
    
    # Fixed prediction box, varying GT sizes
    pred_box = [2.0, 2.0, 2.0, 2.0]
    gt_sizes = [1.0, 2.0, 3.0, 4.0, 5.0]
    offsets = [0.0, 0.5, 1.0]
    ratio = 0.7
    
    print(f"Prediction box: {pred_box} (fixed)")
    print(f"Ratio: {ratio}")
    print()
    
    for gt_size in gt_sizes:
        print(f"GT size: {gt_size}x{gt_size}")
        print("Offset\tRegular IoU\tInner-IoU\tDifference")
        print("-" * 45)
        
        for offset in offsets:
            gt_box = [2.0 + offset, 2.0, gt_size, gt_size]
            pred_tensor = torch.tensor([pred_box])
            gt_tensor = torch.tensor([gt_box])
            regular_iou, inner_iou = calculate_iou_metrics(pred_tensor, gt_tensor, ratio)
            diff = inner_iou - regular_iou
            print(f"{offset:.1f}\t{regular_iou:.4f}\t\t{inner_iou:.4f}\t\t{diff:+.4f}")
        print()

def analyze_ratio_effects():
    """Analyze different ratio effects including ratio > 1.0"""
    print("=" * 80)
    print("3. RATIO EFFECTS ANALYSIS (Including ratio > 1.0)")
    print("=" * 80)
    
    print("When ratio > 1.0, boxes become LARGER (Outer-IoU)")
    print("When ratio < 1.0, boxes become SMALLER (Inner-IoU)")
    print()
    
    test_cases = [
        ([2, 2, 2, 2], [2, 2, 2, 2], "Perfect match"),
        ([2, 2, 2, 2], [2.3, 2, 2, 2], "Small offset"),
        ([2, 2, 2, 2], [2.8, 2, 2, 2], "Medium offset"),
        ([2, 2, 2, 2], [3.5, 2, 2, 2], "Large offset"),
    ]
    
    ratios = [0.5, 0.7, 1.0, 1.2, 1.5]
    
    for pred, gt, desc in test_cases:
        print(f"\n{desc}: Pred={pred}, GT={gt}")
        print("Ratio\tType\t\tRegular IoU\tScaled IoU\tDifference")
        print("-" * 65)
        
        pred_tensor = torch.tensor([pred])
        gt_tensor = torch.tensor([gt])
        
        for ratio in ratios:
            regular_iou, scaled_iou = calculate_iou_metrics(pred_tensor, gt_tensor, ratio)
            diff = scaled_iou - regular_iou
            
            if ratio < 1.0:
                ratio_type = "Inner"
            elif ratio == 1.0:
                ratio_type = "Regular"
            else:
                ratio_type = "Outer"
            
            print(f"{ratio:.1f}\t{ratio_type:<8}\t{regular_iou:.4f}\t\t{scaled_iou:.4f}\t\t{diff:+.4f}")

def create_visualizations():
    """Create all visualizations"""
    print("\n" + "=" * 80)
    print("4. CREATING ALL VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Basic Inner-IoU visualization
    create_basic_inner_iou_viz()
    
    # 2. Aspect ratio comparison
    create_aspect_ratio_viz()
    
    # 3. GT size effect analysis
    create_gt_size_viz()
    
    # 4. Ratio effects analysis
    create_ratio_effects_viz()

def create_basic_inner_iou_viz():
    """Create basic Inner-IoU visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Inner-IoU vs Regular IoU Comparison', fontsize=16, fontweight='bold')
    
    # Overlapping boxes scenario
    box1 = [2.0, 2.0, 2.0, 2.0]
    box2 = [2.5, 2.5, 2.0, 2.0]
    ratio = 0.7
    
    def draw_boxes(ax, title, show_inner=False):
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Box 1 (blue)
        x1, y1, w1, h1 = box1
        rect1 = patches.Rectangle((x1-w1/2, y1-h1/2), w1, h1, 
                                 linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, label='Box1')
        ax.add_patch(rect1)
        ax.text(x1, y1, 'Box1', ha='center', va='center', fontweight='bold', color='blue')
        
        # Box 2 (red)
        x2, y2, w2, h2 = box2
        rect2 = patches.Rectangle((x2-w2/2, y2-h2/2), w2, h2, 
                                 linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, label='Box2')
        ax.add_patch(rect2)
        ax.text(x2, y2, 'Box2', ha='center', va='center', fontweight='bold', color='red')
        
        if show_inner:
            # Inner Box 1 (blue dashed)
            inner_w1, inner_h1 = w1 * ratio, h1 * ratio
            inner_rect1 = patches.Rectangle((x1-inner_w1/2, y1-inner_h1/2), inner_w1, inner_h1, 
                                          linewidth=2, edgecolor='blue', facecolor='none', 
                                          linestyle='--', alpha=0.8, label='Inner Box1')
            ax.add_patch(inner_rect1)
            
            # Inner Box 2 (red dashed)
            inner_w2, inner_h2 = w2 * ratio, h2 * ratio
            inner_rect2 = patches.Rectangle((x2-inner_w2/2, y2-inner_h2/2), inner_w2, inner_h2, 
                                          linewidth=2, edgecolor='red', facecolor='none', 
                                          linestyle='--', alpha=0.8, label='Inner Box2')
            ax.add_patch(inner_rect2)
            
            ax.text(0.02, 0.98, f'Dashed: Inner boxes (ratio={ratio})', 
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    draw_boxes(ax1, "Regular IoU Calculation", show_inner=False)
    draw_boxes(ax2, "Inner-IoU Calculation", show_inner=True)
    
    plt.tight_layout()
    plt.savefig('/home/guo/own_ultralytics/inner_iou_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: inner_iou_visualization.png")

def create_aspect_ratio_viz():
    """Create aspect ratio comparison visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Inner-IoU with Different Aspect Ratios', fontsize=16, fontweight='bold')
    
    # Different aspect ratios
    box1_aspect = [2.0, 2.0, 4.0, 1.0]  # Wide box
    box2_aspect = [2.0, 2.5, 1.0, 4.0]  # Tall box
    ratio = 0.7
    
    def draw_aspect_boxes(ax, title, show_inner=False):
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Box 1 (blue, wide)
        x1, y1, w1, h1 = box1_aspect
        rect1 = patches.Rectangle((x1-w1/2, y1-h1/2), w1, h1, 
                                 linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, label='Wide Box')
        ax.add_patch(rect1)
        ax.text(x1, y1, 'Wide', ha='center', va='center', fontweight='bold', color='blue')
        
        # Box 2 (red, tall)
        x2, y2, w2, h2 = box2_aspect
        rect2 = patches.Rectangle((x2-w2/2, y2-h2/2), w2, h2, 
                                 linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, label='Tall Box')
        ax.add_patch(rect2)
        ax.text(x2, y2, 'Tall', ha='center', va='center', fontweight='bold', color='red')
        
        if show_inner:
            # Inner boxes
            inner_w1, inner_h1 = w1 * ratio, h1 * ratio
            inner_rect1 = patches.Rectangle((x1-inner_w1/2, y1-inner_h1/2), inner_w1, inner_h1, 
                                          linewidth=2, edgecolor='blue', facecolor='none', 
                                          linestyle='--', alpha=0.8)
            ax.add_patch(inner_rect1)
            
            inner_w2, inner_h2 = w2 * ratio, h2 * ratio
            inner_rect2 = patches.Rectangle((x2-inner_w2/2, y2-inner_h2/2), inner_w2, inner_h2, 
                                          linewidth=2, edgecolor='red', facecolor='none', 
                                          linestyle='--', alpha=0.8)
            ax.add_patch(inner_rect2)
    
    draw_aspect_boxes(ax1, "Regular IoU", show_inner=False)
    draw_aspect_boxes(ax2, "Inner-IoU (ratio=0.7)", show_inner=True)
    
    plt.tight_layout()
    plt.savefig('/home/guo/own_ultralytics/inner_iou_aspect_ratio.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: inner_iou_aspect_ratio.png")

def create_gt_size_viz():
    """Create GT size effect visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ground Truth Size Effect on Inner-IoU', fontsize=16, fontweight='bold')
    
    scenarios = [
        {'pred': [2, 2, 2, 2], 'gt': [2, 2, 2, 2], 'title': 'Same Size'},
        {'pred': [2, 2, 2, 2], 'gt': [2, 2, 3, 3], 'title': 'GT Larger'},
        {'pred': [2, 2, 2, 2], 'gt': [2.5, 2, 2, 2], 'title': 'Center Offset'},
        {'pred': [2, 2, 1.5, 1.5], 'gt': [2, 2, 2.5, 2.5], 'title': 'Scale Difference'}
    ]
    
    ratio = 0.7
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 5.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(scenario['title'], fontsize=12, fontweight='bold')
        
        # Prediction box (blue)
        pred = scenario['pred']
        pred_rect = patches.Rectangle((pred[0]-pred[2]/2, pred[1]-pred[3]/2), 
                                     pred[2], pred[3], 
                                     linewidth=2, edgecolor='blue', 
                                     facecolor='blue', alpha=0.3, label='Prediction')
        ax.add_patch(pred_rect)
        
        # GT box (red)
        gt = scenario['gt']
        gt_rect = patches.Rectangle((gt[0]-gt[2]/2, gt[1]-gt[3]/2), 
                                   gt[2], gt[3], 
                                   linewidth=2, edgecolor='red', 
                                   facecolor='red', alpha=0.3, label='Ground Truth')
        ax.add_patch(gt_rect)
        
        # Inner boxes (dashed)
        inner_pred_w, inner_pred_h = pred[2] * ratio, pred[3] * ratio
        inner_pred_rect = patches.Rectangle((pred[0]-inner_pred_w/2, pred[1]-inner_pred_h/2), 
                                           inner_pred_w, inner_pred_h, 
                                           linewidth=2, edgecolor='blue', 
                                           facecolor='none', linestyle='--', alpha=0.8)
        ax.add_patch(inner_pred_rect)
        
        inner_gt_w, inner_gt_h = gt[2] * ratio, gt[3] * ratio
        inner_gt_rect = patches.Rectangle((gt[0]-inner_gt_w/2, gt[1]-inner_gt_h/2), 
                                         inner_gt_w, inner_gt_h, 
                                         linewidth=2, edgecolor='red', 
                                         facecolor='none', linestyle='--', alpha=0.8)
        ax.add_patch(inner_gt_rect)
        
        # Calculate and display metrics
        pred_tensor = torch.tensor([scenario['pred']])
        gt_tensor = torch.tensor([scenario['gt']])
        regular_iou, inner_iou = calculate_iou_metrics(pred_tensor, gt_tensor, ratio)
        
        info_text = f"Regular IoU: {regular_iou:.3f}\n"
        info_text += f"Inner-IoU: {inner_iou:.3f}\n"
        info_text += f"Difference: {inner_iou-regular_iou:+.3f}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/home/guo/own_ultralytics/gt_size_effect_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: gt_size_effect_analysis.png")

def create_ratio_effects_viz():
    """Create ratio effects visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Effect of Different Ratios on Inner-IoU Calculation', fontsize=16, fontweight='bold')
    
    # Test case: slightly offset boxes
    pred_box = [2.0, 2.0, 2.0, 2.0]
    gt_box = [2.5, 2.0, 2.0, 2.0]
    
    ratios = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
    
    for idx, ratio in enumerate(ratios):
        ax = axes[idx // 3, idx % 3]
        ax.set_xlim(0, 5)
        ax.set_ylim(0.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Calculate metrics
        pred_tensor = torch.tensor([pred_box])
        gt_tensor = torch.tensor([gt_box])
        regular_iou, scaled_iou = calculate_iou_metrics(pred_tensor, gt_tensor, ratio)
        
        # Title with ratio and type
        if ratio < 1.0:
            title = f'Ratio = {ratio:.1f} (Inner)'
        elif ratio == 1.0:
            title = f'Ratio = {ratio:.1f} (Original)'
        else:
            title = f'Ratio = {ratio:.1f} (Outer)'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Draw original boxes
        pred_rect = patches.Rectangle((pred_box[0]-pred_box[2]/2, pred_box[1]-pred_box[3]/2), 
                                     pred_box[2], pred_box[3], 
                                     linewidth=2, edgecolor='blue', 
                                     facecolor='blue', alpha=0.3, label='Prediction')
        ax.add_patch(pred_rect)
        
        gt_rect = patches.Rectangle((gt_box[0]-gt_box[2]/2, gt_box[1]-gt_box[3]/2), 
                                   gt_box[2], gt_box[3], 
                                   linewidth=2, edgecolor='red', 
                                   facecolor='red', alpha=0.3, label='Ground Truth')
        ax.add_patch(gt_rect)
        
        # Draw scaled boxes (dashed lines)
        scaled_pred_w, scaled_pred_h = pred_box[2] * ratio, pred_box[3] * ratio
        scaled_pred_rect = patches.Rectangle((pred_box[0]-scaled_pred_w/2, pred_box[1]-scaled_pred_h/2), 
                                           scaled_pred_w, scaled_pred_h, 
                                           linewidth=2, edgecolor='blue', 
                                           facecolor='none', linestyle='--', alpha=0.8)
        ax.add_patch(scaled_pred_rect)
        
        scaled_gt_w, scaled_gt_h = gt_box[2] * ratio, gt_box[3] * ratio
        scaled_gt_rect = patches.Rectangle((gt_box[0]-scaled_gt_w/2, gt_box[1]-scaled_gt_h/2), 
                                         scaled_gt_w, scaled_gt_h, 
                                         linewidth=2, edgecolor='red', 
                                         facecolor='none', linestyle='--', alpha=0.8)
        ax.add_patch(scaled_gt_rect)
        
        # Add metrics text
        info_text = f"Regular IoU: {regular_iou:.3f}\n"
        if ratio < 1.0:
            info_text += f"Inner-IoU: {scaled_iou:.3f}\n"
        elif ratio == 1.0:
            info_text += f"Same IoU: {scaled_iou:.3f}\n"
        else:
            info_text += f"Outer-IoU: {scaled_iou:.3f}\n"
        info_text += f"Difference: {scaled_iou-regular_iou:+.3f}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/home/guo/own_ultralytics/ratio_effects_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: ratio_effects_analysis.png")


def main():
    """Main analysis function"""
    print("Starting Complete Inner-IoU Analysis...")
    print("This script analyzes:")
    print("1. Inner-IoU vs Regular IoU")
    print("2. Ground Truth size effects")
    print("3. Ratio effects (including ratio > 1.0)")
    print("4. Visualizations")
    print()
    
    # Run all analyses
    analyze_vs_regular_iou()
    analyze_gt_size_effects()
    analyze_ratio_effects()
    create_visualizations()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("Key findings:")
    print("• Inner-IoU (ratio < 1.0): Stricter, focuses on object centers")
    print("• Outer-IoU (ratio > 1.0): More lenient, includes surrounding context")
    print("• ratio = 1.2 is used in research for expanded matching")
    print("• GT size affects sensitivity: larger GT → less sensitive to small offsets")
    print("• Choice of ratio depends on task requirements and training strategy")

if __name__ == "__main__":
    main()
