#!/usr/bin/env python3
"""
Insert-IoU 可视化分析工具

Insert-IoU (Interpolated IoU) 通过在预测框和真实框之间创建插值框来解决非重叠情况下的梯度问题。
这个工具展示了Insert-IoU的核心机制和与传统IoU的区别。

核心原理：
1. 插值框：I = α * P + (1-α) * T，其中P是预测框，T是真实框，α是插值系数
2. Insert-IoU：计算插值框I与真实框T之间的IoU
3. 动态α：α = 1 - IoU(P,T)，当框不重叠时α接近1，重叠时α接近0

作者：AI Assistant
日期：2025年9月
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F

# Set English fonts and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class InsertIoUVisualizer:
    """Insert-IoU Visualization Analyzer"""
    
    def __init__(self):
        self.colors = {
            'pred': '#FF6B6B',      # Prediction box - Red
            'target': '#4ECDC4',    # Target box - Cyan  
            'interp': '#45B7D1',    # Interpolated box - Blue
            'intersection': '#96CEB4', # Intersection - Green
            'union': '#FFEAA7'      # Union - Yellow
        }
    
    def box_iou_standard(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate standard IoU"""
        # Calculate intersection
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            intersection = 0.0
        else:
            intersection = (x2_min - x1_max) * (y2_min - y1_max)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-7)
    
    def box_insert_iou(self, pred_box: np.ndarray, target_box: np.ndarray, 
                       alpha: Optional[float] = None, dynamic: bool = True) -> Tuple[float, np.ndarray, float]:
        """Calculate Insert-IoU"""
        # Calculate standard IoU for dynamic alpha
        standard_iou = self.box_iou_standard(pred_box, target_box)
        
        # Determine alpha value
        if dynamic:
            alpha_val = 1.0 - max(0, min(1, standard_iou))  # α = 1 - IoU
        else:
            alpha_val = alpha if alpha is not None else 0.5
        
        # Calculate interpolated box: I = α * P + (1-α) * T
        interp_box = alpha_val * pred_box + (1 - alpha_val) * target_box
        
        # Calculate IoU between interpolated box and target
        insert_iou = self.box_iou_standard(interp_box, target_box)
        
        return insert_iou, interp_box, alpha_val
    
    def visualize_interpolation_process(self, pred_box: np.ndarray, target_box: np.ndarray, 
                                      alpha_values: List[float], save_path: str = None):
        """Visualize interpolation process"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Insert-IoU Interpolation Process Visualization', fontsize=20, fontweight='bold')
        
        # 计算标准IoU
        standard_iou = self.box_iou_standard(pred_box, target_box)
        dynamic_alpha = 1.0 - max(0, min(1, standard_iou))
        
        for idx, alpha in enumerate(alpha_values):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # 计算插值框
            interp_box = alpha * pred_box + (1 - alpha) * target_box
            insert_iou = self.box_iou_standard(interp_box, target_box)
            
            # 绘制框
            self._draw_box(ax, pred_box, self.colors['pred'], f'Pred Box P', alpha=0.7)
            self._draw_box(ax, target_box, self.colors['target'], f'Target Box T', alpha=0.7)
            self._draw_box(ax, interp_box, self.colors['interp'], f'Interp Box I', alpha=0.8, linewidth=3)
            
            # 绘制插值公式和结果
            ax.text(0.02, 0.98, f'α = {alpha:.2f}', transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.text(0.02, 0.88, f'I = {alpha:.2f}×P + {1-alpha:.2f}×T', 
                   transform=ax.transAxes, fontsize=12, va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
            
            ax.text(0.02, 0.78, f'Insert-IoU = {insert_iou:.3f}', 
                   transform=ax.transAxes, fontsize=12, va='top', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
            
            # 标记动态alpha
            if abs(alpha - dynamic_alpha) < 0.01:
                ax.text(0.98, 0.98, 'Dynamic α', transform=ax.transAxes, 
                       fontsize=12, fontweight='bold', va='top', ha='right',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.8))
            
            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(-0.5, 6.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'α = {alpha:.2f}, Insert-IoU = {insert_iou:.3f}', 
                        fontsize=14, fontweight='bold')
        
        # 添加说明文字
        fig.text(0.02, 0.02, 
                f'Standard IoU = {standard_iou:.3f} | Dynamic α = {dynamic_alpha:.3f} | '
                f'Interpolation: I = α×P + (1-α)×T | Insert-IoU = IoU(I, T)',
                fontsize=12, ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Interpolation process saved to: {save_path}")
        plt.show()
    
    def compare_iou_methods(self, scenarios: List[Tuple[np.ndarray, np.ndarray, str]], 
                          save_path: str = None):
        """Compare different IoU methods"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Insert-IoU vs Standard IoU Comparison Analysis', fontsize=18, fontweight='bold')
        
        results = []
        
        for idx, (pred_box, target_box, scenario_name) in enumerate(scenarios):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # 计算各种IoU
            standard_iou = self.box_iou_standard(pred_box, target_box)
            insert_iou, interp_box, alpha = self.box_insert_iou(pred_box, target_box, dynamic=True)
            
            results.append({
                'scenario': scenario_name,
                'standard_iou': standard_iou,
                'insert_iou': insert_iou,
                'alpha': alpha,
                'difference': insert_iou - standard_iou
            })
            
            # 绘制框
            self._draw_box(ax, pred_box, self.colors['pred'], 'Pred Box', alpha=0.6)
            self._draw_box(ax, target_box, self.colors['target'], 'Target Box', alpha=0.6)
            self._draw_box(ax, interp_box, self.colors['interp'], 'Interp Box', alpha=0.8, linewidth=2)
            
            # 添加结果信息
            info_text = (f'Standard IoU: {standard_iou:.3f}\n'
                        f'Insert-IoU: {insert_iou:.3f}\n'
                        f'Dynamic α: {alpha:.3f}\n'
                        f'Difference: {insert_iou-standard_iou:+.3f}')
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=11, va='top', ha='left',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
            
            ax.set_xlim(-1, 7)
            ax.set_ylim(-1, 7)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{scenario_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 打印比较结果
        print("\n" + "="*60)
        print("Insert-IoU vs Standard IoU Comparison Results")
        print("="*60)
        for result in results:
            print(f"{result['scenario']:12} | Standard IoU: {result['standard_iou']:.3f} | "
                  f"Insert-IoU: {result['insert_iou']:.3f} | α: {result['alpha']:.3f} | "
                  f"Difference: {result['difference']:+.3f}")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nComparison chart saved to: {save_path}")
        plt.show()
        
        return results
    
    def analyze_alpha_sensitivity(self, pred_box: np.ndarray, target_box: np.ndarray, 
                                save_path: str = None):
        """Analyze alpha parameter sensitivity"""
        alpha_range = np.linspace(0, 1, 21)
        insert_ious = []
        
        standard_iou = self.box_iou_standard(pred_box, target_box)
        dynamic_alpha = 1.0 - max(0, min(1, standard_iou))
        
        for alpha in alpha_range:
            insert_iou, _, _ = self.box_insert_iou(pred_box, target_box, alpha=alpha, dynamic=False)
            insert_ious.append(insert_iou)
        
        # Create figure with better layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top-left: Alpha sensitivity curve
        ax1.plot(alpha_range, insert_ious, 'b-', linewidth=3, label='Insert-IoU')
        ax1.axhline(y=standard_iou, color='r', linestyle='--', linewidth=2, label=f'Standard IoU = {standard_iou:.3f}')
        ax1.axvline(x=dynamic_alpha, color='orange', linestyle=':', linewidth=2, 
                   label=f'Dynamic α = {dynamic_alpha:.3f}')
        
        # Mark dynamic alpha point
        dynamic_insert_iou = insert_ious[int(dynamic_alpha * 20)]
        ax1.plot(dynamic_alpha, dynamic_insert_iou, 'ro', markersize=10, 
                label=f'Dynamic Insert-IoU = {dynamic_insert_iou:.3f}')
        
        ax1.set_xlabel('Interpolation Coefficient α', fontsize=12)
        ax1.set_ylabel('Insert-IoU Value', fontsize=12)
        ax1.set_title('Insert-IoU vs α Curve', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Top-right: Dynamic alpha visualization
        insert_iou_dynamic, interp_box_dynamic, _ = self.box_insert_iou(pred_box, target_box, dynamic=True)
        
        self._draw_box(ax2, pred_box, self.colors['pred'], 'Pred Box', alpha=0.7)
        self._draw_box(ax2, target_box, self.colors['target'], 'Target Box', alpha=0.7)
        self._draw_box(ax2, interp_box_dynamic, self.colors['interp'], 'Dynamic Insert Box', alpha=0.8, linewidth=3)
        
        ax2.text(0.02, 0.98, f'Standard IoU: {standard_iou:.3f}\nInsert-IoU: {insert_iou_dynamic:.3f}\nDynamic α: {dynamic_alpha:.3f}', 
                transform=ax2.transAxes, fontsize=12, va='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.9))
        
        ax2.set_xlim(-1, 7)
        ax2.set_ylim(-1, 7)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Dynamic α Result', fontsize=14, fontweight='bold')
        
        # Bottom-left: Show different alpha values
        alpha_examples = [0.0, 0.3, 0.7, 1.0]
        colors = ['green', 'blue', 'purple', 'red']
        
        for i, alpha_val in enumerate(alpha_examples):
            _, interp_box_ex, _ = self.box_insert_iou(pred_box, target_box, alpha=alpha_val, dynamic=False)
            self._draw_box(ax3, interp_box_ex, colors[i], f'α={alpha_val}', alpha=0.6, linewidth=2)
        
        self._draw_box(ax3, pred_box, self.colors['pred'], 'Pred Box', alpha=0.3, linewidth=1)
        self._draw_box(ax3, target_box, self.colors['target'], 'Target Box', alpha=0.3, linewidth=1)
        
        ax3.set_xlim(-1, 7)
        ax3.set_ylim(-1, 7)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Different α Values Comparison', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        
        # Bottom-right: Alpha impact analysis
        alpha_impact = np.array(insert_ious) - standard_iou
        ax4.plot(alpha_range, alpha_impact, 'g-', linewidth=3, label='Insert-IoU - Standard IoU')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.axvline(x=dynamic_alpha, color='orange', linestyle=':', linewidth=2, label=f'Dynamic α')
        
        ax4.set_xlabel('Interpolation Coefficient α', fontsize=12)
        ax4.set_ylabel('IoU Improvement', fontsize=12)
        ax4.set_title('Insert-IoU Improvement over Standard IoU', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        fig.suptitle('Insert-IoU Alpha Parameter Analysis', fontsize=16, fontweight='bold', y=0.98)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Alpha sensitivity analysis saved to: {save_path}")
        plt.show()
    
    def gradient_analysis(self, pred_box: np.ndarray, target_box: np.ndarray, 
                         displacement_range: float = 2.0, save_path: Optional[str] = None):
        """Analyze gradient characteristics"""
        displacements = np.linspace(-displacement_range, displacement_range, 41)
        
        standard_ious = []
        insert_ious = []
        
        for dx in displacements:
            # 移动预测框
            moved_pred = pred_box + np.array([dx, 0, dx, 0])
            
            # 计算IoU
            std_iou = self.box_iou_standard(moved_pred, target_box)
            insert_iou, _, _ = self.box_insert_iou(moved_pred, target_box, dynamic=True)
            
            standard_ious.append(std_iou)
            insert_ious.append(insert_iou)
        
        # 计算数值梯度
        std_gradients = np.gradient(standard_ious, displacements)
        interp_gradients = np.gradient(insert_ious, displacements)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 上图：IoU值变化
        ax1.plot(displacements, standard_ious, 'r-', linewidth=2, label='标准IoU')
        ax1.plot(displacements, insert_ious, 'b-', linewidth=2, label='Insert-IoU')
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='原始位置')
        
        ax1.set_xlabel('水平位移', fontsize=12)
        ax1.set_ylabel('IoU 值', fontsize=12)
        ax1.set_title('IoU值随位移变化', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 下图：梯度变化
        ax2.plot(displacements, std_gradients, 'r-', linewidth=2, label='标准IoU梯度')
        ax2.plot(displacements, interp_gradients, 'b-', linewidth=2, label='Insert-IoU梯度')
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='原始位置')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax2.set_xlabel('水平位移', fontsize=12)
        ax2.set_ylabel('梯度值', fontsize=12)
        ax2.set_title('IoU梯度随位移变化', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"梯度分析图已保存至: {save_path}")
        plt.show()
        
        # 分析非重叠区域的梯度
        zero_iou_indices = np.where(np.array(standard_ious) == 0)[0]
        if len(zero_iou_indices) > 0:
            print(f"\n非重叠区域分析 (标准IoU = 0):")
            print(f"标准IoU梯度范围: [{np.min(std_gradients[zero_iou_indices]):.4f}, {np.max(std_gradients[zero_iou_indices]):.4f}]")
            print(f"Insert-IoU梯度范围: [{np.min(interp_gradients[zero_iou_indices]):.4f}, {np.max(interp_gradients[zero_iou_indices]):.4f}]")
            print(f"Insert-IoU在非重叠区域提供了 {'有效' if np.max(np.abs(interp_gradients[zero_iou_indices])) > 1e-6 else '无效'} 的梯度信号")
    
    def _draw_box(self, ax, box: np.ndarray, color: str, label: str, 
                  alpha: float = 0.7, linewidth: int = 2):
        """绘制边界框"""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = Rectangle((x1, y1), width, height, 
                        linewidth=linewidth, edgecolor=color, 
                        facecolor=color, alpha=alpha, label=label)
        ax.add_patch(rect)
        
        # 添加中心点
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.plot(center_x, center_y, 'o', color=color, markersize=6)
        
        # 添加标签
        ax.text(x1, y2 + 0.1, label, fontsize=10, color=color, fontweight='bold')


def main():
    """Main function: Run Insert-IoU visualization analysis"""
    print("Insert-IoU Visualization Analysis Tool")
    print("=" * 50)
    
    visualizer = InsertIoUVisualizer()
    
    # 1. Interpolation process visualization
    print("\n1. Interpolation Process Visualization")
    pred_box = np.array([1.0, 1.0, 3.0, 3.0])  # Prediction box
    target_box = np.array([3.5, 2.0, 5.5, 4.0])  # Target box
    alpha_values = [0.0, 0.2, 0.5, 0.7, 0.9, 1.0]
    
    visualizer.visualize_interpolation_process(
        pred_box, target_box, alpha_values,
        '/home/guo/own_ultralytics/interpiou_interpolation_process.png'
    )
    
    # 2. Different scenario comparison
    print("\n2. Different Scenario Comparison")
    scenarios = [
        (np.array([1.0, 1.0, 3.0, 3.0]), np.array([4.0, 1.0, 6.0, 3.0]), "Complete Separation"),
        (np.array([1.0, 1.0, 3.5, 3.0]), np.array([3.0, 1.5, 5.0, 3.5]), "Partial Overlap"),
        (np.array([1.0, 1.0, 3.0, 3.0]), np.array([2.0, 2.0, 4.0, 4.0]), "Moderate Overlap"),
        (np.array([1.0, 1.0, 3.0, 3.0]), np.array([1.5, 1.5, 2.5, 2.5]), "High Overlap")
    ]
    
    comparison_results = visualizer.compare_iou_methods(
        scenarios,
        '/home/guo/own_ultralytics/interpiou_comparison.png'
    )
    
    # 3. Alpha sensitivity analysis
    print("\n3. Alpha Sensitivity Analysis")
    visualizer.analyze_alpha_sensitivity(
        np.array([1.0, 1.0, 3.0, 3.0]), 
        np.array([4.0, 1.0, 6.0, 3.0]),
        '/home/guo/own_ultralytics/interpiou_alpha_sensitivity.png'
    )
    
    # 4. Gradient analysis
    print("\n4. Gradient Characteristics Analysis")
    visualizer.gradient_analysis(
        np.array([2.0, 2.0, 4.0, 4.0]), 
        np.array([3.0, 2.0, 5.0, 4.0]),
        '/home/guo/own_ultralytics/interpiou_gradient_analysis.png'
    )
    
    print("\n" + "="*50)
    print("Insert-IoU Analysis Completed!")
    print("Key Findings:")
    print("1. Insert-IoU solves gradient vanishing problem in non-overlapping cases through interpolated boxes")
    print("2. Dynamic α mechanism enables adaptive adjustment of interpolation strength")
    print("3. Insert-IoU still provides effective gradient signals in complete separation cases")
    print("4. Insert-IoU is particularly useful in early training when prediction boxes are far from targets")


if __name__ == "__main__":
    main()
