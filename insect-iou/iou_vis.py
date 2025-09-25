import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Tuple, List, Optional

# Set English fonts and style for better compatibility
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class IoUVisualizer:
    """
    A comprehensive visualizer for comparing various IoU-based loss functions,
    including standard IoU, GIoU, DIoU, CIoU, and the novel InterpIoU.
    """
    
    def __init__(self):
        self.colors = {
            'pred': '#FF6B6B',      # Prediction box - Red
            'target': '#4ECDC4',    # Target box - Cyan
            'interp': '#45B7D1',    # Interpolated box - Blue
            'enclosing': '#F7B7A3', # Enclosing box for GIoU - Peach (more visible now)
            'guidance': '#556270',  # Center lines for DIoU/CIoU
            'intersection': '#96CEB4' # Intersection - Green
        }

    ## ---------------------------------------------------
    ## Core IoU Calculation Functions
    ## ---------------------------------------------------

    def _calculate_box_properties(self, box1: np.ndarray, box2: np.ndarray) -> dict:
        """A helper to calculate all necessary geometric properties between two boxes."""
        # Intersection
        x1_max, y1_max = np.maximum(box1[:2], box2[:2])
        x2_min, y2_min = np.minimum(box1[2:], box2[2:])
        
        inter_w = np.maximum(0, x2_min - x1_max)
        inter_h = np.maximum(0, y2_min - y1_max)
        intersection_area = inter_w * inter_h

        # Areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - intersection_area
        
        # Enclosing box (C)
        c_x1, c_y1 = np.minimum(box1[:2], box2[:2])
        c_x2, c_y2 = np.maximum(box1[2:], box2[2:])
        enclosing_area = (c_x2 - c_x1) * (c_y2 - c_y1)
        
        # Center points and distance
        center1 = (box1[:2] + box1[2:]) / 2
        center2 = (box2[:2] + box2[2:]) / 2
        center_dist_sq = np.sum((center1 - center2) ** 2)
        
        # Diagonal of enclosing box
        diag_dist_sq = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2
        
        # Aspect ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

        return {
            "iou": intersection_area / (union_area + 1e-7),
            "intersection": intersection_area,
            "union": union_area,
            "enclosing_area": enclosing_area,
            "enclosing_box": np.array([c_x1, c_y1, c_x2, c_y2]),
            "center_dist_sq": center_dist_sq,
            "diag_dist_sq": diag_dist_sq,
            "w1": w1, "h1": h1, "w2": w2, "h2": h2,
        }

    def calculate_giou(self, box1: np.ndarray, box2: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate Generalized IoU (GIoU) and the enclosing box."""
        props = self._calculate_box_properties(box1, box2)
        giou = props['iou'] - (props['enclosing_area'] - props['union']) / (props['enclosing_area'] + 1e-7)
        return giou, props['enclosing_box']

    def calculate_diou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Distance IoU (DIoU)."""
        props = self._calculate_box_properties(box1, box2)
        diou = props['iou'] - props['center_dist_sq'] / (props['diag_dist_sq'] + 1e-7)
        return diou

    def calculate_ciou(self, box1: np.ndarray, box2: np.ndarray) -> Tuple[float, float]:
        """Calculate Complete IoU (CIoU) and the aspect ratio penalty 'v'."""
        props = self._calculate_box_properties(box1, box2)
        diou = props['iou'] - props['center_dist_sq'] / (props['diag_dist_sq'] + 1e-7)
        
        # Aspect ratio penalty
        arctan_w1_h1 = np.arctan(props['w1'] / (props['h1'] + 1e-7))
        arctan_w2_h2 = np.arctan(props['w2'] / (props['h2'] + 1e-7))
        v = (4 / (np.pi**2)) * ((arctan_w2_h2 - arctan_w1_h1)**2)
        
        # Alpha trade-off factor for CIoU, as per paper
        alpha_tradeoff = v / (1 - props['iou'] + v + 1e-7) if props['iou'] > 0 else 1.0 # Handle IoU=0 case for alpha_tradeoff
        ciou = diou - alpha_tradeoff * v
        return ciou, v

    def calculate_interpiou(self, pred_box: np.ndarray, target_box: np.ndarray, gamma: float = 1.0) -> Tuple[float, np.ndarray, float]:
        """Calculate InterpIoU, the interpolated box, and dynamic alpha."""
        props = self._calculate_box_properties(pred_box, target_box)
        standard_iou = props['iou']
        
        # Dynamic alpha = (1 - IoU)^gamma
        alpha_val = (1.0 - standard_iou)**gamma if standard_iou < 1.0 else 0.0 # Prevent (1-1)^gamma=0 in perfect overlap
        
        # Interpolated box: I = alpha * P + (1-alpha) * T (using your formulation)
        interp_box = alpha_val * pred_box + (1 - alpha_val) * target_box
        
        interp_props = self._calculate_box_properties(interp_box, target_box)
        interp_iou = interp_props['iou']
        
        return interp_iou, interp_box, alpha_val
    
    ## ---------------------------------------------------
    ## Visualization Functions
    ## ---------------------------------------------------

    def _draw_box(self, ax, box: np.ndarray, color: str, label: str, 
                  alpha: float = 0.5, linewidth: int = 2, linestyle: str = '-', show_center=True, zorder=1):
        """Helper to draw a bounding box."""
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, 
                         linewidth=linewidth, edgecolor=color, 
                         facecolor=color if alpha > 0.0 else 'none', # If alpha is 0, don't fill
                         alpha=alpha, label=label, linestyle=linestyle, zorder=zorder)
        ax.add_patch(rect)
        if show_center:
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.plot(center_x, center_y, 'o', color=color, markersize=8, markeredgecolor='white', zorder=zorder+1)
    
    def visualize_guidance_comparison(self, pred_box: np.ndarray, target_box: np.ndarray, 
                                      scenario_name: str, save_path: Optional[str] = None):
        """
        Visualizes and compares the guidance mechanisms of GIoU, DIoU, CIoU, and InterpIoU.
        """
        print(f"\n🚀 Visualizing guidance for scenario: {scenario_name}")
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'Comparison of IoU Guidance Mechanisms\nScenario: {scenario_name}', fontsize=20, fontweight='bold')
        
        # Calculate standard IoU for reference
        std_iou_props = self._calculate_box_properties(pred_box, target_box)
        standard_iou = std_iou_props['iou']

        pred_center = (pred_box[:2] + pred_box[2:]) / 2
        target_center = (target_box[:2] + target_box[2:]) / 2

        # --- 1. GIoU Visualization ---
        ax = axes[0, 0]
        giou, enclosing_box = self.calculate_giou(pred_box, target_box)
        self._draw_box(ax, enclosing_box, self.colors['enclosing'], 'Enclosing Box C', alpha=0.3, linewidth=2, linestyle='--', show_center=False, zorder=0) # Make enclosing more visible
        self._draw_box(ax, pred_box, self.colors['pred'], 'Pred Box (P)', alpha=0.7)
        self._draw_box(ax, target_box, self.colors['target'], 'Target Box (T)', alpha=0.7)
        
        # Draw intersection for clarity
        if standard_iou > 1e-7:
            x1_int, y1_int = np.maximum(pred_box[:2], target_box[:2])
            x2_int, y2_int = np.minimum(pred_box[2:], target_box[2:])
            intersection_box = np.array([x1_int, y1_int, x2_int, y2_int])
            self._draw_box(ax, intersection_box, self.colors['intersection'], 'Intersection', alpha=0.5, linewidth=0, show_center=False, zorder=2)

        ax.set_title(f"GIoU Guidance: Enclosing Box (GIoU={giou:.3f})", fontsize=16, fontweight='bold')
        info_text = f"Standard IoU = {standard_iou:.3f}\nGuidance: Minimize C \\ (P U T) area."
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, va='top', bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8))

        # --- 2. DIoU Visualization ---
        ax = axes[0, 1]
        diou = self.calculate_diou(pred_box, target_box)
        self._draw_box(ax, pred_box, self.colors['pred'], 'Pred Box (P)', alpha=0.7)
        self._draw_box(ax, target_box, self.colors['target'], 'Target Box (T)', alpha=0.7)
        if standard_iou > 1e-7:
            self._draw_box(ax, intersection_box, self.colors['intersection'], 'Intersection', alpha=0.5, linewidth=0, show_center=False, zorder=2)
        ax.plot([pred_center[0], target_center[0]], [pred_center[1], target_center[1]],
                color=self.colors['guidance'], linestyle='--', linewidth=2.5, label='Center Distance', zorder=3)
        ax.set_title(f"DIoU Guidance: Center Distance (DIoU={diou:.3f})", fontsize=16, fontweight='bold')
        info_text = f"Standard IoU = {standard_iou:.3f}\nGuidance: Minimize distance between centers."
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, va='top', bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8))

        # --- 3. CIoU Visualization ---
        ax = axes[1, 0]
        ciou, v = self.calculate_ciou(pred_box, target_box)
        self._draw_box(ax, pred_box, self.colors['pred'], 'Pred Box (P)', alpha=0.7)
        self._draw_box(ax, target_box, self.colors['target'], 'Target Box (T)', alpha=0.7)
        if standard_iou > 1e-7:
            self._draw_box(ax, intersection_box, self.colors['intersection'], 'Intersection', alpha=0.5, linewidth=0, show_center=False, zorder=2)
        ax.plot([pred_center[0], target_center[0]], [pred_center[1], target_center[1]],
                color=self.colors['guidance'], linestyle='--', linewidth=2.5, label='Center Distance', zorder=3)
        ax.set_title(f"CIoU Guidance: Distance + Aspect Ratio (CIoU={ciou:.3f})", fontsize=16, fontweight='bold')
        info_text = f"Standard IoU = {standard_iou:.3f}\nGuidance: DIoU + Aspect Ratio (v={v:.3f}) penalty."
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, va='top', bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8))

        # --- 4. InterpIoU Visualization ---
        ax = axes[1, 1]
        interp_iou, interp_box, alpha = self.calculate_interpiou(pred_box, target_box)
        self._draw_box(ax, pred_box, self.colors['pred'], 'Pred Box (P)', alpha=0.7)
        self._draw_box(ax, target_box, self.colors['target'], 'Target Box (T)', alpha=0.7)
        if standard_iou > 1e-7:
            self._draw_box(ax, intersection_box, self.colors['intersection'], 'Intersection', alpha=0.5, linewidth=0, show_center=False, zorder=2)
        self._draw_box(ax, interp_box, self.colors['interp'], 'Interp Box (I)', alpha=0.7, linewidth=3, linestyle='-', zorder=3)
        ax.set_title(f"InterpIoU Guidance: Interpolation Path (InterpIoU={interp_iou:.3f})", fontsize=16, fontweight='bold')
        info_text = f"Standard IoU = {standard_iou:.3f}\nGuidance: IoU(I, T), Dynamic α={alpha:.3f}."
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, va='top', bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8))
        
        # --- Final Touches ---
        for ax_item in axes.flatten(): # Renamed 'ax' to 'ax_item' to avoid conflict with loop variable
            ax_item.set_aspect('equal')
            ax_item.grid(True, linestyle=':', alpha=0.6)
            ax_item.legend(loc='lower right', fontsize=10)
            # Set consistent limits based on all boxes (P, T, C, I)
            all_relevant_boxes = np.vstack([pred_box, target_box, enclosing_box, interp_box])
            x_min, y_min = np.min(all_relevant_boxes, axis=0)[:2]
            x_max, y_max = np.max(all_relevant_boxes, axis=0)[2:]
            x_range, y_range = x_max - x_min, y_max - y_min
            # Add a bit of padding to the limits
            ax_item.set_xlim(x_min - x_range*0.1, x_max + x_range*0.1)
            ax_item.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)


        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Guidance comparison saved to: {save_path}")
        plt.show()

def main():
    """Main function to run the IoU guidance visualization."""
    print("=" * 60)
    print("🎯 IoU Loss Functions Guidance Visualization Tool 🎯")
    print("=" * 60)
    
    visualizer = IoUVisualizer()
    
    # --- SCENARIO DEFINITION (Revised) ---
    # Prediction box and target box with slight overlap and similar aspect ratios.
    pred_box = np.array([2.0, 2.0, 4.5, 4.5])   # Prediction box
    target_box = np.array([3.0, 3.0, 5.0, 5.0]) # Target box
    scenario_name = "Slight Overlap with Similar Aspect Ratios"

    # Run the comparison visualization
    visualizer.visualize_guidance_comparison(
        pred_box, 
        target_box, 
        scenario_name,
        './iou_guidance_comparison_refined.png' # You can change the save path
    )
    
    print("\n" + "="*60)
    print("Visualization complete. Observe how each method provides a unique form of guidance.")
    print("- GIoU pushes the prediction away from the edges of the enclosing box.")
    print("- DIoU & CIoU directly pull the prediction's center towards the target's center.")
    print("- InterpIoU creates a proxy target and calculates IoU against it, keeping the optimization goal consistent.")

if __name__ == "__main__":
    main()