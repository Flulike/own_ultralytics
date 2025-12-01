from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


# Load a model
# model = YOLO("runs/detect/ours_112/weights/best.pt")  # load a custom model
model = YOLO("runs/detect/ours/weights/last.pt")  # load a pretrained model (recommended for training)
project = "runs/cctv_results/test/"
name = "baseline"
device = [0]

# Define path to the image file
source = "/mnt/vmlqnap02/home/guo/dataset/carclassyolo/valid/labels/"
gt_labels_dir = "/mnt/vmlqnap02/home/guo/dataset/carclassyolo/valid/labels/"
show_labels = False


def load_gt_labels(label_path: Path):
    """Load YOLO-format labels for a single image."""
    if not label_path.exists():
        return []
    items = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            items.append((cls, cx, cy, w, h))
    return items


def yolo_to_xyxy(label, width: int, height: int):
    """Convert YOLO (cx, cy, w, h) relative coords to absolute xyxy."""
    cls, cx, cy, w, h = label
    cx *= width
    cy *= height
    w *= width
    h *= height
    x1 = max(0.0, cx - w / 2.0)
    y1 = max(0.0, cy - h / 2.0)
    x2 = min(float(width - 1), cx + w / 2.0)
    y2 = min(float(height - 1), cy + h / 2.0)
    return cls, [x1, y1, x2, y2]


def visualize_ground_truth(image_dir: str, label_dir: str, save_dir: Path, class_names, draw_labels: bool) -> None:
    """Render GT annotations with Ultralytics-style overlays."""
    image_dir_path = Path(image_dir)
    label_dir_path = Path(label_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in image_dir_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}])
    for image_path in image_paths:
        label_path = label_dir_path / f"{image_path.stem}.txt"
        gt_items = load_gt_labels(label_path)
        if not gt_items:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        annotator = Annotator(image, line_width=2, example=str(class_names))
        for cls, xyxy in [yolo_to_xyxy(item, width, height) for item in gt_items]:
            label_text = class_names.get(cls, str(cls)) if draw_labels else None
            annotator.box_label(xyxy, label_text, color=colors(cls, bgr=True))
        out_path = save_dir / image_path.name
        cv2.imwrite(str(out_path), annotator.result())


# Run inference on the source
results = model.predict(
    source=source,
    project=project,
    name=name,
    save=True,
    save_txt=True,
    device=device,
    show_labels=show_labels,
    show_conf=False,
)


# Render ground-truth overlays to mirror prediction outputs
# visualize_ground_truth(
#     image_dir=source,
#     label_dir=gt_labels_dir,
#     save_dir=Path(project) / f"{name}_gt",
#     class_names=model.names,
#     draw_labels=show_labels,
# )
