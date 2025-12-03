#!/usr/bin/env python3
"""
Convert binary mask images (black background, white foreground) into YOLO-format txt labels.

Each connected component in the mask becomes one YOLO box entry where:
    class_id x_center y_center width height
Values are normalized by the image width/height. Example usage:

python tools/masks_to_yolo_labels.py --mask-dir path/to/masks --output-dir labels --class-id 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ultralytics.utils import LOGGER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True, help="Folder containing binary mask images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write YOLO txt files.")
    parser.add_argument(
        "--class-id",
        type=int,
        default=1,
        help="Class id to assign to every detection (default: 1). Note YOLO typically expects 0-based ids.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Binarization threshold applied to grayscale masks (0-255).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=10.0,
        help="Ignore contours with pixel area below this value.",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".png,.jpg,.jpeg,.bmp,.tif,.tiff",
        help="Comma-separated list of file extensions to scan for masks.",
    )
    return parser.parse_args()


def discover_masks(root: Path, extensions: list[str]) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Mask directory not found: {root}")
    masks = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in extensions and p.is_file()]
    if not masks:
        raise FileNotFoundError(f"No mask images with extensions {extensions} were found under {root}")
    return masks


def mask_to_boxes(image_path: Path, threshold: int, min_area: float) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    """Return bounding boxes (x, y, w, h) for each connected component plus image shape."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read mask image: {image_path}")
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    return boxes, img.shape


def box_to_yolo_line(box: tuple[int, int, int, int], shape: tuple[int, int], class_id: int) -> str:
    x, y, w, h = box
    img_h, img_w = shape
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def process_mask(image_path: Path, out_dir: Path, args: argparse.Namespace) -> int:
    boxes, shape = mask_to_boxes(image_path, args.threshold, args.min_area)
    lines = [box_to_yolo_line(box, shape, args.class_id) for box in boxes]
    out_path = out_dir / f"{image_path.stem}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
    return len(lines)


def main() -> None:
    args = parse_args()
    exts = [e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}" for e in args.exts.split(",")]
    masks = discover_masks(args.mask_dir, exts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_boxes = 0
    for mask_path in masks:
        n = process_mask(mask_path, args.output_dir, args)
        total_boxes += n
    LOGGER.info(f"Converted {len(masks)} masks → {total_boxes} YOLO boxes saved under {args.output_dir}")


if __name__ == "__main__":
    main()
