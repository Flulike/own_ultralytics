#!/usr/bin/env python3
"""
Compute detection mAP scores directly from YOLO-format TXT folders.

The script pairs ground-truth and prediction files (matching relative paths) and feeds
them through Ultralytics' DetMetrics implementation, so the reported values match what
`model.val()` would return. Example:

python tools/calc_map_from_txt.py --gt-dir path/to/labels --pred-dir runs/detect/exp/labels
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import yaml

from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import DetMetrics, box_iou


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory with ground-truth txt files.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with prediction txt files.")
    parser.add_argument(
        "--names",
        type=str,
        default=None,
        help="Comma separated list or YAML path providing class names; falls back to class_i.",
    )
    parser.add_argument(
        "--suffix", type=str, default=".txt", help="Label file suffix to collect (default: .txt)."
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=None,
        help="Optional confidence threshold applied to the prediction txt entries.",
    )
    parser.add_argument(
        "--use-scipy",
        action="store_true",
        help="Use the Hungarian matching (requires scipy) identical to validator --use_scipy flag.",
    )
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Print per-class metrics in addition to the global summary.",
    )
    parser.add_argument(
        "--iou-min",
        type=float,
        default=0.5,
        help="Lower bound for IoU range (default: 0.5).",
    )
    parser.add_argument(
        "--iou-max",
        type=float,
        default=0.95,
        help="Upper bound for IoU range (default: 0.95).",
    )
    parser.add_argument(
        "--iou-steps",
        type=int,
        default=10,
        help="Number of IoU thresholds to evaluate between iou-min/max (default: 10).",
    )
    return parser.parse_args()


def discover_label_files(root: Path, suffix: str) -> Dict[str, Path]:
    """Recursively collect txt files under root, keeping their relative paths."""
    if not root.exists():
        return {}
    files = {}
    for file in sorted(root.rglob(f"*{suffix}")):
        if file.name.lower() == "classes.txt":
            continue
        files[file.relative_to(root).as_posix()] = file
    return files


def load_detection_file(
    path: Path | None, is_prediction: bool, conf_thres: float | None
) -> dict[str, np.ndarray]:
    """Load YOLO-format txt file into numpy arrays."""
    boxes, classes, confs = [], [], []
    if path is None:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "cls": np.zeros((0,), dtype=np.int64),
            "conf": np.zeros((0,), dtype=np.float32),
        }
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"{path}: line {line_no} should contain at least 5 values, got: '{raw}'")
            cls = int(float(fields[0]))
            xywh = list(map(float, fields[1:5]))
            conf = float(fields[5]) if is_prediction and len(fields) >= 6 else 1.0
            if conf_thres is not None and conf < conf_thres:
                continue
            boxes.append(xywh)
            classes.append(cls)
            if is_prediction:
                confs.append(conf)
    if not confs and is_prediction:
        confs = [1.0] * len(boxes)
    return {
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if boxes else np.zeros((0, 4), dtype=np.float32),
        "cls": np.asarray(classes, dtype=np.int64),
        "conf": np.asarray(confs, dtype=np.float32),
    }


def match_detections(
    pred_classes: torch.Tensor,
    target_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
    use_scipy: bool = False,
) -> torch.Tensor:
    """Vectorized prediction-to-target matching aligned with BaseValidator.match_predictions."""
    correct = torch.zeros((pred_classes.shape[0], iouv.numel()), dtype=torch.bool, device=pred_classes.device)
    if target_classes.numel() == 0 or pred_classes.numel() == 0:
        return correct
    correct_class = target_classes[:, None] == pred_classes
    iou = iou * correct_class.float()
    iou_np = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        if use_scipy:
            import scipy  # local import

            cost_matrix = iou_np * (iou_np >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
        else:
            matches = np.nonzero(iou_np >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    order = iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]
                    matches = matches[order]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
    return correct


def build_name_map(names_arg: str | None, max_index: int) -> dict[int, str]:
    """Resolve class names from CLI argument or fall back to synthetic placeholders."""
    if max_index < 0:
        raise ValueError("No labels were found in the provided folders.")
    default_map = {i: f"class_{i}" for i in range(max_index + 1)}
    if not names_arg:
        return default_map
    path = Path(names_arg)
    data: Iterable | dict | None = None
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict) and "names" in loaded:
            data = loaded["names"]
        else:
            data = loaded
    else:
        data = [name.strip() for name in names_arg.split(",") if name.strip()]
    if isinstance(data, dict):
        return {i: str(data.get(i, default_map[i])) for i in range(max_index + 1)}
    if isinstance(data, list):
        if len(data) <= max_index:
            raise ValueError(f"Provided names list has {len(data)} entries but needs {max_index + 1}.")
        return {i: str(data[i]) for i in range(max_index + 1)}
    raise ValueError(f"Unsupported names specification: {data}")


def prepare_samples(
    gt_dir: Path,
    pred_dir: Path,
    suffix: str,
    conf_thres: float | None,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], int]:
    """Load paired txt files and return a mapping keyed by relative path."""
    gt_files = discover_label_files(gt_dir, suffix)
    pred_files = discover_label_files(pred_dir, suffix)
    keys = sorted(set(gt_files) | set(pred_files))
    if not keys:
        raise FileNotFoundError("No matching txt files found in either directory.")
    samples, max_class = {}, -1
    for key in keys:
        gt = load_detection_file(gt_files.get(key), is_prediction=False, conf_thres=None)
        pred = load_detection_file(pred_files.get(key), is_prediction=True, conf_thres=conf_thres)
        samples[key] = {"gt": gt, "pred": pred}
        for arr in (gt["cls"], pred["cls"]):
            if arr.size:
                max_class = max(max_class, int(arr.max()))
    return samples, max_class


def compute_stats_for_sample(
    sample: dict[str, dict[str, np.ndarray]], iouv: torch.Tensor, use_scipy: bool
) -> dict[str, np.ndarray]:
    """Convert numpy data into the structures expected by DetMetrics."""
    gt = sample["gt"]
    pred = sample["pred"]
    gt_boxes = torch.from_numpy(gt["boxes"]) if gt["boxes"].size else torch.zeros((0, 4), dtype=torch.float32)
    pred_boxes = torch.from_numpy(pred["boxes"]) if pred["boxes"].size else torch.zeros((0, 4), dtype=torch.float32)
    if gt_boxes.numel():
        gt_boxes = ops.xywh2xyxy(gt_boxes.clone())
    if pred_boxes.numel():
        pred_boxes = ops.xywh2xyxy(pred_boxes.clone())
    target_cls = torch.from_numpy(gt["cls"]) if gt["cls"].size else torch.zeros((0,), dtype=torch.int64)
    pred_cls = torch.from_numpy(pred["cls"]) if pred["cls"].size else torch.zeros((0,), dtype=torch.int64)
    if pred_boxes.numel() and gt_boxes.numel():
        iou = box_iou(gt_boxes, pred_boxes)
        tp = match_detections(pred_cls, target_cls, iou, iouv, use_scipy)
    else:
        tp = torch.zeros((pred_cls.shape[0], iouv.numel()), dtype=torch.bool)
    return {
        "tp": tp.cpu().numpy(),
        "conf": pred["conf"],
        "pred_cls": pred["cls"],
        "target_cls": gt["cls"],
        "target_img": np.unique(gt["cls"]).astype(int),
    }


def main() -> None:
    """Entry point."""
    args = parse_args()
    samples, max_class = prepare_samples(args.gt_dir, args.pred_dir, args.suffix, args.conf_thres)
    total_labels = int(sum(sample["gt"]["cls"].size for sample in samples.values()))
    if total_labels == 0:
        raise ValueError("No ground-truth labels were found; mAP cannot be computed.")
    names = build_name_map(args.names, max_class)
    iouv = torch.linspace(args.iou_min, args.iou_max, args.iou_steps)
    metrics = DetMetrics(names=names)
    for key, sample in samples.items():
        stats = compute_stats_for_sample(sample, iouv, args.use_scipy)
        metrics.update_stats(stats)
    metrics.process()
    LOGGER.info(f"Processed {len(samples)} files: {args.gt_dir} (GT) vs {args.pred_dir} (predictions)")
    for k, v in metrics.results_dict.items():
        LOGGER.info(f"{k}: {v:.4f}")
    if args.per_class:
        for summary in metrics.summary():
            LOGGER.info(summary)


if __name__ == "__main__":
    main()
