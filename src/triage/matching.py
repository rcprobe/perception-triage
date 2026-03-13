from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .schema import Box3D


@dataclass(frozen=True)
class Match:
    pred_idx: int
    gt_idx: int
    iou: float
    match_type: str


@dataclass(frozen=True)
class FrameMatchResult:
    matches: List[Match]
    false_positives: List[int]
    false_negatives: List[int]


def iou3d_axis_aligned(a: Box3D, b: Box3D) -> float:
    """Axis-aligned 3D IoU (ignores yaw)."""
    ax_min, ax_max = a.x - a.dx / 2.0, a.x + a.dx / 2.0
    ay_min, ay_max = a.y - a.dy / 2.0, a.y + a.dy / 2.0
    az_min, az_max = a.z - a.dz / 2.0, a.z + a.dz / 2.0

    bx_min, bx_max = b.x - b.dx / 2.0, b.x + b.dx / 2.0
    by_min, by_max = b.y - b.dy / 2.0, b.y + b.dy / 2.0
    bz_min, bz_max = b.z - b.dz / 2.0, b.z + b.dz / 2.0

    inter_x = max(0.0, min(ax_max, bx_max) - max(ax_min, bx_min))
    inter_y = max(0.0, min(ay_max, by_max) - max(ay_min, by_min))
    inter_z = max(0.0, min(az_max, bz_max) - max(az_min, bz_min))
    inter = inter_x * inter_y * inter_z

    if inter <= 0.0:
        return 0.0

    vol_a = max(a.dx, 0.0) * max(a.dy, 0.0) * max(a.dz, 0.0)
    vol_b = max(b.dx, 0.0) * max(b.dy, 0.0) * max(b.dz, 0.0)
    union = vol_a + vol_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def match_frame(
    predictions: List[Box3D],
    ground_truth: List[Box3D],
    iou_threshold: float = 0.5,
    loc_threshold: Optional[float] = 0.1,
    iou_fn: Callable[[Box3D, Box3D], float] = iou3d_axis_aligned,
    class_aware: bool = True,
) -> FrameMatchResult:
    if loc_threshold is not None and loc_threshold > iou_threshold:
        raise ValueError("loc_threshold must be <= iou_threshold")

    matches: List[Match] = []
    false_positives: List[int] = []
    gt_matched = [False] * len(ground_truth)

    for pred_idx, pred in enumerate(predictions):
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(ground_truth):
            if class_aware and pred.class_name != gt.class_name:
                continue
            iou = iou_fn(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            matches.append(Match(pred_idx=pred_idx, gt_idx=best_gt_idx, iou=best_iou, match_type="TP"))
            gt_matched[best_gt_idx] = True
            continue

        if best_gt_idx is not None and loc_threshold is not None and best_iou >= loc_threshold:
            matches.append(Match(pred_idx=pred_idx, gt_idx=best_gt_idx, iou=best_iou, match_type="LOC"))
            gt_matched[best_gt_idx] = True
            continue

        false_positives.append(pred_idx)

    false_negatives = [gt_idx for gt_idx, matched in enumerate(gt_matched) if not matched]

    return FrameMatchResult(
        matches=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
