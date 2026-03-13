from __future__ import annotations  # Postpone evaluation of annotations

from typing import List, Optional  # type hints

from .matching import FrameMatchResult  # match outcome per frame
from .schema import Box3D, FailureRecord  # core data types


def build_failure_records(
    frame_id: str,
    image_path: Optional[str],
    predictions: List[Box3D],
    ground_truth: List[Box3D],
    match_result: FrameMatchResult,
) -> List[FailureRecord]:
    # Collect all failure records for a single frame.
    failures: List[FailureRecord] = []

    # False positives: predictions with no valid match.
    # Tradeoff: we store metadata only, not the full predicted box geometry.
    for pred_idx in match_result.false_positives:
        pred = predictions[pred_idx]
        failures.append(
            FailureRecord(
                frame_id=frame_id,
                failure_type="FP",
                class_name=pred.class_name,
                distance_m=pred.distance_m,
                box_height_px=pred.box_height_px,
                num_points=pred.num_points,
                occlusion=pred.occlusion,
                truncation=pred.truncation,
                confidence=pred.score,
                image_path=image_path,
                iou=None,
            )
        )

    # False negatives: GT boxes with no match.
    for gt_idx in match_result.false_negatives:
        gt = ground_truth[gt_idx]
        failures.append(
            FailureRecord(
                frame_id=frame_id,
                failure_type="FN",
                class_name=gt.class_name,
                distance_m=gt.distance_m,
                box_height_px=gt.box_height_px,
                num_points=gt.num_points,
                occlusion=gt.occlusion,
                truncation=gt.truncation,
                confidence=None,
                image_path=image_path,
                iou=None,
            )
        )

    # Localization errors: matched, but below TP threshold.
    # We attach GT metadata plus pred confidence and IoU.
    for match in match_result.matches:
        if match.match_type != "LOC":
            continue
        pred = predictions[match.pred_idx]
        gt = ground_truth[match.gt_idx]
        failures.append(
            FailureRecord(
                frame_id=frame_id,
                failure_type="LOC",
                class_name=gt.class_name,
                distance_m=gt.distance_m,
                box_height_px=gt.box_height_px,
                num_points=gt.num_points,
                occlusion=gt.occlusion,
                truncation=gt.truncation,
                confidence=pred.score,
                image_path=image_path,
                iou=match.iou,
            )
        )

    return failures
