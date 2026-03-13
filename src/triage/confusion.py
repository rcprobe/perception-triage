from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Sequence

from .matching import FrameMatchResult
from .schema import Box3D


def build_confusion_matrix(
    predictions: List[Box3D],
    ground_truth: List[Box3D],
    match_result: FrameMatchResult,
) -> Dict[str, Dict[str, int]]:
    """Build a GT-vs-Pred confusion matrix from matched pairs.

    Uses the match_result.matches list (TP + LOC by default) and counts
    how often each GT class is paired with each predicted class.
    """
    confusion: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

    for match in match_result.matches:
        gt_class = ground_truth[match.gt_idx].class_name
        pred_class = predictions[match.pred_idx].class_name
        confusion[gt_class][pred_class] += 1

    return {gt: dict(preds) for gt, preds in confusion.items()}


def merge_confusion_matrices(
    matrices: Iterable[Dict[str, Dict[str, int]]],
) -> Dict[str, Dict[str, int]]:
    """Aggregate multiple per-frame confusion matrices into one matrix."""
    merged: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

    for matrix in matrices:
        for gt_class, pred_counts in matrix.items():
            for pred_class, count in pred_counts.items():
                merged[gt_class][pred_class] += count

    return {gt: dict(preds) for gt, preds in merged.items()}


def confusion_matrix_to_table(
    confusion: Dict[str, Dict[str, int]],
    classes: Sequence[str] | None = None,
) -> tuple[List[str], List[List[int]]]:
    """Convert a confusion matrix dict into tabular headers and rows."""
    if classes is None:
        class_names = sorted(
            {
                *confusion.keys(),
                *(pred_class for pred_counts in confusion.values() for pred_class in pred_counts.keys()),
            }
        )
    else:
        class_names = list(classes)

    headers = ["gt\\pred", *class_names]
    rows: List[List[int]] = []

    for gt_class in class_names:
        row = [gt_class]
        pred_counts = confusion.get(gt_class, {})
        for pred_class in class_names:
            row.append(pred_counts.get(pred_class, 0))
        rows.append(row)

    return headers, rows
