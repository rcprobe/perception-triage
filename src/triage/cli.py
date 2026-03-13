from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional

from tabulate import tabulate

from .confusion import build_confusion_matrix, confusion_matrix_to_table, merge_confusion_matrices
from .db import init_db, insert_failures, query_failures
from .failures import build_failure_records
from .matching import match_frame
from .schema import Box3D


def _box_from_dict(d: Dict[str, Any]) -> Box3D:
    return Box3D(
        x=float(d["x"]),
        y=float(d["y"]),
        z=float(d.get("z", 0.0)),
        dx=float(d["dx"]),
        dy=float(d["dy"]),
        dz=float(d["dz"]),
        yaw=float(d.get("yaw", 0.0)),
        class_name=str(d.get("class", d.get("class_name"))),
        score=float(d["score"]) if d.get("score") is not None else None,
        occlusion=int(d["occlusion"]) if d.get("occlusion") is not None else None,
        truncation=float(d["truncation"]) if d.get("truncation") is not None else None,
        num_points=int(d["num_points"]) if d.get("num_points") is not None else None,
        box_height_px=float(d["box_height_px"]) if d.get("box_height_px") is not None else None,
    )


def _load_frames(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames: Dict[str, Dict[str, Any]] = {}
    for item in data:
        frame_id = str(item["frame_id"])
        frames[frame_id] = item
    return frames


def _build_db(
    predictions_path: str,
    ground_truth_path: str,
    db_path: str,
    iou_threshold: float,
    loc_threshold: Optional[float],
) -> None:
    init_db(db_path)

    preds = _load_frames(predictions_path)
    gts = _load_frames(ground_truth_path)

    total_failures = 0

    for frame_id, gt_item in gts.items():
        pred_item = preds.get(frame_id)
        if pred_item is None:
            pred_list: List[Box3D] = []
            image_path = gt_item.get("image_path")
        else:
            pred_list = [_box_from_dict(p) for p in pred_item.get("predictions", [])]
            image_path = pred_item.get("image_path", gt_item.get("image_path"))

        gt_list = [_box_from_dict(g) for g in gt_item.get("ground_truth", [])]

        match_result = match_frame(
            predictions=pred_list,
            ground_truth=gt_list,
            iou_threshold=iou_threshold,
            loc_threshold=loc_threshold,
        )

        failures = build_failure_records(
            frame_id=frame_id,
            image_path=image_path,
            predictions=pred_list,
            ground_truth=gt_list,
            match_result=match_result,
        )

        total_failures += insert_failures(db_path, failures)

    print(f"Inserted {total_failures} failure records into {db_path}")


def _query_db(
    db_path: str,
    failure_type: Optional[str],
    class_name: Optional[str],
    min_distance: Optional[float],
    max_distance: Optional[float],
    max_points: Optional[int],
    limit: int,
) -> None:
    rows = query_failures(
        db_path=db_path,
        failure_type=failure_type,
        class_name=class_name,
        min_distance=min_distance,
        max_distance=max_distance,
        max_points=max_points,
        limit=limit,
    )
    if not rows:
        print("No failures found.")
        return

    table = [
        [
            r["frame_id"],
            r["failure_type"],
            r["class"],
            r["distance_m"],
            r["box_height_px"],
            r["num_points"],
            r["confidence"],
            r["iou"],
            r["image_path"],
        ]
        for r in rows
    ]

    print(
        tabulate(
            table,
            headers=[
                "frame_id",
                "type",
                "class",
                "distance_m",
                "box_height_px",
                "num_points",
                "confidence",
                "iou",
                "image_path",
            ],
        )
    )


def _build_confusion(
    predictions_path: str,
    ground_truth_path: str,
    iou_threshold: float,
    loc_threshold: Optional[float],
    class_aware: bool,
    as_json: bool,
) -> None:
    preds = _load_frames(predictions_path)
    gts = _load_frames(ground_truth_path)
    matrices = []

    for frame_id, gt_item in gts.items():
        pred_item = preds.get(frame_id)
        pred_list = []
        if pred_item is not None:
            pred_list = [_box_from_dict(p) for p in pred_item.get("predictions", [])]

        gt_list = [_box_from_dict(g) for g in gt_item.get("ground_truth", [])]
        match_result = match_frame(
            predictions=pred_list,
            ground_truth=gt_list,
            iou_threshold=iou_threshold,
            loc_threshold=loc_threshold,
            class_aware=class_aware,
        )
        matrices.append(build_confusion_matrix(pred_list, gt_list, match_result))

    confusion = merge_confusion_matrices(matrices)

    if as_json:
        print(json.dumps(confusion, indent=2, sort_keys=True))
        return

    headers, rows = confusion_matrix_to_table(confusion)
    if not rows:
        print("No matched boxes found for confusion matrix.")
        return
    print(tabulate(rows, headers=headers))


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Perception failure triage CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build failures DB from JSON inputs")
    build.add_argument("--predictions", required=True)
    build.add_argument("--ground-truth", required=True)
    build.add_argument("--db", required=True)
    build.add_argument("--iou-threshold", type=float, default=0.5)
    build.add_argument("--loc-threshold", type=float, default=0.1)

    query = sub.add_parser("query", help="Query failures DB")
    query.add_argument("--db", required=True)
    query.add_argument("--type", dest="failure_type", default=None)
    query.add_argument("--class", dest="class_name", default=None)
    query.add_argument("--min-distance", type=float, default=None)
    query.add_argument("--max-distance", type=float, default=None)
    query.add_argument("--max-points", type=int, default=None)
    query.add_argument("--limit", type=int, default=50)

    confusion = sub.add_parser("confusion", help="Build a confusion matrix from JSON inputs")
    confusion.add_argument("--predictions", required=True)
    confusion.add_argument("--ground-truth", required=True)
    confusion.add_argument("--iou-threshold", type=float, default=0.5)
    confusion.add_argument("--loc-threshold", type=float, default=0.1)
    confusion.add_argument(
        "--class-aware",
        action="store_true",
        help="Only match predictions to GT of the same class. Off by default so misclassifications appear in the matrix.",
    )
    confusion.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        _build_db(
            predictions_path=args.predictions,
            ground_truth_path=args.ground_truth,
            db_path=args.db,
            iou_threshold=args.iou_threshold,
            loc_threshold=args.loc_threshold,
        )
        return

    if args.cmd == "query":
        _query_db(
            db_path=args.db,
            failure_type=args.failure_type,
            class_name=args.class_name,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            max_points=args.max_points,
            limit=args.limit,
        )
        return

    if args.cmd == "confusion":
        _build_confusion(
            predictions_path=args.predictions,
            ground_truth_path=args.ground_truth,
            iou_threshold=args.iou_threshold,
            loc_threshold=args.loc_threshold,
            class_aware=args.class_aware,
            as_json=args.as_json,
        )
        return


if __name__ == "__main__":
    main()
