#!/usr/bin/env python3
"""Generate a small synthetic dataset for quick pipeline validation."""

import json
import random

random.seed(42)

CLASSES = ["Car", "Pedestrian", "Cyclist"]


def random_box():
    return {
        "x": random.uniform(-20, 20),
        "y": random.uniform(-20, 20),
        "z": 0.0,
        "dx": 1.5,
        "dy": 3.5,
        "dz": 1.5,
        "yaw": 0.0,
    }


def main() -> None:
    frames_gt = []
    frames_pred = []

    for i in range(5):
        frame_id = f"{i:06d}"

        gt_boxes = []
        pred_boxes = []

        # Create ground truth boxes
        for _ in range(3):
            box = random_box()
            box["class"] = random.choice(CLASSES)
            gt_boxes.append(box)

            # Create prediction near GT (TP / LOC)
            pred = box.copy()
            pred["x"] += random.uniform(-0.5, 0.5)
            pred["y"] += random.uniform(-0.5, 0.5)
            pred["score"] = random.uniform(0.7, 0.95)

            # Sometimes misclassify
            if random.random() < 0.2:
                pred["class"] = random.choice(CLASSES)
            else:
                pred["class"] = box["class"]

            pred_boxes.append(pred)

        # Add a false positive
        fp = random_box()
        fp["class"] = random.choice(CLASSES)
        fp["score"] = 0.6
        pred_boxes.append(fp)

        frames_gt.append({
            "frame_id": frame_id,
            "image_path": f"frame_{frame_id}.png",
            "ground_truth": gt_boxes,
        })

        frames_pred.append({
            "frame_id": frame_id,
            "image_path": f"frame_{frame_id}.png",
            "predictions": pred_boxes,
        })

    with open("gt.json", "w", encoding="utf-8") as f:
        json.dump(frames_gt, f, indent=2)

    with open("preds.json", "w", encoding="utf-8") as f:
        json.dump(frames_pred, f, indent=2)

    print("Wrote gt.json and preds.json")


if __name__ == "__main__":
    main()
