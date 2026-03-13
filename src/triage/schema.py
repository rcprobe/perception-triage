from __future__ import annotations  # Postpone evaluation of type annotations (forward refs, less runtime cost)

from dataclasses import dataclass, asdict  # dataclass for boilerplate, asdict for serialization
from math import sqrt  # used for planar distance computation
from typing import Optional, Dict, Any  # type hints for optional fields and dicts


@dataclass(frozen=True)  # Immutable boxes: safer, but updates require creating new objects
class Box3D:
    """Basic 3D bounding box.

    Coordinates follow KITTI-style LiDAR frame by convention, but this is a
    general-purpose container. Yaw is in radians.
    """

    x: float  # center x (meters)
    y: float  # center y (meters)
    z: float  # center z (meters)
    dx: float  # size along x (meters)
    dy: float  # size along y (meters)
    dz: float  # size along z (meters)
    yaw: float  # rotation around z (radians)
    class_name: str  # class label (string); renamed to \"class\" in JSON/DB
    score: Optional[float] = None  # confidence for predictions; None for GT
    occlusion: Optional[int] = None  # KITTI-style occlusion (0/1/2)
    truncation: Optional[float] = None  # KITTI truncation ratio
    num_points: Optional[int] = None  # LiDAR points inside box
    box_height_px: Optional[float] = None  # projected box height in image (if available)

    @property
    def distance_m(self) -> float:
        # Uses planar distance (x,y). This is typical for ground-plane analysis,
        # but ignores z and thus is not true Euclidean range.
        return sqrt(self.x ** 2 + self.y ** 2)

    @property
    def volume(self) -> float:
        # Clamp dimensions at 0 to avoid negative volumes from bad inputs.
        return max(self.dx, 0.0) * max(self.dy, 0.0) * max(self.dz, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        # Convert to a JSON-friendly dict, renaming class_name -> class.
        d = asdict(self)
        d["class"] = d.pop("class_name")
        return d


@dataclass(frozen=True)  # Immutable failure records for safe DB insertion
class FailureRecord:
    frame_id: str  # frame identifier (e.g., "000042")
    failure_type: str  # "FP", "FN", or "LOC"
    class_name: str  # object class
    distance_m: float  # distance from sensor (planar)
    box_height_px: Optional[float]  # apparent image size
    num_points: Optional[int]  # LiDAR points inside box
    occlusion: Optional[int]  # KITTI occlusion label
    truncation: Optional[float]  # KITTI truncation label
    confidence: Optional[float]  # prediction score (None for FN)
    image_path: Optional[str]  # path to image for visual triage
    iou: Optional[float]  # IoU for LOC errors (None for FP/FN)

    def to_row(self) -> Dict[str, Any]:
        # Convert to a dict suitable for SQLite insertion.
        d = asdict(self)
        d["class"] = d.pop("class_name")
        return d
