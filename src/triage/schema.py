from __future__ import annotations

from dataclasses import dataclass, asdict
from math import sqrt
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class Box3D:
    """Basic 3D bounding box.

    Coordinates follow KITTI-style LiDAR frame by convention, but this is a
    general-purpose container. Yaw is in radians.
    """

    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    yaw: float
    class_name: str
    score: Optional[float] = None
    occlusion: Optional[int] = None
    truncation: Optional[float] = None
    num_points: Optional[int] = None
    box_height_px: Optional[float] = None

    @property
    def distance_m(self) -> float:
        return sqrt(self.x ** 2 + self.y ** 2)

    @property
    def volume(self) -> float:
        return max(self.dx, 0.0) * max(self.dy, 0.0) * max(self.dz, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["class"] = d.pop("class_name")
        return d


@dataclass(frozen=True)
class FailureRecord:
    frame_id: str
    failure_type: str
    class_name: str
    distance_m: float
    box_height_px: Optional[float]
    num_points: Optional[int]
    occlusion: Optional[int]
    truncation: Optional[float]
    confidence: Optional[float]
    image_path: Optional[str]
    iou: Optional[float]

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["class"] = d.pop("class_name")
        return d
