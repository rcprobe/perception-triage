"""Perception error triage engine (MVP)."""

from .schema import Box3D, FailureRecord
from .matching import match_frame
from .db import init_db, insert_failures, query_failures
from .confusion import build_confusion_matrix

__all__ = [
    "Box3D",
    "FailureRecord",
    "match_frame",
    "init_db",
    "insert_failures",
    "query_failures",
    "build_confusion_matrix",
]
