"""Perception error triage engine (MVP)."""  # Package docstring

from .schema import Box3D, FailureRecord  # public data models
from .matching import match_frame  # public matcher
from .db import init_db, insert_failures, query_failures  # public DB helpers
from .confusion import build_confusion_matrix  # confusion matrix helper

__all__ = [
    "Box3D",
    "FailureRecord",
    "match_frame",
    "init_db",
    "insert_failures",
    "query_failures",
    "build_confusion_matrix",
]
