from __future__ import annotations  # Postpone evaluation of annotations

import sqlite3  # lightweight embedded DB
from typing import Iterable, List, Optional, Tuple, Any  # type hints

from .schema import FailureRecord  # record model


def init_db(db_path: str) -> None:
    # Create the failures table and indexes if they don't exist.
    with sqlite3.connect(db_path) as conn:
        # Schema is intentionally minimal; no uniqueness constraints or run IDs.
        # Note: column name \"class\" is fine for SQLite but can be reserved elsewhere.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                class TEXT NOT NULL,
                distance_m REAL,
                box_height_px REAL,
                num_points INTEGER,
                occlusion INTEGER,
                truncation REAL,
                confidence REAL,
                image_path TEXT,
                iou REAL
            )
            """
        )
        # Indexes for common filters.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_failures_type ON failures(failure_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_failures_class ON failures(class)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_failures_distance ON failures(distance_m)")


def insert_failures(db_path: str, failures: Iterable[FailureRecord]) -> int:
    # Convert to dict rows for executemany insertion.
    rows = [f.to_row() for f in failures]
    if not rows:
        return 0

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO failures (
                frame_id, failure_type, class, distance_m, box_height_px,
                num_points, occlusion, truncation, confidence, image_path, iou
            ) VALUES (
                :frame_id, :failure_type, :class, :distance_m, :box_height_px,
                :num_points, :occlusion, :truncation, :confidence, :image_path, :iou
            )
            """,
            rows,
        )
        return len(rows)


def _build_where(
    failure_type: Optional[str],
    class_name: Optional[str],
    min_distance: Optional[float],
    max_distance: Optional[float],
    max_points: Optional[int],
) -> Tuple[str, List[Any]]:
    # Build a WHERE clause and parameter list safely.
    clauses: List[str] = []
    params: List[Any] = []

    if failure_type:
        clauses.append("failure_type = ?")
        params.append(failure_type)
    if class_name:
        clauses.append("class = ?")
        params.append(class_name)
    if min_distance is not None:
        clauses.append("distance_m >= ?")
        params.append(min_distance)
    if max_distance is not None:
        clauses.append("distance_m <= ?")
        params.append(max_distance)
    if max_points is not None:
        clauses.append("num_points <= ?")
        params.append(max_points)

    if clauses:
        return " WHERE " + " AND ".join(clauses), params
    return "", params


def query_failures(
    db_path: str,
    failure_type: Optional[str] = None,
    class_name: Optional[str] = None,
    min_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
    max_points: Optional[int] = None,
    limit: Optional[int] = 50,
) -> List[sqlite3.Row]:
    # Build SQL query with optional filters.
    where_sql, params = _build_where(
        failure_type=failure_type,
        class_name=class_name,
        min_distance=min_distance,
        max_distance=max_distance,
        max_points=max_points,
    )

    # Use parameterized queries to avoid SQL injection and handle typing.
    sql = (
        "SELECT frame_id, failure_type, class, distance_m, box_height_px, "
        "num_points, occlusion, truncation, confidence, image_path, iou "
        "FROM failures"
    )
    if where_sql:
        sql += where_sql
    sql += " ORDER BY frame_id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return list(conn.execute(sql, params))
