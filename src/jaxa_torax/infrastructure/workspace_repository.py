"""Durable desktop-workspace snapshots stored alongside the local DARTS cache."""

from __future__ import annotations

import json
from pathlib import Path

from .science_core import (
    DATABASE_ACCESS_LOCK,
    DB_PATH,
    STORAGE_SCHEMA_LOCK,
    duckdb,
    require_duckdb,
    serialized_database_access,
    utc_now_text,
)

WORKSPACE_SCHEMA_VERSION = 1
ACTIVE_WORKSPACE_KEY = "desktop.active_workspace"
_WORKSPACE_SCHEMA_READY: dict[Path, tuple[int, int]] = {}


def _json(value) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def _ensure_workspace_schema_unlocked(db_path: Path | str = DB_PATH) -> None:
    """Create additive, versioned tables without disturbing catalog/cache tables."""
    require_duckdb()
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS desktop_workspaces (
                workspace_id TEXT PRIMARY KEY,
                target_name TEXT NOT NULL,
                region_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS desktop_workspace_observations (
                workspace_id TEXT NOT NULL,
                record_key TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                visible BOOLEAN NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY (workspace_id, record_key)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS desktop_workspace_slices (
                workspace_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                slice_json TEXT NOT NULL,
                PRIMARY KEY (workspace_id, ordinal)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS desktop_workspace_rois (
                workspace_id TEXT NOT NULL,
                roi_name TEXT NOT NULL,
                roi_json TEXT NOT NULL,
                PRIMARY KEY (workspace_id, roi_name)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS desktop_workspace_preferences (
                preference_key TEXT PRIMARY KEY,
                preference_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
    finally:
        con.close()


def ensure_workspace_schema(db_path: Path | str = DB_PATH) -> None:
    """Serialize additive workspace DDL with the cache schema initializer."""
    with DATABASE_ACCESS_LOCK, STORAGE_SCHEMA_LOCK:
        path = Path(db_path)
        resolved = path.resolve()
        if path.exists():
            stat = path.stat()
            if _WORKSPACE_SCHEMA_READY.get(resolved) == (stat.st_dev, stat.st_ino):
                return
        _ensure_workspace_schema_unlocked(db_path)
        stat = path.stat()
        _WORKSPACE_SCHEMA_READY[resolved] = (stat.st_dev, stat.st_ino)


@serialized_database_access
def save_workspace(snapshot: dict, db_path: Path | str = DB_PATH) -> str:
    """Atomically store one workspace snapshot and make it the active workspace."""
    ensure_workspace_schema(db_path)
    workspace_id = str(snapshot["workspace_id"])
    now = utc_now_text()
    observations = list(snapshot.get("observations", []))
    slices = list(snapshot.get("slices", []))
    rois = dict(snapshot.get("rois", {}))
    con = duckdb.connect(str(db_path))
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(
            """
            INSERT INTO desktop_workspaces AS workspace
                (workspace_id, target_name, region_json, state_json, schema_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (workspace_id) DO UPDATE SET
                target_name = excluded.target_name,
                region_json = excluded.region_json,
                state_json = excluded.state_json,
                schema_version = excluded.schema_version,
                updated_at = excluded.updated_at
            """,
            [
                workspace_id,
                str(snapshot.get("target_name") or "Sky region"),
                _json(snapshot["region"]),
                _json(snapshot.get("state", {})),
                int(snapshot.get("schema_version", WORKSPACE_SCHEMA_VERSION)),
                now,
                now,
            ],
        )
        for table in (
            "desktop_workspace_observations",
            "desktop_workspace_slices",
            "desktop_workspace_rois",
        ):
            con.execute(f"DELETE FROM {table} WHERE workspace_id = ?", [workspace_id])
        for ordinal, item in enumerate(observations):
            con.execute(
                """
                INSERT INTO desktop_workspace_observations
                    (workspace_id, record_key, ordinal, visible, record_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    workspace_id,
                    str(item["record_key"]),
                    ordinal,
                    bool(item.get("visible", True)),
                    _json(item["record"]),
                ],
            )
        for ordinal, item in enumerate(slices):
            con.execute(
                "INSERT INTO desktop_workspace_slices VALUES (?, ?, ?)",
                [workspace_id, ordinal, _json(item)],
            )
        for name, value in rois.items():
            if value is not None:
                con.execute(
                    "INSERT INTO desktop_workspace_rois VALUES (?, ?, ?)",
                    [workspace_id, str(name), _json(value)],
                )
        con.execute(
            """
            INSERT INTO desktop_workspace_preferences AS preference
                (preference_key, preference_value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT (preference_key) DO UPDATE SET
                preference_value = excluded.preference_value,
                updated_at = excluded.updated_at
            """,
            [ACTIVE_WORKSPACE_KEY, workspace_id, now],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return workspace_id


@serialized_database_access
def load_workspace(workspace_id: str, db_path: Path | str = DB_PATH) -> dict | None:
    ensure_workspace_schema(db_path)
    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            """
            SELECT target_name, region_json, state_json, schema_version, updated_at
            FROM desktop_workspaces WHERE workspace_id = ?
            """,
            [str(workspace_id)],
        ).fetchone()
        if row is None:
            return None
        observations = con.execute(
            """
            SELECT record_key, visible, record_json
            FROM desktop_workspace_observations WHERE workspace_id = ? ORDER BY ordinal
            """,
            [str(workspace_id)],
        ).fetchall()
        slices = con.execute(
            """
            SELECT slice_json FROM desktop_workspace_slices
            WHERE workspace_id = ? ORDER BY ordinal
            """,
            [str(workspace_id)],
        ).fetchall()
        rois = con.execute(
            "SELECT roi_name, roi_json FROM desktop_workspace_rois WHERE workspace_id = ?",
            [str(workspace_id)],
        ).fetchall()
    finally:
        con.close()
    return {
        "workspace_id": str(workspace_id),
        "target_name": str(row[0]),
        "region": json.loads(row[1]),
        "state": json.loads(row[2]),
        "schema_version": int(row[3]),
        "updated_at": str(row[4]),
        "observations": [
            {"record_key": str(key), "visible": bool(visible), "record": json.loads(record)}
            for key, visible, record in observations
        ],
        "slices": [json.loads(item[0]) for item in slices],
        "rois": {str(name): json.loads(value) for name, value in rois},
    }


@serialized_database_access
def load_active_workspace(db_path: Path | str = DB_PATH) -> dict | None:
    ensure_workspace_schema(db_path)
    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            """
            SELECT preference_value FROM desktop_workspace_preferences
            WHERE preference_key = ? LIMIT 1
            """,
            [ACTIVE_WORKSPACE_KEY],
        ).fetchone()
    finally:
        con.close()
    return load_workspace(str(row[0]), db_path) if row else None


@serialized_database_access
def list_workspaces(db_path: Path | str = DB_PATH) -> list[dict]:
    ensure_workspace_schema(db_path)
    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            """
            SELECT workspace_id, target_name, region_json, updated_at
            FROM desktop_workspaces ORDER BY updated_at DESC
            """
        ).fetchall()
    finally:
        con.close()
    return [
        {
            "workspace_id": str(workspace_id),
            "target_name": str(target_name),
            "region": json.loads(region_json),
            "updated_at": str(updated_at),
        }
        for workspace_id, target_name, region_json, updated_at in rows
    ]
