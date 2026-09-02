"""Destructive, user-confirmed reset of generated UDON3 application state."""

from __future__ import annotations

import shutil
from pathlib import Path

from jaxa_udon3.infrastructure.science_core import APP_DATA_ROOT


def clear_application_storage(app_data_root: Path | str = APP_DATA_ROOT) -> list[Path]:
    """Remove generated cache/workspace state while preserving data and exports."""
    root = Path(app_data_root).expanduser().resolve()
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError(f"Refusing unsafe application-data root: {root}")

    removed: list[Path] = []
    for directory in (root / "data_cache", root / "logs"):
        if directory.exists():
            shutil.rmtree(directory)
            removed.append(directory)
    for database in (
        root / "darts_events.duckdb",
        root / "darts_events.duckdb.wal",
        root / "darts_events.duckdb.tmp",
    ):
        if database.exists():
            database.unlink()
            removed.append(database)
    return removed
