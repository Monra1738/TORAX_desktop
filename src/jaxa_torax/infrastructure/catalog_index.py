"""Search, indexing, loading, and export helpers for the PyVista prototype."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:  # The UI can still show a useful install message.
    duckdb = None

try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
except ImportError:
    u = None
    SkyCoord = None
    WCS = None


from .cache_repository import *
from .catalog_service import *
from .event_sources import *
from .images import *
from .previews import *
from .science_core import *


@serialized_database_access
def build_duckdb_index(
    records: Sequence[EventFile] | None = None,
    db_path: Path | str = DB_PATH,
    data_dir: Path | str = DATA_DIR,
    cell_size_deg: float = INDEX_CELL_DEG,
) -> dict:
    require_duckdb()
    require_astropy()
    records = list(records if records is not None else discover_event_files(data_dir))
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS files")
    con.execute("DROP TABLE IF EXISTS sky_cells")
    con.execute(
        """
        CREATE TABLE files (
            file_id INTEGER,
            key TEXT,
            mission TEXT,
            instrument TEXT,
            observation_id TEXT,
            object TEXT,
            parquet_path TEXT,
            header_path TEXT,
            ra_pnt DOUBLE,
            dec_pnt DOUBLE,
            date_obs TEXT,
            date_end TEXT,
            total_events BIGINT,
            ra_min DOUBLE,
            ra_max DOUBLE,
            dec_min DOUBLE,
            dec_max DOUBLE,
            pi_min DOUBLE,
            pi_max DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE sky_cells (
            cell_x INTEGER,
            cell_y INTEGER,
            file_id INTEGER,
            key TEXT,
            mission TEXT,
            instrument TEXT,
            observation_id TEXT,
            event_count BIGINT,
            ra_min DOUBLE,
            ra_max DOUBLE,
            dec_min DOUBLE,
            dec_max DOUBLE,
            pi_min DOUBLE,
            pi_max DOUBLE
        )
        """
    )

    indexed_files = 0
    indexed_cells = 0
    skipped: list[str] = []
    data_dir = Path(data_dir)

    for file_id, record in enumerate(records):
        try:
            frame, metadata = read_events_with_sky(record)
            if frame.empty:
                continue
            key = record_key(record)
            file_df = pd.DataFrame(
                [
                    {
                        "file_id": file_id,
                        "key": key,
                        "mission": record.mission,
                        "instrument": record.instrument,
                        "observation_id": record.observation_id,
                        "object": metadata.get("OBJECT", ""),
                        "parquet_path": str(record.parquet_path.relative_to(data_dir)),
                        "header_path": str(record.header_path.relative_to(data_dir)),
                        "ra_pnt": metadata.get("RA_PNT", metadata.get("TCRVL_X")),
                        "dec_pnt": metadata.get("DEC_PNT", metadata.get("TCRVL_Y")),
                        "date_obs": metadata.get("DATE-OBS", ""),
                        "date_end": metadata.get("DATE-END", ""),
                        "total_events": len(frame),
                        "ra_min": float(frame.RA.min()),
                        "ra_max": float(frame.RA.max()),
                        "dec_min": float(frame.DEC.min()),
                        "dec_max": float(frame.DEC.max()),
                        "pi_min": float(frame.PI.min()),
                        "pi_max": float(frame.PI.max()),
                    }
                ]
            )
            con.register("file_df", file_df)
            con.execute("INSERT INTO files SELECT * FROM file_df")
            con.unregister("file_df")

            cells = pd.DataFrame(
                {
                    "cell_x": np.floor(frame.RA.to_numpy(float) / cell_size_deg).astype(int),
                    "cell_y": np.floor(
                        (frame.DEC.to_numpy(float) + 90.0) / cell_size_deg
                    ).astype(int),
                    "RA": frame.RA.to_numpy(float),
                    "DEC": frame.DEC.to_numpy(float),
                    "PI": frame.PI.to_numpy(float),
                }
            )
            grouped = (
                cells.groupby(["cell_x", "cell_y"], as_index=False)
                .agg(
                    event_count=("PI", "size"),
                    ra_min=("RA", "min"),
                    ra_max=("RA", "max"),
                    dec_min=("DEC", "min"),
                    dec_max=("DEC", "max"),
                    pi_min=("PI", "min"),
                    pi_max=("PI", "max"),
                )
            )
            grouped["file_id"] = file_id
            grouped["key"] = key
            grouped["mission"] = record.mission
            grouped["instrument"] = record.instrument
            grouped["observation_id"] = record.observation_id
            grouped = grouped[
                [
                    "cell_x",
                    "cell_y",
                    "file_id",
                    "key",
                    "mission",
                    "instrument",
                    "observation_id",
                    "event_count",
                    "ra_min",
                    "ra_max",
                    "dec_min",
                    "dec_max",
                    "pi_min",
                    "pi_max",
                ]
            ]
            con.register("cells_df", grouped)
            con.execute("INSERT INTO sky_cells SELECT * FROM cells_df")
            con.unregister("cells_df")
            indexed_files += 1
            indexed_cells += len(grouped)
        except Exception as error:
            skipped.append(f"{record_key(record)}: {error}")

    con.execute("CREATE INDEX IF NOT EXISTS files_key_idx ON files(key)")
    con.execute("CREATE INDEX IF NOT EXISTS sky_cell_idx ON sky_cells(cell_x, cell_y)")
    con.close()
    return {
        "db_path": str(db_path),
        "files": indexed_files,
        "cells": indexed_cells,
        "skipped": skipped,
    }


def duckdb_index_exists(db_path: Path | str = DB_PATH) -> bool:
    return Path(db_path).exists()


@serialized_database_access
def query_duckdb_candidate_keys(
    center_ra: float,
    center_dec: float,
    radius_deg: float,
    selected_pairs: Iterable[str] | None,
    db_path: Path | str = DB_PATH,
    cell_size_deg: float = INDEX_CELL_DEG,
) -> list[str]:
    require_duckdb()
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    selected_pairs = {str(item).lower() for item in selected_pairs or []}
    ra_min = max(0.0, (center_ra % 360.0) - radius_deg)
    ra_max = min(360.0, (center_ra % 360.0) + radius_deg)
    dec_min = max(-90.0, center_dec - radius_deg)
    dec_max = min(90.0, center_dec + radius_deg)
    cell_x_min = int(np.floor(ra_min / cell_size_deg))
    cell_x_max = int(np.floor(ra_max / cell_size_deg))
    cell_y_min = int(np.floor((dec_min + 90.0) / cell_size_deg))
    cell_y_max = int(np.floor((dec_max + 90.0) / cell_size_deg))

    con = duckdb.connect(str(db_path))
    params = [cell_x_min, cell_x_max, cell_y_min, cell_y_max, ra_min, ra_max, dec_min, dec_max]
    pair_clause = ""
    if selected_pairs:
        pair_clause = " AND concat(mission, '/', instrument) IN ({})".format(
            ", ".join(["?"] * len(selected_pairs))
        )
        params.extend(sorted(selected_pairs))

    rows = con.execute(
        f"""
        SELECT DISTINCT key
        FROM sky_cells
        WHERE cell_x BETWEEN ? AND ?
          AND cell_y BETWEEN ? AND ?
          AND ra_max >= ?
          AND ra_min <= ?
          AND dec_max >= ?
          AND dec_min <= ?
          {pair_clause}
        ORDER BY key
        """,
        params,
    ).fetchall()
    con.close()
    return [row[0] for row in rows]
