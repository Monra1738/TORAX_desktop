"""Search, indexing, loading, and export helpers for the PyVista prototype."""

from __future__ import annotations

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
from .event_sources import *
from .science_core import *
from .selections import *


def _energy_stratified_sample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    max_rows = max(1, int(max_rows))
    if len(frame) <= max_rows:
        return frame.copy()
    bin_count = max(1, min(32, int(frame["PI"].nunique()), max_rows))
    labels = pd.qcut(frame["PI"], q=bin_count, labels=False, duplicates="drop")
    groups = frame.assign(_ENERGY_GROUP=labels).groupby(
        "_ENERGY_GROUP",
        sort=True,
        observed=True,
        dropna=False,
    )
    group_count = max(1, groups.ngroups)
    per_group = max(1, max_rows // group_count)
    sampled_parts = [
        group.sample(min(len(group), per_group), random_state=42)
        for _label, group in groups
    ]
    sampled = pd.concat(sampled_parts).drop(columns="_ENERGY_GROUP")
    if len(sampled) < max_rows:
        remaining = frame.drop(index=sampled.index)
        if not remaining.empty:
            sampled = pd.concat(
                [
                    sampled,
                    remaining.sample(
                        min(len(remaining), max_rows - len(sampled)),
                        random_state=43,
                    ),
                ]
            )
    return sampled.sort_index().head(max_rows)


def preview_product_key(
    record: EventFile,
    max_rows: int = DEFAULT_PREVIEW_ROWS,
    metadata: dict | None = None,
) -> str:
    signature = compact_hash(
        {
            "version": PREVIEW_VERSION,
            "calibration": CALIBRATION_VERSION,
            "rows": int(max_rows),
            "factor": pi_to_kev_factor(record),
            "wcs": wcs_metadata_signature(metadata or {}),
        }
    )
    return f"preview:{record_key(record)}:{signature}"


def read_preview_source(
    record: EventFile,
    max_rows: int = DEFAULT_PREVIEW_ROWS,
    db_path: Path | str = DB_PATH,
) -> tuple[pd.DataFrame, int]:
    """Read a deterministic PI-stratified sample without retaining remote raw data."""
    max_rows = max(1, int(max_rows))
    if record.parquet_path.exists():
        frame = pd.read_parquet(record.parquet_path, columns=REQUIRED_COLUMNS)
        frame = frame.dropna(subset=REQUIRED_COLUMNS).copy()
        frame = frame.loc[frame["PI"] >= 0].copy()
        total_events = len(frame)
        frame["SOURCE_ROW"] = frame.index.to_numpy(dtype=np.uint32)
        return _energy_stratified_sample(frame, max_rows), total_events

    if record.parquet_url and duckdb is not None:
        con = duckdb.connect()
        try:
            frame = con.execute(
                """
                WITH source AS (
                    SELECT
                        row_number() OVER () - 1 AS SOURCE_ROW,
                        TIME, PI, X, Y,
                        count(*) OVER () AS TOTAL_EVENTS
                    FROM read_parquet(?)
                    WHERE TIME IS NOT NULL AND PI IS NOT NULL AND PI >= 0
                      AND X IS NOT NULL AND Y IS NOT NULL
                ), energy_groups AS (
                    SELECT *, ntile(32) OVER (ORDER BY PI, SOURCE_ROW) AS energy_group
                    FROM source
                ), ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY energy_group ORDER BY hash(SOURCE_ROW)
                    ) AS sample_rank
                    FROM energy_groups
                )
                SELECT SOURCE_ROW, TIME, PI, X, Y, TOTAL_EVENTS
                FROM ranked
                WHERE sample_rank <= ceil(? / 32.0)
                ORDER BY SOURCE_ROW
                LIMIT ?
                """,
                [record.parquet_url, max_rows, max_rows],
            ).fetchdf()
            if not frame.empty:
                total_events = int(frame.pop("TOTAL_EVENTS").iloc[0])
                return frame, total_events
        except Exception:
            # A local fallback keeps the app usable when httpfs is unavailable.
            pass
        finally:
            con.close()

    cached = ensure_cached(record, db_path=db_path)
    frame = pd.read_parquet(cached.parquet_path, columns=REQUIRED_COLUMNS)
    frame = frame.dropna(subset=REQUIRED_COLUMNS).copy()
    frame = frame.loc[frame["PI"] >= 0].copy()
    total_events = len(frame)
    frame["SOURCE_ROW"] = frame.index.to_numpy(dtype=np.uint32)
    return _energy_stratified_sample(frame, max_rows), total_events


def build_compact_preview(
    record: EventFile,
    max_rows: int = DEFAULT_PREVIEW_ROWS,
    db_path: Path | str = DB_PATH,
) -> tuple[pd.DataFrame, dict, int]:
    metadata = get_record_header(record, db_path=db_path)
    frame, total_events = read_preview_source(
        record,
        max_rows=max_rows,
        db_path=db_path,
    )
    wcs = native_wcs(metadata)
    ra, dec = wcs.all_pix2world(
        frame["X"].to_numpy(dtype=float),
        frame["Y"].to_numpy(dtype=float),
        1,
    )
    frame["RA"] = np.mod(ra, 360.0).astype(np.float32)
    frame["DEC"] = np.asarray(dec, dtype=np.float32)
    frame["KEV"] = (
        frame["PI"].to_numpy(dtype=float) * pi_to_kev_factor(record)
    ).astype(np.float32)
    frame = frame.astype(
        {
            "TIME": "float64",
            "PI": "int32",
            "X": "int16",
            "Y": "int16",
            "SOURCE_ROW": "uint32",
            "RA": "float32",
            "DEC": "float32",
            "KEV": "float32",
        }
    )
    product_key = preview_product_key(record, max_rows, metadata=metadata)
    signature = product_key.rsplit(":", 1)[-1]
    path = cache_product_path(record, "preview", ".parquet", token=signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        frame.to_parquet(partial, index=False, compression="zstd")
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()
    register_cache_entry(
        product_key,
        record_key(record),
        "preview",
        path,
        parameters_hash=signature,
        metadata={
            "total_events": total_events,
            "preview_events": len(frame),
            "preview_version": PREVIEW_VERSION,
            "calibration_version": CALIBRATION_VERSION,
        },
        db_path=db_path,
    )
    enforce_cache_limit(db_path=db_path)
    return frame.reset_index(drop=True), metadata, total_events


def read_compact_preview(
    record: EventFile,
    max_rows: int = DEFAULT_PREVIEW_ROWS,
    db_path: Path | str = DB_PATH,
) -> tuple[pd.DataFrame, dict, int, bool]:
    metadata = get_record_header(record, db_path=db_path)
    key = preview_product_key(record, max_rows, metadata=metadata)
    entry = cache_entry(key, db_path=db_path)
    if entry is None:
        frame, metadata, total = build_compact_preview(record, max_rows, db_path=db_path)
        return frame, metadata, total, False
    frame = pd.read_parquet(entry["path"])
    total = int(entry["metadata"].get("total_events", len(frame)))
    return frame, metadata, total, True


def _region_pixel_bounds(wcs, selection: SkySelection) -> tuple[float, float, float, float]:
    world_ra, world_dec = selection_world_points(selection)
    x_pixel, y_pixel = wcs.all_world2pix(world_ra, world_dec, 1)
    valid = np.isfinite(x_pixel) & np.isfinite(y_pixel)
    if not np.any(valid):
        raise ValueError("Requested sky region does not project into this observation WCS")
    padding = 2.0
    return (
        float(np.min(x_pixel[valid]) - padding),
        float(np.max(x_pixel[valid]) + padding),
        float(np.min(y_pixel[valid]) - padding),
        float(np.max(y_pixel[valid]) + padding),
    )


def _read_region_candidates(
    record: EventFile,
    bounds: tuple[float, float, float, float],
    db_path: Path | str,
) -> pd.DataFrame:
    x_min, x_max, y_min, y_max = bounds
    if record.parquet_path.exists():
        frame = pd.read_parquet(record.parquet_path, columns=REQUIRED_COLUMNS)
        frame["SOURCE_ROW"] = frame.index.to_numpy(dtype=np.uint32)
        return frame.loc[
            frame["X"].between(x_min, x_max)
            & frame["Y"].between(y_min, y_max)
        ].copy()
    if record.parquet_url and duckdb is not None:
        con = duckdb.connect()
        try:
            return con.execute(
                """
                WITH source AS (
                    SELECT row_number() OVER () - 1 AS SOURCE_ROW, TIME, PI, X, Y
                    FROM read_parquet(?)
                )
                SELECT * FROM source
                WHERE X BETWEEN ? AND ? AND Y BETWEEN ? AND ?
                  AND TIME IS NOT NULL AND PI IS NOT NULL
                  AND X IS NOT NULL AND Y IS NOT NULL
                """,
                [record.parquet_url, x_min, x_max, y_min, y_max],
            ).fetchdf()
        except Exception:
            pass
        finally:
            con.close()
    cached = ensure_cached(record, db_path=db_path)
    frame = pd.read_parquet(cached.parquet_path, columns=REQUIRED_COLUMNS)
    frame["SOURCE_ROW"] = frame.index.to_numpy(dtype=np.uint32)
    return frame.loc[
        frame["X"].between(x_min, x_max)
        & frame["Y"].between(y_min, y_max)
    ].copy()


def read_region_preview(
    record: EventFile,
    region: SkySelection,
    max_rows: int = DEFAULT_PREVIEW_ROWS,
    db_path: Path | str = DB_PATH,
) -> tuple[pd.DataFrame, dict, int, bool]:
    """Read and cache a deterministic event preview inside a spatial selection."""
    max_rows = max(1, int(max_rows))
    metadata = get_record_header(record, db_path=db_path)
    signature_data = {
        "version": PREVIEW_VERSION,
        "calibration": CALIBRATION_VERSION,
        "record": record_key(record),
        "rows": max_rows,
        "region": region.signature(),
        "factor": pi_to_kev_factor(record),
        "wcs": wcs_metadata_signature(metadata),
    }
    signature = compact_hash(signature_data)
    product_key = f"region_preview:{record_key(record)}:{signature}"
    entry = cache_entry(product_key, db_path=db_path)
    if entry is not None:
        frame = pd.read_parquet(entry["path"])
        total = int(entry["metadata"].get("events_in_region", len(frame)))
        return frame, metadata, total, True

    # A display-budget change must not force calibrated source re-reading when
    # an otherwise identical region preview already exists. Reuse the nearest
    # cached row allocation and downsample it deterministically when needed.
    compatible = []
    for candidate in cache_entries_for_observation(
        record_key(record), "region_preview", db_path=db_path
    ):
        saved = candidate["metadata"]
        if all(saved.get(name) == signature_data.get(name) for name in (
            "version", "calibration", "record", "region", "factor", "wcs"
        )):
            compatible.append(candidate)
    if compatible:
        compatible.sort(key=lambda item: (
            int(item["metadata"].get("rows", 0)) < max_rows,
            abs(int(item["metadata"].get("rows", 0)) - max_rows),
        ))
        chosen = compatible[0]
        frame = pd.read_parquet(chosen["path"])
        if len(frame) > max_rows:
            indexes = np.linspace(0, len(frame) - 1, max_rows, dtype=np.int64)
            frame = frame.iloc[indexes].reset_index(drop=True)
        total = int(chosen["metadata"].get("events_in_region", len(frame)))
        return frame, metadata, total, True

    wcs = native_wcs(metadata)
    candidates = _read_region_candidates(
        record, _region_pixel_bounds(wcs, region), db_path=db_path
    )
    candidates = candidates.dropna(subset=REQUIRED_COLUMNS).copy()
    if not candidates.empty:
        ra, dec = wcs.all_pix2world(
            candidates["X"].to_numpy(dtype=float),
            candidates["Y"].to_numpy(dtype=float),
            1,
        )
        valid = (
            np.isfinite(ra)
            & np.isfinite(dec)
            & selection_contains(region, ra, dec)
        )
        candidates = candidates.loc[valid].copy()
        candidates["RA"] = np.mod(np.asarray(ra)[valid], 360.0)
        candidates["DEC"] = np.asarray(dec)[valid]
    events_in_region = len(candidates)
    frame = _energy_stratified_sample(candidates, max_rows).copy()
    if "RA" not in frame:
        frame["RA"] = pd.Series(dtype="float32")
        frame["DEC"] = pd.Series(dtype="float32")
    frame["KEV"] = frame["PI"].to_numpy(dtype=float) * pi_to_kev_factor(record)
    dtype_map = {
        "TIME": "float64", "PI": "int32", "X": "int16", "Y": "int16",
        "SOURCE_ROW": "uint32", "RA": "float32", "DEC": "float32", "KEV": "float32",
    }
    frame = frame.astype({key: value for key, value in dtype_map.items() if key in frame})
    path = cache_product_path(record, "region_preview", ".parquet", token=signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        frame.to_parquet(partial, index=False, compression="zstd")
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()
    register_cache_entry(
        product_key,
        record_key(record),
        "region_preview",
        path,
        parameters_hash=signature,
        metadata={**signature_data, "events_in_region": events_in_region},
        db_path=db_path,
    )
    enforce_cache_limit(db_path=db_path)
    return frame.reset_index(drop=True), metadata, events_in_region, False

