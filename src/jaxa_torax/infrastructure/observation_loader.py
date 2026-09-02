"""Search, indexing, loading, and export helpers for the PyVista prototype."""

from __future__ import annotations

import random
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


from .catalog_index import *
from .catalog_service import *
from .event_sources import *
from .science_core import *
from .selections import *


def _project_loaded_frame(
    frame: pd.DataFrame,
    target_wcs,
    center_ra: float,
    center_dec: float,
) -> pd.DataFrame:
    local = frame.copy()
    common_x, common_y = project_to_common_plane(
        target_wcs,
        center_ra,
        center_dec,
        local.RA.to_numpy(float),
        local.DEC.to_numpy(float),
    )
    local["COMMON_X_DEG"] = common_x
    local["COMMON_Y_DEG"] = common_y
    return local


def _result_summary(observations: Sequence[LoadedObservation]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mission": item.record.mission,
                "instrument": item.record.instrument,
                "observation_id": item.record.observation_id,
                "object": item.metadata.get("OBJECT", ""),
                "min_sep_deg": item.minimum_separation_deg,
                "events_in_region": item.events_in_region,
                "displayed_events": item.displayed_events,
            }
            for item in observations
        ]
    )


def load_records(
    records: Sequence[EventFile],
    center_ra: float | None = None,
    center_dec: float | None = None,
) -> SearchResult:
    require_astropy()
    records = list(records)
    if not records:
        return SearchResult([], _result_summary([]), [], 0.0, 0.0, None)

    records[0] = ensure_cached(records[0])
    first_meta = read_header(records[0].header_path)
    center_ra = float(center_ra if center_ra is not None else first_meta.get("RA_PNT", first_meta.get("TCRVL_X", 0.0)))
    center_dec = float(center_dec if center_dec is not None else first_meta.get("DEC_PNT", first_meta.get("TCRVL_Y", 0.0)))
    target_wcs = common_tan_wcs(center_ra, center_dec)
    observations: list[LoadedObservation] = []
    skipped: list[str] = []

    for record in records:
        try:
            frame, metadata = read_events_with_sky(record)
            frame = _project_loaded_frame(frame, target_wcs, center_ra, center_dec)
            observations.append(
                LoadedObservation(
                    record=record,
                    frame=frame.reset_index(drop=True),
                    metadata=metadata,
                    total_events=len(frame),
                    events_in_region=len(frame),
                    displayed_events=len(frame),
                    minimum_separation_deg=None,
                )
            )
        except Exception as error:
            skipped.append(f"{record_key(record)}: {error}")

    return SearchResult(
        observations,
        _result_summary(observations),
        skipped,
        center_ra,
        center_dec,
        None,
    )


def search_region(
    records: Sequence[EventFile],
    center_ra: float,
    center_dec: float,
    radius_deg: float,
    selected_pairs: Iterable[str] | None = None,
    use_duckdb: bool = True,
    db_path: Path | str = DB_PATH,
) -> SearchResult:
    require_astropy()
    if not 0.0 < radius_deg <= 180.0:
        raise ValueError("Radius must be in the range (0, 180] deg")
    if not -90.0 <= center_dec <= 90.0:
        raise ValueError("DEC must be between -90 and 90 deg")

    records_map = records_by_key(records)
    selected_pair_set = {str(item).lower() for item in selected_pairs or []}
    candidate_records = list(records)

    if selected_pair_set:
        candidate_records = filter_records_by_pairs(candidate_records, selected_pair_set)

    if use_duckdb and duckdb is not None and duckdb_index_exists(db_path):
        candidate_keys = query_duckdb_candidate_keys(
            center_ra,
            center_dec,
            radius_deg,
            selected_pair_set,
            db_path=db_path,
        )
        candidate_records = [records_map[key] for key in candidate_keys if key in records_map]

    center_ra = center_ra % 360.0
    requested_center = SkyCoord(center_ra * u.deg, center_dec * u.deg, frame="icrs")
    target_wcs = common_tan_wcs(center_ra, center_dec)
    candidates: list[tuple[float, LoadedObservation]] = []
    skipped: list[str] = []

    for record in candidate_records:
        try:
            frame, metadata = read_events_with_sky(record)
            if frame.empty:
                continue
            sky = SkyCoord(
                frame.RA.to_numpy(float) * u.deg,
                frame.DEC.to_numpy(float) * u.deg,
                frame="icrs",
            )
            separation = sky.separation(requested_center).deg
            inside = separation <= radius_deg
            if not np.any(inside):
                continue

            local = frame.loc[inside].copy()
            local = _project_loaded_frame(local, target_wcs, center_ra, center_dec)
            observation = LoadedObservation(
                record=record,
                frame=local.reset_index(drop=True),
                metadata=metadata,
                total_events=len(frame),
                events_in_region=len(local),
                displayed_events=len(local),
                minimum_separation_deg=float(np.min(separation[inside])),
            )
            candidates.append((observation.minimum_separation_deg, observation))
        except Exception as error:
            skipped.append(f"{record_key(record)}: {error}")

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1].record.mission,
            item[1].record.instrument,
            item[1].record.observation_id,
        )
    )
    observations = [item[1] for item in candidates]
    return SearchResult(
        observations,
        _result_summary(observations),
        skipped,
        center_ra,
        center_dec,
        radius_deg,
    )


def random_record(records: Sequence[EventFile], selected_pairs=None) -> EventFile | None:
    candidates = list(records)
    if selected_pairs:
        candidates = filter_records_by_pairs(candidates, selected_pairs)
    if not candidates:
        return None
    return random.choice(candidates)


def export_observations(
    observations: Sequence[LoadedObservation],
    export_dir: Path | str = EXPORT_DIR,
    stem: str = "selected_events",
) -> dict:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if not observations:
        raise ValueError("No loaded events to export")

    combined = pd.concat([item.frame for item in observations], ignore_index=True)
    csv_path = export_dir / f"{stem}.csv"
    parquet_path = export_dir / f"{stem}.parquet"
    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)
    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "rows": len(combined),
    }
