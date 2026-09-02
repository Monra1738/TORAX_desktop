"""Search, indexing, loading, and export helpers for the PyVista prototype."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
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
from .images import *
from .previews import *
from .science_core import *


def _compact_catalog_search_text(value: str) -> str:
    """Make catalog name searches insensitive to spaces, underscores, and case."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def catalog_dataframe(records: Sequence[EventFile]) -> pd.DataFrame:
    rows = []
    for record in records:
        try:
            metadata = read_header(record.header_path)
        except Exception:
            metadata = {}
        rows.append(
            {
                "key": record_key(record),
                "label": record_label(record),
                "mission": record.mission,
                "instrument": record.instrument,
                "observation_id": record.observation_id,
                "object": metadata.get("OBJECT", ""),
                "ra_pnt": metadata.get("RA_PNT", metadata.get("TCRVL_X")),
                "dec_pnt": metadata.get("DEC_PNT", metadata.get("TCRVL_Y")),
                "date_obs": metadata.get("DATE-OBS", ""),
                "date_end": metadata.get("DATE-END", ""),
                "parquet_path": str(record.parquet_path),
            }
        )
    return pd.DataFrame(rows)


def mission_catalog_url(mission: str) -> str:
    return f"{TORAX_BASE_URL}/{safe_token(mission)}/catalog.csv"


def fetch_mission_catalog(mission: str) -> pd.DataFrame:
    mission = safe_token(mission)
    text = fetch_text(mission_catalog_url(mission))
    frame = pd.read_csv(
        StringIO(text),
        dtype={
            "obsid": str,
            "instrument": str,
            "OBJECT": str,
            "DATE-OBS": str,
            "DATE-END": str,
        },
        keep_default_na=False,
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"obsid", "instrument", "RA_PNT", "DEC_PNT"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{mission} catalog missing columns: {', '.join(sorted(missing))}")
    frame["mission"] = mission
    frame["obsid"] = frame["obsid"].astype(str).str.strip()
    frame["instrument"] = frame["instrument"].astype(str).str.strip().str.lower()
    frame["OBJECT"] = frame.get("OBJECT", "").astype(str)
    frame["DATE-OBS"] = frame.get("DATE-OBS", "").astype(str)
    frame["DATE-END"] = frame.get("DATE-END", "").astype(str)
    return frame


def _sortable_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")


@serialized_database_access
def build_server_catalog(
    missions: Sequence[str] = TORAX_MISSIONS,
    db_path: Path | str = DB_PATH,
    cache_dir: Path | str = CACHE_DIR,
) -> dict:
    require_duckdb()
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)

    rows = []
    skipped: list[str] = []
    file_id = 0

    for mission in missions:
        try:
            catalog = fetch_mission_catalog(mission)
        except Exception as error:
            skipped.append(f"{mission}: {error}")
            continue

        catalog["date_obs_sort"] = _sortable_datetime(catalog["DATE-OBS"])
        catalog["date_end_sort"] = _sortable_datetime(catalog["DATE-END"])

        for values in catalog.to_dict("records"):
            mission_name = safe_token(values["mission"])
            instrument = safe_token(values["instrument"])
            observation_id = str(values["obsid"]).strip()
            record = remote_event_record(
                mission_name,
                instrument,
                observation_id,
                cache_dir=cache_dir,
            )
            rows.append(
                {
                    "file_id": file_id,
                    "key": record_key(record),
                    "mission": mission_name,
                    "instrument": instrument,
                    "observation_id": observation_id,
                    "object": str(values.get("OBJECT", "")),
                    "ra_pnt": pd.to_numeric(values.get("RA_PNT"), errors="coerce"),
                    "dec_pnt": pd.to_numeric(values.get("DEC_PNT"), errors="coerce"),
                    "date_obs": str(values.get("DATE-OBS", "")),
                    "date_end": str(values.get("DATE-END", "")),
                    "date_obs_sort": str(values.get("date_obs_sort", "")),
                    "date_end_sort": str(values.get("date_end_sort", "")),
                    "tstart": pd.to_numeric(values.get("TSTART"), errors="coerce"),
                    "tstop": pd.to_numeric(values.get("TSTOP"), errors="coerce"),
                    "parquet_url": record.parquet_url,
                    "header_url": record.header_url,
                    "parquet_cache_path": str(record.parquet_path),
                    "header_cache_path": str(record.header_path),
                    "cached": bool(record.parquet_path.exists() and record.header_path.exists()),
                }
            )
            file_id += 1

    server_df = pd.DataFrame(rows)
    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS server_files")
    con.execute(
        """
        CREATE TABLE server_files (
            file_id INTEGER,
            key TEXT,
            mission TEXT,
            instrument TEXT,
            observation_id TEXT,
            object TEXT,
            ra_pnt DOUBLE,
            dec_pnt DOUBLE,
            date_obs TEXT,
            date_end TEXT,
            date_obs_sort TEXT,
            date_end_sort TEXT,
            tstart DOUBLE,
            tstop DOUBLE,
            parquet_url TEXT,
            header_url TEXT,
            parquet_cache_path TEXT,
            header_cache_path TEXT,
            cached BOOLEAN
        )
        """
    )
    if not server_df.empty:
        con.register("server_df", server_df)
        con.execute("INSERT INTO server_files SELECT * FROM server_df")
        con.unregister("server_df")
    con.execute("CREATE INDEX IF NOT EXISTS server_files_key_idx ON server_files(key)")
    con.execute("CREATE INDEX IF NOT EXISTS server_files_pair_idx ON server_files(mission, instrument)")
    con.execute("CREATE INDEX IF NOT EXISTS server_files_obs_idx ON server_files(observation_id)")
    con.close()
    ensure_storage_schema(db_path)

    return {
        "db_path": str(db_path),
        "files": len(server_df),
        "missions": sorted(set(server_df["mission"])) if not server_df.empty else [],
        "skipped": skipped,
    }


@serialized_database_access
def server_catalog_exists(db_path: Path | str = DB_PATH) -> bool:
    if duckdb is None:
        return False
    path = Path(db_path)
    if not path.exists():
        return False
    con = duckdb.connect(str(path))
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if "server_files" not in tables:
            return False
        count = con.execute("SELECT COUNT(*) FROM server_files").fetchone()[0]
        return count > 0
    finally:
        con.close()


@serialized_database_access
def server_catalog_count(db_path: Path | str = DB_PATH) -> int:
    if not server_catalog_exists(db_path):
        return 0
    con = duckdb.connect(str(db_path))
    try:
        return int(con.execute("SELECT COUNT(*) FROM server_files").fetchone()[0])
    finally:
        con.close()


@serialized_database_access
def search_server_catalog(
    object_text: str = "",
    observation_text: str = "",
    selected_pairs: Iterable[str] | None = None,
    center_ra: float | None = None,
    center_dec: float | None = None,
    radius_deg: float | None = None,
    pointing_margin_deg: float = 0.0,
    date_start: str = "",
    date_end: str = "",
    limit: int = DEFAULT_SEARCH_LIMIT,
    db_path: Path | str = DB_PATH,
) -> pd.DataFrame:
    require_duckdb()
    if not server_catalog_exists(db_path):
        return pd.DataFrame()

    clauses = ["1=1"]
    params: list = []
    object_text = str(object_text or "").strip()
    observation_text = str(observation_text or "").strip()
    selected_pair_set = {str(item).lower() for item in selected_pairs or []}

    if object_text:
        compact_object_text = _compact_catalog_search_text(object_text)
        if compact_object_text:
            clauses.append(
                "regexp_replace(lower(object), '[^a-z0-9]+', '', 'g') LIKE ?"
            )
            params.append(f"%{compact_object_text}%")

    if observation_text:
        clauses.append("observation_id LIKE ?")
        params.append(f"%{observation_text}%")

    if selected_pair_set:
        clauses.append(
            "concat(mission, '/', instrument) IN ({})".format(
                ", ".join(["?"] * len(selected_pair_set))
            )
        )
        params.extend(sorted(selected_pair_set))

    if date_start:
        clauses.append("(date_end_sort = '' OR date_end_sort >= ?)")
        params.append(str(date_start))

    if date_end:
        clauses.append("(date_obs_sort = '' OR date_obs_sort <= ?)")
        params.append(str(date_end))

    sql = f"""
        SELECT *
        FROM server_files
        WHERE {' AND '.join(clauses)}
        ORDER BY mission, instrument, observation_id
    """
    con = duckdb.connect(str(db_path))
    try:
        frame = con.execute(sql, params).fetchdf()
    finally:
        con.close()

    if frame.empty:
        return frame

    frame["separation_deg"] = np.nan
    if center_ra is not None and center_dec is not None and radius_deg is not None:
        require_astropy()
        valid = frame.ra_pnt.notna() & frame.dec_pnt.notna()
        if valid.any():
            target = SkyCoord(float(center_ra) % 360.0 * u.deg, float(center_dec) * u.deg)
            coords = SkyCoord(
                frame.loc[valid, "ra_pnt"].astype(float).to_numpy() * u.deg,
                frame.loc[valid, "dec_pnt"].astype(float).to_numpy() * u.deg,
            )
            separations = coords.separation(target).deg
            frame.loc[valid, "separation_deg"] = separations
            frame = frame.loc[
                frame["separation_deg"].le(
                    float(radius_deg) + max(0.0, float(pointing_margin_deg))
                )
            ].copy()
            frame = frame.sort_values(
                ["separation_deg", "mission", "instrument", "observation_id"],
                na_position="last",
            )

    limit = max(1, int(limit or DEFAULT_SEARCH_LIMIT))
    frame = frame.head(limit).copy()
    frame["label"] = (
        frame["mission"].str.upper()
        + " / "
        + frame["instrument"].str.upper()
        + " / "
        + frame["observation_id"].astype(str)
        + " / "
        + frame["object"].astype(str)
    )
    return frame.reset_index(drop=True)


def server_records_from_dataframe(
    frame: pd.DataFrame,
) -> list[EventFile]:
    records: list[EventFile] = []
    if frame.empty:
        return records
    for row in frame.to_dict("records"):
        records.append(
            normalized_cache_record(EventFile(
                mission=str(row["mission"]),
                instrument=str(row["instrument"]),
                observation_id=str(row["observation_id"]),
                parquet_path=Path(row["parquet_cache_path"]),
                header_path=Path(row["header_cache_path"]),
                parquet_url=str(row["parquet_url"]),
                header_url=str(row["header_url"]),
                source="remote",
            ))
        )
    return records


@serialized_database_access
def server_catalog_records(
    mission: str = "",
    search_text: str = "",
    limit: int | None = None,
    missing_headers_only: bool = False,
    db_path: Path | str = DB_PATH,
) -> list[EventFile]:
    if not server_catalog_exists(db_path):
        return []
    ensure_storage_schema(db_path)
    clauses = ["1=1"]
    params: list = []
    mission = safe_token(mission) if mission else ""
    search_text = str(search_text or "").strip().lower()
    if mission:
        clauses.append("sf.mission = ?")
        params.append(mission)
    if search_text:
        clauses.append(
            "(lower(sf.object) LIKE ? OR lower(sf.observation_id) LIKE ? "
            "OR lower(sf.key) LIKE ?)"
        )
        token = f"%{search_text}%"
        params.extend([token, token, token])
    if missing_headers_only:
        clauses.append("sh.key IS NULL")
    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT ?"
        params.append(max(0, int(limit)))
    con = duckdb.connect(str(db_path))
    try:
        frame = con.execute(
            f"""
            SELECT sf.*
            FROM server_files sf
            LEFT JOIN server_headers sh ON sh.key = sf.key
            WHERE {' AND '.join(clauses)}
            ORDER BY sf.mission, sf.instrument, sf.observation_id
            {limit_sql}
            """,
            params,
        ).fetchdf()
    finally:
        con.close()
    return server_records_from_dataframe(frame)


def _fetch_header_payload(record: EventFile) -> tuple[EventFile, dict, str, str]:
    if not record.header_url:
        raise FileNotFoundError(f"No header URL for {record_key(record)}")
    request = urllib.request.Request(record.header_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.loads(response.read().decode("utf-8", errors="replace"))
        return (
            record,
            metadata,
            response.headers.get("ETag", ""),
            response.headers.get("Last-Modified", ""),
        )


def sync_server_headers(
    records: Sequence[EventFile] | None = None,
    workers: int = 8,
    db_path: Path | str = DB_PATH,
    progress=None,
) -> dict:
    records = list(
        records
        if records is not None
        else server_catalog_records(missing_headers_only=True, db_path=db_path)
    )
    completed = 0
    skipped = []
    local = []
    remote = []
    for record in records:
        if cached_server_header(record, db_path=db_path) is not None:
            completed += 1
        elif record.header_path.exists():
            local.append(record)
        else:
            remote.append(record)

    for record in local:
        try:
            store_server_header(record, read_header(record.header_path), db_path=db_path)
            completed += 1
            if progress:
                progress(completed, len(records), record_key(record))
        except Exception as error:
            skipped.append(f"{record_key(record)}: {error}")

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(_fetch_header_payload, record): record for record in remote}
        for future in as_completed(futures):
            record = futures[future]
            try:
                item, metadata, etag, last_modified = future.result()
                store_server_header(
                    item,
                    metadata,
                    etag=etag,
                    last_modified=last_modified,
                    db_path=db_path,
                )
                completed += 1
                if progress:
                    progress(completed, len(records), record_key(item))
            except Exception as error:
                skipped.append(f"{record_key(record)}: {error}")
    return {"requested": len(records), "completed": completed, "skipped": skipped}


def build_preview_batch(
    records: Sequence[EventFile],
    max_rows: int = DEFAULT_PREVIEW_ROWS,
    db_path: Path | str = DB_PATH,
    progress=None,
) -> dict:
    records = list(records)
    completed = 0
    cache_hits = 0
    skipped = []
    for record in records:
        try:
            _frame, _metadata, _total, hit = read_compact_preview(
                record,
                max_rows=max_rows,
                db_path=db_path,
            )
            completed += 1
            cache_hits += int(hit)
            if progress:
                progress(completed, len(records), record_key(record))
        except Exception as error:
            skipped.append(f"{record_key(record)}: {error}")
    return {
        "requested": len(records),
        "completed": completed,
        "cache_hits": cache_hits,
        "skipped": skipped,
        "cache": cache_status(db_path=db_path),
    }
