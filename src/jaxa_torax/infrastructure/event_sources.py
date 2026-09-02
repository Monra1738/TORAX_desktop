"""Search, indexing, loading, and export helpers for the PyVista prototype."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
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
from .science_core import *


def known_pair_options() -> list[str]:
    pairs = []
    for mission, instruments in KNOWN_INSTRUMENTS.items():
        for instrument in instruments:
            pairs.append(pair_key(mission, instrument))
    return pairs


def remote_key(
    mission: str,
    instrument: str,
    observation_id: str,
) -> str:
    return pair_key(mission, instrument) + f"/{observation_id}"


def torax_file_url(
    mission: str,
    observation_id: str,
    filename: str,
) -> str:
    return f"{TORAX_BASE_URL}/{mission}/{observation_id}/{filename}"


def remote_cache_paths(
    mission: str,
    instrument: str,
    observation_id: str,
    cache_dir: Path | str = CACHE_DIR,
) -> tuple[Path, Path]:
    cache_root = Path(cache_dir) / mission / observation_id
    stem = f"{observation_id}_{instrument}"
    return (
        cache_root / f"{stem}_events.parquet",
        cache_root / f"{stem}_hdr.json",
    )


def remote_event_record(
    mission: str,
    instrument: str,
    observation_id: str,
    cache_dir: Path | str = CACHE_DIR,
) -> EventFile:
    mission = safe_token(mission)
    instrument = safe_token(instrument)
    observation_id = str(observation_id).strip()
    stem = f"{observation_id}_{instrument}"
    parquet_path, header_path = remote_cache_paths(
        mission,
        instrument,
        observation_id,
        cache_dir,
    )
    return EventFile(
        mission=mission,
        instrument=instrument,
        observation_id=observation_id,
        parquet_path=parquet_path,
        header_path=header_path,
        parquet_url=torax_file_url(mission, observation_id, f"{stem}_events.parquet"),
        header_url=torax_file_url(mission, observation_id, f"{stem}_hdr.json"),
        source="remote",
    )


def normalized_cache_record(
    record: EventFile,
    cache_dir: Path | str = CACHE_DIR,
) -> EventFile:
    """Relocate persisted remote records away from stale application roots."""
    if str(record.source).lower() != "remote":
        return record
    parquet_path, header_path = remote_cache_paths(
        str(record.mission), str(record.instrument), str(record.observation_id), cache_dir
    )
    return EventFile(
        mission=str(record.mission),
        instrument=str(record.instrument),
        observation_id=str(record.observation_id),
        parquet_path=parquet_path,
        header_path=header_path,
        parquet_url=record.parquet_url,
        header_url=record.header_url,
        source="remote",
    )


def discover_event_files(data_dir: Path | str = DATA_DIR) -> list[EventFile]:
    root = Path(data_dir)
    records: list[EventFile] = []
    if not root.exists():
        return records

    for parquet_path in sorted(root.rglob("*_events.parquet")):
        relative = parquet_path.relative_to(root)
        mission = relative.parts[0].lower() if len(relative.parts) > 1 else "unknown"
        instrument, observation_id = instrument_and_obsid(parquet_path)
        header_path = parquet_path.with_name(
            parquet_path.name.replace("_events.parquet", "_hdr.json")
        )
        if header_path.exists():
            records.append(
                EventFile(
                    mission=mission,
                    instrument=instrument,
                    observation_id=observation_id,
                    parquet_path=parquet_path,
                    header_path=header_path,
                )
            )
    return records


def records_by_key(records: Sequence[EventFile]) -> dict[str, EventFile]:
    return {record_key(record): record for record in records}


def filter_records_by_pairs(records: Sequence[EventFile], selected_pairs) -> list[EventFile]:
    if not selected_pairs:
        return []
    selected = {str(item).lower() for item in selected_pairs}
    return [
        record
        for record in records
        if pair_key(record.mission, record.instrument).lower() in selected
    ]


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def download_url(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
        partial.replace(destination)
        return destination
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def ensure_cached(
    record: EventFile,
    db_path: Path | str = DB_PATH,
) -> EventFile:
    record = normalized_cache_record(record)
    cached_header = cached_server_header(record, db_path=db_path)
    if not record.header_path.exists() and cached_header is not None:
        record.header_path.parent.mkdir(parents=True, exist_ok=True)
        partial_header = record.header_path.with_name(record.header_path.name + ".part")
        try:
            partial_header.write_text(
                json.dumps(cached_header, separators=(",", ":"), default=str),
                encoding="utf-8",
            )
            partial_header.replace(record.header_path)
        finally:
            if partial_header.exists():
                partial_header.unlink()
    header_available = (
        record.header_path.exists()
        or cached_header is not None
    )
    if record.parquet_path.exists() and header_available:
        register_cache_entry(
            f"raw:{record_key(record)}",
            record_key(record),
            "raw",
            record.parquet_path,
            db_path=db_path,
        )
        return record

    if not record.parquet_url or not record.header_url:
        missing = [
            str(path)
            for path in (record.parquet_path, record.header_path)
            if not path.exists()
        ]
        raise FileNotFoundError(f"Missing local file(s): {', '.join(missing)}")

    if not header_available:
        download_url(record.header_url, record.header_path)

    if not record.parquet_path.exists():
        download_url(record.parquet_url, record.parquet_path)

    register_cache_entry(
        f"raw:{record_key(record)}",
        record_key(record),
        "raw",
        record.parquet_path,
        db_path=db_path,
    )
    if record.header_path.exists():
        store_server_header(record, read_header(record.header_path), db_path=db_path)

    return record



def native_wcs(metadata: dict):
    require_astropy()
    required = (
        "TCTYP_X",
        "TCTYP_Y",
        "TCRPX_X",
        "TCRPX_Y",
        "TCRVL_X",
        "TCRVL_Y",
        "TCDLT_X",
        "TCDLT_Y",
    )
    missing = [key for key in required if metadata.get(key) is None]
    if missing:
        raise ValueError(f"Missing WCS values: {', '.join(missing)}")

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [metadata["TCTYP_X"], metadata["TCTYP_Y"]]
    wcs.wcs.crpix = [metadata["TCRPX_X"], metadata["TCRPX_Y"]]
    wcs.wcs.crval = [metadata["TCRVL_X"], metadata["TCRVL_Y"]]
    wcs.wcs.cdelt = [metadata["TCDLT_X"], metadata["TCDLT_Y"]]
    wcs.wcs.cunit = [metadata.get("TCUNI_X") or "deg", metadata.get("TCUNI_Y") or "deg"]
    wcs.wcs.set()
    return wcs


def common_tan_wcs(center_ra: float, center_dec: float):
    require_astropy()
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crval = [float(center_ra) % 360.0, float(center_dec)]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.cdelt = [-COMMON_BIN_DEG, COMMON_BIN_DEG]
    wcs.wcs.set()
    return wcs


def project_to_common_plane(
    target_wcs,
    center_ra: float,
    center_dec: float,
    ra: np.ndarray,
    dec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center_x, center_y = target_wcs.all_world2pix([center_ra], [center_dec], 1)
    pixel_x, pixel_y = target_wcs.all_world2pix(ra, dec, 1)
    x_deg = (pixel_x - center_x[0]) * target_wcs.wcs.cdelt[0]
    y_deg = (pixel_y - center_y[0]) * target_wcs.wcs.cdelt[1]
    return x_deg.astype(float), y_deg.astype(float)


def read_events_with_sky(record: EventFile) -> tuple[pd.DataFrame, dict]:
    record = ensure_cached(record)
    metadata = get_record_header(record)
    wcs = native_wcs(metadata)
    frame = pd.read_parquet(record.parquet_path, columns=REQUIRED_COLUMNS)
    frame = frame.dropna(subset=REQUIRED_COLUMNS).copy()
    # FITS event tables commonly use negative PI sentinels for rejected events.
    # They are not physical channels and must never reach PI→keV calibration.
    frame = frame.loc[frame["PI"] >= 0].copy()
    ra, dec = wcs.all_pix2world(
        frame.X.to_numpy(dtype=float),
        frame.Y.to_numpy(dtype=float),
        1,
    )
    frame["RA"] = np.mod(ra, 360.0)
    frame["DEC"] = np.asarray(dec, dtype=float)
    frame["MISSION"] = record.mission
    frame["INSTRUMENT"] = record.instrument
    frame["OBSERVATION_ID"] = record.observation_id
    frame["SOURCE_ROW"] = frame.index.to_numpy(dtype=np.int64)
    return frame, metadata
