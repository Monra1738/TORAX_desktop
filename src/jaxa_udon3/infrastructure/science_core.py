"""Search, indexing, loading, and export helpers for the PyVista prototype."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from functools import lru_cache, wraps
from pathlib import Path
from threading import RLock

from jaxa_udon3.domain import (  # noqa: F401
    EventFile,
    LoadedObservation,
    RGBBandConfig,
    SearchResult,
    SkyRectangle,
    SkyRegion,
    SmoothingConfig,
)

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


APP_DATA_ROOT = Path(
    os.environ.get("UDON3_DATA_ROOT", Path.cwd() / "var")
).expanduser().resolve()
DATA_DIR = APP_DATA_ROOT / "data"
CACHE_DIR = APP_DATA_ROOT / "data_cache"
PRODUCT_CACHE_DIR = CACHE_DIR / "_products"
DB_PATH = APP_DATA_ROOT / "darts_events.duckdb"
EXPORT_DIR = APP_DATA_ROOT / "exports"
REQUIRED_COLUMNS = ["TIME", "PI", "X", "Y"]
COMMON_BIN_DEG = 1.0 / 3600.0
INDEX_CELL_DEG = 0.05
RESOLVE_PI_CHANNELS_PER_KEV = 2_000.0
UDON3_BASE_URL = "https://data.darts.isas.jaxa.jp/pub/app-data/udon3"
UDON3_MISSIONS = ("asca", "suzaku", "hitomi", "xrism")
KNOWN_INSTRUMENTS = {
    "asca": ("gis", "sis"),
    "suzaku": ("xis",),
    "hitomi": ("sxs", "sxi", "hxi"),
    "xrism": ("resolve", "xtend"),
}
DEFAULT_SEARCH_LIMIT = 200
USER_AGENT = "Mozilla/5.0 DARTS-UDON3-PyVista/1.0"
DEFAULT_CACHE_LIMIT_BYTES = 5 * 1024**3
DEFAULT_PREVIEW_ROWS = 5_000
PREVIEW_VERSION = 1
CALIBRATION_VERSION = 3
STORAGE_SCHEMA_LOCK = RLock()
DATABASE_ACCESS_LOCK = RLock()
_STORAGE_SCHEMA_READY: dict[Path, tuple[int, int]] = {}
PI_TO_KEV_FACTORS = {
    # Standard ASCA GIS PI has 1024 channels. Some UDON3 files are 256-channel
    # products; pi_to_kev_factor detects those from parquet statistics below.
    ("asca", "gis"): 1.0 / 84.9,
    ("asca", "sis"): 1.0 / 68.5,
    ("suzaku", "xis"): 0.00365,
    ("hitomi", "sxs"): 0.0005,
    ("hitomi", "sxi"): 0.006,
    ("hitomi", "hxi"): 0.1,
    ("xrism", "resolve"): 0.0005,
    ("xrism", "xtend"): 0.006,
}


def dependency_messages() -> list[str]:
    messages: list[str] = []
    if WCS is None or SkyCoord is None or u is None:
        messages.append("Missing astropy: python -m pip install astropy")
    if duckdb is None:
        messages.append("Missing duckdb: python -m pip install duckdb")
    return messages


def require_astropy():
    if WCS is None or SkyCoord is None or u is None:
        raise ImportError("Missing astropy. Install with: python -m pip install astropy")


def require_duckdb():
    if duckdb is None:
        raise ImportError("Missing duckdb. Install with: python -m pip install duckdb")


def serialized_database_access(function):
    """Serialize short transactions against the shared on-disk DuckDB file."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with DATABASE_ACCESS_LOCK:
            return function(*args, **kwargs)

    return wrapped


def safe_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value).strip().lower())
    return token or "unknown"


def instrument_and_obsid(path: Path) -> tuple[str, str]:
    stem = path.name.removesuffix("_events.parquet")
    if "_" not in stem:
        return "unknown", stem
    observation_id, instrument = stem.rsplit("_", 1)
    return instrument.lower(), observation_id


def record_key(record: EventFile) -> str:
    return f"{record.mission}/{record.instrument}/{record.observation_id}"


def pair_key(mission: str, instrument: str) -> str:
    return f"{mission}/{instrument}"


def record_label(record: EventFile) -> str:
    return f"{record.mission.upper()} / {record.instrument.upper()} / {record.observation_id}"


def utc_now_text() -> str:
    return datetime.now(UTC).isoformat()


def compact_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


@lru_cache(maxsize=4096)
def _parquet_pi_max(path_text: str, url: str, size: int, mtime_ns: int) -> float | None:
    path = Path(path_text)
    if path.exists():
        try:
            import pyarrow.parquet as parquet

            parquet_file = parquet.ParquetFile(path)
            metadata = parquet_file.metadata
            column_index = parquet_file.schema_arrow.get_field_index("PI")
            maxima = []
            for index in range(metadata.num_row_groups):
                statistics = metadata.row_group(index).column(column_index).statistics
                if statistics is not None and statistics.has_min_max:
                    maxima.append(float(statistics.max))
            if maxima:
                return max(maxima)
        except (ImportError, OSError, ValueError):
            pass
    if url and duckdb is not None:
        con = duckdb.connect()
        try:
            value = con.execute(
                "SELECT max(PI) FROM read_parquet(?)", [url]
            ).fetchone()[0]
            return None if value is None else float(value)
        except Exception:
            return None
        finally:
            con.close()
    return None


def pi_to_kev_factor(record: EventFile) -> float:
    key = (str(record.mission).lower(), str(record.instrument).lower())
    if key == ("asca", "gis"):
        path = Path(record.parquet_path)
        if path.exists():
            stat = path.stat()
            size, mtime_ns = stat.st_size, stat.st_mtime_ns
        else:
            size = mtime_ns = 0
        maximum = _parquet_pi_max(
            str(path), str(record.parquet_url or ""), size, mtime_ns
        )
        # The ASCA ABC Guide states N_10bit=84.9E; for 8-bit products N is
        # divided by four. A maximum <=255 identifies that representation.
        if maximum is not None and maximum <= 255.0:
            return 4.0 / 84.9
    return float(PI_TO_KEV_FACTORS.get(key, 1.0 / RESOLVE_PI_CHANNELS_PER_KEV))


def wcs_metadata_signature(metadata: dict) -> str:
    keys = (
        "TCTYP_X", "TCTYP_Y", "TCRPX_X", "TCRPX_Y", "TCRVL_X", "TCRVL_Y",
        "TCDLT_X", "TCDLT_Y", "TCUNI_X", "TCUNI_Y",
    )
    return compact_hash({key: metadata.get(key) for key in keys})


def source_identity(record: EventFile) -> dict:
    if record.parquet_path.exists():
        stat = record.parquet_path.stat()
        return {
            "key": record_key(record),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    identity = {"key": record_key(record), "url": record.parquet_url or ""}
    if not record.parquet_url:
        return identity
    try:
        request = urllib.request.Request(
            record.parquet_url,
            method="HEAD",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            identity.update(
                {
                    "etag": response.headers.get("ETag", ""),
                    "last_modified": response.headers.get("Last-Modified", ""),
                    "content_length": response.headers.get("Content-Length", ""),
                }
            )
    except Exception:
        pass
    return identity


def ensure_storage_schema(db_path: Path | str = DB_PATH) -> None:
    """Create additive cache tables without replacing the server catalog."""
    require_duckdb()
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preview workers can reach schema setup at the same time on a fresh cache.
    # Serialize the additive DDL so DuckDB never sees competing catalog writes.
    with DATABASE_ACCESS_LOCK, STORAGE_SCHEMA_LOCK:
        resolved = path.resolve()
        if path.exists():
            stat = path.stat()
            if _STORAGE_SCHEMA_READY.get(resolved) == (stat.st_dev, stat.st_ino):
                return
        con = duckdb.connect(str(path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS server_headers (
                    key TEXT,
                    mission TEXT,
                    instrument TEXT,
                    observation_id TEXT,
                    header_json TEXT,
                    cached_at TEXT
                )
                """
            )
            for name, sql_type in (
                ("etag", "TEXT"),
                ("last_modified", "TEXT"),
                ("tctyp_x", "TEXT"),
                ("tctyp_y", "TEXT"),
                ("tcrpx_x", "DOUBLE"),
                ("tcrpx_y", "DOUBLE"),
                ("tcrvl_x", "DOUBLE"),
                ("tcrvl_y", "DOUBLE"),
                ("tcdlt_x", "DOUBLE"),
                ("tcdlt_y", "DOUBLE"),
                ("tcuni_x", "TEXT"),
                ("tcuni_y", "TEXT"),
            ):
                con.execute(
                    f"ALTER TABLE server_headers ADD COLUMN IF NOT EXISTS {name} {sql_type}"
                )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    product_key TEXT,
                    observation_key TEXT,
                    kind TEXT,
                    path TEXT,
                    size_bytes BIGINT,
                    parameters_hash TEXT,
                    source_etag TEXT,
                    metadata_json TEXT,
                    created_at TEXT,
                    last_accessed TEXT,
                    pinned BOOLEAN
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS server_headers_key_idx ON server_headers(key)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS cache_entries_key_idx ON cache_entries(product_key)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS cache_entries_access_idx "
                "ON cache_entries(last_accessed)"
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS resolved_targets (
                    normalized_name TEXT,
                    target_name TEXT,
                    ra_deg DOUBLE,
                    dec_deg DOUBLE,
                    resolver TEXT,
                    resolved_at TEXT
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS resolved_targets_name_idx "
                "ON resolved_targets(normalized_name)"
            )
        finally:
            con.close()
        stat = path.stat()
        _STORAGE_SCHEMA_READY[resolved] = (stat.st_dev, stat.st_ino)



def read_header(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
