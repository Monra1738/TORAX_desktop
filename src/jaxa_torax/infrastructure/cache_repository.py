"""Search, indexing, loading, and export helpers for the PyVista prototype."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

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


from .science_core import *

_CACHE_LIMIT_CALLS: dict[Path, int] = {}
_CACHE_LIMIT_CHECK_EVERY = 25


def _header_row_values(record: EventFile, metadata: dict, etag: str, last_modified: str):
    return (
        record_key(record),
        record.mission,
        record.instrument,
        record.observation_id,
        json.dumps(metadata, separators=(",", ":"), default=str),
        utc_now_text(),
        str(etag or ""),
        str(last_modified or ""),
        metadata.get("TCTYP_X"),
        metadata.get("TCTYP_Y"),
        metadata.get("TCRPX_X"),
        metadata.get("TCRPX_Y"),
        metadata.get("TCRVL_X"),
        metadata.get("TCRVL_Y"),
        metadata.get("TCDLT_X"),
        metadata.get("TCDLT_Y"),
        metadata.get("TCUNI_X") or "deg",
        metadata.get("TCUNI_Y") or "deg",
    )


@serialized_database_access
def store_server_header(
    record: EventFile,
    metadata: dict,
    etag: str = "",
    last_modified: str = "",
    db_path: Path | str = DB_PATH,
) -> None:
    ensure_storage_schema(db_path)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("DELETE FROM server_headers WHERE key = ?", [record_key(record)])
        con.execute(
            """
            INSERT INTO server_headers (
                key, mission, instrument, observation_id, header_json, cached_at,
                etag, last_modified, tctyp_x, tctyp_y, tcrpx_x, tcrpx_y,
                tcrvl_x, tcrvl_y, tcdlt_x, tcdlt_y, tcuni_x, tcuni_y
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _header_row_values(record, metadata, etag, last_modified),
        )
    finally:
        con.close()


@serialized_database_access
def cached_server_header(
    record: EventFile,
    db_path: Path | str = DB_PATH,
) -> dict | None:
    if duckdb is None or not Path(db_path).exists():
        return None
    ensure_storage_schema(db_path)
    # Use the same read/write connection configuration everywhere in-process.
    # DuckDB rejects opening one file as read-only while preview workers or the
    # workspace autosaver already have it open read/write.
    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT header_json FROM server_headers WHERE key = ? LIMIT 1",
            [record_key(record)],
        ).fetchone()
    finally:
        con.close()
    if not row or not row[0]:
        return None
    return json.loads(row[0])


def fetch_server_header(
    record: EventFile,
    db_path: Path | str = DB_PATH,
) -> dict:
    if not record.header_url:
        raise FileNotFoundError(f"No header URL for {record_key(record)}")
    request = urllib.request.Request(record.header_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.loads(response.read().decode("utf-8", errors="replace"))
        store_server_header(
            record,
            metadata,
            etag=response.headers.get("ETag", ""),
            last_modified=response.headers.get("Last-Modified", ""),
            db_path=db_path,
        )
    return metadata


def get_record_header(
    record: EventFile,
    allow_remote: bool = True,
    db_path: Path | str = DB_PATH,
) -> dict:
    metadata = cached_server_header(record, db_path=db_path)
    if metadata is not None:
        return metadata
    if record.header_path.exists():
        metadata = read_header(record.header_path)
        store_server_header(record, metadata, db_path=db_path)
        return metadata
    if allow_remote:
        return fetch_server_header(record, db_path=db_path)
    raise FileNotFoundError(f"Header is not cached for {record_key(record)}")


def cache_product_path(record: EventFile, kind: str, suffix: str, token: str = "") -> Path:
    root = PRODUCT_CACHE_DIR / kind / record.mission / record.observation_id
    stem = f"{record.observation_id}_{record.instrument}"
    if token:
        stem += f"_{safe_token(token)}"
    return root / f"{stem}{suffix}"


@serialized_database_access
def register_cache_entry(
    product_key: str,
    observation_key: str,
    kind: str,
    path: Path | str,
    parameters_hash: str = "",
    source_etag: str = "",
    metadata: dict | None = None,
    pinned: bool = False,
    db_path: Path | str = DB_PATH,
) -> None:
    path = Path(path)
    if not path.exists():
        return
    ensure_storage_schema(db_path)
    now = utc_now_text()
    con = duckdb.connect(str(db_path))
    try:
        con.execute("DELETE FROM cache_entries WHERE product_key = ?", [product_key])
        con.execute(
            """
            INSERT INTO cache_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                product_key,
                observation_key,
                kind,
                str(path.resolve()),
                int(path.stat().st_size),
                parameters_hash,
                source_etag,
                json.dumps(metadata or {}, separators=(",", ":"), default=str),
                now,
                now,
                bool(pinned),
            ],
        )
    finally:
        con.close()


@serialized_database_access
def cache_entry(product_key: str, db_path: Path | str = DB_PATH) -> dict | None:
    if duckdb is None or not Path(db_path).exists():
        return None
    ensure_storage_schema(db_path)
    con = duckdb.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT * FROM cache_entries WHERE product_key = ? LIMIT 1",
            [product_key],
        ).fetchdf()
        if row.empty:
            return None
        path = Path(str(row.iloc[0]["path"]))
        if not path.exists():
            con.execute("DELETE FROM cache_entries WHERE product_key = ?", [product_key])
            return None
        now = utc_now_text()
        con.execute(
            "UPDATE cache_entries SET last_accessed = ?, size_bytes = ? "
            "WHERE product_key = ?",
            [now, int(path.stat().st_size), product_key],
        )
        result = row.iloc[0].to_dict()
        result["path"] = str(path)
        result["metadata"] = json.loads(result.get("metadata_json") or "{}")
        return result
    finally:
        con.close()


@serialized_database_access
def cache_entries_for_observation(
    observation_key: str,
    kind: str,
    db_path: Path | str = DB_PATH,
) -> list[dict]:
    """Return existing products for compatibility reuse across display budgets."""
    if duckdb is None or not Path(db_path).exists():
        return []
    ensure_storage_schema(db_path)
    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            """
            SELECT product_key, path, metadata_json
            FROM cache_entries
            WHERE observation_key = ? AND kind = ?
            ORDER BY last_accessed DESC
            """,
            [str(observation_key), str(kind)],
        ).fetchall()
        result = []
        for product_key, raw_path, metadata_json in rows:
            path = Path(str(raw_path))
            if not path.exists():
                con.execute("DELETE FROM cache_entries WHERE product_key = ?", [product_key])
                continue
            result.append({
                "product_key": str(product_key),
                "path": str(path),
                "metadata": json.loads(metadata_json or "{}"),
            })
        return result
    finally:
        con.close()


def register_existing_cache(
    cache_dir: Path | str = CACHE_DIR,
    db_path: Path | str = DB_PATH,
) -> dict:
    root = Path(cache_dir)
    ensure_storage_schema(db_path)
    registered = 0
    for path in root.rglob("*_events.parquet") if root.exists() else []:
        if PRODUCT_CACHE_DIR in path.parents:
            continue
        instrument, observation_id = instrument_and_obsid(path)
        relative = path.relative_to(root)
        mission = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        key = f"{mission}/{instrument}/{observation_id}"
        register_cache_entry(
            f"raw:{key}",
            key,
            "raw",
            path,
            metadata={"imported": True},
            db_path=db_path,
        )
        registered += 1
    return {"registered": registered, **cache_status(db_path=db_path)}


@serialized_database_access
def cache_status(db_path: Path | str = DB_PATH) -> dict:
    ensure_storage_schema(db_path)
    con = duckdb.connect(str(db_path))
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        catalog_files = (
            int(con.execute("SELECT COUNT(*) FROM server_files").fetchone()[0])
            if "server_files" in tables
            else 0
        )
        cached_headers = int(
            con.execute("SELECT COUNT(DISTINCT key) FROM server_headers").fetchone()[0]
        )
        rows = con.execute(
            "SELECT product_key, kind, path, size_bytes FROM cache_entries"
        ).fetchall()
        missing = [row[0] for row in rows if not Path(str(row[2])).exists()]
        if missing:
            con.executemany(
                "DELETE FROM cache_entries WHERE product_key = ?",
                [(key,) for key in missing],
            )
        valid = [row for row in rows if row[0] not in set(missing)]
    finally:
        con.close()
    by_kind: dict[str, int] = {}
    for _key, kind, path, _stored_size in valid:
        size = Path(str(path)).stat().st_size
        by_kind[str(kind)] = by_kind.get(str(kind), 0) + int(size)
    return {
        "entries": len(valid),
        "bytes": sum(by_kind.values()),
        "by_kind": by_kind,
        "limit_bytes": DEFAULT_CACHE_LIMIT_BYTES,
        "catalog_files": catalog_files,
        "cached_headers": cached_headers,
    }


@serialized_database_access
def enforce_cache_limit(
    limit_bytes: int = DEFAULT_CACHE_LIMIT_BYTES,
    db_path: Path | str = DB_PATH,
) -> dict:
    resolved = Path(db_path).resolve()
    calls = _CACHE_LIMIT_CALLS.get(resolved, 0) + 1
    _CACHE_LIMIT_CALLS[resolved] = calls
    # A large load can create hundreds of compact previews. Scanning the full
    # cache after every small file is quadratic work; checking every 25 files
    # keeps the size bound close while avoiding a database/file-system storm.
    if calls > 1 and calls % _CACHE_LIMIT_CHECK_EVERY:
        return {"removed": [], "deferred": True, "check_every": _CACHE_LIMIT_CHECK_EVERY}
    ensure_storage_schema(db_path)
    con = duckdb.connect(str(db_path))
    removed = []
    try:
        entries = con.execute(
            """
            SELECT product_key, kind, path, size_bytes, last_accessed, created_at
            FROM cache_entries
            WHERE NOT coalesce(pinned, false)
            ORDER BY last_accessed ASC, created_at ASC
            """
        ).fetchall()
        sizes = {}
        totals = {"raw": 0, "derived": 0}
        for product_key, kind, path_text, stored_size, _accessed, _created in entries:
            path = Path(str(path_text))
            size = path.stat().st_size if path.exists() else int(stored_size or 0)
            sizes[product_key] = size
            group = "raw" if kind == "raw" else "derived"
            totals[group] += size

        targets = {
            "raw": int(limit_bytes * 0.70),
            "derived": int(limit_bytes * 0.30),
        }

        def remove_entry(product_key, path_text, group):
            path = Path(str(path_text))
            size = sizes.get(product_key, 0)
            if path.exists():
                path.unlink()
            con.execute("DELETE FROM cache_entries WHERE product_key = ?", [product_key])
            totals[group] -= size
            removed.append(str(product_key))

        for group in ("raw", "derived"):
            for product_key, kind, path_text, _size, _accessed, _created in entries:
                entry_group = "raw" if kind == "raw" else "derived"
                if entry_group != group or product_key in removed:
                    continue
                if totals[group] <= targets[group]:
                    break
                remove_entry(product_key, path_text, group)

        total = totals["raw"] + totals["derived"]
        for product_key, kind, path_text, _size, _accessed, _created in entries:
            if total <= int(limit_bytes):
                break
            if product_key in removed:
                continue
            group = "raw" if kind == "raw" else "derived"
            size = sizes.get(product_key, 0)
            remove_entry(product_key, path_text, group)
            total -= size
    finally:
        con.close()
    return {"removed": removed, **cache_status(db_path=db_path)}


@serialized_database_access
def clear_cache(
    kinds: Sequence[str] = ("raw", "preview", "image"),
    db_path: Path | str = DB_PATH,
) -> dict:
    ensure_storage_schema(db_path)
    selected = {str(kind) for kind in kinds}
    con = duckdb.connect(str(db_path))
    removed = []
    try:
        rows = con.execute(
            "SELECT product_key, kind, path FROM cache_entries"
        ).fetchall()
        for product_key, kind, path_text in rows:
            if kind not in selected:
                continue
            path = Path(str(path_text))
            if path.exists():
                path.unlink()
            con.execute("DELETE FROM cache_entries WHERE product_key = ?", [product_key])
            removed.append(str(product_key))
    finally:
        con.close()
    return {"removed": removed, **cache_status(db_path=db_path)}
