"""Search, indexing, loading, and export helpers for the PyVista prototype."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:  # The UI can still show a useful install message.
    duckdb = None

try:
    import pyarrow.dataset as arrow_dataset
except ImportError:  # pandas remains a functional, less memory-efficient fallback.
    arrow_dataset = None

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


def read_exact_energy_columns(
    record: EventFile,
    low_pi: float,
    high_pi: float,
    db_path: Path | str = DB_PATH,
) -> pd.DataFrame:
    columns = ["PI", "X", "Y"]
    filters = [("PI", ">=", low_pi), ("PI", "<=", high_pi)]
    if record.parquet_path.exists():
        register_cache_entry(
            f"raw:{record_key(record)}",
            record_key(record),
            "raw",
            record.parquet_path,
            db_path=db_path,
        )
        return pd.read_parquet(record.parquet_path, columns=columns, filters=filters)
    if record.parquet_url and duckdb is not None:
        con = duckdb.connect()
        try:
            return con.execute(
                """
                SELECT PI, X, Y
                FROM read_parquet(?)
                WHERE PI >= ? AND PI <= ?
                """,
                [record.parquet_url, float(low_pi), float(high_pi)],
            ).fetchdf()
        except Exception:
            pass
        finally:
            con.close()
    record = ensure_cached(record, db_path=db_path)
    return pd.read_parquet(record.parquet_path, columns=columns, filters=filters)


def iter_exact_energy_columns(
    record: EventFile,
    low_pi: float | None,
    high_pi: float | None,
    db_path: Path | str = DB_PATH,
    batch_size: int = 262_144,
):
    """Yield PI/X/Y batches without materializing a complete observation.

    Exact products can cover tens of millions of events.  Keeping the operation
    out-of-core is essential: VTK still receives a bounded interactive preview,
    while every matching parquet event contributes to the scientific image.
    """
    columns = ["PI", "X", "Y"]
    batch_size = max(16_384, int(batch_size))
    if record.parquet_path.exists() and arrow_dataset is not None:
        expression = None
        if low_pi is not None:
            expression = arrow_dataset.field("PI") >= float(low_pi)
        if high_pi is not None:
            upper = arrow_dataset.field("PI") <= float(high_pi)
            expression = upper if expression is None else expression & upper
        scanner = arrow_dataset.dataset(
            str(record.parquet_path), format="parquet"
        ).scanner(
            columns=columns,
            filter=expression,
            batch_size=batch_size,
            use_threads=True,
        )
        for batch in scanner.to_batches():
            if batch.num_rows:
                yield batch.to_pandas()
        return

    if record.parquet_url and duckdb is not None and not record.parquet_path.exists():
        clauses = []
        parameters: list[object] = [record.parquet_url]
        if low_pi is not None:
            clauses.append("PI >= ?")
            parameters.append(float(low_pi))
        if high_pi is not None:
            clauses.append("PI <= ?")
            parameters.append(float(high_pi))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        con = duckdb.connect()
        try:
            reader = con.execute(
                f"SELECT PI, X, Y FROM read_parquet(?){where}",
                parameters,
            ).fetch_record_batch(rows_per_batch=batch_size)
            for batch in reader:
                if batch.num_rows:
                    yield batch.to_pandas()
            return
        except Exception:
            # A server without range support is handled by the normal cache path.
            pass
        finally:
            con.close()

    cached = ensure_cached(record, db_path=db_path)
    if arrow_dataset is not None:
        yield from iter_exact_energy_columns(
            cached, low_pi, high_pi, db_path=db_path, batch_size=batch_size
        )
        return
    filters = []
    if low_pi is not None:
        filters.append(("PI", ">=", float(low_pi)))
    if high_pi is not None:
        filters.append(("PI", "<=", float(high_pi)))
    yield pd.read_parquet(
        cached.parquet_path,
        columns=columns,
        filters=filters or None,
    )


def exact_energy_image(
    records: Sequence[EventFile],
    low_kev: float | None,
    high_kev: float | None,
    bins: int,
    db_path: Path | str = DB_PATH,
    region: SkySelection | None = None,
) -> dict:
    records = list(records)
    if not records:
        raise ValueError("No observations selected for exact image")
    bins = max(16, int(bins))
    all_events = low_kev is None and high_kev is None
    if (low_kev is None) != (high_kev is None):
        raise ValueError("Exact image energy bounds must both be set or both be omitted")
    if not all_events:
        low_kev = float(low_kev)
        high_kev = float(high_kev)
    if not all_events and high_kev <= low_kev:
        raise ValueError("Exact image energy maximum must exceed minimum")
    metadata_by_key = {
        record_key(record): get_record_header(record, db_path=db_path)
        for record in records
    }
    sources = []
    for record in records:
        source = source_identity(record)
        source["wcs"] = wcs_metadata_signature(metadata_by_key[record_key(record)])
        source["kev_factor"] = pi_to_kev_factor(record)
        sources.append(source)
    signature_data = {
        "records": sorted(sources, key=lambda item: item["key"]),
        "energy_scope": "all_events" if all_events else "band",
        "low_kev": None if all_events else round(low_kev, 8),
        "high_kev": None if all_events else round(high_kev, 8),
        "bins": bins,
        "calibration": CALIBRATION_VERSION,
        "region": region.signature() if region is not None else None,
    }
    signature = compact_hash(signature_data)
    product_key = f"image:{signature}"
    entry = cache_entry(product_key, db_path=db_path)
    if entry is not None:
        with np.load(entry["path"]) as stored:
            return {
                "hist": stored["hist"],
                "x_edges": stored["x_edges"],
                "y_edges": stored["y_edges"],
                "event_count": int(stored["event_count"][0]),
                "low_kev": float(stored["energy_bounds"][0]),
                "high_kev": float(stored["energy_bounds"][1]),
                "source_keys": [str(value) for value in stored["source_keys"]],
                "calibration_version": int(stored["calibration_version"][0]),
                "exact": True,
                "cache_hit": True,
            }

    ra_parts = []
    dec_parts = []
    event_count = 0
    actual_low_kev = np.inf
    actual_high_kev = -np.inf
    accumulated = np.zeros((bins, bins), dtype=np.uint64) if region is not None else None
    if region is not None:
        image_range = selection_image_range(region)
        fixed_x_edges = np.linspace(image_range[0][0], image_range[0][1], bins + 1)
        fixed_y_edges = np.linspace(image_range[1][0], image_range[1][1], bins + 1)
    for record in records:
        metadata = metadata_by_key[record_key(record)]
        factor = pi_to_kev_factor(record)
        wcs = native_wcs(metadata)
        # Negative PI values are instrument sentinels, not physical energies.
        # "All events" therefore means all finite, scientifically valid PI.
        low_pi = 0.0 if all_events else low_kev / factor
        high_pi = None if all_events else high_kev / factor
        for frame in iter_exact_energy_columns(
            record, low_pi, high_pi, db_path=db_path
        ):
            frame = frame.dropna(subset=["PI", "X", "Y"])
            if not all_events:
                frame = frame.loc[
                    frame["PI"].between(low_pi, high_pi, inclusive="both")
                ]
            if frame.empty:
                continue
            ra, dec = wcs.all_pix2world(
                frame["X"].to_numpy(dtype=float),
                frame["Y"].to_numpy(dtype=float),
                1,
            )
            valid = np.isfinite(ra) & np.isfinite(dec)
            if region is not None:
                valid &= selection_contains(region, ra, dec)
            if not np.any(valid):
                continue
            selected_pi = frame["PI"].to_numpy(dtype=float)[valid]
            actual_low_kev = min(actual_low_kev, float(np.min(selected_pi)) * factor)
            actual_high_kev = max(actual_high_kev, float(np.max(selected_pi)) * factor)
            selected_ra = np.mod(np.asarray(ra)[valid], 360.0)
            selected_dec = np.asarray(dec)[valid]
            event_count += len(selected_ra)
            if region is not None:
                selected_ra = unwrap_ra_for_selection(selected_ra, region)
                partial_hist, _, _ = np.histogram2d(
                    selected_ra, selected_dec, bins=(fixed_x_edges, fixed_y_edges)
                )
                accumulated += partial_hist.T.astype(np.uint64)
            else:
                ra_parts.append(selected_ra)
                dec_parts.append(selected_dec)

    if event_count == 0:
        scope = "all energies" if all_events else f"{low_kev:.3f}-{high_kev:.3f} keV"
        raise ValueError(f"No exact events in {scope}")
    if region is not None:
        hist = accumulated
        x_edges, y_edges = fixed_x_edges, fixed_y_edges
    else:
        ra = np.concatenate(ra_parts)
        dec = np.concatenate(dec_parts)
        hist, x_edges, y_edges = np.histogram2d(ra, dec, bins=bins)
        hist = hist.T.astype(np.uint64)
    stored_low = float(actual_low_kev) if all_events else float(low_kev)
    stored_high = float(actual_high_kev) if all_events else float(high_kev)
    path = PRODUCT_CACHE_DIR / "image" / f"{signature}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with partial.open("wb") as stream:
            np.savez_compressed(
                stream,
                hist=hist.astype(np.uint64),
                x_edges=x_edges.astype(np.float64),
                y_edges=y_edges.astype(np.float64),
                event_count=np.asarray([event_count], dtype=np.int64),
                energy_bounds=np.asarray([stored_low, stored_high], dtype=np.float64),
                source_keys=np.asarray(
                    sorted(record_key(record) for record in records), dtype="U128"
                ),
                calibration_version=np.asarray([CALIBRATION_VERSION], dtype=np.int32),
            )
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()
    register_cache_entry(
        product_key,
        ",".join(sorted(record_key(record) for record in records)),
        "image",
        path,
        parameters_hash=signature,
        metadata=signature_data,
        db_path=db_path,
    )
    enforce_cache_limit(db_path=db_path)
    return {
        "hist": hist.astype(np.uint64),
        "x_edges": x_edges,
        "y_edges": y_edges,
        "event_count": event_count,
        "low_kev": stored_low,
        "high_kev": stored_high,
        "source_keys": sorted(record_key(record) for record in records),
        "calibration_version": CALIBRATION_VERSION,
        "exact": True,
        "cache_hit": False,
    }


def exact_all_events_image(
    records: Sequence[EventFile],
    bins: int,
    db_path: Path | str = DB_PATH,
    region: SkySelection | None = None,
) -> dict:
    """Compress every spatially matching parquet event into one exact sky image."""
    return exact_energy_image(
        records, None, None, bins=bins, db_path=db_path, region=region
    )


def exact_rgb_image(
    records: Sequence[EventFile],
    config: RGBBandConfig,
    bins: int,
    db_path: Path | str = DB_PATH,
    region: SkySelection | None = None,
) -> dict:
    """Build three exact, cache-backed energy images on a shared region grid."""
    bands = config.bands()
    channels = [
        exact_energy_image(
            records, low, high, bins=bins, db_path=db_path, region=region
        )
        for low, high in bands
    ]
    reference_x = np.asarray(channels[0]["x_edges"], dtype=float)
    reference_y = np.asarray(channels[0]["y_edges"], dtype=float)
    arrays = []
    for channel in channels:
        values = np.asarray(channel["hist"], dtype=float)
        x_edges = np.asarray(channel["x_edges"], dtype=float)
        y_edges = np.asarray(channel["y_edges"], dtype=float)
        if not (
            np.allclose(x_edges, reference_x) and np.allclose(y_edges, reference_y)
        ):
            source_x = 0.5 * (x_edges[:-1] + x_edges[1:])
            source_y = 0.5 * (y_edges[:-1] + y_edges[1:])
            target_x = 0.5 * (reference_x[:-1] + reference_x[1:])
            target_y = 0.5 * (reference_y[:-1] + reference_y[1:])
            x_index = np.abs(source_x[:, None] - target_x[None, :]).argmin(axis=0)
            y_index = np.abs(source_y[:, None] - target_y[None, :]).argmin(axis=0)
            values = values[np.ix_(y_index, x_index)]
        arrays.append(values)
    return {
        "channels": np.stack(arrays, axis=-1),
        "x_edges": reference_x,
        "y_edges": reference_y,
        "event_counts": [int(channel["event_count"]) for channel in channels],
        "bands": bands,
        "exact": True,
        "cache_hit": all(bool(channel.get("cache_hit")) for channel in channels),
    }
