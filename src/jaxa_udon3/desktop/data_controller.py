from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jaxa_udon3.infrastructure import science as backend

INSTRUMENTS: tuple[tuple[str, str], ...] = (
    ("xrism", "resolve"),
    ("xrism", "xtend"),
    ("hitomi", "sxs"),
    ("hitomi", "sxi"),
    ("hitomi", "hxi"),
    ("suzaku", "xis"),
    ("asca", "gis"),
    ("asca", "sis"),
)
COMBINED_PREVIEW_MAXIMUM = 600_000
PER_OBSERVATION_MAXIMUM = 15_000
MINIMUM_PREVIEW_ALLOCATION = 250


@dataclass(frozen=True)
class SearchPayload:
    region: object
    region_cache_hit: bool
    frame: pd.DataFrame
    records_by_key: dict[str, backend.EventFile]
    target_label: str


@dataclass(frozen=True)
class LoadPayload:
    observations: list[backend.LoadedObservation]
    skipped: list[str]


def _record_key(record: backend.EventFile) -> str:
    return f"{record.mission}/{record.instrument}/{record.observation_id}"


def search_catalog(
    *,
    mode: str,
    target_name: str,
    ra_value: str | float,
    dec_value: str | float,
    radius_arcmin: float,
    selected_pairs: Iterable[str],
    object_text: str = "",
    observation_text: str = "",
    date_start: str = "",
    date_end: str = "",
    limit: int = 300,
) -> SearchPayload:
    radius_deg = float(radius_arcmin) / 60.0
    # A normal target search should also find catalog entries explicitly named
    # for that target.  The catalog search normalizes spelling variants such as
    # ``Cas A`` and ``CAS_A_C1O``.  An advanced Object contains value remains
    # an explicit override.
    if str(mode or "").strip().lower() == "target" and not str(object_text).strip():
        object_text = backend.canonical_target_name(target_name)
    region, cache_hit = backend.parse_sky_region(
        mode,
        ra_value,
        dec_value,
        radius_deg,
        target_name=target_name,
    )
    if not backend.server_catalog_exists():
        backend.build_server_catalog()
    frame = backend.search_server_catalog(
        object_text=object_text,
        observation_text=observation_text,
        selected_pairs=list(selected_pairs),
        center_ra=region.center_ra_deg,
        center_dec=region.center_dec_deg,
        radius_deg=region.radius_deg,
        pointing_margin_deg=max(0.35, region.radius_deg),
        date_start=date_start,
        date_end=date_end,
        limit=limit,
    )
    records = backend.server_records_from_dataframe(frame)
    mapping = {_record_key(record): record for record in records}
    target_label = region.label or target_name or f"RA {region.center_ra_deg:.4f}"
    return SearchPayload(region, cache_hit, frame, mapping, target_label)


def balanced_row_limits(records: Sequence[backend.EventFile], max_points: int) -> list[int]:
    """Allocate display rows fairly: mission, then instrument, then observation."""
    records = list(records)
    if not records:
        return []
    missions: dict[str, dict[str, int]] = {}
    for record in records:
        mission = str(record.mission).lower()
        instrument = str(record.instrument).lower()
        groups = missions.setdefault(mission, {})
        groups[instrument] = groups.get(instrument, 0) + 1
    maximum = min(COMBINED_PREVIEW_MAXIMUM, max(1, int(max_points)))
    mission_count = len(missions)
    raw = []
    for record in records:
        instruments = missions[str(record.mission).lower()]
        count = instruments[str(record.instrument).lower()]
        raw.append(maximum / mission_count / len(instruments) / count)
    limits = np.floor(raw).astype(int)
    limits = np.clip(limits, MINIMUM_PREVIEW_ALLOCATION, PER_OBSERVATION_MAXIMUM)
    # Distribute spare rows deterministically without exceeding either hard cap.
    spare = max(0, maximum - int(limits.sum()))
    while spare and np.any(limits < PER_OBSERVATION_MAXIMUM):
        eligible = np.flatnonzero(limits < PER_OBSERVATION_MAXIMUM)
        share = spare // len(eligible)
        if share == 0:
            eligible = eligible[:spare]
            additions = np.ones(len(eligible), dtype=int)
        else:
            additions = np.minimum(PER_OBSERVATION_MAXIMUM - limits[eligible], share)
        limits[eligible] += additions
        consumed = int(additions.sum())
        spare -= consumed
        if consumed == 0:
            break
    return [int(value) for value in limits]


def load_observation_preview(record: backend.EventFile, region, row_limit: int):
    row_limit = max(
        MINIMUM_PREVIEW_ALLOCATION,
        min(PER_OBSERVATION_MAXIMUM, int(row_limit)),
    )
    frame, metadata, events_in_region, _cache_hit = backend.read_region_preview(
        record, region, max_rows=row_limit
    )
    frame = frame.copy()
    if len(frame) > row_limit:
        index = np.linspace(0, len(frame) - 1, row_limit, dtype=int)
        frame = frame.iloc[index].copy()
    frame["MISSION"] = str(record.mission)
    frame["INSTRUMENT"] = str(record.instrument)
    frame["OBSERVATION_ID"] = str(record.observation_id)
    if "SOURCE_ROW" not in frame:
        frame["SOURCE_ROW"] = np.arange(len(frame), dtype=np.uint32)
    return backend.LoadedObservation(
        record=record,
        frame=frame.reset_index(drop=True),
        metadata=metadata,
        total_events=int(events_in_region),
        events_in_region=int(events_in_region),
        displayed_events=len(frame),
        minimum_separation_deg=None,
    )


def load_previews(
    records: Sequence[backend.EventFile],
    region,
    max_points: int = 120_000,
) -> LoadPayload:
    """Load independent observation previews concurrently with a deliberately small worker cap."""
    records = list(records)
    limits = balanced_row_limits(records, max_points)
    if not records:
        return LoadPayload([], [])

    def load_one(index: int, record, row_limit: int):
        return index, load_observation_preview(record, region, row_limit)

    ordered: dict[int, backend.LoadedObservation] = {}
    skipped: list[str] = []
    max_workers = max(1, min(3, len(records)))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="udon3-preview") as pool:
        futures = {
            pool.submit(load_one, index, record, limit): record
            for index, (record, limit) in enumerate(zip(records, limits))
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                index, observation = future.result()
                ordered[index] = observation
            except Exception as error:
                skipped.append(f"{_record_key(record)}: {error}")
    observations = [ordered[index] for index in sorted(ordered)]
    return LoadPayload(observations, skipped)


def exact_energy(
    records, region, low: float, high: float, bins: int, progress_callback=None,
) -> dict:
    records = list(records)
    if progress_callback is None or len(records) <= 1:
        return backend.exact_energy_image(records, low, high, bins=bins, region=region)

    def one(record):
        return record, backend.exact_energy_image(
            [record], low, high, bins=bins, region=region
        )

    products, failures = [], {}
    with ThreadPoolExecutor(
        max_workers=min(3, len(records)), thread_name_prefix="udon3-exact"
    ) as pool:
        futures = {pool.submit(one, record): record for record in records}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            key = _record_key(record)
            try:
                _record, product = future.result()
                products.append(product)
            except Exception as error:
                failures[key] = str(error)
            progress_callback(completed, len(records), key)
    if not products:
        details = "; ".join(f"{key}: {message}" for key, message in failures.items())
        raise ValueError(f"No exact events could be loaded. {details}")
    first = products[0]
    return {
        **first,
        "hist": np.sum([np.asarray(item["hist"], dtype=np.uint64) for item in products], axis=0),
        "event_count": sum(int(item["event_count"]) for item in products),
        "source_keys": [key for item in products for key in item.get("source_keys", [])],
        "cache_hit": all(bool(item.get("cache_hit")) for item in products),
        "failures": failures,
    }


def exact_all_events(records, region, bins: int, progress_callback=None) -> dict:
    """Build an exact RA/DEC image from every event in every selected parquet."""
    records = list(records)
    if not records:
        raise ValueError("No observations selected for the all-event image")

    def one(record):
        return record, backend.exact_all_events_image(
            [record], bins=bins, region=region
        )

    products, failures = [], {}
    # Each worker is itself out-of-core. Keep this small so disk/network scans
    # and WCS transforms remain predictable on desktop machines.
    with ThreadPoolExecutor(
        max_workers=min(2, len(records)), thread_name_prefix="udon3-all-events"
    ) as pool:
        futures = {pool.submit(one, record): record for record in records}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            key = _record_key(record)
            try:
                _record, product = future.result()
                products.append(product)
            except Exception as error:
                failures[key] = str(error)
            if progress_callback is not None:
                progress_callback(completed, len(records), key)
    if not products:
        details = "; ".join(f"{key}: {message}" for key, message in failures.items())
        raise ValueError(f"No all-event image could be built. {details}")
    first = products[0]
    return {
        **first,
        "hist": np.sum(
            [np.asarray(item["hist"], dtype=np.uint64) for item in products], axis=0
        ),
        "event_count": sum(int(item["event_count"]) for item in products),
        "low_kev": min(float(item["low_kev"]) for item in products),
        "high_kev": max(float(item["high_kev"]) for item in products),
        "source_keys": [
            key for item in products for key in item.get("source_keys", [])
        ],
        "cache_hit": all(bool(item.get("cache_hit")) for item in products),
        "failures": failures,
        "all_events": True,
    }


def exact_rgb(records, region, centers, widths, bins: int) -> dict:
    config = backend.RGBBandConfig(
        float(centers[0]), float(widths[0]),
        float(centers[1]), float(widths[1]),
        float(centers[2]), float(widths[2]),
    )
    return backend.exact_rgb_image(records, config, bins=bins, region=region)


def export_preview_frame(frame: pd.DataFrame, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path
