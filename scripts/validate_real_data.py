#!/usr/bin/env python3
"""Validate cached DARTS parquet/header pairs without modifying source data.

Usage examples::

    python scripts/validate_real_data.py --data-root var --json var/exports/data_validation.json
    python scripts/validate_real_data.py --data-root var --require-all

The command is intentionally explicit and opt-in because it reads real parquet files and
may take minutes.  It validates every cached record, or fails if any of the eight supported
instrument families are absent when ``--require-all`` is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jaxa_udon3.infrastructure import event_sources, science

EXPECTED_PAIRS = {
    "asca/gis", "asca/sis", "suzaku/xis", "hitomi/sxs", "hitomi/sxi", "hitomi/hxi",
    "xrism/resolve", "xrism/xtend",
}


def validate_record(record) -> dict:
    frame, metadata = event_sources.read_events_with_sky(record)
    preview, total = science.read_preview_source(record, max_rows=min(1_000, max(1, len(frame))))
    if len(frame) != total:
        raise ValueError(f"{science.record_key(record)}: total event count mismatch")
    if not set(science.REQUIRED_COLUMNS) <= set(frame):
        raise ValueError(f"{science.record_key(record)}: missing required columns")
    if frame.empty:
        raise ValueError(f"{science.record_key(record)}: no valid events")
    for column in ("PI", "X", "Y", "RA", "DEC"):
        values = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{science.record_key(record)}: non-finite {column}")
    if np.any(frame["PI"].to_numpy(dtype=float) < 0.0):
        raise ValueError(f"{science.record_key(record)}: negative PI sentinel")
    if not np.all((frame["RA"] >= 0.0) & (frame["RA"] < 360.0)):
        raise ValueError(f"{science.record_key(record)}: RA outside [0, 360)")
    if not np.all((frame["DEC"] >= -90.0) & (frame["DEC"] <= 90.0)):
        raise ValueError(f"{science.record_key(record)}: DEC outside [-90, 90]")
    factor = science.pi_to_kev_factor(record)
    energy = preview["PI"].to_numpy(dtype=float) * factor
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError(f"{science.record_key(record)}: invalid calibrated energy")
    return {
        "record_key": science.record_key(record),
        "mission": record.mission,
        "instrument": record.instrument,
        "observation_id": record.observation_id,
        "rows": int(total),
        "preview_rows": len(preview),
        "pi_min": float(frame["PI"].min()),
        "pi_max": float(frame["PI"].max()),
        "energy_min_kev": float(energy.min()),
        "energy_max_kev": float(energy.max()),
        "ra_min": float(frame["RA"].min()),
        "ra_max": float(frame["RA"].max()),
        "dec_min": float(frame["DEC"].min()),
        "dec_max": float(frame["DEC"].max()),
        "wcs_keys": sorted(key for key in metadata if key.startswith("T")),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("var"))
    parser.add_argument("--json", type=Path, help="write a machine-readable validation manifest")
    parser.add_argument("--require-all", action="store_true", help="require all eight instrument pairs")
    args = parser.parse_args(argv)
    records = event_sources.discover_event_files(args.data_root / "data")
    if not records:
        records = event_sources.discover_event_files(args.data_root / "data_cache")
    if not records:
        raise SystemExit("No cached parquet/header pairs found")
    results = []
    failures = []
    for record in records:
        try:
            results.append(validate_record(record))
        except Exception as error:  # isolate one corrupt observation from the report
            failures.append({"record_key": science.record_key(record), "error": str(error)})
    pairs = {f"{item['mission']}/{item['instrument']}" for item in results}
    missing_pairs = sorted(EXPECTED_PAIRS - pairs)
    report = {"records": results, "failures": failures, "missing_pairs": missing_pairs}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures or (args.require_all and missing_pairs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
