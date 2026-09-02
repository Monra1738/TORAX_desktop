"""Cas A reference bands, records, and reproducible product metadata."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .event_sources import remote_event_record
from .science_core import CACHE_DIR, DB_PATH, SkyRegion, safe_token

CAS_A_REGION = SkyRegion(
    350.8584,
    58.8113,
    0.20,
    label="Cas A",
    source="JAXA Theme 2 PDF",
)

CAS_A_BANDS = (
    (1.55, 1.75, "Low continuum"),
    (1.75, 1.95, "Si He alpha"),
    (2.35, 2.52, "S He alpha"),
    (3.93, 6.23, "High continuum"),
)


@dataclass(frozen=True)
class CasARecordSpec:
    mission: str
    instrument: str
    observation_id: str
    role: str

    @property
    def key(self) -> str:
        return (
            f"{safe_token(self.mission)}/{safe_token(self.instrument)}/"
            f"{self.observation_id}"
        )


# Holt et al. (PASJ 46, L151, 1994) used the two August 1993 Cas A pointings.
# Keep those four parquet products distinct from later mission comparisons so
# a raw-count image is never mislabeled as the restored literature figure.
CAS_A_RECORDS = (
    CasARecordSpec("asca", "gis", "50018000", "1993 paper observation"),
    CasARecordSpec("asca", "sis", "50018000", "1993 paper observation"),
    CasARecordSpec("asca", "gis", "50018010", "1993 paper observation"),
    CasARecordSpec("asca", "sis", "50018010", "1993 paper observation"),
    CasARecordSpec("suzaku", "xis", "507038010", "Suzaku comparison"),
    CasARecordSpec("xrism", "resolve", "000129000", "XRISM comparison"),
    CasARecordSpec("xrism", "resolve", "000130000", "XRISM comparison"),
)

CAS_A_XTEND_RECORDS = (
    CasARecordSpec("xrism", "xtend", "000129000", "XRISM wide-field"),
    CasARecordSpec("xrism", "xtend", "000130000", "XRISM wide-field"),
)


def cas_a_record_specs(include_xtend: bool = False) -> tuple[CasARecordSpec, ...]:
    return CAS_A_RECORDS + (CAS_A_XTEND_RECORDS if include_xtend else ())


def cas_a_records(
    cache_dir: Path | str = CACHE_DIR,
    include_xtend: bool = False,
):
    return [
        remote_event_record(
            spec.mission,
            spec.instrument,
            spec.observation_id,
            cache_dir=cache_dir,
        )
        for spec in cas_a_record_specs(include_xtend)
    ]


def cas_a_manifest(
    records: Iterable[CasARecordSpec] | None = None,
    *,
    image_bins: int,
    output_dir: Path | str,
    db_path: Path | str = DB_PATH,
    downloaded: bool = False,
) -> dict:
    selected = tuple(records or CAS_A_RECORDS)
    return {
        "target": "Cas A",
        "reference": "docs/8470b964-a30c-4a14-b994-56cb1beb73d6_2026_JAXA_Intern_Theme_2.pdf",
        "literature": {
            "citation": "Holt et al., PASJ 46, L151-L155 (1994)",
            "doi": "10.1093/pasj/46.4.L151",
            "asca_observation_ids": ["50018000", "50018010"],
            "published_product": "PSF-restored images at approximately 30 arcsec FWHM",
        },
        "region": {
            "ra_deg": CAS_A_REGION.center_ra_deg,
            "dec_deg": CAS_A_REGION.center_dec_deg,
            "radius_deg": CAS_A_REGION.radius_deg,
        },
        "bands": [
            {"low_kev": low, "high_kev": high, "label": label}
            for low, high, label in CAS_A_BANDS
        ],
        "records": [asdict(spec) for spec in selected],
        "image_bins": int(image_bins),
        "output_dir": str(Path(output_dir).resolve()),
        "db_path": str(Path(db_path).resolve()),
        "downloaded": bool(downloaded),
        "created_at": datetime.now(UTC).isoformat(),
    }
