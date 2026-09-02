"""Framework-independent models and validation rules."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isclose
from pathlib import Path
from typing import Any, Literal


def stable_key(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EventFile:
    mission: str
    instrument: str
    observation_id: str
    parquet_path: Path
    header_path: Path
    parquet_url: str | None = None
    header_url: str | None = None
    source: str = "local"


@dataclass
class LoadedObservation:
    record: EventFile
    frame: Any
    metadata: dict
    total_events: int
    events_in_region: int
    displayed_events: int
    minimum_separation_deg: float | None = None


@dataclass
class SearchResult:
    observations: list[LoadedObservation]
    summary: Any
    skipped: list[str]
    center_ra: float
    center_dec: float
    radius_deg: float | None


@dataclass(frozen=True)
class SkyRegion:
    center_ra_deg: float
    center_dec_deg: float
    radius_deg: float
    label: str = ""
    source: str = "degrees"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_ra_deg", float(self.center_ra_deg) % 360.0)
        if not -90.0 <= float(self.center_dec_deg) <= 90.0:
            raise ValueError("Declination must be between -90 and 90 degrees")
        if not 0.0 < float(self.radius_deg) <= 180.0:
            raise ValueError("Radius must be greater than 0 and at most 180 degrees")

    def signature(self) -> dict:
        return {
            "type": "circle",
            "ra": round(float(self.center_ra_deg), 8),
            "dec": round(float(self.center_dec_deg), 8),
            "radius": round(float(self.radius_deg), 8),
        }


@dataclass(frozen=True)
class SkyRectangle:
    """Independent RA/DEC bounds; RA minimum greater than maximum crosses zero."""

    ra_min_deg: float
    ra_max_deg: float
    dec_min_deg: float
    dec_max_deg: float
    label: str = ""
    source: str = "rectangle"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ra_min_deg", float(self.ra_min_deg) % 360.0)
        object.__setattr__(self, "ra_max_deg", float(self.ra_max_deg) % 360.0)
        if not -90.0 <= float(self.dec_min_deg) < float(self.dec_max_deg) <= 90.0:
            raise ValueError(
                "Declination minimum must be below maximum within -90 to 90 degrees"
            )
        if isclose(float(self.ra_min_deg), float(self.ra_max_deg)):
            raise ValueError("RA minimum and maximum must differ")

    @property
    def crosses_ra_zero(self) -> bool:
        return self.ra_min_deg > self.ra_max_deg

    @property
    def center_ra_deg(self) -> float:
        span = (self.ra_max_deg - self.ra_min_deg) % 360.0
        return (self.ra_min_deg + span / 2.0) % 360.0

    @property
    def center_dec_deg(self) -> float:
        return (self.dec_min_deg + self.dec_max_deg) / 2.0

    def signature(self) -> dict:
        return {
            "type": "rectangle",
            "ra_min": round(float(self.ra_min_deg), 8),
            "ra_max": round(float(self.ra_max_deg), 8),
            "dec_min": round(float(self.dec_min_deg), 8),
            "dec_max": round(float(self.dec_max_deg), 8),
        }


ScienceSelection = SkyRegion | SkyRectangle


@dataclass(frozen=True)
class RGBBandConfig:
    red_center_kev: float
    red_width_kev: float
    green_center_kev: float
    green_width_kev: float
    blue_center_kev: float
    blue_width_kev: float

    def bands(self) -> tuple[tuple[float, float], ...]:
        values = []
        for center, width in (
            (self.red_center_kev, self.red_width_kev),
            (self.green_center_kev, self.green_width_kev),
            (self.blue_center_kev, self.blue_width_kev),
        ):
            if float(width) <= 0:
                raise ValueError("RGB band widths must be greater than zero")
            values.append((float(center) - float(width) / 2, float(center) + float(width) / 2))
        return tuple(values)


@dataclass(frozen=True)
class SmoothingConfig:
    spatial_sigma_arcmin: float = 0.0
    energy_sigma_kev: float = 0.0
    size_strength: float = 0.0
    opacity_strength: float = 0.0


@dataclass(frozen=True)
class CircleSelection:
    ra_deg: float
    dec_deg: float
    radius_deg: float

    def __post_init__(self) -> None:
        if not -90 <= self.dec_deg <= 90 or not 0 < self.radius_deg <= 180:
            raise ValueError("Invalid circle selection")

    def key(self) -> str:
        return stable_key({"kind": "circle", **asdict(self), "ra_deg": self.ra_deg % 360})


@dataclass(frozen=True)
class RectangleSelection:
    ra_min_deg: float
    ra_max_deg: float
    dec_min_deg: float
    dec_max_deg: float

    def __post_init__(self) -> None:
        if not -90 <= self.dec_min_deg < self.dec_max_deg <= 90:
            raise ValueError("Invalid rectangle DEC bounds")

    @property
    def crosses_ra_zero(self) -> bool:
        return self.ra_min_deg % 360 > self.ra_max_deg % 360

    def key(self) -> str:
        return stable_key({"kind": "rectangle", **asdict(self)})


SkySelection = CircleSelection | RectangleSelection


@dataclass(frozen=True)
class EnergyFilter:
    minimum_kev: float
    maximum_kev: float

    def __post_init__(self) -> None:
        if self.minimum_kev >= self.maximum_kev:
            raise ValueError("Energy minimum must be lower than maximum")


@dataclass(frozen=True)
class EnergyBand:
    center_kev: float
    width_kev: float
    enabled: bool = True
    color: str = "#d71920"
    opacity: float = 0.18

    def __post_init__(self) -> None:
        if self.width_kev <= 0 or not 0 <= self.opacity <= 1:
            raise ValueError("Invalid energy band")


@dataclass(frozen=True)
class RGBBands:
    red: EnergyBand
    green: EnergyBand
    blue: EnergyBand
    brightness: float = 1.25
    gamma: float = 1.0


@dataclass(frozen=True)
class CatalogQuery:
    selected_pairs: tuple[str, ...] = ()
    object_text: str = ""
    observation_text: str = ""
    filename_text: str = ""
    date_start: str = ""
    date_end: str = ""
    selection: SkySelection | None = None
    limit: int = 250


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


@dataclass(frozen=True)
class JobRequest:
    kind: str
    payload: dict
    workspace_id: str | None = None
    dedupe_key: str | None = None

    def key(self) -> str:
        return self.dedupe_key or stable_key({"kind": self.kind, "payload": self.payload})


@dataclass(frozen=True)
class JobRecord:
    id: str
    kind: str
    status: JobStatus
    payload: dict
    workspace_id: str | None
    dedupe_key: str
    progress_current: int = 0
    progress_total: int = 0
    progress_message: str = ""
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False
