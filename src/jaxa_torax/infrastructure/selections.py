"""Search, indexing, loading, and export helpers for the PyVista prototype."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

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


def _normalized_target_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def canonical_target_name(name: str) -> str:
    """Keep common target aliases stable in the search field and saved workspace."""
    text = " ".join(str(name or "").strip().split())
    aliases = {
        "cas": "Cas A",
        "cas a": "Cas A",
        "casa": "Cas A",
        "cassiopeia a": "Cas A",
    }
    return aliases.get(_normalized_target_name(text), text)


_RA_DEC_LABEL = re.compile(
    r"^\s*ra\s*(?:=|:)?\s*"
    r"(?P<ra>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:deg|°)?\s*,?\s*"
    r"dec\s*(?:=|:)?\s*"
    r"(?P<dec>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:deg|°)?\s*$",
    re.IGNORECASE,
)


def _coordinates_from_label(value: str) -> tuple[float, float] | None:
    """Recognize the coordinate label displayed by the desktop search UI.

    A completed degree search is shown as ``RA …, DEC …`` in the top bar.  It
    must be accepted as coordinates when the user searches again, rather than
    being sent to Sesame as an object name.
    """
    match = _RA_DEC_LABEL.match(str(value or ""))
    if match is None:
        return None
    return float(match.group("ra")), float(match.group("dec"))


@serialized_database_access
def _cached_target_coordinates(normalized: str, db_path: Path | str):
    con = duckdb.connect(str(db_path))
    try:
        return con.execute(
            """
            SELECT ra_deg, dec_deg FROM resolved_targets
            WHERE normalized_name = ? ORDER BY resolved_at DESC LIMIT 1
            """,
            [normalized],
        ).fetchone()
    finally:
        con.close()


@serialized_database_access
def _store_target_coordinates(
    normalized: str,
    target_name: str,
    ra_deg: float,
    dec_deg: float,
    db_path: Path | str,
):
    con = duckdb.connect(str(db_path))
    try:
        con.execute("DELETE FROM resolved_targets WHERE normalized_name = ?", [normalized])
        con.execute(
            "INSERT INTO resolved_targets VALUES (?, ?, ?, ?, ?, ?)",
            [normalized, target_name, ra_deg, dec_deg, "Sesame", utc_now_text()],
        )
    finally:
        con.close()


def resolve_target_name(
    name: str,
    db_path: Path | str = DB_PATH,
    resolver=None,
) -> tuple[float, float, bool]:
    """Resolve a target through Sesame and persist successful results."""
    require_astropy()
    require_duckdb()
    target_name = canonical_target_name(name)
    normalized = _normalized_target_name(target_name)
    if not normalized:
        raise ValueError("Enter a target name")
    ensure_storage_schema(db_path)
    row = _cached_target_coordinates(normalized, db_path)
    if row is not None:
        return float(row[0]) % 360.0, float(row[1]), True

    coordinate = (
        resolver(target_name)
        if resolver is not None
        else SkyCoord.from_name(target_name, parse=True)
    )
    ra_deg = float(coordinate.icrs.ra.deg) % 360.0
    dec_deg = float(coordinate.icrs.dec.deg)
    _store_target_coordinates(normalized, target_name, ra_deg, dec_deg, db_path)
    return ra_deg, dec_deg, False


def parse_sky_region(
    mode: str,
    ra_value,
    dec_value,
    radius_deg,
    target_name: str = "",
    db_path: Path | str = DB_PATH,
    resolver=None,
) -> tuple[SkyRegion, bool]:
    """Parse degree, sexagesimal, or Sesame target input into one ICRS region."""
    require_astropy()
    mode = str(mode or "degrees").strip().lower()
    cache_hit = False
    if mode == "target":
        displayed_coordinates = _coordinates_from_label(target_name)
        if displayed_coordinates is None:
            ra_deg, dec_deg, cache_hit = resolve_target_name(
                target_name, db_path=db_path, resolver=resolver
            )
            label = canonical_target_name(target_name)
        else:
            ra_deg, dec_deg = displayed_coordinates
            label = f"RA {ra_deg:.6f}, DEC {dec_deg:.6f}"
            mode = "degrees"
    elif mode == "sexagesimal":
        try:
            coordinate = SkyCoord(
                str(ra_value).strip(),
                str(dec_value).strip(),
                unit=(u.hourangle, u.deg),
                frame="icrs",
            )
        except Exception as error:
            raise ValueError(
                "Use RA as HH:MM:SS.s and DEC as signed DD:MM:SS.s"
            ) from error
        ra_deg = float(coordinate.ra.deg)
        dec_deg = float(coordinate.dec.deg)
        label = f"{ra_value} {dec_value}"
    elif mode == "degrees":
        try:
            ra_deg = float(ra_value)
            dec_deg = float(dec_value)
        except (TypeError, ValueError) as error:
            raise ValueError("RA, DEC, and radius must be numeric") from error
        label = f"RA {ra_deg:.6f}, DEC {dec_deg:.6f}"
    else:
        raise ValueError(f"Unknown coordinate mode: {mode}")
    try:
        radius = float(radius_deg)
    except (TypeError, ValueError) as error:
        raise ValueError("Radius must be numeric") from error
    return SkyRegion(ra_deg, dec_deg, radius, label=label, source=mode), cache_hit


def angular_separation_deg(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    center_ra_deg: float,
    center_dec_deg: float,
) -> np.ndarray:
    """Vectorized great-circle separation, stable at RA wrap and the poles."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    center_ra = np.deg2rad(float(center_ra_deg))
    center_dec = np.deg2rad(float(center_dec_deg))
    cosine = (
        np.sin(dec) * np.sin(center_dec)
        + np.cos(dec) * np.cos(center_dec) * np.cos(ra - center_ra)
    )
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def parse_sky_rectangle(
    ra_min_deg,
    ra_max_deg,
    dec_min_deg,
    dec_max_deg,
) -> SkyRectangle:
    try:
        values = [
            float(ra_min_deg), float(ra_max_deg),
            float(dec_min_deg), float(dec_max_deg),
        ]
    except (TypeError, ValueError) as error:
        raise ValueError("Rectangle RA and DEC bounds must be numeric") from error
    return SkyRectangle(
        *values,
        label=(
            f"RA {values[0]:.6f}-{values[1]:.6f} deg | "
            f"DEC {values[2]:.6f}-{values[3]:.6f} deg"
        ),
    )


def selection_contains(
    selection: SkySelection,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> np.ndarray:
    ra = np.mod(np.asarray(ra_deg, dtype=float), 360.0)
    dec = np.asarray(dec_deg, dtype=float)
    if isinstance(selection, SkyRegion):
        return angular_separation_deg(
            ra, dec, selection.center_ra_deg, selection.center_dec_deg
        ) <= selection.radius_deg
    dec_valid = (dec >= selection.dec_min_deg) & (dec <= selection.dec_max_deg)
    if selection.crosses_ra_zero:
        ra_valid = (ra >= selection.ra_min_deg) | (ra <= selection.ra_max_deg)
    else:
        ra_valid = (ra >= selection.ra_min_deg) & (ra <= selection.ra_max_deg)
    return dec_valid & ra_valid


def selection_world_points(selection: SkySelection) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(selection, SkyRegion):
        center = SkyCoord(selection.center_ra_deg * u.deg, selection.center_dec_deg * u.deg)
        angles = np.linspace(0.0, 360.0, 73) * u.deg
        ring = center.directional_offset_by(angles, selection.radius_deg * u.deg)
        return (
            np.concatenate(([selection.center_ra_deg], ring.ra.deg)),
            np.concatenate(([selection.center_dec_deg], ring.dec.deg)),
        )
    return (
        np.asarray(
            [selection.ra_min_deg, selection.ra_min_deg, selection.ra_max_deg,
             selection.ra_max_deg, selection.center_ra_deg],
            dtype=float,
        ),
        np.asarray(
            [selection.dec_min_deg, selection.dec_max_deg, selection.dec_min_deg,
             selection.dec_max_deg, selection.center_dec_deg],
            dtype=float,
        ),
    )


def selection_image_range(selection: SkySelection) -> tuple[tuple[float, float], tuple[float, float]]:
    if isinstance(selection, SkyRegion):
        cosine = max(abs(np.cos(np.deg2rad(selection.center_dec_deg))), 0.01)
        ra_radius = min(selection.radius_deg / cosine, 180.0)
        return (
            (selection.center_ra_deg - ra_radius, selection.center_ra_deg + ra_radius),
            (selection.center_dec_deg - selection.radius_deg, selection.center_dec_deg + selection.radius_deg),
        )
    ra_max = selection.ra_max_deg + (360.0 if selection.crosses_ra_zero else 0.0)
    return (
        (selection.ra_min_deg, ra_max),
        (selection.dec_min_deg, selection.dec_max_deg),
    )


def unwrap_ra_for_selection(ra_deg: np.ndarray, selection: SkySelection) -> np.ndarray:
    ra = np.mod(np.asarray(ra_deg, dtype=float), 360.0)
    if isinstance(selection, SkyRectangle) and selection.crosses_ra_zero:
        return np.where(ra < selection.ra_min_deg, ra + 360.0, ra)
    center = selection.center_ra_deg
    return center + ((ra - center + 180.0) % 360.0 - 180.0)


def selection_catalog_radius_deg(selection: SkySelection) -> float:
    if isinstance(selection, SkyRegion):
        return selection.radius_deg
    center = SkyCoord(selection.center_ra_deg * u.deg, selection.center_dec_deg * u.deg)
    ra, dec = selection_world_points(selection)
    corners = SkyCoord(ra * u.deg, dec * u.deg)
    return float(np.max(center.separation(corners).deg))
