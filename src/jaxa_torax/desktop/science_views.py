"""Fast preview products and color transforms for the native desktop front-end.

Exact products still come from ``jaxa_torax.infrastructure.science`` so the desktop
application shares the same calibrated/cache-backed science implementation as Theme 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from jaxa_torax.desktop.theme import ENERGY_COLORS, MISSION_COLORS

# Scalar image palettes are deliberately separate from the scientific RGB and
# 3D event-colour systems.  Their first anchor is visible (not black): exact
# zero-count pixels are masked to black after lookup instead.
SCALAR_PALETTES: dict[str, tuple[str, ...]] = {
    "none": ("#000000", "#ffffff"),
    "gray": ("#1c1c1c", "#ffffff"),
    "inferno": ("#250f28", "#7a1f5c", "#d94b3d", "#f6a13a", "#fcfdbf"),
    "viridis": ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"),
    "hot": ("#4a0000", "#b30000", "#ffcc00", "#ffffff"),
    "rainbow": ("#6a00a8", "#0074d9", "#00a878", "#f5c518", "#d62828"),
    "magma": ("#1b0c41", "#721f81", "#f1605d", "#fcfdbf"),
    "plasma": ("#0d0887", "#9c179e", "#ed7953", "#f0f921"),
}
SCALAR_PALETTE_LABELS = {
    "none": "No colormap (grayscale)",
    "gray": "Grayscale",
    "inferno": "Inferno",
    "viridis": "Viridis",
    "hot": "Hot",
    "rainbow": "Rainbow",
    "magma": "Magma",
    "plasma": "Plasma",
}
SCALAR_STRETCHES = ("linear", "sqrt", "log")
LOG_STRETCH_K = 100.0
SPECTRUM_SCALES = ("linear", "log_y", "log_log")


@dataclass(frozen=True)
class EnergySceneTransform:
    """Reversible display transform for the mixed sky/energy 3D scene."""

    center_ra_deg: float
    center_dec_deg: float
    reference_kev: float
    energy_scale: float = 1.0
    absolute_coordinates: bool = False

    def coordinates(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.empty((0, 3), dtype=np.float32)
        ra = wrapped_ra(frame["RA"].to_numpy(float), self.center_ra_deg)
        dec = frame["DEC"].to_numpy(float)
        energy = frame["KEV"].to_numpy(float)
        if self.absolute_coordinates:
            # Preserve the same absolute precision as the 2D image coordinates;
            # converting RA/DEC near 350° to float32 visibly rounds tick-aligned
            # event positions in the 3D view.
            return np.column_stack((ra, dec, energy)).astype(np.float64, copy=False)
        cosine = max(abs(np.cos(np.deg2rad(self.center_dec_deg))), 0.02)
        x = -(ra - self.center_ra_deg) * cosine * 60.0
        y = (dec - self.center_dec_deg) * 60.0
        z = (energy - self.reference_kev) * float(self.energy_scale)
        return np.column_stack((x, y, z)).astype(np.float32, copy=False)

    def energy_to_scene(self, energy):
        if self.absolute_coordinates:
            return np.asarray(energy, dtype=float)
        return (np.asarray(energy, dtype=float) - self.reference_kev) * self.energy_scale


def normalize_spectrum_scale(value: object) -> str:
    value = str(value or "linear").lower()
    return value if value in SPECTRUM_SCALES else "linear"


def energy_to_plot_x(energy, scale: str = "linear"):
    """Map canonical keV values to PyQtGraph view coordinates."""
    values = np.asarray(energy, dtype=float)
    if normalize_spectrum_scale(scale) == "log_log":
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(values > 0.0, np.log10(values), np.nan)
    else:
        result = values
    return float(result) if result.ndim == 0 else result


def plot_x_to_energy(plot_x, scale: str = "linear"):
    values = np.asarray(plot_x, dtype=float)
    result = np.power(10.0, values) if normalize_spectrum_scale(scale) == "log_log" else values
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class SkyViewport:
    """One continuous, unwrapped RA/DEC workspace viewport."""

    center_ra_deg: float
    center_dec_deg: float
    radius_deg: float
    ra_min_deg: float
    ra_max_deg: float
    dec_min_deg: float
    dec_max_deg: float

    @classmethod
    def from_region(cls, region):
        if region is None:
            return None
        if hasattr(region, "radius_deg"):
            center_ra = float(region.center_ra_deg) % 360.0
            center_dec = float(region.center_dec_deg)
            radius = float(region.radius_deg)
            cosine = max(abs(np.cos(np.deg2rad(center_dec))), 0.02)
            half_ra = min(radius / cosine, 180.0)
            return cls(
                center_ra, center_dec, radius,
                center_ra - half_ra, center_ra + half_ra,
                max(-90.0, center_dec - radius), min(90.0, center_dec + radius),
            )
        a = float(region.ra_min_deg)
        b = float(region.ra_max_deg)
        if a > b:
            b += 360.0
        center_ra = 0.5 * (a + b)
        center_dec = 0.5 * (float(region.dec_min_deg) + float(region.dec_max_deg))
        return cls(
            center_ra, center_dec, max(0.5 * (b - a), 1e-12), a, b,
            float(region.dec_min_deg), float(region.dec_max_deg),
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.ra_min_deg, self.ra_max_deg, self.dec_min_deg, self.dec_max_deg

    @property
    def x_range(self) -> tuple[float, float]:
        return self.ra_min_deg, self.ra_max_deg

    @property
    def y_range(self) -> tuple[float, float]:
        return self.dec_min_deg, self.dec_max_deg

    def normalize_ra(self, value: float) -> float:
        return float(value) % 360.0

    def unwrap_ra(self, value: float) -> float:
        return float(self.center_ra_deg) + ((float(value) - float(self.center_ra_deg) + 180.0) % 360.0 - 180.0)

    def clamp_rectangle(self, rectangle, *, minimum_size: float = 1e-8):
        if rectangle is None:
            return None
        a, b, c, d = map(float, rectangle)
        a, b = sorted((self.unwrap_ra(a), self.unwrap_ra(b)))
        c, d = sorted((c, d))
        width = min(max(b - a, minimum_size), self.ra_max_deg - self.ra_min_deg)
        height = min(max(d - c, minimum_size), self.dec_max_deg - self.dec_min_deg)
        a = min(max(a, self.ra_min_deg), self.ra_max_deg - width)
        c = min(max(c, self.dec_min_deg), self.dec_max_deg - height)
        return a, a + width, c, c + height

    def clamp_view(self, view):
        if view is None:
            return self.bounds
        return self.clamp_rectangle(view)

    def to_payload(self) -> dict:
        return {
            "center_ra_deg": self.center_ra_deg,
            "center_dec_deg": self.center_dec_deg,
            "radius_deg": self.radius_deg,
            "ra_min_deg": self.ra_min_deg,
            "ra_max_deg": self.ra_max_deg,
            "dec_min_deg": self.dec_min_deg,
            "dec_max_deg": self.dec_max_deg,
        }


@dataclass(frozen=True)
class ImageProduct:
    values: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    count: int
    low_kev: float | None = None
    high_kev: float | None = None
    exact: bool = False
    cache_hit: bool = False


@dataclass(frozen=True)
class SpectrumProduct:
    x: np.ndarray
    counts: np.ndarray
    edges: np.ndarray
    smoothed_counts: np.ndarray | None = None


def combine_frames(observations) -> pd.DataFrame:
    parts = []
    for item in observations:
        frame = getattr(item, "frame", None)
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        record = item.record
        local["MISSION"] = str(record.mission)
        local["INSTRUMENT"] = str(record.instrument)
        local["OBSERVATION_ID"] = str(record.observation_id)
        local["RECORD_KEY"] = f"{record.mission}/{record.instrument}/{record.observation_id}"
        parts.append(local)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    # Repeated string labels dominate memory when many observations are loaded.
    # Categorical columns preserve their values while using compact integer codes.
    for column in ("MISSION", "INSTRUMENT", "OBSERVATION_ID", "RECORD_KEY"):
        if column in combined:
            combined[column] = combined[column].astype("category")
    return combined


def wrapped_ra(values: np.ndarray, center_ra: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return float(center_ra) + ((values - float(center_ra) + 180.0) % 360.0 - 180.0)


def sky_image_range(region) -> tuple[tuple[float, float], tuple[float, float]] | None:
    viewport = SkyViewport.from_region(region)
    return None if viewport is None else (viewport.x_range, viewport.y_range)


def filter_energy(frame: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    if frame.empty or "KEV" not in frame:
        return frame.iloc[0:0]
    mask = frame["KEV"].between(float(low), float(high), inclusive="both")
    return frame.loc[mask]


def filter_rectangle(
    frame: pd.DataFrame,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> pd.DataFrame:
    if frame.empty or "RA" not in frame or "DEC" not in frame:
        return frame.iloc[0:0]
    ra = np.mod(frame["RA"].to_numpy(float), 360.0)
    dec = frame["DEC"].to_numpy(float)
    a = float(ra_min) % 360.0
    b = float(ra_max) % 360.0
    full_ra = abs(float(ra_max) - float(ra_min)) >= 360.0 - 1e-9
    if full_ra:
        ra_mask = np.ones(len(frame), dtype=bool)
    elif a <= b:
        ra_mask = (ra >= a) & (ra <= b)
    else:
        ra_mask = (ra >= a) | (ra <= b)
    mask = ra_mask & (dec >= float(dec_min)) & (dec <= float(dec_max))
    return frame.loc[mask]


def spectrum(
    frame: pd.DataFrame,
    bins: int = 240,
    low: float | None = None,
    high: float | None = None,
    smoothing_sigma_bins: float = 0.0,
) -> SpectrumProduct:
    bins = max(16, int(bins))
    if frame.empty or "KEV" not in frame:
        edges = np.linspace(0.0, 12.0, bins + 1)
        counts = np.zeros(bins)
        smoothed = counts.copy() if smoothing_sigma_bins > 0 else None
        return SpectrumProduct(0.5 * (edges[:-1] + edges[1:]), counts, edges, smoothed)
    energy = frame["KEV"].to_numpy(float)
    energy = energy[np.isfinite(energy)]
    if energy.size == 0:
        return spectrum(pd.DataFrame(), bins=bins, low=low, high=high, smoothing_sigma_bins=smoothing_sigma_bins)
    lo = float(np.nanmin(energy) if low is None else low)
    hi = float(np.nanmax(energy) if high is None else high)
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(energy, bins=bins, range=(lo, hi))
    counts = counts.astype(float)
    smoothed = None
    if smoothing_sigma_bins > 0:
        smoothed = gaussian_filter(counts, sigma=float(smoothing_sigma_bins), mode="nearest")
    return SpectrumProduct(0.5 * (edges[:-1] + edges[1:]), counts, edges, smoothed)


def energy_image(
    frame: pd.DataFrame,
    low: float,
    high: float,
    bins: int = 240,
    center_ra: float | None = None,
    region=None,
    smoothing_sigma_pixels: float = 0.0,
) -> ImageProduct:
    bins = max(16, int(bins))
    selected = filter_energy(frame, low, high)
    image_range = sky_image_range(region)
    if selected.empty:
        if image_range is None:
            x_edges = np.linspace(0.0, 1.0, bins + 1)
            y_edges = np.linspace(0.0, 1.0, bins + 1)
        else:
            x_edges = np.linspace(*image_range[0], bins + 1)
            y_edges = np.linspace(*image_range[1], bins + 1)
        return ImageProduct(np.zeros((bins, bins)), x_edges, y_edges, 0, float(low), float(high))

    ra = selected["RA"].to_numpy(float)
    dec = selected["DEC"].to_numpy(float)
    if center_ra is not None:
        ra = wrapped_ra(ra, float(center_ra))
    if image_range is None:
        hist, x_edges, y_edges = np.histogram2d(ra, dec, bins=bins)
    else:
        hist, x_edges, y_edges = np.histogram2d(ra, dec, bins=bins, range=image_range)
    image = hist.T.astype(float)
    if smoothing_sigma_pixels > 0:
        image = gaussian_filter(image, sigma=float(smoothing_sigma_pixels), mode="nearest")
    return ImageProduct(image, x_edges, y_edges, len(selected), float(low), float(high))


def _normalize_channel(values: np.ndarray, gain: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    scale = float(np.percentile(positive, 99.5)) if positive.size else 0.0
    if scale > 0:
        values = values / scale
    return np.clip(values * float(gain), 0.0, 1.0)


def rgb_from_channels(
    channels: np.ndarray,
    gains: Sequence[float] = (1.0, 1.0, 1.0),
    brightness: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    values = np.asarray(channels, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("RGB channels must have shape (ny, nx, 3)")
    rgb = np.stack([_normalize_channel(values[..., i], gains[i]) for i in range(3)], axis=-1)
    rgb *= max(0.0, float(brightness))
    gamma = max(float(gamma), 0.05)
    return np.power(np.clip(rgb, 0.0, 1.0), 1.0 / gamma)


def rgb_image(
    frame: pd.DataFrame,
    bands: Sequence[tuple[float, float]],
    bins: int = 240,
    center_ra: float | None = None,
    region=None,
    smoothing_sigma_pixels: float = 0.0,
    gains: Sequence[float] = (1.0, 1.0, 1.0),
    brightness: float = 1.0,
    gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    products = [
        energy_image(
            frame,
            low,
            high,
            bins=bins,
            center_ra=center_ra,
            region=region,
            smoothing_sigma_pixels=smoothing_sigma_pixels,
        )
        for low, high in bands
    ]
    channels = np.stack([product.values for product in products], axis=-1)
    rgb = rgb_from_channels(channels, gains=gains, brightness=brightness, gamma=gamma)
    return rgb, products[0].x_edges, products[0].y_edges, [p.count for p in products]


def _hex_to_rgb(value: str) -> np.ndarray:
    text = str(value).lstrip("#")
    return np.asarray([int(text[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


def energy_gradient_colors(energy: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    """Map energy to the required red(low) -> blue(high) scientific color ramp."""
    values = np.asarray(energy, dtype=float)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    finite = np.isfinite(values)
    lo = float(np.nanmin(values[finite]) if low is None and np.any(finite) else (low or 0.0))
    hi = float(np.nanmax(values[finite]) if high is None and np.any(finite) else (high or lo + 1.0))
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    anchors = np.stack([_hex_to_rgb(c) for c in ENERGY_COLORS])
    positions = np.linspace(0.0, 1.0, len(anchors))
    rgb = np.column_stack([np.interp(t, positions, anchors[:, channel]) for channel in range(3)])
    rgb[~finite] = 140.0
    return np.asarray(np.round(rgb), dtype=np.uint8)


def mission_event_colors(missions: Sequence[str]) -> np.ndarray:
    rows = []
    for mission in missions:
        rows.append(_hex_to_rgb(MISSION_COLORS.get(str(mission).lower(), "#9aa9b8")))
    return np.asarray(np.round(rows), dtype=np.uint8)


def rgb_event_colors(
    energy: np.ndarray,
    centers: Sequence[float],
    widths: Sequence[float],
    gains: Sequence[float] = (1.0, 1.0, 1.0),
    brightness: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    channels = []
    for center, width, gain in zip(centers, widths, gains):
        sigma = max(float(width) / 2.355, 1e-6)
        channels.append(np.exp(-0.5 * ((energy - float(center)) / sigma) ** 2) * float(gain))
    rgb = np.stack(channels, axis=1)
    maximum = np.max(rgb, axis=1, keepdims=True)
    maximum[maximum <= 0] = 1.0
    rgb /= maximum
    rgb *= max(0.0, float(brightness))
    gamma = max(float(gamma), 0.05)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / gamma)
    return np.asarray(np.round(rgb * 255.0), dtype=np.uint8)


def point_density(frame: pd.DataFrame, bins: tuple[int, int, int] = (48, 48, 48)) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=float)
    values = np.column_stack(
        [frame["RA"].to_numpy(float), frame["DEC"].to_numpy(float), frame["KEV"].to_numpy(float)]
    )
    finite = np.all(np.isfinite(values), axis=1)
    density = np.zeros(len(frame), dtype=float)
    if not np.any(finite):
        return density
    valid = values[finite]
    hist, edges = np.histogramdd(valid, bins=bins)
    indices = []
    for axis in range(3):
        idx = np.searchsorted(edges[axis], valid[:, axis], side="right") - 1
        idx = np.clip(idx, 0, hist.shape[axis] - 1)
        indices.append(idx)
    vals = hist[indices[0], indices[1], indices[2]]
    if vals.size and np.max(vals) > 0:
        vals = np.log1p(vals) / np.log1p(np.max(vals))
    density[finite] = vals
    return density


def local_spectrum_points(
    frame: pd.DataFrame,
    center_ra: float,
    center_dec: float,
    spatial_bin_arcmin: float = 1.0,
    max_points: int = 20_000,
) -> pd.DataFrame:
    """Summarize each spatial cell by event count and mean energy.

    The result is intended for a continuous spectral-field overlay: neighboring
    cells remain spatially ordered while color encodes the local mean energy and
    size encodes the local spectrum signal.
    """
    columns = ["RA", "DEC", "KEV", "COUNT", "MEAN_KEV", "ENERGY_STD"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    ra = wrapped_ra(frame["RA"].to_numpy(float), center_ra)
    dec = frame["DEC"].to_numpy(float)
    energy = frame["KEV"].to_numpy(float)
    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(energy)
    if not np.any(finite):
        return pd.DataFrame(columns=columns)
    cos_dec = max(abs(np.cos(np.deg2rad(float(center_dec)))), 0.02)
    x = -(ra[finite] - float(center_ra)) * cos_dec * 60.0
    y = (dec[finite] - float(center_dec)) * 60.0
    step = max(float(spatial_bin_arcmin), 0.05)
    ix = np.floor(x / step).astype(np.int64)
    iy = np.floor(y / step).astype(np.int64)
    local = pd.DataFrame({"x": x, "y": y, "energy": energy[finite], "ix": ix, "iy": iy})
    grouped = local.groupby(["ix", "iy"], sort=False)
    result = grouped["energy"].agg(COUNT="size", MEAN_KEV="mean", ENERGY_STD="std").reset_index()
    result["ENERGY_STD"] = result["ENERGY_STD"].fillna(0.0)
    result["RA"] = float(center_ra) - (
        (result["ix"].to_numpy(float) + 0.5) * step / cos_dec / 60.0
    )
    result["DEC"] = float(center_dec) + (
        (result["iy"].to_numpy(float) + 0.5) * step / 60.0
    )
    result = result.sort_values(["ix", "iy"]).drop(columns=["ix", "iy"])
    if len(result) > int(max_points):
        result = result.nlargest(int(max_points), "COUNT").sort_values(["RA", "DEC"])
    return result[["RA", "DEC", "COUNT", "MEAN_KEV", "ENERGY_STD"]]


def voxel_histogram(
    frame: pd.DataFrame,
    center_ra: float,
    center_dec: float,
    spatial_voxel_arcmin: float,
    energy_voxel_kev: float,
    smooth_spatial: float = 0.0,
    smooth_energy: float = 0.0,
    max_cells: int = 1_500_000,
):
    if frame.empty:
        return None
    ra = wrapped_ra(frame["RA"].to_numpy(float), center_ra)
    dec = frame["DEC"].to_numpy(float)
    energy = frame["KEV"].to_numpy(float)
    cos_dec = max(abs(np.cos(np.deg2rad(float(center_dec)))), 0.02)
    x = ra
    y = dec
    z = energy
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(finite):
        return None
    x, y, z = x[finite], y[finite], z[finite]
    sx = max(float(spatial_voxel_arcmin) / (60.0 * cos_dec), 1.0e-6)
    sy = max(float(spatial_voxel_arcmin) / 60.0, 1.0e-6)
    # Resolve's useful resolution is a few eV.  Keep the numerical guard at
    # 5 eV (0.005 keV), matching the native Inspector control.
    sz = max(float(energy_voxel_kev), 0.005)
    ranges = [
        (float(np.nanmin(x)), float(np.nanmax(x))),
        (float(np.nanmin(y)), float(np.nanmax(y))),
        (float(np.nanmin(z)), float(np.nanmax(z))),
    ]
    bins = []
    for (lo, hi), step in zip(ranges, (sx, sy, sz)):
        if hi <= lo:
            hi = lo + step
        bins.append(max(1, min(220, int(np.ceil((hi - lo) / step)))))
    # Keep the dense histogram bounded. Increase voxel size automatically rather
    # than allocating a huge cube that would freeze the desktop application.
    cells = int(np.prod(bins))
    if cells > max(1, int(max_cells)):
        scale = (cells / float(max_cells)) ** (1.0 / 3.0)
        bins = [max(1, int(value / scale)) for value in bins]
    hist, edges = np.histogramdd(np.column_stack([x, y, z]), bins=tuple(bins), range=ranges)
    if smooth_spatial > 0 or smooth_energy > 0:
        sigma = (
            # Spatial histogram axes are in degrees; the Inspector controls
            # are in arcminutes.  Missing this conversion creates a kernel
            # 60x too wide after switching the 3D scene to absolute RA/DEC.
            max(0.0, float(smooth_spatial) / 60.0 / sx),
            max(0.0, float(smooth_spatial) / 60.0 / sy),
            max(0.0, float(smooth_energy) / sz),
        )
        hist = gaussian_filter(hist, sigma=sigma, mode="nearest")
    return hist, edges


def scalar_display_values(
    counts: np.ndarray,
    stretch: str = "log",
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Map raw event counts to display intensities without changing science data.

    Positive pixels are robustly normalized at the 99.5th percentile.  The
    zero mask is restored after every operation so empty sky is *always*
    black, including with colourful palettes.  ``LOG_STRETCH_K`` is fixed at
    100 and intentionally documented here so sessions render reproducibly.
    """
    values = np.asarray(counts, dtype=float)
    finite_positive = np.isfinite(values) & (values > 0)
    display = np.zeros_like(values, dtype=float)
    if not np.any(finite_positive):
        return display
    high = float(np.percentile(values[finite_positive], 99.5))
    if not np.isfinite(high) or high <= 0:
        high = float(np.max(values[finite_positive]))
    if not np.isfinite(high) or high <= 0:
        return display
    display = np.clip(np.where(np.isfinite(values), values / high, 0.0), 0.0, 1.0)
    stretch = str(stretch).lower()
    if stretch == "sqrt":
        display = np.sqrt(display)
    elif stretch == "log":
        display = np.log1p(LOG_STRETCH_K * display) / np.log1p(LOG_STRETCH_K)
    elif stretch != "linear":
        raise ValueError(f"Unknown scalar stretch: {stretch}")
    display *= max(0.0, float(brightness))
    display = (display - 0.5) * max(0.0, float(contrast)) + 0.5
    display = np.clip(display, 0.0, 1.0)
    display[~finite_positive] = 0.0
    return display


def scalar_to_rgb(
    values: np.ndarray,
    palette: str = "gray",
    stretch: str = "log",
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Render cached scalar counts with a display-only palette lookup table."""
    palette = str(palette).lower()
    if palette not in SCALAR_PALETTES:
        raise ValueError(f"Unknown scalar palette: {palette}")
    display = scalar_display_values(values, stretch, brightness, contrast)
    anchors = np.stack([_hex_to_rgb(c) for c in SCALAR_PALETTES[palette]])
    positions = np.linspace(0.0, 1.0, len(anchors))
    rgb = np.stack(
        [np.interp(display, positions, anchors[:, channel]) for channel in range(3)], axis=-1
    )
    # Do this after the LUT: a palette must never give visual weight to sky
    # pixels that have no events.
    rgb[~(np.isfinite(values) & (np.asarray(values, dtype=float) > 0))] = 0.0
    return np.asarray(np.round(rgb), dtype=np.uint8)
