from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import numpy as np
import pandas as pd

_slice_ids = count(1)


@dataclass
class EnergySlice:
    """One independently movable energy slice shown in 3D and selectable in 2D."""

    low_kev: float
    high_kev: float
    label: str | None = None
    color: str = "#36c98f"
    opacity: float = 0.62
    visible: bool = True
    show_plane: bool = True
    show_points: bool = True
    cached_event_indices: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    cached_scalar_image: object | None = field(default=None, repr=False)
    membership_signature: tuple | None = field(default=None, repr=False)
    uid: str = field(default_factory=lambda: f"slice-{next(_slice_ids)}")

    @property
    def center_kev(self) -> float:
        return 0.5 * (self.low_kev + self.high_kev)

    @property
    def width_kev(self) -> float:
        return self.high_kev - self.low_kev

    @property
    def title(self) -> str:
        return self.label or f"{self.low_kev:.2f}–{self.high_kev:.2f} keV"


@dataclass
class DesktopState:
    """Desktop interaction state with inexpensive derived-frame caches."""

    # Search region remains fixed until the user deliberately performs another search.
    region: object | None = None
    region_cache_hit: bool = False
    target_name: str = ""
    workspace_id: str = ""
    search_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    search_records: dict[str, object] = field(default_factory=dict)

    # Observations remain loaded in memory until explicitly removed.
    observation_cache: dict[str, object] = field(default_factory=dict)
    loaded_observations: list = field(default_factory=list)
    combined_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    visible_record_keys: set[str] = field(default_factory=set)
    failed_observations: dict[str, str] = field(default_factory=dict)
    pending_observation_keys: list[str] = field(default_factory=list)

    selected_event: dict | None = None
    selected_observation_key: str | None = None
    spatial_rectangle: tuple[float, float, float, float] | None = None
    comparison_rectangle: tuple[float, float, float, float] | None = None

    # Global display band is separate from persistent analysis slices.
    energy_band: tuple[float, float] = (2.0, 6.0)
    slices: list[EnergySlice] = field(default_factory=list)
    selected_slice_uid: str | None = None
    content_mode: str = "all"
    global_detail_mode: str = "auto"
    global_custom_count: int = 300_000
    observation_detail_overrides: dict[str, dict] = field(default_factory=dict)
    top_image_mode: str = "off"
    top_image_source: str = "global"
    top_image_opacity: float = 0.75

    # Preview/performance controls. Analysis can still use exact backend products.
    preview_rows_per_observation: int = 15_000
    combined_preview_maximum: int = 600_000
    interactive_point_budget: int = 160_000
    image_bins: int = 220
    spectrum_bins: int = 260
    spectrum_smoothing_bins: float = 1.25
    spectrum_smooth_visible: bool = True
    spectrum_scale: str = "linear"
    auto_image_quality: bool = True
    auto_spectrum_binning: bool = True
    energy_scan_speed_hz: int = 4
    analysis_debounce_ms: int = 140
    viewer_debounce_ms: int = 95

    # Workspace layout.
    layout_mode: str = "split"  # split, 3d, 2d
    two_d_product: str = "energy"  # energy, rgb, slice, sky
    render_mode: str = "events"  # events, density, voxels
    filter_3d_by_energy: bool = False
    filter_2d_by_energy: bool = False
    # Keep the spectrum selection driving both views by default.  Users can
    # disable this to use the independent all-events/filter controls below.
    spectrum_linked: bool = True
    energy_reference_kev: float = 6.70
    energy_display_scale: float = 1.0
    show_coordinate_triad: bool = True
    show_energy_reference_plane: bool = True
    w49b_centroid_surface: bool = False
    show_slice_planes: bool = True
    show_grid_backdrop: bool = True
    show_coordinate_values: bool = True
    show_slice_window: bool = True
    camera_preset: str = "isometric"
    two_d_zoom: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)

    energy_image_exact: bool = False
    energy_image_exact_scope: str = "band"  # band or all_events
    rgb_image_exact: bool = False
    energy_image_cache_hit: bool = False
    rgb_image_cache_hit: bool = False

    # Event display.
    event_color_mode: str = "energy"
    point_size: float = 4.0
    point_opacity: float = 0.82
    event_spatial_smoothing_arcmin: float = 0.0
    event_energy_smoothing_kev: float = 0.0
    density_size_strength: float = 0.7
    density_opacity_strength: float = 0.7

    # Scalar image display is display-only; it never participates in a
    # scientific histogram/cache identity.  RGB remains independent below.
    image_smoothing_pixels: float = 0.8
    image_palette: str = "gray"
    image_stretch: str = "log"
    image_brightness: float = 1.0
    image_contrast: float = 1.0
    rgb_centers: tuple[float, float, float] = (1.85, 2.44, 6.40)
    rgb_widths: tuple[float, float, float] = (0.20, 0.20, 0.40)
    rgb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rgb_brightness: float = 1.1
    rgb_gamma: float = 1.0

    # Conservative voxel defaults keep the UI responsive.
    spatial_voxel_arcmin: float = 0.80
    energy_voxel_kev: float = 0.30
    voxel_spatial_smoothing_arcmin: float = 1.0
    voxel_energy_smoothing_kev: float = 1.0
    voxel_threshold_fraction: float = 0.03
    voxel_max_cells: int = 450_000
    voxel_energy_source: str = "selected_slice"
    voxel_opacity: float = 0.72
    voxel_show_edges: bool = False

    # These caches avoid repeated pandas .isin/.between work during one interaction.
    _display_cache_signature: tuple | None = field(default=None, init=False, repr=False)
    _display_cache: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _energy_cache_signature: tuple | None = field(default=None, init=False, repr=False)
    _energy_cache: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _spectrum_cache_signature: tuple | None = field(default=None, init=False, repr=False)
    _spectrum_cache: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def record_key(self, obs) -> str:
        return f"{obs.record.mission}/{obs.record.instrument}/{obs.record.observation_id}"

    @property
    def sky_viewport(self):
        from jaxa_udon3.desktop.science_views import SkyViewport

        return SkyViewport.from_region(self.region)

    def clamp_rectangle(self, rectangle):
        viewport = self.sky_viewport
        return rectangle if viewport is None else viewport.clamp_rectangle(rectangle)

    def visibility_signature(self) -> tuple:
        return id(self.combined_frame), tuple(sorted(self.visible_record_keys))

    def data_signature(self) -> tuple:
        region_signature = None
        if self.region is not None:
            try:
                value = self.region.signature()
                region_signature = tuple(sorted(value.items())) if isinstance(value, dict) else tuple(value)
            except Exception:
                region_signature = (
                    round(float(self.region.center_ra_deg), 7),
                    round(float(self.region.center_dec_deg), 7),
                    round(float(self.region.radius_deg), 7),
                )
        return (*self.visibility_signature(), region_signature)

    def clear_derived_caches(self) -> None:
        self._display_cache_signature = None
        self._display_cache = None
        self._energy_cache_signature = None
        self._energy_cache = None
        self._spectrum_cache_signature = None
        self._spectrum_cache = None

    def displayed_frame(self) -> pd.DataFrame:
        signature = self.visibility_signature()
        if self._display_cache_signature == signature and self._display_cache is not None:
            return self._display_cache
        if self.combined_frame.empty or "RECORD_KEY" not in self.combined_frame:
            result = self.combined_frame
        elif not self.visible_record_keys:
            result = self.combined_frame.iloc[0:0]
        else:
            result = self.combined_frame.loc[
                self.combined_frame["RECORD_KEY"].isin(self.visible_record_keys)
            ]
        self._display_cache_signature = signature
        self._display_cache = result
        return result

    def energy_filtered_frame(self) -> pd.DataFrame:
        signature = self.visibility_signature() + tuple(round(float(v), 6) for v in self.energy_band)
        if self._energy_cache_signature == signature and self._energy_cache is not None:
            return self._energy_cache
        frame = self.displayed_frame()
        if frame.empty or "KEV" not in frame:
            result = frame
        else:
            low, high = self.energy_band
            values = frame["KEV"].to_numpy(dtype=np.float32, copy=False)
            result = frame.loc[(values >= low) & (values <= high)]
        self._energy_cache_signature = signature
        self._energy_cache = result
        return result

    def spectrum_frame(self) -> pd.DataFrame:
        return self.spectrum_frame_for(self.spatial_rectangle)

    def spectrum_frame_for(self, rectangle) -> pd.DataFrame:
        rectangle = None if rectangle is None else tuple(
            round(float(value), 7) for value in rectangle
        )
        signature = (*self.visibility_signature(), "spectrum", rectangle)
        if self._spectrum_cache_signature == signature and self._spectrum_cache is not None:
            return self._spectrum_cache
        frame = self.displayed_frame()
        if frame.empty or rectangle is None:
            result = frame
        else:
            from jaxa_udon3.desktop.science_views import filter_rectangle

            result = filter_rectangle(frame, *rectangle)
        self._spectrum_cache_signature = signature
        self._spectrum_cache = result
        return result

    def visible_observations(self) -> list:
        return [
            obs for obs in self.loaded_observations
            if self.record_key(obs) in self.visible_record_keys
        ]

    def energy_range(self) -> tuple[float, float] | None:
        frame = self.displayed_frame()
        if frame.empty or "KEV" not in frame:
            return None
        values = frame["KEV"].to_numpy(dtype=np.float32, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return float(finite.min()), float(finite.max())

    def effective_image_smoothing(self, event_count: int) -> float:
        """Choose readable preview smoothing automatically, while preserving a custom override."""
        if not self.auto_image_quality:
            return float(self.image_smoothing_pixels)
        count = max(0, int(event_count))
        if count < 120:
            return 2.0
        if count < 500:
            return 1.6
        if count < 2_000:
            return 1.25
        if count < 10_000:
            return 0.9
        return 0.6

    def effective_spectrum_bins(self, event_count: int) -> int:
        if not self.auto_spectrum_binning:
            return int(self.spectrum_bins)
        count = max(0, int(event_count))
        if count < 500:
            return 120
        if count < 2_000:
            return 180
        if count < 15_000:
            return 260
        if count < 60_000:
            return 360
        return 520

    def rgb_bands(self) -> list[tuple[float, float]]:
        return [
            (center - width / 2.0, center + width / 2.0)
            for center, width in zip(self.rgb_centers, self.rgb_widths)
        ]

    def selected_slice(self) -> EnergySlice | None:
        for item in self.slices:
            if item.uid == self.selected_slice_uid:
                return item
        return self.slices[0] if self.slices else None

    def add_slice(
        self,
        low: float | None = None,
        high: float | None = None,
        *,
        label: str | None = None,
        color: str | None = None,
    ) -> EnergySlice:
        if low is None or high is None:
            low, high = self.energy_band
        palette = ("#36c98f", "#f1c75b", "#e86f92", "#66a3ff", "#a78bfa", "#ef8a62")
        item = EnergySlice(
            float(low),
            float(high),
            label=label,
            color=color or palette[len(self.slices) % len(palette)],
        )
        existing = {slice_item.uid for slice_item in self.slices}
        while item.uid in existing:
            item.uid = f"slice-{next(_slice_ids)}"
        self.slices.append(item)
        self.selected_slice_uid = item.uid
        return item

    def add_cas_a_reference_slices(self) -> list[EnergySlice]:
        """Bands quoted in Ken's Cas A/ASCA reference task."""
        definitions = (
            (1.55, 1.75, "Low continuum", "#ef8a62"),
            (1.75, 1.95, "Si He α", "#f6c85f"),
            (2.35, 2.52, "S He α", "#58c4a3"),
            (3.93, 6.23, "High continuum", "#5b8ff9"),
        )
        existing = {(round(item.low_kev, 3), round(item.high_kev, 3)) for item in self.slices}
        created = []
        for low, high, label, color in definitions:
            key = (round(low, 3), round(high, 3))
            if key in existing:
                continue
            created.append(self.add_slice(low, high, label=label, color=color))
            existing.add(key)
        return created

    def remove_slice(self, uid: str) -> None:
        self.slices = [item for item in self.slices if item.uid != uid]
        if self.selected_slice_uid == uid:
            self.selected_slice_uid = self.slices[0].uid if self.slices else None

    def slice_event_indices(self, item: EnergySlice, frame: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
        """Return per-observation inclusive memberships, caching by immutable frame identity."""
        frame = self.displayed_frame() if frame is None else frame
        signature = (id(frame), tuple(round(float(v), 6) for v in (item.low_kev, item.high_kev)))
        if item.membership_signature == signature:
            return item.cached_event_indices
        result: dict[str, np.ndarray] = {}
        if not frame.empty and {"KEV", "RECORD_KEY"}.issubset(frame.columns):
            values = frame["KEV"].to_numpy(dtype=float, copy=False)
            keys = frame["RECORD_KEY"].astype(str).to_numpy()
            for key in np.unique(keys):
                positions = np.flatnonzero((keys == key) & (values >= item.low_kev) & (values <= item.high_kev))
                result[str(key)] = positions
        item.cached_event_indices = result
        item.membership_signature = signature
        item.cached_scalar_image = None
        return result
