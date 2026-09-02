"""Simple UDON3 server search with Plotly 2D and PyVista 3D viewers."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from aiohttp import web
from astropy.io import fits
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout

try:
    from trame.widgets import plotly as trame_plotly
except ImportError as error:
    raise ImportError(
        "Missing trame-plotly. Install it with: "
        "python -m pip install -U trame-plotly plotly"
    ) from error

from trame.widgets import vuetify3 as vuetify

try:
    import pyvista as pv
    from trame.widgets import vtk as vtk_widgets
except ImportError as error:
    pv = None
    vtk_widgets = None
    PYVISTA_IMPORT_ERROR = error
else:
    PYVISTA_IMPORT_ERROR = None

import darts_backend as backend


JAXA_BLUE = "#003366"
DEFAULT_LIMIT = 250
DEFAULT_MAX_POINTS = 30_000
DEFAULT_PYVISTA_POINT_LIMIT = 50_000
DEFAULT_PYVISTA_POINT_SIZE = 4
MAX_RENDERED_VOXELS = 20_000
MAX_SLICES = 20
DEFAULT_SLICE_POINT_LIMIT = 20_000
DEFAULT_SLICE_IMAGE_BINS = 160
DEFAULT_GC_IMAGE_BINS = 240
LIVE_SEARCH_MIN_CHARS = 2
EXPORT_DIR = Path(__file__).resolve().parent / "exports"
DEFAULT_CHANNELS_PER_KEV = backend.RESOLVE_PI_CHANNELS_PER_KEV
MISSION_COLORS = {
    "ASCA": "#2563eb",
    "SUZAKU": "#059669",
    "HITOMI": "#dc2626",
    "XRISM": JAXA_BLUE,
}
PI_TO_KEV_FACTORS = backend.PI_TO_KEV_FACTORS
PI_TO_KEV_LABELS = {
    ("asca", "gis"): "ASCA GIS: PI/84.9",
    ("asca", "sis"): "ASCA SIS: PI/68.5",
    ("suzaku", "xis"): "Suzaku XIS: PI*0.00365",
    ("hitomi", "sxs"): "Hitomi SXS: PI/2000",
    ("hitomi", "sxi"): "Hitomi SXI: PI*0.006",
    ("xrism", "resolve"): "XRISM Resolve: PI/2000",
    ("xrism", "xtend"): "XRISM Xtend: PI*0.006",
}
MISSION_TIME_ORIGINS = {
    "asca": "1993-01-01T00:00:00Z",
    "suzaku": "2000-01-01T00:00:00Z",
    "hitomi": "2016-01-01T00:00:00Z",
    "xrism": "2019-01-01T00:00:00Z",
}
SLICE_COLORS = [
    "#d71920",
    "#f59e0b",
    "#0891b2",
    "#7c3aed",
    "#16a34a",
    "#db2777",
    "#4f46e5",
    "#0f766e",
]
SLICE_FIELDS = ("enabled", "center_kev", "width_kev", "color", "opacity")
SLICE_CHANGE_KEYS = [
    f"slice_{field}_{index}"
    for index in range(MAX_SLICES)
    for field in SLICE_FIELDS
]
GC_CENTER_RA = 266.41683
GC_CENTER_DEC = -29.00781
GC_RADIUS_DEG = 2.6
PALETTE_OPTIONS = {
    "inferno": {
        "label": "Inferno",
        "plotly": "Inferno",
        "pyvista": "inferno",
        "preview": "linear-gradient(90deg,#000004,#781c6d,#ed6925,#fcffa4)",
    },
    "magma": {
        "label": "Magma",
        "plotly": "Magma",
        "pyvista": "magma",
        "preview": "linear-gradient(90deg,#000004,#721f81,#f1605d,#fcfdbf)",
    },
    "turbo": {
        "label": "Turbo",
        "plotly": "Turbo",
        "pyvista": "turbo",
        "preview": "linear-gradient(90deg,#30123b,#1ae4b6,#f9e721,#7a0402)",
    },
    "viridis": {
        "label": "Viridis",
        "plotly": "Viridis",
        "pyvista": "viridis",
        "preview": "linear-gradient(90deg,#440154,#31688e,#35b779,#fde725)",
    },
    "plasma": {
        "label": "Plasma",
        "plotly": "Plasma",
        "pyvista": "plasma",
        "preview": "linear-gradient(90deg,#0d0887,#9c179e,#ed7953,#f0f921)",
    },
    "gray": {
        "label": "Gray",
        "plotly": "Gray",
        "pyvista": "gray",
        "preview": "linear-gradient(90deg,#000000,#666666,#bdbdbd,#ffffff)",
    },
    "hot": {
        "label": "Hot",
        "plotly": "Hot",
        "pyvista": "hot",
        "preview": "linear-gradient(90deg,#000000,#b30000,#ffcc00,#ffffff)",
    },
}
ENERGY_MAP_NAMES = {
    "fe64": "Energy map 1",
    "fe67": "Energy map 2",
}
MISSION_KEYS = {
    mission: f"mission_{mission}"
    for mission in backend.UDON3_MISSIONS
}
INSTRUMENT_KEYS = {
    (mission, instrument): f"instrument_{mission}_{instrument}"
    for mission, instruments in backend.KNOWN_INSTRUMENTS.items()
    for instrument in instruments
}

RESULTS = pd.DataFrame()
RESULT_RECORDS_BY_LABEL = {}
LOADED_OBSERVATION_CACHE = OrderedDict()
VIEWER_FIGURE_CACHE = OrderedDict()
MAX_MEMORY_CACHE_BYTES = 512 * 1024**2
MAX_FIGURE_CACHE_ENTRIES = 32
RESULTS_CHART = None
VIEWER_CHART = None
GC_IMAGE_CHARTS = {"fe64": None, "fe67": None}
RGB_IMAGE_CHART = None
RGB_COMPOSITE_ACTOR = None
GC_IMAGE_STATE_KEYS = {
    "fe64": "energy_map_fe64_figure",
    "fe67": "energy_map_fe67_figure",
}
PYVISTA_AVAILABLE = pv is not None and vtk_widgets is not None
PYVISTA_PLOTTER = None
PYVISTA_VIEW = None
PYVISTA_ACTORS = []
PYVISTA_DATA_ACTORS = []
PYVISTA_PICK_LOOKUP = {}
PYVISTA_SELECTED_ACTOR = None
CURRENT_SEARCH_RESULT = None
CURRENT_SKY_REGION = None
INSTRUMENT_SYNC_ACTIVE = False
REGION_REQUEST_GENERATION = 0
CURRENT_PYVISTA_POINTS = pd.DataFrame()
CURRENT_PYVISTA_POINTS_KEY = None
CURRENT_PYVISTA_SCENE_POINTS = pd.DataFrame()
CURRENT_PYVISTA_TRANSFORM = {}
SLICE_ACTORS = {}
SLICE_TEXTURE_ACTORS = {}
SLICE_IMAGE_CHARTS = {}
SLICE_PROFILE_CHART = None
SLICE_IMAGE_DATA = {}
ACTIVE_SLICE_IMAGE_DATA = {}
PROFILE_ENDPOINTS = []


def parse_float(value, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a number") from error


def selected_pairs_from_state(state) -> list[str]:
    return [
        backend.pair_key(mission, instrument)
        for (mission, instrument), key in INSTRUMENT_KEYS.items()
        if bool(state[key])
    ]


def obsid_from_filename(filename: str) -> str:
    stem = Path(str(filename or "").strip()).name
    for suffix in ("_events.parquet", "_hdr.json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if "_" not in stem:
        return ""
    return stem.rsplit("_", 1)[0]


def filename_for_row(row: pd.Series) -> str:
    return f"{row['observation_id']}_{row['instrument']}_events.parquet"


def cache_state(row: pd.Series) -> str:
    parquet_path = Path(str(row["parquet_cache_path"]))
    return "raw" if parquet_path.exists() else "remote"


def format_bytes(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0


def refresh_cache_status(message: str = ""):
    try:
        info = backend.cache_status()
        used = int(info["bytes"])
        limit = max(1, int(info["limit_bytes"]))
        kinds = info.get("by_kind", {})
        state.cache_usage_fraction = min(1.0, used / limit)
        state.cache_summary = (
            f"{format_bytes(used)} / {format_bytes(limit)} | "
            f"raw {format_bytes(kinds.get('raw', 0))}, "
            f"previews {format_bytes(kinds.get('preview', 0))}, "
            f"images {format_bytes(kinds.get('image', 0))} | "
            f"{info.get('cached_headers', 0):,} headers"
        )
        state.cache_message = message or "LRU cache ready"
    except Exception as error:
        state.cache_usage_fraction = 0.0
        state.cache_summary = "Cache status unavailable"
        state.cache_message = str(error)
    state.dirty("cache_usage_fraction", "cache_summary", "cache_message")


def clear_cache_products(kinds, label: str):
    try:
        result = backend.clear_cache(kinds=kinds)
        LOADED_OBSERVATION_CACHE.clear()
        VIEWER_FIGURE_CACHE.clear()
        refresh_cache_status(f"Cleared {len(result['removed'])} {label} cache item(s)")
    except Exception as error:
        refresh_cache_status(str(error))


def clear_raw_cache():
    clear_cache_products(("raw",), "raw")


def clear_derived_cache():
    clear_cache_products(("preview", "image"), "derived")


def result_label(row: pd.Series) -> str:
    return (
        f"{str(row['mission']).upper()} / "
        f"{str(row['instrument']).upper()} / "
        f"{row['observation_id']} / "
        f"{row.get('object', '')}"
    )


def apply_filename_filter(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    filename = str(filename or "").strip().lower()
    if frame.empty or not filename:
        return frame
    filtered = frame.copy()
    filtered["parquet_filename"] = filtered.apply(filename_for_row, axis=1)
    mask = filtered["parquet_filename"].str.lower().str.contains(filename, regex=False)
    return filtered.loc[mask].reset_index(drop=True)


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": "#64748b"},
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 30, "r": 30, "t": 35, "b": 30},
        paper_bgcolor="#f8fafc",
        plot_bgcolor="white",
    )
    return figure


def slice_empty_figure(message: str, color: str = JAXA_BLUE) -> go.Figure:
    figure = empty_figure(message)
    figure.update_layout(
        height=320,
        margin={"l": 45, "r": 18, "t": 42, "b": 42},
        title={"text": message, "font": {"size": 13, "color": color}},
    )
    return figure


def slice_state_name(index: int, field: str) -> str:
    return f"slice_{field}_{index}"


def slice_config(index: int) -> dict:
    center = parse_float(getattr(state, slice_state_name(index, "center_kev")), "Slice center")
    width = max(
        0.001,
        parse_float(getattr(state, slice_state_name(index, "width_kev")), "Slice width"),
    )
    low = center - width / 2.0
    high = center + width / 2.0
    return {
        "index": index,
        "enabled": bool(getattr(state, slice_state_name(index, "enabled"))),
        "center": center,
        "width": width,
        "low": low,
        "high": high,
        "color": str(getattr(state, slice_state_name(index, "color"))),
        "opacity": float(getattr(state, slice_state_name(index, "opacity"))),
    }


def active_slice_indices() -> range:
    count = max(0, min(MAX_SLICES, int(float(state.slice_count))))
    return range(count)


def update_slice_chart(index: int, figure: go.Figure):
    chart = SLICE_IMAGE_CHARTS.get(index)
    if chart is not None:
        chart.update(figure)
        state.dirty(f"slice_image_figure_{index}")


def slice_heatmap_figure(index: int, points: pd.DataFrame, config: dict) -> go.Figure:
    global ACTIVE_SLICE_IMAGE_DATA
    label = f"Slice {index + 1}: {config['low']:.3f}-{config['high']:.3f} keV"
    if points.empty:
        SLICE_IMAGE_DATA.pop(index, None)
        if index == int(getattr(state, "active_slice_index", 0)):
            ACTIVE_SLICE_IMAGE_DATA = {}
        return slice_empty_figure(f"{label} | no events", config["color"])

    bins = max(16, int(float(state.slice_image_bins)))
    ra = points["RA"].to_numpy(dtype=float)
    dec = points["DEC"].to_numpy(dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[valid]
    dec = dec[valid]
    if len(ra) == 0:
        SLICE_IMAGE_DATA.pop(index, None)
        if index == int(getattr(state, "active_slice_index", 0)):
            ACTIVE_SLICE_IMAGE_DATA = {}
        return slice_empty_figure(f"{label} | no finite RA/DEC", config["color"])

    hist, x_edges, y_edges = np.histogram2d(ra, dec, bins=bins)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    files = points["FILE_LABEL"].nunique() if "FILE_LABEL" in points.columns else 0
    missions = points["MISSION"].nunique() if "MISSION" in points.columns else 0
    image_data = {
        "index": index,
        "hist": hist.T.astype(float),
        "x_edges": x_edges.astype(float),
        "y_edges": y_edges.astype(float),
        "x_centers": x_centers.astype(float),
        "y_centers": y_centers.astype(float),
        "config": dict(config),
        "event_count": int(len(points)),
        "quality": "Preview",
    }
    SLICE_IMAGE_DATA[index] = image_data
    if index == int(getattr(state, "active_slice_index", 0)):
        ACTIVE_SLICE_IMAGE_DATA = image_data
    display_hist = smoothed_slice_histogram(image_data)

    figure = go.Figure(
        data=[
            go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=display_hist,
                colorscale="Viridis",
                colorbar={"title": "Events"},
                hovertemplate=(
                    "RA=%{x:.6f} deg<br>"
                    "DEC=%{y:.6f} deg<br>"
                    "Events=%{z:.0f}<extra></extra>"
                ),
            )
        ]
    )
    if PROFILE_ENDPOINTS:
        endpoint_x = [point[0] for point in PROFILE_ENDPOINTS]
        endpoint_y = [point[1] for point in PROFILE_ENDPOINTS]
        figure.add_trace(
            go.Scatter(
                x=endpoint_x,
                y=endpoint_y,
                mode="lines+markers" if len(PROFILE_ENDPOINTS) == 2 else "markers",
                line={"color": "white", "width": 3},
                marker={
                    "size": 10,
                    "color": ["#00e5ff", "#ffea00"][: len(PROFILE_ENDPOINTS)],
                    "line": {"color": JAXA_BLUE, "width": 1},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=320,
        margin={"l": 50, "r": 18, "t": 52, "b": 44},
        paper_bgcolor="#f8fafc",
        plot_bgcolor="white",
        title={
            "text": f"{label} | {len(points):,} events | {files} file(s), {missions} mission(s)",
            "font": {"size": 13, "color": config["color"]},
        },
        font={"family": "Arial, sans-serif", "color": "#172033"},
    )
    figure.update_xaxes(title_text="Right Ascension (deg)", autorange="reversed")
    figure.update_yaxes(title_text="Declination (deg)")
    return figure


def exact_slice_heatmap_figure(index: int, exact: dict, config: dict) -> go.Figure:
    global ACTIVE_SLICE_IMAGE_DATA
    hist = np.asarray(exact["hist"], dtype=float)
    x_edges = np.asarray(exact["x_edges"], dtype=float)
    y_edges = np.asarray(exact["y_edges"], dtype=float)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    image_data = {
        "index": index,
        "hist": hist,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "config": dict(config),
        "event_count": int(exact["event_count"]),
        "quality": "Exact",
    }
    SLICE_IMAGE_DATA[index] = image_data
    if index == int(getattr(state, "active_slice_index", 0)):
        ACTIVE_SLICE_IMAGE_DATA = image_data
    display_hist = smoothed_slice_histogram(image_data)

    cache_label = "cache hit" if exact.get("cache_hit") else "computed"
    figure = go.Figure(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=display_hist,
            colorscale="Viridis",
            colorbar={"title": "Events"},
            hovertemplate=(
                "RA=%{x:.6f} deg<br>DEC=%{y:.6f} deg<br>"
                "Events=%{z:.0f}<extra></extra>"
            ),
        )
    )
    if PROFILE_ENDPOINTS:
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in PROFILE_ENDPOINTS],
                y=[point[1] for point in PROFILE_ENDPOINTS],
                mode="lines+markers" if len(PROFILE_ENDPOINTS) == 2 else "markers",
                line={"color": "white", "width": 3},
                marker={
                    "size": 10,
                    "color": ["#00e5ff", "#ffea00"][: len(PROFILE_ENDPOINTS)],
                    "line": {"color": JAXA_BLUE, "width": 1},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=320,
        margin={"l": 50, "r": 18, "t": 52, "b": 44},
        paper_bgcolor="#f8fafc",
        plot_bgcolor="white",
        title={
            "text": (
                f"Exact slice {index + 1}: {config['low']:.3f}-"
                f"{config['high']:.3f} keV | {exact['event_count']:,} events | "
                f"{cache_label}"
            ),
            "font": {"size": 13, "color": config["color"]},
        },
        font={"family": "Arial, sans-serif", "color": "#172033"},
    )
    figure.update_xaxes(title_text="Right Ascension (deg)", autorange="reversed")
    figure.update_yaxes(title_text="Declination (deg)")
    return figure


def update_exact_slice_image(index: int):
    if CURRENT_SEARCH_RESULT is None or not CURRENT_SEARCH_RESULT.observations:
        return
    config = slice_config(index)
    if not config["enabled"]:
        return
    status_key = slice_state_name(index, "status")
    setattr(state, status_key, "Exact image loading...")
    state.dirty(status_key)
    try:
        exact = backend.exact_energy_image(
            [item.record for item in CURRENT_SEARCH_RESULT.observations],
            config["low"],
            config["high"],
            bins=max(16, int(float(state.slice_image_bins))),
            region=CURRENT_SKY_REGION,
        )
        update_slice_chart(index, exact_slice_heatmap_figure(index, exact, config))
        add_slice_texture(index, SLICE_IMAGE_DATA.get(index, {}))
        cache_label = "cached" if exact.get("cache_hit") else "computed"
        setattr(
            state,
            status_key,
            f"Exact | {config['low']:.3f}-{config['high']:.3f} keV | "
            f"{exact['event_count']:,} events | {cache_label}",
        )
        state.dirty(status_key)
        refresh_cache_status()
        update_profile_chart()
    except Exception as error:
        setattr(state, status_key, f"Preview | exact image unavailable: {error}")
        state.dirty(status_key)


def profile_empty_figure(message: str = "Click two points on any slice image") -> go.Figure:
    figure = empty_figure(message)
    figure.update_layout(height=310, margin={"l": 55, "r": 18, "t": 42, "b": 45})
    return figure


def bilinear_image_values(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0
    return (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x1] * dx * (1 - dy)
        + image[y1, x0] * (1 - dx) * dy
        + image[y1, x1] * dx * dy
    )


def image_profile_values(
    image_data: dict,
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    width_pixels: int,
) -> np.ndarray | None:
    image = np.asarray(image_data["hist"], dtype=float)
    x_centers = np.asarray(image_data["x_centers"], dtype=float)
    y_centers = np.asarray(image_data["y_centers"], dtype=float)
    if image.size == 0 or len(x_centers) < 2 or len(y_centers) < 2:
        return None

    x_order = np.argsort(x_centers)
    y_order = np.argsort(y_centers)
    sample_x = np.interp(
        ra_samples,
        x_centers[x_order],
        np.arange(len(x_centers), dtype=float)[x_order],
    )
    sample_y = np.interp(
        dec_samples,
        y_centers[y_order],
        np.arange(len(y_centers), dtype=float)[y_order],
    )
    valid = (
        (ra_samples >= float(np.min(x_centers)))
        & (ra_samples <= float(np.max(x_centers)))
        & (dec_samples >= float(np.min(y_centers)))
        & (dec_samples <= float(np.max(y_centers)))
    )
    if not np.any(valid):
        return None

    dx = float(sample_x[-1] - sample_x[0])
    dy = float(sample_y[-1] - sample_y[0])
    norm = max(float(np.hypot(dx, dy)), 1e-9)
    perpendicular_x = -dy / norm
    perpendicular_y = dx / norm
    offsets = np.arange(width_pixels, dtype=float) - (width_pixels - 1) / 2.0
    profiles = [
        bilinear_image_values(
            image,
            sample_x + perpendicular_x * offset,
            sample_y + perpendicular_y * offset,
        )
        for offset in offsets
    ]
    values = np.mean(profiles, axis=0)
    values[~valid] = np.nan
    return values


def comparative_profile_data() -> dict:
    if len(PROFILE_ENDPOINTS) != 2 or not SLICE_IMAGE_DATA:
        raise ValueError("Select points A and B on a slice image first")
    (ra0, dec0), (ra1, dec1) = PROFILE_ENDPOINTS
    width_pixels = max(1, int(float(state.profile_width_pixels)))
    sample_count = max(
        100,
        min(
            800,
            3 * max(
                max(len(item["x_centers"]), len(item["y_centers"]))
                for item in SLICE_IMAGE_DATA.values()
            ),
        ),
    )
    fraction = np.linspace(0.0, 1.0, sample_count)
    delta_ra = (ra1 - ra0 + 180.0) % 360.0 - 180.0
    ra_samples = np.mod(ra0 + fraction * delta_ra, 360.0)
    dec_samples = dec0 + fraction * (dec1 - dec0)
    mean_dec = np.deg2rad(0.5 * (dec0 + dec1))
    dra = delta_ra * np.cos(mean_dec)
    distance_arcmin = np.hypot(dra, dec1 - dec0) * 60.0
    distance = np.linspace(0.0, distance_arcmin, sample_count)
    profiles = []
    for index in active_slice_indices():
        image_data = SLICE_IMAGE_DATA.get(index)
        if not image_data or not slice_config(index)["enabled"]:
            continue
        values = image_profile_values(
            image_data,
            ra_samples,
            dec_samples,
            width_pixels,
        )
        if values is None:
            continue
        config = image_data["config"]
        quality = image_data.get("quality", "Preview")
        label = (
            f"Slice {index + 1} | {config['low']:.3f}-"
            f"{config['high']:.3f} keV | {quality}"
        )
        profiles.append(
            {
                "index": index,
                "label": label,
                "config": config,
                "quality": quality,
                "values": values,
            }
        )

    if not profiles:
        raise ValueError("The selected line is outside the slice images")
    return {
        "point_a": (ra0, dec0),
        "point_b": (ra1, dec1),
        "width_pixels": width_pixels,
        "distance_arcmin": distance,
        "ra_deg": ra_samples,
        "dec_deg": dec_samples,
        "profiles": profiles,
    }


def profile_figure() -> go.Figure:
    try:
        comparison = comparative_profile_data()
    except ValueError as error:
        return profile_empty_figure(str(error))

    figure = go.Figure()
    for profile in comparison["profiles"]:
        label = profile["label"]
        config = profile["config"]
        figure.add_trace(
            go.Scattergl(
                x=comparison["distance_arcmin"],
                y=profile["values"],
                mode="lines",
                name=label,
                line={"color": config["color"], "width": 2.5},
                hovertemplate=(
                    f"{label}<br>Distance=%{{x:.3f}} arcmin<br>"
                    "Events/pixel=%{y:.3f}<extra></extra>"
                ),
            )
        )
    profile_count = len(comparison["profiles"])
    width_pixels = comparison["width_pixels"]
    figure.update_layout(
        template="plotly_white",
        height=310,
        margin={"l": 58, "r": 18, "t": 54, "b": 48},
        title={
            "text": (
                f"Comparative line profiles | {profile_count} image(s) | "
                f"width {width_pixels} pixel(s)"
            ),
            "font": {"size": 13},
        },
        xaxis_title="Distance from point A (arcmin)",
        yaxis_title="Mean events / pixel",
        legend={"orientation": "h", "y": 1.16, "x": 0},
    )
    return figure


def update_profile_chart():
    if SLICE_PROFILE_CHART is not None:
        SLICE_PROFILE_CHART.update(profile_figure())
        state.dirty("slice_profile_figure")
    if hasattr(state, "profile_download_ready"):
        state.profile_download_ready = False
        state.profile_download_csv_href = ""
        state.profile_download_json_href = ""
        state.profile_download_status = (
            "Profile changed; prepare the download again"
            if len(PROFILE_ENDPOINTS) == 2
            else "Select A and B to prepare profile data"
        )
        state.dirty(
            "profile_download_ready",
            "profile_download_csv_href",
            "profile_download_json_href",
            "profile_download_status",
        )


def slice_click_coordinates(event) -> tuple[float, float] | None:
    if isinstance(event, (list, tuple)) and len(event) >= 2:
        try:
            return float(event[0]), float(event[1])
        except (TypeError, ValueError):
            return None
    if not isinstance(event, dict):
        return None
    points = event.get("points")
    if isinstance(points, list) and points:
        point = points[0]
    else:
        point = event.get("point", event)
    if not isinstance(point, dict):
        return None
    try:
        return float(point["x"]), float(point["y"])
    except (KeyError, TypeError, ValueError):
        return None


def redraw_slice_image(index: int):
    if index not in active_slice_indices():
        return
    config = slice_config(index)
    current = SLICE_IMAGE_DATA.get(index)
    if current and current.get("quality") == "Exact":
        exact = {
            "hist": current["hist"],
            "x_edges": current["x_edges"],
            "y_edges": current["y_edges"],
            "event_count": current["event_count"],
            "cache_hit": True,
        }
        update_slice_chart(index, exact_slice_heatmap_figure(index, exact, config))
        return
    source = slice_source_points(config)
    update_slice_chart(index, slice_heatmap_figure(index, source, config))


def redraw_all_slice_images():
    for index in active_slice_indices():
        if slice_config(index)["enabled"]:
            redraw_slice_image(index)


def slice_image_clicked(index: int, event=None, **_):
    coordinates = slice_click_coordinates(event)
    if coordinates is None:
        state.profile_status = "Could not read the selected image position"
        state.dirty("profile_status")
        return
    if index != int(state.active_slice_index):
        set_active_slice(index)
    if len(PROFILE_ENDPOINTS) >= 2:
        PROFILE_ENDPOINTS.clear()
    PROFILE_ENDPOINTS.append(coordinates)
    image_count = sum(
        1
        for slice_index in active_slice_indices()
        if slice_config(slice_index)["enabled"] and slice_index in SLICE_IMAGE_DATA
    )
    state.profile_status = (
        f"Point A shared across {image_count} image(s); click point B"
        if len(PROFILE_ENDPOINTS) == 1
        else f"Comparing the A-B profile across {image_count} image(s)"
    )
    state.dirty("profile_status")
    redraw_all_slice_images()
    update_profile_chart()


def clear_slice_profile():
    PROFILE_ENDPOINTS.clear()
    state.profile_status = "Click point A, then point B on any slice image"
    state.dirty("profile_status")
    redraw_all_slice_images()
    update_profile_chart()


def update_gc_image_chart(mode: str, figure: go.Figure):
    chart = GC_IMAGE_CHARTS.get(mode)
    if chart is not None:
        chart.update(figure)
        state.dirty(GC_IMAGE_STATE_KEYS[mode])


def gc_empty_figure(message: str) -> go.Figure:
    figure = empty_figure(message)
    figure.update_layout(
        height=560,
        margin={"l": 58, "r": 28, "t": 56, "b": 52},
        title={"text": message, "font": {"size": 15, "color": JAXA_BLUE}},
    )
    return figure


def gaussian_smooth_image(image: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(0.0, float(sigma))
    if sigma <= 0:
        return image
    radius = max(1, min(24, int(np.ceil(sigma * 3.0))))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"),
        axis=0,
        arr=image,
    )
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"),
        axis=1,
        arr=smoothed,
    )
    return smoothed


def rgb_band_config() -> backend.RGBBandConfig:
    return backend.RGBBandConfig(
        parse_float(state.rgb_red_center, "Red center"),
        parse_float(state.rgb_red_width, "Red width"),
        parse_float(state.rgb_green_center, "Green center"),
        parse_float(state.rgb_green_width, "Green width"),
        parse_float(state.rgb_blue_center, "Blue center"),
        parse_float(state.rgb_blue_width, "Blue width"),
    )


def rgb_event_colors(energy_kev: np.ndarray) -> np.ndarray:
    config = rgb_band_config()
    config.bands()
    red = float(config.red_center_kev)
    green = float(config.green_center_kev)
    blue = float(config.blue_center_kev)
    energy = np.asarray(energy_kev, dtype=float)
    colors = np.zeros((len(energy), 3), dtype=float)
    below_green = energy <= green
    red_mix = np.clip((energy - red) / max(green - red, 1e-9), 0.0, 1.0)
    blue_mix = np.clip((energy - green) / max(blue - green, 1e-9), 0.0, 1.0)
    colors[below_green, 0] = 1.0 - red_mix[below_green]
    colors[below_green, 1] = red_mix[below_green]
    colors[~below_green, 1] = 1.0 - blue_mix[~below_green]
    colors[~below_green, 2] = blue_mix[~below_green]
    colors[energy <= red] = (1.0, 0.0, 0.0)
    colors[energy >= blue] = (0.0, 0.0, 1.0)
    gains = np.asarray(
        [state.rgb_red_gain, state.rgb_green_gain, state.rgb_blue_gain], dtype=float
    )
    colors *= gains.reshape(1, 3)
    gamma = max(parse_float(state.rgb_gamma, "RGB gamma"), 0.05)
    colors = np.power(np.clip(colors, 0.0, 1.0), 1.0 / gamma)
    colors *= max(parse_float(state.rgb_brightness, "RGB brightness"), 0.0)
    return np.asarray(np.clip(colors, 0.0, 1.0) * 255.0, dtype=np.uint8)


def _smooth_volume_axis(volume: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    sigma = max(0.0, float(sigma))
    if sigma <= 0:
        return volume
    radius = max(1, min(12, int(np.ceil(3 * sigma)), (volume.shape[axis] - 1) // 2))
    if radius < 1:
        return volume
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.apply_along_axis(
        lambda values: np.convolve(values, kernel, mode="same"), axis, volume
    )


def event_density_values(points: pd.DataFrame) -> np.ndarray:
    spatial_sigma = max(parse_float(state.event_spatial_sigma, "Spatial smoothing"), 0.0)
    energy_sigma = max(parse_float(state.event_energy_sigma, "Energy smoothing"), 0.0)
    if spatial_sigma <= 0 and energy_sigma <= 0:
        return np.ones(len(points), dtype=np.float32)
    values = points[["X_3D", "Y_3D", "KEV"]].to_numpy(dtype=float)
    dimensions = (48, 48, 36)
    minimum = np.nanmin(values, axis=0)
    maximum = np.nanmax(values, axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    normalized = np.clip((values - minimum) / span, 0.0, 1.0)
    histogram, _ = np.histogramdd(normalized, bins=dimensions, range=((0, 1),) * 3)
    spatial_bin_arcmin = max(
        CURRENT_PYVISTA_TRANSFORM.get("spatial_span", 1.0) / dimensions[0], 1e-6
    )
    energy_bin_kev = max(CURRENT_PYVISTA_TRANSFORM.get("z_span", 1.0) / dimensions[2], 1e-6)
    histogram = _smooth_volume_axis(histogram, spatial_sigma / spatial_bin_arcmin, 0)
    histogram = _smooth_volume_axis(histogram, spatial_sigma / spatial_bin_arcmin, 1)
    histogram = _smooth_volume_axis(histogram, energy_sigma / energy_bin_kev, 2)
    indices = np.minimum(
        (normalized * np.asarray(dimensions)).astype(int), np.asarray(dimensions) - 1
    )
    density = histogram[indices[:, 0], indices[:, 1], indices[:, 2]]
    scale = float(np.percentile(density, 99)) if len(density) else 0.0
    if scale <= 0:
        return np.zeros(len(points), dtype=np.float32)
    return np.asarray(np.clip(density / scale, 0.0, 1.0), dtype=np.float32)


def smoothed_slice_histogram(image_data: dict) -> np.ndarray:
    hist = np.asarray(image_data["hist"], dtype=float)
    sigma_arcmin = max(parse_float(state.image_spatial_sigma, "Image smoothing"), 0.0)
    x_edges = np.asarray(image_data["x_edges"], dtype=float)
    if sigma_arcmin <= 0 or len(x_edges) < 2:
        return hist
    pixel_arcmin = max(
        abs(float(np.median(np.diff(x_edges))))
        * np.cos(np.deg2rad(float(np.mean(image_data["y_centers"]))))
        * 60.0,
        1e-9,
    )
    return gaussian_smooth_image(hist, sigma_arcmin / pixel_arcmin)


def display_scaled_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    values = gaussian_smooth_image(
        values,
        parse_float(state.gc_smoothing_sigma, "GC smoothing"),
    )
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)

    clean = np.where(finite, values, 0.0)
    stretch = str(state.gc_stretch)
    if stretch == "sqrt":
        low = float(np.nanmin(clean))
        clean = np.sqrt(np.maximum(clean - low, 0.0))
    elif stretch == "log":
        low = float(np.nanmin(clean))
        clean = np.log1p(np.maximum(clean - low, 0.0))

    vmin = float(np.nanmin(clean))
    vmax = float(np.nanmax(clean))
    if vmax <= vmin:
        scaled = np.zeros_like(clean)
    else:
        scaled = (clean - vmin) / (vmax - vmin)

    brightness = parse_float(state.gc_brightness, "GC brightness")
    contrast = parse_float(state.gc_contrast, "GC contrast")
    scaled = scaled * brightness
    scaled = (scaled - 0.5) * contrast + 0.5
    return np.clip(scaled, 0.0, 1.0)


def finite_loaded_gc_points() -> pd.DataFrame:
    if CURRENT_PYVISTA_POINTS.empty:
        return pd.DataFrame()
    required = {"RA", "DEC", "KEV"}
    if not required.issubset(CURRENT_PYVISTA_POINTS.columns):
        return pd.DataFrame()
    points = CURRENT_PYVISTA_POINTS.copy()
    valid = np.isfinite(points["RA"].to_numpy(dtype=float))
    valid &= np.isfinite(points["DEC"].to_numpy(dtype=float))
    valid &= np.isfinite(points["KEV"].to_numpy(dtype=float))
    return points.loc[valid].copy()


def histogram_for_bands(
    points: pd.DataFrame,
    bands: tuple[tuple[float, float], ...],
    bins: int,
    ra_range: tuple[float, float],
    dec_range: tuple[float, float],
) -> tuple[np.ndarray, int]:
    if points.empty:
        return np.zeros((bins, bins), dtype=float), 0
    kev = points["KEV"].to_numpy(dtype=float)
    mask = np.zeros(len(points), dtype=bool)
    for low, high in bands:
        mask |= (kev >= low) & (kev <= high)
    selected = points.loc[mask]
    if selected.empty:
        return np.zeros((bins, bins), dtype=float), 0
    hist, _x_edges, _y_edges = np.histogram2d(
        selected["RA"].to_numpy(dtype=float),
        selected["DEC"].to_numpy(dtype=float),
        bins=bins,
        range=[ra_range, dec_range],
    )
    return hist.T.astype(float), len(selected)


def gc_image_arrays(
    points: pd.DataFrame, mode: str
) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    bins = max(24, int(parse_float(state.gc_image_bins, "GC image bins")))
    ra = points["RA"].to_numpy(dtype=float)
    dec = points["DEC"].to_numpy(dtype=float)
    ra_range = (float(np.nanmin(ra)), float(np.nanmax(ra)))
    dec_range = (float(np.nanmin(dec)), float(np.nanmax(dec)))
    if ra_range[0] == ra_range[1]:
        ra_range = (ra_range[0] - 0.01, ra_range[1] + 0.01)
    if dec_range[0] == dec_range[1]:
        dec_range = (dec_range[0] - 0.01, dec_range[1] + 0.01)

    center = parse_float(getattr(state, f"gc_center_{mode}"), f"{mode} center")
    width = max(
        0.001,
        parse_float(getattr(state, f"gc_width_{mode}"), f"{mode} width"),
    )
    band = ((center - width / 2.0, center + width / 2.0),)
    counts = {"low_kev": band[0][0], "high_kev": band[0][1]}
    raw, counts[mode] = histogram_for_bands(
        points,
        band,
        bins,
        ra_range,
        dec_range,
    )

    x_centers = np.linspace(ra_range[0], ra_range[1], bins)
    y_centers = np.linspace(dec_range[0], dec_range[1], bins)
    counts["raw_min"] = float(np.nanmin(raw)) if raw.size else 0.0
    counts["raw_max"] = float(np.nanmax(raw)) if raw.size else 0.0
    return raw, counts, x_centers, y_centers


def gc_image_figure(mode: str, palette_key: str) -> go.Figure:
    points = finite_loaded_gc_points()
    if points.empty:
        setattr(state, f"gc_status_{mode}", "Load event data first")
        state.dirty(f"gc_status_{mode}")
        return gc_empty_figure("Load event data first")

    try:
        raw, counts, x_centers, y_centers = gc_image_arrays(points, mode)
        display = display_scaled_image(raw)
    except Exception as error:
        setattr(state, f"gc_status_{mode}", str(error))
        state.dirty(f"gc_status_{mode}")
        return gc_empty_figure(str(error))

    low_kev = counts["low_kev"]
    high_kev = counts["high_kev"]
    mode_label = f"{low_kev:.3f}-{high_kev:.3f} keV"
    files = points["FILE_LABEL"].nunique() if "FILE_LABEL" in points.columns else 0
    missions = points["MISSION"].nunique() if "MISSION" in points.columns else 0
    event_count = sum(
        value
        for key, value in counts.items()
        if key not in {"raw_min", "raw_max", "low_kev", "high_kev"}
    )
    palette_config = PALETTE_OPTIONS.get(
        palette_key.lower(), PALETTE_OPTIONS["inferno"]
    )
    palette = palette_config["plotly"]
    setattr(state, f"gc_status_{mode}", (
        f"{mode_label}: {event_count:,} selected events | "
        f"{files} file(s), {missions} mission(s)"
    ))
    state.dirty(f"gc_status_{mode}")

    figure = go.Figure(
        data=[
            go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=display,
                customdata=raw,
                colorscale=palette,
                colorbar={"title": "Scaled"},
                hovertemplate=(
                    "RA=%{x:.6f} deg<br>"
                    "DEC=%{y:.6f} deg<br>"
                    "Display=%{z:.3f}<br>"
                    "Raw=%{customdata:.3f}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        template="plotly_white",
        height=560,
        margin={"l": 62, "r": 28, "t": 64, "b": 54},
        paper_bgcolor="#f8fafc",
        plot_bgcolor="white",
        title={
            "text": (
                f"{ENERGY_MAP_NAMES[mode]} | {mode_label} | "
                f"{palette_config['label']}"
            ),
            "font": {"size": 16, "color": JAXA_BLUE},
        },
        font={"family": "Arial, sans-serif", "color": "#172033"},
    )
    figure.update_xaxes(title_text="Right Ascension (deg)", autorange="reversed")
    figure.update_yaxes(title_text="Declination (deg)")
    return figure


def update_gc_image():
    for mode in ("fe64", "fe67"):
        palette_key = str(getattr(state, f"gc_palette_{mode}"))
        update_gc_image_chart(mode, gc_image_figure(mode, palette_key))


def _rgb_display_array(channels: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    values = np.asarray(channels, dtype=float).copy()
    sigma_arcmin = max(parse_float(state.image_spatial_sigma, "Image smoothing"), 0.0)
    if sigma_arcmin > 0 and len(x_edges) > 1:
        pixel_arcmin = max(
            abs(float(np.median(np.diff(x_edges))))
            * np.cos(np.deg2rad(float(np.mean(y_edges))))
            * 60.0,
            1e-9,
        )
        for channel in range(3):
            values[..., channel] = gaussian_smooth_image(
                values[..., channel], sigma_arcmin / pixel_arcmin
            )
    gains = np.asarray(
        [state.rgb_red_gain, state.rgb_green_gain, state.rgb_blue_gain], dtype=float
    )
    for channel in range(3):
        scale = float(np.percentile(values[..., channel], 99.5))
        if scale > 0:
            values[..., channel] /= scale
        values[..., channel] *= gains[channel]
    values *= max(parse_float(state.rgb_brightness, "RGB brightness"), 0.0)
    gamma = max(parse_float(state.rgb_gamma, "RGB gamma"), 0.05)
    values = np.power(np.clip(values, 0.0, 1.0), 1.0 / gamma)
    return np.asarray(values * 255.0, dtype=np.uint8)


def preview_rgb_image() -> dict:
    points = finite_loaded_gc_points()
    if points.empty:
        raise ValueError("Load event data first")
    config = rgb_band_config()
    bands = config.bands()
    bins = max(24, int(float(state.gc_image_bins)))
    ra = points["RA"].to_numpy(dtype=float)
    dec = points["DEC"].to_numpy(dtype=float)
    if CURRENT_SKY_REGION is not None:
        ra = backend.unwrap_ra_for_selection(ra, CURRENT_SKY_REGION)
        spatial_range = list(backend.selection_image_range(CURRENT_SKY_REGION))
    else:
        spatial_range = [
            (float(np.nanmin(ra)), float(np.nanmax(ra))),
            (float(np.nanmin(dec)), float(np.nanmax(dec))),
        ]
    energy = points["KEV"].to_numpy(dtype=float)
    energy_sigma = max(parse_float(state.event_energy_sigma, "Energy smoothing"), 0.0)
    channel_images = []
    event_counts = []
    for (low, high), center, width in zip(
        bands,
        (config.red_center_kev, config.green_center_kev, config.blue_center_kev),
        (config.red_width_kev, config.green_width_kev, config.blue_width_kev),
    ):
        if energy_sigma > 0:
            sigma = np.hypot(float(width) / 2.355, energy_sigma)
            weights = np.exp(-0.5 * ((energy - float(center)) / max(sigma, 1e-9)) ** 2)
            mask = weights >= 1e-4
        else:
            mask = (energy >= low) & (energy <= high)
            weights = np.ones(len(energy), dtype=float)
        hist, x_edges, y_edges = np.histogram2d(
            ra[mask], dec[mask], bins=bins, range=spatial_range, weights=weights[mask]
        )
        channel_images.append(hist.T)
        event_counts.append(int(np.count_nonzero(mask)))
    return {
        "channels": np.stack(channel_images, axis=-1),
        "x_edges": x_edges,
        "y_edges": y_edges,
        "event_counts": event_counts,
        "bands": bands,
        "exact": False,
        "cache_hit": False,
    }


def update_rgb_image_chart(rgb_data: dict):
    if RGB_IMAGE_CHART is None:
        return
    image = _rgb_display_array(
        rgb_data["channels"], rgb_data["x_edges"], rgb_data["y_edges"]
    )
    x_edges = np.asarray(rgb_data["x_edges"], dtype=float)
    y_edges = np.asarray(rgb_data["y_edges"], dtype=float)
    quality = "Exact" if rgb_data.get("exact") else "Preview"
    figure = go.Figure(
        go.Image(
            z=image,
            x0=float(x_edges[0]),
            dx=float((x_edges[-1] - x_edges[0]) / image.shape[1]),
            y0=float(y_edges[0]),
            dy=float((y_edges[-1] - y_edges[0]) / image.shape[0]),
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=560,
        margin={"l": 62, "r": 20, "t": 64, "b": 54},
        title={
            "text": f"RGB energy composite | {quality} | R/G/B events {rgb_data['event_counts']}",
            "font": {"size": 15, "color": JAXA_BLUE},
        },
    )
    figure.update_xaxes(title_text="Right Ascension (deg)", autorange="reversed")
    figure.update_yaxes(title_text="Declination (deg)")
    RGB_IMAGE_CHART.update(figure)
    state.dirty("rgb_image_figure")


def update_rgb_plane(rgb_data: dict):
    global RGB_COMPOSITE_ACTOR
    if RGB_COMPOSITE_ACTOR is not None and PYVISTA_PLOTTER is not None:
        try:
            PYVISTA_PLOTTER.remove_actor(RGB_COMPOSITE_ACTOR, render=False)
        except Exception:
            pass
        if RGB_COMPOSITE_ACTOR in PYVISTA_ACTORS:
            PYVISTA_ACTORS.remove(RGB_COMPOSITE_ACTOR)
        RGB_COMPOSITE_ACTOR = None
    if (
        not bool(state.show_rgb_plane)
        or PYVISTA_PLOTTER is None
        or not CURRENT_PYVISTA_TRANSFORM
    ):
        return
    image = _rgb_display_array(
        rgb_data["channels"], rgb_data["x_edges"], rgb_data["y_edges"]
    )
    x_edges = np.asarray(rgb_data["x_edges"], dtype=float)
    y_edges = np.asarray(rgb_data["y_edges"], dtype=float)
    center_ra = CURRENT_PYVISTA_TRANSFORM["center_ra"]
    center_dec = CURRENT_PYVISTA_TRANSFORM["center_dec"]
    x = ((np.asarray([x_edges[0], x_edges[-1]]) - center_ra + 180) % 360 - 180)
    x *= np.cos(np.deg2rad(center_dec)) * 60.0
    y = (np.asarray([y_edges[0], y_edges[-1]]) - center_dec) * 60.0
    plane = pv.Plane(
        center=(float(np.mean(x)), float(np.mean(y)), z_value_to_scene(float(state.rgb_green_center))),
        direction=(0, 0, 1),
        i_size=max(abs(float(np.diff(x)[0])), 1e-6),
        j_size=max(abs(float(np.diff(y)[0])), 1e-6),
    )
    plane.texture_map_to_plane(inplace=True)
    RGB_COMPOSITE_ACTOR = PYVISTA_PLOTTER.add_mesh(
        plane,
        texture=pv.numpy_to_texture(np.flipud(image)),
        lighting=False,
        pickable=False,
        name="rgb_energy_image_plane",
        reset_camera=False,
    )
    PYVISTA_ACTORS.append(RGB_COMPOSITE_ACTOR)


def update_rgb_composite(exact: bool = False):
    try:
        config = rgb_band_config()
        config.bands()
        if exact:
            if CURRENT_SEARCH_RESULT is None or not CURRENT_SEARCH_RESULT.observations:
                raise ValueError("Load observations first")
            state.rgb_status = "Computing exact RGB image..."
            state.dirty("rgb_status")
            rgb_data = backend.exact_rgb_image(
                [item.record for item in CURRENT_SEARCH_RESULT.observations],
                config,
                bins=max(24, int(float(state.gc_image_bins))),
                region=CURRENT_SKY_REGION,
            )
        else:
            rgb_data = preview_rgb_image()
        update_rgb_image_chart(rgb_data)
        update_rgb_plane(rgb_data)
        quality = "Exact" if rgb_data.get("exact") else "Preview"
        state.rgb_status = f"{quality} RGB image | channel events {rgb_data['event_counts']}"
        state.dirty("rgb_status")
        pyvista_update_view()
    except Exception as error:
        state.rgb_status = str(error)
        state.dirty("rgb_status")


def select_energy_palette(mode: str, palette_key: str):
    if mode not in ENERGY_MAP_NAMES or palette_key not in PALETTE_OPTIONS:
        return
    state[f"gc_palette_pending_{mode}"] = palette_key
    state[f"gc_palette_{mode}"] = palette_key
    state.dirty(f"gc_palette_pending_{mode}", f"gc_palette_{mode}")
    update_gc_image()


def apply_energy_map_settings():
    try:
        for mode in ENERGY_MAP_NAMES:
            parse_float(getattr(state, f"gc_center_{mode}"), f"{mode} center")
            width = parse_float(getattr(state, f"gc_width_{mode}"), f"{mode} width")
            if width <= 0:
                raise ValueError(f"{ENERGY_MAP_NAMES[mode]} width must be greater than zero")
    except ValueError as error:
        state.status_message = str(error)
        state.dirty("status_message")
        return

    dirty_names = []
    for mode in ENERGY_MAP_NAMES:
        palette_key = str(getattr(state, f"gc_palette_pending_{mode}")).lower()
        if palette_key not in PALETTE_OPTIONS:
            palette_key = "inferno"
        setattr(state, f"gc_palette_{mode}", palette_key)
        dirty_names.append(f"gc_palette_{mode}")
    state.dirty(*dirty_names)
    update_gc_image()
    state.status_message = "Energy-map ranges and color palettes applied"
    state.dirty("status_message")


def select_pyvista_palette(palette_key: str):
    if palette_key not in PALETTE_OPTIONS:
        return
    state.pyvista_colormap_pending = palette_key
    state.pyvista_colormap = palette_key
    state.pyvista_color_mode = "pi"
    state.dirty(
        "pyvista_colormap_pending", "pyvista_colormap", "pyvista_color_mode"
    )
    if CURRENT_SEARCH_RESULT is None:
        state.pyvista_status = "3D color selected; load event data to display it"
        state.dirty("pyvista_status")
    else:
        rebuild_pyvista_from_current()


def apply_pyvista_palette():
    palette_key = str(state.pyvista_colormap_pending).lower()
    if palette_key not in PALETTE_OPTIONS:
        palette_key = "turbo"
    state.pyvista_colormap = palette_key
    state.pyvista_color_mode = "pi"
    state.dirty("pyvista_colormap", "pyvista_color_mode")
    if CURRENT_SEARCH_RESULT is None:
        state.pyvista_status = "3D color selected; load event data to display it"
        state.dirty("pyvista_status")
        return
    rebuild_pyvista_from_current()


def results_table_figure(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return empty_figure("No matching UDON3 server files")

    visible = frame.copy()
    visible["parquet"] = visible.apply(filename_for_row, axis=1)
    visible["cached_text"] = [cache_state(row) for _, row in visible.iterrows()]
    sep_values = (
        visible["separation_deg"].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.5f}"
        )
        if "separation_deg" in visible.columns
        else pd.Series([""] * len(visible))
    )

    figure = go.Figure(
        data=[
            go.Table(
                columnwidth=[48, 68, 72, 100, 210, 155, 230, 74, 92],
                header={
                    "values": [
                        "#",
                        "Mission",
                        "Instr.",
                        "Obs ID",
                        "Object",
                        "Date",
                        "Parquet file",
                        "Cached",
                        "Sep deg",
                    ],
                    "fill_color": JAXA_BLUE,
                    "font": {"color": "white", "size": 12},
                    "align": "left",
                    "height": 28,
                },
                cells={
                    "values": [
                        np.arange(1, len(visible) + 1),
                        visible["mission"].str.upper(),
                        visible["instrument"].str.upper(),
                        visible["observation_id"].astype(str),
                        visible["object"].astype(str),
                        visible["date_obs"].astype(str),
                        visible["parquet"],
                        visible["cached_text"],
                        sep_values,
                    ],
                    "fill_color": "#ffffff",
                    "font": {"color": "#172033", "size": 11},
                    "align": "left",
                    "height": 24,
                },
            )
        ]
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        # Keep every returned catalog row in the figure. The surrounding page
        # provides the scroll area, so the table itself must not truncate rows.
        height=max(520, 80 + len(visible) * 24),
        paper_bgcolor="#f8fafc",
    )
    return figure


def loaded_viewer_figure(result: backend.SearchResult, max_points: int) -> go.Figure:
    frames = []
    for observation in result.observations:
        frame = observation.frame.copy()
        frame["FILE"] = backend.record_label(observation.record)
        frame["OBJECT"] = str(observation.metadata.get("OBJECT", ""))
        frames.append(frame)

    if not frames:
        return empty_figure("No event rows loaded")

    points = pd.concat(frames, ignore_index=True)
    if len(points) > max_points:
        points = points.sample(max_points, random_state=42).sort_index()

    has_kev = "KEV" in points.columns
    if has_kev:
        customdata = np.column_stack(
            (
                points["FILE"].astype(str),
                points["OBJECT"].astype(str),
                points["TIME"].to_numpy(float),
                points["X"].to_numpy(float),
                points["Y"].to_numpy(float),
                points["KEV"].to_numpy(float),
            )
        )
        sky_hover = (
            "%{customdata[0]}<br>"
            "Object=%{customdata[1]}<br>"
            "RA=%{x:.6f} deg<br>DEC=%{y:.6f} deg<br>"
            "PI=%{marker.color:.0f}<br>"
            "keV=%{customdata[5]:.4f}<br>"
            "TIME=%{customdata[2]:.6f}<br>"
            "X=%{customdata[3]:.1f}, Y=%{customdata[4]:.1f}<extra></extra>"
        )
        time_hover = (
            "%{customdata[0]}<br>"
            "TIME=%{x:.6f}<br>PI=%{y:.0f}<br>"
            "keV=%{customdata[5]:.4f}<extra></extra>"
        )
    else:
        customdata = np.column_stack(
            (
                points["FILE"].astype(str),
                points["OBJECT"].astype(str),
                points["TIME"].to_numpy(float),
                points["X"].to_numpy(float),
                points["Y"].to_numpy(float),
            )
        )
        sky_hover = (
            "%{customdata[0]}<br>"
            "Object=%{customdata[1]}<br>"
            "RA=%{x:.6f} deg<br>DEC=%{y:.6f} deg<br>"
            "PI=%{marker.color:.0f}<br>"
            "TIME=%{customdata[2]:.6f}<br>"
            "X=%{customdata[3]:.1f}, Y=%{customdata[4]:.1f}<extra></extra>"
        )
        time_hover = (
            "%{customdata[0]}<br>"
            "TIME=%{x:.6f}<br>PI=%{y:.0f}<extra></extra>"
        )

    figure = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.14,
        subplot_titles=("Loaded events: RA / DEC", "Loaded events: TIME / PI"),
    )
    figure.add_trace(
        go.Scattergl(
            x=points["RA"],
            y=points["DEC"],
            mode="markers",
            marker={
                "size": 3,
                "opacity": 0.65,
                "color": points["PI"],
                "colorscale": "Turbo",
                "reversescale": True,
                "showscale": True,
                "colorbar": {"title": "PI"},
            },
            customdata=customdata,
            hovertemplate=sky_hover,
            name="sky",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=points["TIME"],
            y=points["PI"],
            mode="markers",
            marker={
                "size": 3,
                "opacity": 0.45,
                "color": points["PI"],
                "colorscale": "Turbo",
                "reversescale": True,
                "showscale": False,
            },
            customdata=customdata,
            hovertemplate=time_hover,
            name="time-pi",
        ),
        row=2,
        col=1,
    )
    figure.update_xaxes(title_text="Right Ascension (deg)", row=1, col=1)
    figure.update_yaxes(title_text="Declination (deg)", row=1, col=1)
    figure.update_xaxes(title_text="TIME (mission seconds)", row=2, col=1)
    figure.update_yaxes(title_text="PI channel", row=2, col=1)
    figure.update_layout(
        template="plotly_white",
        height=760,
        margin={"l": 70, "r": 30, "t": 70, "b": 50},
        paper_bgcolor="#f8fafc",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": "#172033"},
    )
    return figure


def pyvista_state_set(**values):
    current_state = globals().get("state")
    if current_state is None:
        return
    dirty_names = []
    for name, value in values.items():
        setattr(current_state, name, value)
        dirty_names.append(name)
    if dirty_names:
        current_state.dirty(*dirty_names)


def pyvista_update_view(reset_camera: bool = False):
    if PYVISTA_PLOTTER is None:
        return
    if reset_camera:
        PYVISTA_PLOTTER.reset_camera()
    if PYVISTA_VIEW is not None:
        if reset_camera and hasattr(PYVISTA_VIEW, "reset_camera"):
            PYVISTA_VIEW.reset_camera()
        elif hasattr(PYVISTA_VIEW, "update"):
            PYVISTA_VIEW.update()


def clear_slice_actors(index: int | None = None):
    if PYVISTA_PLOTTER is None:
        return
    indices = list(SLICE_ACTORS) if index is None else [index]
    for slice_index in indices:
        for actor in list(SLICE_ACTORS.get(slice_index, [])):
            try:
                PYVISTA_PLOTTER.remove_actor(actor, render=False)
            except Exception:
                pass
        SLICE_ACTORS.pop(slice_index, None)
        SLICE_TEXTURE_ACTORS.pop(slice_index, None)


def clear_pyvista_scene(message: str = "", reset_camera: bool = False):
    global PYVISTA_ACTORS, PYVISTA_DATA_ACTORS
    global PYVISTA_PICK_LOOKUP, PYVISTA_SELECTED_ACTOR, RGB_COMPOSITE_ACTOR

    if PYVISTA_PLOTTER is None:
        pyvista_state_set(
            pyvista_status=(
                "PyVista is not available. Install pyvista and trame-vtk."
            )
        )
        return

    clear_slice_actors()
    for actor in list(PYVISTA_ACTORS):
        try:
            PYVISTA_PLOTTER.remove_actor(actor, render=False)
        except Exception:
            pass
    try:
        PYVISTA_PLOTTER.remove_scalar_bar("PI")
    except Exception:
        pass
    PYVISTA_ACTORS = []
    PYVISTA_DATA_ACTORS = []
    PYVISTA_PICK_LOOKUP = {}
    RGB_COMPOSITE_ACTOR = None

    if PYVISTA_SELECTED_ACTOR is not None:
        try:
            PYVISTA_PLOTTER.remove_actor(PYVISTA_SELECTED_ACTOR, render=False)
        except Exception:
            pass
        PYVISTA_SELECTED_ACTOR = None

    if message:
        actor = PYVISTA_PLOTTER.add_text(
            message,
            position="upper_left",
            font_size=12,
            color=JAXA_BLUE,
        )
        PYVISTA_ACTORS.append(actor)
        pyvista_state_set(pyvista_status=message)
    pyvista_update_view(reset_camera=reset_camera)


def selected_point_value(value, precision: int = 6) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def pyvista_point_picked(point, picker):
    global PYVISTA_SELECTED_ACTOR

    if pv is None or picker is None:
        return
    dataset = picker.GetDataSet()
    point_id = picker.GetPointId()
    if dataset is None or point_id < 0:
        return

    mesh = pv.wrap(dataset)
    if "PICK_ID" not in mesh.point_data:
        return
    pick_id = int(mesh["PICK_ID"][point_id])
    details = PYVISTA_PICK_LOOKUP.get(pick_id)
    if not details:
        return

    pyvista_state_set(
        pyvista_has_selection=True,
        pyvista_selected_file=details["file"],
        pyvista_selected_object=details["object"],
        pyvista_selected_row=str(details["source_row"]),
        pyvista_selected_time=selected_point_value(details["time"], precision=6),
        pyvista_selected_mission_datetime=details["mission_datetime"],
        pyvista_selected_pi=selected_point_value(details["pi"], precision=0),
        pyvista_selected_kev=selected_point_value(details.get("kev"), precision=4),
        pyvista_selected_ra=selected_point_value(details["ra"], precision=6),
        pyvista_selected_dec=selected_point_value(details["dec"], precision=6),
        pyvista_selected_xy=(
            f"X {selected_point_value(details['x'], precision=1)}, "
            f"Y {selected_point_value(details['y'], precision=1)}"
        ),
        pyvista_status=f"Selected {details['file']} row {details['source_row']}",
    )

    if PYVISTA_PLOTTER is not None:
        if PYVISTA_SELECTED_ACTOR is not None:
            try:
                PYVISTA_PLOTTER.remove_actor(PYVISTA_SELECTED_ACTOR, render=False)
            except Exception:
                pass
        try:
            selected_point = np.asarray(point, dtype=float).reshape(1, 3)
            if not np.isfinite(selected_point).all():
                raise ValueError("Picked point is not finite")
        except Exception:
            selected_point = np.asarray(mesh.points[point_id], dtype=float).reshape(1, 3)
        point_size = DEFAULT_PYVISTA_POINT_SIZE
        current_state = globals().get("state")
        if current_state is not None:
            point_size = int(float(current_state.pyvista_point_size))
        PYVISTA_SELECTED_ACTOR = PYVISTA_PLOTTER.add_points(
            selected_point,
            color="#d71920",
            point_size=max(point_size * 2, 9),
            render_points_as_spheres=True,
            reset_camera=False,
        )
        pyvista_update_view()


def initialize_pyvista_plotter():
    global PYVISTA_AVAILABLE, PYVISTA_IMPORT_ERROR

    if not PYVISTA_AVAILABLE:
        return None

    try:
        pv.OFF_SCREEN = True
        plotter = pv.Plotter(off_screen=True, window_size=(1200, 720))
        plotter.set_background("#ffffff")
        try:
            plotter.enable_anti_aliasing("fxaa")
        except Exception:
            pass
        try:
            plotter.add_axes(line_width=2, color=JAXA_BLUE)
        except Exception:
            pass
        plotter.enable_point_picking(
            callback=pyvista_point_picked,
            use_picker=True,
            picker="point",
            left_clicking=True,
            show_point=True,
            color="#d71920",
            point_size=12,
            tolerance=0.025,
            show_message=False,
        )
        return plotter
    except Exception as error:
        PYVISTA_AVAILABLE = False
        PYVISTA_IMPORT_ERROR = error
        return None


def pyvista_points_key(result: backend.SearchResult) -> tuple:
    return tuple(
        (
            backend.record_key(observation.record),
            id(observation.frame),
            len(observation.frame),
            tuple(observation.frame.columns),
        )
        for observation in result.observations
    )


def build_pyvista_event_points(result: backend.SearchResult) -> pd.DataFrame:
    frames = []
    for observation_index, observation in enumerate(result.observations):
        frame = observation.frame.copy()
        if frame.empty:
            continue
        required = ["RA", "DEC", "PI", "TIME", "X", "Y", "SOURCE_ROW"]
        if any(column not in frame.columns for column in required):
            continue
        frame["OBS_INDEX"] = observation_index
        frame["FILE_LABEL"] = backend.record_label(observation.record)
        frame["OBJECT"] = str(observation.metadata.get("OBJECT", ""))
        frame["MISSION"] = observation.record.mission.upper()
        frame["INSTRUMENT"] = observation.record.instrument.upper()
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    points = pd.concat(frames, ignore_index=True)
    numeric_columns = ["RA", "DEC", "PI", "TIME", "X", "Y"]
    if "KEV" in points.columns:
        numeric_columns.append("KEV")
    valid = np.ones(len(points), dtype=bool)
    for column in numeric_columns:
        valid &= np.isfinite(points[column].to_numpy(dtype=float))
    points = points.loc[valid].copy()
    if points.empty:
        return points

    return points.reset_index(drop=True)


def pyvista_event_points(result: backend.SearchResult, point_limit: int) -> pd.DataFrame:
    global CURRENT_PYVISTA_POINTS, CURRENT_PYVISTA_POINTS_KEY

    key = pyvista_points_key(result)
    if key != CURRENT_PYVISTA_POINTS_KEY:
        CURRENT_PYVISTA_POINTS = build_pyvista_event_points(result)
        CURRENT_PYVISTA_POINTS_KEY = key

    if CURRENT_PYVISTA_POINTS.empty:
        return CURRENT_PYVISTA_POINTS

    point_limit = max(1, int(point_limit))
    if len(CURRENT_PYVISTA_POINTS) > point_limit:
        return (
            CURRENT_PYVISTA_POINTS
            .sample(point_limit, random_state=42)
            .sort_index()
            .reset_index(drop=True)
        )
    return CURRENT_PYVISTA_POINTS.copy()


def scene_transform_from_points(points: pd.DataFrame, z_source: np.ndarray) -> dict:
    ra = points["RA"].to_numpy(dtype=float)
    dec = points["DEC"].to_numpy(dtype=float)
    if CURRENT_SKY_REGION is not None:
        center_ra = float(CURRENT_SKY_REGION.center_ra_deg)
        center_dec = float(CURRENT_SKY_REGION.center_dec_deg)
    else:
        center_ra = float(np.nanmedian(ra))
        center_dec = float(np.nanmedian(dec))
    ra_offset = (ra - center_ra + 180.0) % 360.0 - 180.0
    x_arcmin = ra_offset * np.cos(np.deg2rad(center_dec)) * 60.0
    y_arcmin = (dec - center_dec) * 60.0
    x_span = max(float(np.nanmax(x_arcmin) - np.nanmin(x_arcmin)), 1e-6)
    y_span = max(float(np.nanmax(y_arcmin) - np.nanmin(y_arcmin)), 1e-6)
    z_min = float(np.nanmin(z_source))
    z_max = float(np.nanmax(z_source))
    return {
        "center_ra": center_ra,
        "center_dec": center_dec,
        "spatial_span": max(x_span, y_span, 1.0),
        "x_min": float(np.nanmin(x_arcmin)),
        "x_max": float(np.nanmax(x_arcmin)),
        "y_min": float(np.nanmin(y_arcmin)),
        "y_max": float(np.nanmax(y_arcmin)),
        "z_min": z_min,
        "z_max": z_max,
        "z_span": max(z_max - z_min, 1e-6),
    }


def apply_scene_transform(
    points: pd.DataFrame,
    transform: dict,
    z_column: str,
) -> pd.DataFrame:
    transformed = points.copy()
    ra = transformed["RA"].to_numpy(dtype=float)
    dec = transformed["DEC"].to_numpy(dtype=float)
    z_source = transformed[z_column].to_numpy(dtype=float)
    ra_offset = (ra - transform["center_ra"] + 180.0) % 360.0 - 180.0
    transformed["X_3D"] = ra_offset * np.cos(np.deg2rad(transform["center_dec"])) * 60.0
    transformed["Y_3D"] = (dec - transform["center_dec"]) * 60.0
    transformed["Z_3D"] = (
        (z_source - transform["z_min"])
        / transform["z_span"]
        * transform["spatial_span"]
    )
    return transformed


def z_value_to_scene(z_value: float) -> float:
    if not CURRENT_PYVISTA_TRANSFORM:
        return 0.0
    return (
        (float(z_value) - CURRENT_PYVISTA_TRANSFORM["z_min"])
        / CURRENT_PYVISTA_TRANSFORM["z_span"]
        * CURRENT_PYVISTA_TRANSFORM["spatial_span"]
    )


def voxel_source_points(points: pd.DataFrame) -> pd.DataFrame:
    """Apply the existing slice-mode semantics before aggregating voxel cells."""
    if "KEV" not in points.columns or points.empty:
        return pd.DataFrame()
    enabled = [
        slice_config(index)
        for index in active_slice_indices()
        if slice_config(index)["enabled"]
    ]
    mode = str(getattr(state, "slice_3d_mode", "all"))
    if mode == "cloud" or not enabled:
        return points.copy()
    if mode == "active":
        index = max(0, min(int(state.active_slice_index), len(enabled) - 1))
        config = next(
            (item for item in enabled if item["index"] == index), enabled[0]
        )
        return points.loc[points["KEV"].between(config["low"], config["high"])].copy()
    mask = np.zeros(len(points), dtype=bool)
    energy = points["KEV"].to_numpy(dtype=float)
    for config in enabled:
        mask |= (energy >= config["low"]) & (energy <= config["high"])
    return points.loc[mask].copy()


def hex_to_rgb(color: str) -> np.ndarray:
    value = str(color).lstrip("#")
    if len(value) != 6:
        return np.asarray([100, 116, 139], dtype=np.uint8)
    return np.asarray([int(value[index:index + 2], 16) for index in (0, 2, 4)], dtype=np.uint8)


def build_voxel_table(points: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    source = voxel_source_points(points)
    if source.empty:
        return pd.DataFrame(), {"input_events": 0, "capped": False}
    spatial_size = max(parse_float(state.voxel_spatial_size, "Voxel spatial size"), 0.01)
    energy_size = max(parse_float(state.voxel_energy_size, "Voxel energy size"), 0.001)
    source = source.copy()
    x_origin = np.floor(float(source["X_3D"].min()) / spatial_size) * spatial_size
    y_origin = np.floor(float(source["Y_3D"].min()) / spatial_size) * spatial_size
    energy_origin = np.floor(float(source["KEV"].min()) / energy_size) * energy_size
    source["VOXEL_X"] = np.floor((source["X_3D"] - x_origin) / spatial_size).astype(int)
    source["VOXEL_Y"] = np.floor((source["Y_3D"] - y_origin) / spatial_size).astype(int)
    source["VOXEL_Z"] = np.floor((source["KEV"] - energy_origin) / energy_size).astype(int)

    rows = []
    color_mode = str(state.pyvista_color_mode)
    for (x_index, y_index, z_index), group in source.groupby(
        ["VOXEL_X", "VOXEL_Y", "VOXEL_Z"], sort=False
    ):
        count = len(group)
        if color_mode == "mission":
            mission = group["MISSION"].value_counts().sort_index().idxmax()
            rgb = hex_to_rgb(MISSION_COLORS.get(str(mission), "#64748b"))
        elif color_mode == "rgb":
            rgb = np.mean(rgb_event_colors(group["KEV"].to_numpy(dtype=float)), axis=0)
        else:
            rgb = np.asarray([0, 0, 0], dtype=np.uint8)
        rows.append(
            {
                "VOXEL_X": int(x_index),
                "VOXEL_Y": int(y_index),
                "VOXEL_Z": int(z_index),
                "COUNT": count,
                "MEAN_PI": float(group["PI"].mean()),
                "MEAN_KEV": float(group["KEV"].mean()),
                "R": int(rgb[0]), "G": int(rgb[1]), "B": int(rgb[2]),
            }
        )
    voxels = pd.DataFrame(rows)
    voxels = voxels.sort_values(
        ["COUNT", "VOXEL_X", "VOXEL_Y", "VOXEL_Z"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    capped = len(voxels) > MAX_RENDERED_VOXELS
    voxels = voxels.head(MAX_RENDERED_VOXELS).copy()
    voxels["X_3D"] = x_origin + (voxels["VOXEL_X"] + 0.5) * spatial_size
    voxels["Y_3D"] = y_origin + (voxels["VOXEL_Y"] + 0.5) * spatial_size
    energy_centers = energy_origin + (voxels["VOXEL_Z"] + 0.5) * energy_size
    voxels["Z_3D"] = [z_value_to_scene(value) for value in energy_centers]
    maximum = max(int(voxels["COUNT"].max()), 1)
    opacity = np.clip(
        55 + 200 * np.sqrt(voxels["COUNT"].to_numpy(dtype=float) / maximum),
        55,
        255,
    ).astype(np.uint8)
    if color_mode == "pi":
        low = float(voxels["MEAN_PI"].min())
        high = max(float(voxels["MEAN_PI"].max()), low + 1.0)
        normalized = np.clip((voxels["MEAN_PI"] - low) / (high - low), 0.0, 1.0)
        # The palette mapper below handles PI colors; use opaque white RGBA here.
        voxels[["R", "G", "B"]] = 255
    voxels["RGBA"] = [
        np.asarray([red, green, blue, alpha], dtype=np.uint8)
        for red, green, blue, alpha in zip(voxels.R, voxels.G, voxels.B, opacity)
    ]
    return voxels, {
        "input_events": len(source),
        "capped": capped,
        "spatial_size": spatial_size,
        "energy_size": energy_size,
        "color_mode": color_mode,
    }


def add_voxel_volume(points: pd.DataFrame) -> tuple[int, dict]:
    voxels, info = build_voxel_table(points)
    if voxels.empty:
        return 0, info
    cloud = pv.PolyData(voxels[["X_3D", "Y_3D", "Z_3D"]].to_numpy(dtype=float))
    cloud["VOXEL_COUNT"] = voxels["COUNT"].to_numpy(dtype=np.int32)
    cloud["MEAN_PI"] = voxels["MEAN_PI"].to_numpy(dtype=np.float32)
    cloud["RGBA"] = np.vstack(voxels["RGBA"].to_numpy())
    cube = pv.Cube(
        center=(0, 0, 0),
        x_length=info["spatial_size"],
        y_length=info["spatial_size"],
        z_length=max(
            info["energy_size"]
            / max(CURRENT_PYVISTA_TRANSFORM["z_span"], 1e-9)
            * CURRENT_PYVISTA_TRANSFORM["spatial_span"],
            1e-4,
        ),
    )
    mesh = cloud.glyph(geom=cube, scale=False, orient=False)
    cells_per_voxel = max(int(cube.n_cells), 1)
    mesh.cell_data["VOXEL_COUNT"] = np.repeat(
        voxels["COUNT"].to_numpy(dtype=np.int32), cells_per_voxel
    )
    mesh.cell_data["MEAN_PI"] = np.repeat(
        voxels["MEAN_PI"].to_numpy(dtype=np.float32), cells_per_voxel
    )
    mesh.cell_data["RGBA"] = np.repeat(
        np.vstack(voxels["RGBA"].to_numpy()), cells_per_voxel, axis=0
    )
    if info["color_mode"] == "pi":
        actor = PYVISTA_PLOTTER.add_mesh(
            mesh,
            scalars="MEAN_PI",
            cmap=PALETTE_OPTIONS.get(
                str(state.pyvista_colormap).lower(), PALETTE_OPTIONS["turbo"]
            )["pyvista"],
            opacity=0.88,
            name="pyvista_voxels",
            reset_camera=False,
        )
    else:
        actor = PYVISTA_PLOTTER.add_mesh(
            mesh,
            scalars="RGBA",
            rgb=True,
            opacity=0.96,
            name="pyvista_voxels",
            reset_camera=False,
        )
    PYVISTA_ACTORS.append(actor)
    PYVISTA_DATA_ACTORS.append(actor)
    return len(voxels), info


def add_slice_texture(index: int, image_data: dict) -> object | None:
    if PYVISTA_PLOTTER is None or not CURRENT_PYVISTA_TRANSFORM or not image_data:
        return None
    previous = SLICE_TEXTURE_ACTORS.pop(index, None)
    if previous is not None:
        try:
            PYVISTA_PLOTTER.remove_actor(previous, render=False)
        except Exception:
            pass
        if previous in SLICE_ACTORS.get(index, []):
            SLICE_ACTORS[index].remove(previous)

    display = smoothed_slice_histogram(image_data)
    if display.size == 0 or float(np.nanmax(display)) <= 0:
        return None
    normalized = np.log1p(np.maximum(display, 0.0))
    normalized /= max(float(np.nanmax(normalized)), 1e-9)
    color = str(image_data["config"]["color"]).lstrip("#")
    rgb = np.asarray([int(color[offset:offset + 2], 16) for offset in (0, 2, 4)])
    rgba = np.empty((*normalized.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb.reshape(1, 1, 3)
    rgba[..., 3] = np.asarray(np.clip(normalized * 235.0, 0, 235), dtype=np.uint8)
    rgba = np.flipud(rgba)

    x_edges = np.asarray(image_data["x_edges"], dtype=float)
    y_edges = np.asarray(image_data["y_edges"], dtype=float)
    center_ra = CURRENT_PYVISTA_TRANSFORM["center_ra"]
    center_dec = CURRENT_PYVISTA_TRANSFORM["center_dec"]
    x_offsets = (
        (np.asarray([x_edges[0], x_edges[-1]]) - center_ra + 180.0) % 360.0 - 180.0
    ) * np.cos(np.deg2rad(center_dec)) * 60.0
    y_offsets = (np.asarray([y_edges[0], y_edges[-1]]) - center_dec) * 60.0
    x_size = max(abs(float(x_offsets[1] - x_offsets[0])), 1e-6)
    y_size = max(abs(float(y_offsets[1] - y_offsets[0])), 1e-6)
    plane = pv.Plane(
        center=(
            float(np.mean(x_offsets)),
            float(np.mean(y_offsets)),
            z_value_to_scene(image_data["config"]["center"]),
        ),
        direction=(0, 0, 1),
        i_size=x_size,
        j_size=y_size,
    )
    plane.texture_map_to_plane(inplace=True)
    actor = PYVISTA_PLOTTER.add_mesh(
        plane,
        texture=pv.numpy_to_texture(rgba),
        lighting=False,
        pickable=False,
        name=f"slice_{index}_image_texture",
        reset_camera=False,
    )
    SLICE_TEXTURE_ACTORS[index] = actor
    SLICE_ACTORS.setdefault(index, []).append(actor)
    return actor


def update_pyvista_scene(result: backend.SearchResult, point_limit: int | None = None):
    global PYVISTA_PICK_LOOKUP, CURRENT_SEARCH_RESULT, CURRENT_PYVISTA_SCENE_POINTS
    global CURRENT_PYVISTA_TRANSFORM

    CURRENT_SEARCH_RESULT = result
    state.pyvista_has_selection = False
    state.dirty("pyvista_has_selection")
    if PYVISTA_PLOTTER is None:
        clear_pyvista_scene()
        return

    point_limit = (
        int(point_limit)
        if point_limit is not None
        else int(float(state.pyvista_point_limit))
    )
    points = pyvista_event_points(result, point_limit=point_limit)
    if points.empty:
        clear_pyvista_scene("No finite RA/DEC/PI rows available for PyVista 3D", reset_camera=True)
        return

    clear_pyvista_scene(reset_camera=False)

    pi_values = points["PI"].to_numpy(dtype=float)
    has_kev = "KEV" in points.columns and points["KEV"].notna().any()
    z_column = "KEV" if has_kev else "PI"
    z_source = points[z_column].to_numpy(dtype=float)
    CURRENT_PYVISTA_TRANSFORM = scene_transform_from_points(points, z_source)
    points = apply_scene_transform(points, CURRENT_PYVISTA_TRANSFORM, z_column)
    smoothing_enabled = (
        float(state.event_spatial_sigma) > 0 or float(state.event_energy_sigma) > 0
    )
    points["DENSITY"] = event_density_values(points)
    CURRENT_PYVISTA_SCENE_POINTS = points.copy()

    pi_min = float(np.nanmin(pi_values))
    pi_max = max(float(np.nanmax(pi_values)), pi_min + 1.0)
    pick_id_start = 0
    scalar_bar_added = False
    color_mode = str(getattr(state, "pyvista_color_mode", "mission"))
    use_mission_colors = color_mode == "mission"
    use_rgb_colors = color_mode == "rgb" and has_kev

    if str(getattr(state, "pyvista_display_mode", "points")) == "voxels":
        voxel_count, voxel_info = add_voxel_volume(points)
        if voxel_count == 0:
            clear_pyvista_scene("No events available for the selected voxel slice mode", reset_camera=True)
            return
        bounds_actor = PYVISTA_PLOTTER.show_bounds(
            xtitle="RA offset (arcmin)",
            ytitle="DEC offset (arcmin)",
            ztitle="Energy scale",
            color=JAXA_BLUE,
            grid="back",
            location="outer",
            font_size=10,
        )
        if bounds_actor is not None:
            PYVISTA_ACTORS.append(bounds_actor)
        missions = sorted(points["MISSION"].dropna().astype(str).unique())
        title_actor = PYVISTA_PLOTTER.add_text(
            "UDON3 PyVista 3D voxel volume | " + " / ".join(missions),
            position="upper_left",
            font_size=12,
            color=JAXA_BLUE,
        )
        PYVISTA_ACTORS.append(title_actor)
        cap_text = " | capped at 20,000" if voxel_info["capped"] else ""
        pyvista_state_set(
            pyvista_status=(
                f"PyVista 3D voxels: {voxel_count:,} occupied cells from "
                f"{voxel_info['input_events']:,} preview events | "
                f"{voxel_info['spatial_size']:g} arcmin x "
                f"{voxel_info['energy_size']:g} keV | color by {color_mode}{cap_text}"
            )
        )
        update_all_slices(reset_camera=False)
        update_gc_image()
        update_rgb_composite(exact=False)
        pyvista_update_view(reset_camera=True)
        return

    for observation_index, group in points.groupby("OBS_INDEX", sort=False):
        coordinates = group[["X_3D", "Y_3D", "Z_3D"]].to_numpy(dtype=float)
        cloud = pv.PolyData(coordinates)
        count = len(group)
        pick_ids = np.arange(pick_id_start, pick_id_start + count, dtype=np.int64)
        pick_id_start += count

        cloud["PICK_ID"] = pick_ids
        cloud["PI"] = group["PI"].to_numpy(dtype=np.float32)
        cloud["TIME"] = group["TIME"].to_numpy(dtype=np.float64)
        cloud["RA"] = group["RA"].to_numpy(dtype=np.float64)
        cloud["DEC"] = group["DEC"].to_numpy(dtype=np.float64)
        cloud["X_RAW"] = group["X"].to_numpy(dtype=np.float32)
        cloud["Y_RAW"] = group["Y"].to_numpy(dtype=np.float32)
        cloud["SOURCE_ROW"] = group["SOURCE_ROW"].to_numpy(dtype=np.int64)
        if has_kev and "KEV" in group.columns:
            cloud["KEV"] = group["KEV"].to_numpy(dtype=np.float32)
        density = group["DENSITY"].to_numpy(dtype=np.float32)
        base_scale = max(CURRENT_PYVISTA_TRANSFORM["spatial_span"] * 0.004, 0.01)
        cloud["SPLAT_SCALE"] = np.asarray(
            base_scale
            * (1.0 + float(state.density_size_strength) * 2.5 * density),
            dtype=np.float32,
        )
        cloud["SPLAT_OPACITY"] = np.asarray(
            np.clip(
                0.18
                + (0.62 + 0.18 * float(state.density_opacity_strength)) * density,
                0.05,
                1.0,
            ),
            dtype=np.float32,
        )

        for pick_id, (_, row) in zip(pick_ids, group.iterrows()):
            PYVISTA_PICK_LOOKUP[int(pick_id)] = {
                "file": str(row["FILE_LABEL"]),
                "object": str(row["OBJECT"]),
                "mission": str(row["MISSION"]),
                "instrument": str(row["INSTRUMENT"]),
                "source_row": int(row["SOURCE_ROW"]),
                "time": float(row["TIME"]),
                "mission_datetime": str(row.get("MISSION_DATETIME", "-")),
                "pi": float(row["PI"]),
                "kev": float(row["KEV"]) if has_kev and "KEV" in row else None,
                "ra": float(row["RA"]),
                "dec": float(row["DEC"]),
                "x": float(row["X"]),
                "y": float(row["Y"]),
            }

        mission = str(group["MISSION"].iloc[0])
        actor_name = (
            "pyvista_"
            f"{mission.lower()}_"
            f"{str(group['INSTRUMENT'].iloc[0]).lower()}_"
            f"{int(observation_index)}"
        )
        point_style = "points_gaussian" if smoothing_enabled else "points"
        if use_mission_colors:
            actor = PYVISTA_PLOTTER.add_points(
                cloud,
                color=MISSION_COLORS.get(mission, "#64748b"),
                point_size=int(float(state.pyvista_point_size)),
                opacity=0.78,
                style=point_style,
                render_points_as_spheres=not smoothing_enabled,
                name=actor_name,
                reset_camera=False,
            )
        elif use_rgb_colors:
            cloud["RGB"] = rgb_event_colors(group["KEV"].to_numpy(dtype=float))
            actor = PYVISTA_PLOTTER.add_points(
                cloud,
                scalars="RGB",
                rgb=True,
                point_size=int(float(state.pyvista_point_size)),
                opacity=0.86,
                style=point_style,
                render_points_as_spheres=not smoothing_enabled,
                name=actor_name,
                reset_camera=False,
            )
        else:
            actor = PYVISTA_PLOTTER.add_points(
                cloud,
                scalars="PI",
                cmap=PALETTE_OPTIONS.get(
                    str(state.pyvista_colormap).lower(),
                    PALETTE_OPTIONS["turbo"],
                )["pyvista"],
                clim=(pi_min, pi_max),
                point_size=int(float(state.pyvista_point_size)),
                opacity=0.78,
                style=point_style,
                render_points_as_spheres=not smoothing_enabled,
                show_scalar_bar=not scalar_bar_added,
                scalar_bar_args={
                    "title": "PI",
                    "color": JAXA_BLUE,
                    "title_font_size": 12,
                    "label_font_size": 10,
                },
                name=actor_name,
                reset_camera=False,
            )
            scalar_bar_added = True
        if smoothing_enabled:
            try:
                actor.mapper.SetScaleArray("SPLAT_SCALE")
                actor.mapper.SetScaleFactor(1.0)
                actor.mapper.SetOpacityArray("SPLAT_OPACITY")
            except Exception:
                pass
        PYVISTA_ACTORS.append(actor)
        PYVISTA_DATA_ACTORS.append(actor)

    bounds_actor = PYVISTA_PLOTTER.show_bounds(
        xtitle="RA offset (arcmin)",
        ytitle="DEC offset (arcmin)",
        ztitle="Energy scale",
        color=JAXA_BLUE,
        grid="back",
        location="outer",
        font_size=10,
    )
    if bounds_actor is not None:
        PYVISTA_ACTORS.append(bounds_actor)
    missions = sorted(points["MISSION"].dropna().astype(str).unique())
    mission_text = " / ".join(missions)
    title_actor = PYVISTA_PLOTTER.add_text(
        f"UDON3 PyVista 3D event cloud | {mission_text}",
        position="upper_left",
        font_size=12,
        color=JAXA_BLUE,
    )
    PYVISTA_ACTORS.append(title_actor)
    if use_mission_colors:
        color_text = "Mission colors: " + " | ".join(
            f"{mission} {MISSION_COLORS.get(mission, '#64748b')}"
            for mission in missions
        )
        color_actor = PYVISTA_PLOTTER.add_text(
            color_text,
            position="lower_left",
            font_size=9,
            color=JAXA_BLUE,
        )
        PYVISTA_ACTORS.append(color_actor)
    z_units = "keV" if has_kev else "PI"
    pyvista_state_set(
        pyvista_status=(
            f"PyVista 3D: {len(points):,} points, "
            f"{points['OBS_INDEX'].nunique()} file(s), "
            f"{len(missions)} mission(s), "
            f"color by {'mission' if use_mission_colors else ('RGB energy' if use_rgb_colors else 'PI')}, "
            f"Z scaled from {CURRENT_PYVISTA_TRANSFORM['z_min']:.4g}-"
            f"{CURRENT_PYVISTA_TRANSFORM['z_max']:.4g} {z_units}"
        )
    )
    update_all_slices(reset_camera=False)
    update_gc_image()
    update_rgb_composite(exact=False)
    pyvista_update_view(reset_camera=True)


def slice_source_points(config: dict) -> pd.DataFrame:
    if CURRENT_PYVISTA_POINTS.empty or "KEV" not in CURRENT_PYVISTA_POINTS.columns:
        return pd.DataFrame()
    kev = CURRENT_PYVISTA_POINTS["KEV"].to_numpy(dtype=float)
    mask = (
        np.isfinite(kev)
        & (kev >= config["low"])
        & (kev <= config["high"])
    )
    return CURRENT_PYVISTA_POINTS.loc[mask].copy()


def update_slice_status(index: int, event_count: int, config: dict):
    setattr(
        state,
        slice_state_name(index, "status"),
        f"Preview | {config['low']:.3f}-{config['high']:.3f} keV | "
        f"{event_count:,} events",
    )
    state.dirty(slice_state_name(index, "status"))


def update_slice_overlay(index: int, point_limit: int | None = None):
    clear_slice_actors(index)
    config = slice_config(index)
    if not config["enabled"]:
        update_slice_status(index, 0, config)
        update_slice_chart(index, slice_empty_figure(f"Slice {index + 1} hidden", config["color"]))
        return

    source = slice_source_points(config)
    update_slice_status(index, len(source), config)
    update_slice_chart(index, slice_heatmap_figure(index, source, config))
    if (
        PYVISTA_PLOTTER is None
        or source.empty
        or not CURRENT_PYVISTA_TRANSFORM
        or "KEV" not in source.columns
    ):
        return

    point_limit = max(
        1,
        int(
            point_limit
            if point_limit is not None
            else float(state.slice_point_limit)
        ),
    )
    if len(source) > point_limit:
        source = source.sample(point_limit, random_state=42).sort_index()
    source = apply_scene_transform(source, CURRENT_PYVISTA_TRANSFORM, "KEV")
    coordinates = source[["X_3D", "Y_3D", "Z_3D"]].to_numpy(dtype=float)
    actors = []

    if len(coordinates):
        cloud = pv.PolyData(coordinates)
        pick_ids = -(
            (index + 1) * 10_000_000
            + np.arange(len(source), dtype=np.int64)
            + 1
        )
        cloud["PICK_ID"] = pick_ids
        cloud["KEV"] = source["KEV"].to_numpy(dtype=np.float32)
        cloud["PI"] = source["PI"].to_numpy(dtype=np.float32)
        for pick_id, (_, row) in zip(pick_ids, source.iterrows()):
            PYVISTA_PICK_LOOKUP[int(pick_id)] = {
                "file": str(row.get("FILE_LABEL", "-")),
                "object": str(row.get("OBJECT", "-")),
                "mission": str(row.get("MISSION", "-")),
                "instrument": str(row.get("INSTRUMENT", "-")),
                "source_row": int(row.get("SOURCE_ROW", -1)),
                "time": float(row.get("TIME", np.nan)),
                "mission_datetime": str(row.get("MISSION_DATETIME", "-")),
                "pi": float(row["PI"]),
                "kev": float(row["KEV"]),
                "ra": float(row["RA"]),
                "dec": float(row["DEC"]),
                "x": float(row.get("X", np.nan)),
                "y": float(row.get("Y", np.nan)),
            }
        actor = PYVISTA_PLOTTER.add_points(
            cloud,
            color=config["color"],
            point_size=max(int(float(state.pyvista_point_size)) + 2, 4),
            opacity=0.92,
            render_points_as_spheres=True,
            name=f"slice_{index}_points",
            reset_camera=False,
        )
        actors.append(actor)

    x_size = max(
        CURRENT_PYVISTA_TRANSFORM["x_max"] - CURRENT_PYVISTA_TRANSFORM["x_min"],
        1.0,
    )
    y_size = max(
        CURRENT_PYVISTA_TRANSFORM["y_max"] - CURRENT_PYVISTA_TRANSFORM["y_min"],
        1.0,
    )
    x_center = 0.5 * (
        CURRENT_PYVISTA_TRANSFORM["x_min"] + CURRENT_PYVISTA_TRANSFORM["x_max"]
    )
    y_center = 0.5 * (
        CURRENT_PYVISTA_TRANSFORM["y_min"] + CURRENT_PYVISTA_TRANSFORM["y_max"]
    )
    for boundary_index, kev_value in enumerate((config["low"], config["high"])):
        z_value = z_value_to_scene(kev_value)
        plane = pv.Plane(
            center=(x_center, y_center, z_value),
            direction=(0, 0, 1),
            i_size=x_size,
            j_size=y_size,
        )
        actor = PYVISTA_PLOTTER.add_mesh(
            plane,
            color=config["color"],
            opacity=config["opacity"],
            lighting=False,
            pickable=False,
            name=f"slice_{index}_plane_{boundary_index}",
            reset_camera=False,
        )
        actors.append(actor)

    SLICE_ACTORS[index] = actors
    add_slice_texture(index, SLICE_IMAGE_DATA.get(index, {}))


def update_all_slices(reset_camera: bool = False):
    count = max(0, min(MAX_SLICES, int(float(state.slice_count))))
    active_index = max(0, min(count - 1, int(state.active_slice_index))) if count else 0
    if active_index != int(state.active_slice_index):
        state.active_slice_index = active_index
        state.dirty("active_slice_index")
    enabled_indices = [
        index
        for index in active_slice_indices()
        if slice_config(index)["enabled"]
    ]
    mode = str(getattr(state, "slice_3d_mode", "all"))
    if mode == "active":
        visible_indices = [active_index] if active_index in enabled_indices else []
    else:
        visible_indices = enabled_indices
    show_full_cloud = (
        mode == "cloud"
        or not visible_indices
        or str(getattr(state, "pyvista_display_mode", "points")) == "voxels"
    )
    for actor in PYVISTA_DATA_ACTORS:
        actor.SetVisibility(show_full_cloud)

    total_point_limit = max(1, int(float(state.slice_point_limit)))
    per_slice_limit = max(1, total_point_limit // max(len(visible_indices), 1))

    for index in range(MAX_SLICES):
        if index in active_slice_indices():
            config = slice_config(index)
            if index in visible_indices:
                update_slice_overlay(index, point_limit=per_slice_limit)
            else:
                clear_slice_actors(index)
                source = slice_source_points(config)
                update_slice_status(index, len(source), config)
                update_slice_chart(index, slice_heatmap_figure(index, source, config))
        else:
            clear_slice_actors(index)
            SLICE_IMAGE_DATA.pop(index, None)
            setattr(state, slice_state_name(index, "status"), "Inactive")
            state.dirty(slice_state_name(index, "status"))
    mode_label = {
        "all": "all enabled slices",
        "active": "active slice only",
        "cloud": "full cloud with enabled slices",
    }.get(mode, "all enabled slices")
    state.slice_status_message = (
        f"3D shows {mode_label} | {len(visible_indices)} slice(s) | "
        f"up to {total_point_limit:,} slice points total"
    )
    state.dirty("slice_status_message")
    pyvista_update_view(reset_camera=reset_camera)


def set_active_slice(index: int):
    count = max(0, min(MAX_SLICES, int(float(state.slice_count))))
    if not 0 <= index < count:
        return
    state.active_slice_index = index
    state.workspace_tab = "slice"
    state.slice_download_ready = False
    state.dirty("active_slice_index", "workspace_tab", "slice_download_ready")
    update_all_slices(reset_camera=False)
    update_profile_chart()


def add_slice():
    count = max(0, min(MAX_SLICES, int(float(state.slice_count))))
    if count >= MAX_SLICES:
        state.slice_status_message = f"Maximum {MAX_SLICES} slices"
        state.dirty("slice_status_message")
        return

    index = count
    state.slice_count = count + 1
    state.active_slice_index = index
    setattr(state, slice_state_name(index, "enabled"), True)
    if not CURRENT_PYVISTA_POINTS.empty and "KEV" in CURRENT_PYVISTA_POINTS.columns:
        kev_values = CURRENT_PYVISTA_POINTS["KEV"].to_numpy(dtype=float)
        finite = kev_values[np.isfinite(kev_values)]
        if finite.size:
            centers = np.percentile(
                finite,
                np.linspace(20, 80, MAX_SLICES),
            )
            setattr(state, slice_state_name(index, "center_kev"), float(centers[index]))
    state.slice_status_message = f"Added slice {index + 1}"
    state.dirty(
        "slice_count",
        "active_slice_index",
        slice_state_name(index, "enabled"),
        slice_state_name(index, "center_kev"),
        "slice_status_message",
    )
    update_all_slices()


def remove_slice():
    count = max(0, min(MAX_SLICES, int(float(state.slice_count))))
    if count <= 0:
        state.slice_status_message = "No slices to remove"
        state.dirty("slice_status_message")
        return
    index = count - 1
    clear_slice_actors(index)
    setattr(state, slice_state_name(index, "enabled"), False)
    state.slice_count = index
    state.slice_status_message = f"Removed slice {index + 1}"
    state.dirty("slice_count", slice_state_name(index, "enabled"), "slice_status_message")
    update_slice_chart(index, slice_empty_figure(f"Slice {index + 1} inactive"))
    update_all_slices()


def reset_pyvista_camera():
    if PYVISTA_PLOTTER is None:
        pyvista_state_set(pyvista_status="PyVista is not available")
        return
    pyvista_update_view(reset_camera=True)


def rebuild_pyvista_from_current():
    if CURRENT_SEARCH_RESULT is None:
        return
    try:
        point_limit = max(1, int(parse_float(state.pyvista_point_limit, "3D point limit")))
        point_size = max(1, int(parse_float(state.pyvista_point_size, "3D point size")))
    except ValueError as error:
        state.pyvista_status = str(error)
        state.status_message = str(error)
        state.dirty("pyvista_status", "status_message")
        return
    state.pyvista_point_limit = point_limit
    state.pyvista_point_size = point_size
    state.dirty("pyvista_point_limit", "pyvista_point_size")
    update_pyvista_scene(
        CURRENT_SEARCH_RESULT,
        point_limit=point_limit,
    )


PYVISTA_PLOTTER = initialize_pyvista_plotter()
if PYVISTA_PLOTTER is not None:
    clear_pyvista_scene("Select result files, then load the PyVista 3D view", reset_camera=True)


def update_results_chart(figure: go.Figure):
    if RESULTS_CHART is not None:
        RESULTS_CHART.update(figure)


def update_viewer_chart(figure: go.Figure):
    if VIEWER_CHART is not None:
        VIEWER_CHART.update(figure)


def refresh_results_table_if_visible():
    if bool(state.show_result_table):
        update_results_chart(results_table_figure(RESULTS))


def selected_record_keys(records) -> tuple[str, ...]:
    return tuple(backend.record_key(record) for record in records)


def kev_filter_values() -> tuple[float, float, float] | None:
    if not bool(state.use_kev_filter):
        return None

    low = parse_float(state.kev_min, "keV minimum")
    high = parse_float(state.kev_max, "keV maximum")
    channels = parse_float(state.channels_per_kev, "Fallback PI channels per keV")
    if channels <= 0:
        raise ValueError("Fallback PI channels per keV must be greater than zero")
    if low > high:
        raise ValueError("keV minimum must be less than or equal to keV maximum")
    return low, high, channels


def kev_filter_signature() -> tuple:
    values = kev_filter_values()
    if values is None:
        return ("kev_off",)
    low, high, channels = values
    return ("kev_on", round(low, 6), round(high, 6), round(channels, 6))


def kev_filter_label() -> str:
    values = kev_filter_values()
    if values is None:
        return ""
    low, high, _channels = values
    return f"{low:g}-{high:g} keV"


def kev_factor_for_record(record: backend.EventFile | None) -> tuple[float, str]:
    if record is not None:
        key = (str(record.mission).lower(), str(record.instrument).lower())
        factor = PI_TO_KEV_FACTORS.get(key)
        if factor is not None:
            return factor, PI_TO_KEV_LABELS[key]

    channels = parse_float(state.channels_per_kev, "Fallback PI channels per keV")
    if channels <= 0:
        raise ValueError("Fallback PI channels per keV must be greater than zero")
    return 1.0 / channels, f"Fallback: PI/{channels:g}"


def add_kev_column(frame: pd.DataFrame, record: backend.EventFile | None) -> pd.DataFrame:
    factor, scale_label = kev_factor_for_record(record)
    converted = frame.copy()
    converted["KEV"] = converted["PI"].to_numpy(dtype=float) * factor
    converted["KEV_SCALE"] = scale_label
    return converted


def apply_kev_filter(frame: pd.DataFrame, record: backend.EventFile | None = None) -> pd.DataFrame:
    frame = add_kev_column(frame, record)
    values = kev_filter_values()
    if values is None:
        return frame
    low, high, _channels = values
    return frame.loc[frame["KEV"].between(low, high, inclusive="both")]


def add_mission_time_columns(
    frame: pd.DataFrame,
    record: backend.EventFile | None,
) -> pd.DataFrame:
    if record is None:
        return frame
    origin_text = MISSION_TIME_ORIGINS.get(str(record.mission).lower())
    if not origin_text:
        return frame

    converted = frame.copy()
    origin = pd.Timestamp(origin_text)
    mission_datetime = origin + pd.to_timedelta(
        converted["TIME"].to_numpy(dtype=float),
        unit="s",
    )
    converted["MISSION_TIME_ORIGIN"] = origin_text
    converted["MISSION_DATETIME"] = mission_datetime.astype(str)
    return converted


def observation_cache_key(record: backend.EventFile, row_limit: int) -> tuple:
    region_signature = (
        tuple(sorted(CURRENT_SKY_REGION.signature().items()))
        if CURRENT_SKY_REGION is not None
        else None
    )
    return (
        backend.record_key(record), int(row_limit), kev_filter_signature(), region_signature
    )


def observation_memory_bytes(observation: backend.LoadedObservation) -> int:
    return int(observation.frame.memory_usage(index=True, deep=True).sum())


def store_memory_observation(cache_key, observation: backend.LoadedObservation):
    LOADED_OBSERVATION_CACHE[cache_key] = observation
    LOADED_OBSERVATION_CACHE.move_to_end(cache_key)
    total = sum(
        observation_memory_bytes(item)
        for item in LOADED_OBSERVATION_CACHE.values()
    )
    while total > MAX_MEMORY_CACHE_BYTES and len(LOADED_OBSERVATION_CACHE) > 1:
        _old_key, old = LOADED_OBSERVATION_CACHE.popitem(last=False)
        total -= observation_memory_bytes(old)


def memory_observation(cache_key):
    observation = LOADED_OBSERVATION_CACHE.get(cache_key)
    if observation is not None:
        LOADED_OBSERVATION_CACHE.move_to_end(cache_key)
    return observation


def store_viewer_figure(cache_key, figure: go.Figure):
    VIEWER_FIGURE_CACHE[cache_key] = figure
    VIEWER_FIGURE_CACHE.move_to_end(cache_key)
    while len(VIEWER_FIGURE_CACHE) > MAX_FIGURE_CACHE_ENTRIES:
        VIEWER_FIGURE_CACHE.popitem(last=False)


async def download_file_response(request):
    filename = str(request.match_info.get("filename", ""))
    if (
        not filename
        or Path(filename).name != filename
        or not filename.endswith((".csv", ".json", ".fits"))
    ):
        raise web.HTTPBadRequest(text="Invalid download filename")

    path = EXPORT_DIR / filename
    if not path.exists() or not path.is_file():
        raise web.HTTPNotFound(text="Download file not found")

    if filename.endswith(".json"):
        content_type = "application/json"
    elif filename.endswith(".fits"):
        content_type = "application/fits"
    else:
        content_type = "text/csv"
    return web.FileResponse(
        path,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": content_type,
            "X-Content-Type-Options": "nosniff",
        },
    )


def configure_download_routes(wslink_server):
    wslink_server.app.router.add_get(
        "/download-file/{filename}",
        download_file_response,
        allow_head=True,
    )


def effective_max_points() -> int:
    if bool(state.load_all_points):
        return 10**12
    return max(1, int(parse_float(state.max_points, "Point limit")))


def point_mode_label() -> str:
    return "all points" if bool(state.load_all_points) else f"{effective_max_points():,} point limit"


def angular_separation_deg(frame: pd.DataFrame, center_ra: float, center_dec: float) -> np.ndarray:
    ra1 = np.deg2rad(float(center_ra) % 360.0)
    dec1 = np.deg2rad(float(center_dec))
    ra2 = np.deg2rad(frame["RA"].to_numpy(dtype=float))
    dec2 = np.deg2rad(frame["DEC"].to_numpy(dtype=float))
    cos_sep = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra2 - ra1)
    )
    return np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0)))


def selected_records_from_state():
    selected = state.selected_files or []
    if isinstance(selected, str):
        selected = [selected]
    if not selected and state.selected_file:
        selected = [state.selected_file]
    return [
        RESULT_RECORDS_BY_LABEL[label]
        for label in selected
        if label in RESULT_RECORDS_BY_LABEL
    ]


def current_result_labels() -> list[str]:
    options = state.result_options or []
    if isinstance(options, str):
        options = [options]
    return [
        str(label)
        for label in options
        if str(label) in RESULT_RECORDS_BY_LABEL
    ]


def labels_for_records(records) -> list[str]:
    label_by_key = {
        backend.record_key(record): label
        for label, record in RESULT_RECORDS_BY_LABEL.items()
    }
    labels = []
    for record in records:
        label = label_by_key.get(backend.record_key(record))
        if label and label not in labels:
            labels.append(label)
    return labels


def labels_for_observations(observations) -> list[str]:
    return labels_for_records([observation.record for observation in observations])


def sync_selected_files_to_loaded(observations):
    loaded_labels = labels_for_observations(observations)
    state.selected_files = loaded_labels
    if state.selected_file and state.selected_file not in loaded_labels:
        state.selected_file = ""
        state.dirty("selected_file", "selected_files")
    else:
        state.dirty("selected_files")
    return loaded_labels


def select_all_current_results():
    labels = current_result_labels()
    if not labels:
        state.status_message = "Search first, then select current results"
        state.dirty("status_message")
        return
    state.selected_file = ""
    state.selected_files = labels
    state.status_message = f"Selected {len(labels):,} current search result file(s)"
    state.dirty("selected_file", "selected_files", "status_message")


def load_current_results_in_3d():
    labels = current_result_labels()
    if not labels:
        state.status_message = "Search first, then load current results"
        state.dirty("status_message")
        return
    records = [
        RESULT_RECORDS_BY_LABEL[label]
        for label in labels
        if label in RESULT_RECORDS_BY_LABEL
    ]
    state.selected_file = ""
    state.show_pyvista_3d = True
    state.pyvista_status = (
        f"Loading {len(records):,} current search result file(s) into PyVista..."
    )
    state.status_message = state.pyvista_status
    state.dirty(
        "selected_file",
        "show_pyvista_3d",
        "pyvista_status",
        "status_message",
    )
    load_selected_files(records_override=records)


def refresh_catalog():
    try:
        state.status_message = "Refreshing UDON3 catalog from server..."
        state.dirty("status_message")
        info = backend.build_server_catalog()
        state.catalog_summary = (
            f"{info['files']:,} server files indexed from "
            f"{', '.join(mission.upper() for mission in info['missions'])}"
        )
        state.status_message = state.catalog_summary
        if info["skipped"]:
            state.status_message += f" | skipped {len(info['skipped'])} mission(s)"
        state.dirty("catalog_summary", "status_message")
        show_all_files()
    except Exception as error:
        state.status_message = str(error)
        state.catalog_summary = str(error)
        state.dirty("status_message", "catalog_summary")


def search_files(show_all: bool = False, balanced_instruments: bool = False):
    global RESULTS, RESULT_RECORDS_BY_LABEL

    try:
        if not backend.server_catalog_exists():
            state.status_message = "Server catalog is missing. Click Refresh catalog."
            state.dirty("status_message")
            if bool(state.show_result_table):
                update_results_chart(empty_figure("Server catalog is missing"))
            return

        selected_pairs = selected_pairs_from_state(state)
        if not selected_pairs:
            state.status_message = "Select at least one mission"
            state.dirty("status_message")
            if bool(state.show_result_table):
                update_results_chart(empty_figure("Select at least one mission"))
            return

        center_ra = center_dec = radius_deg = None
        if bool(state.use_sky_filter) and not show_all:
            center_ra = parse_float(state.center_ra, "RA") % 360.0
            center_dec = parse_float(state.center_dec, "DEC")
            radius_deg = parse_float(state.radius_deg, "Radius")

        object_text = "" if show_all else str(state.object_query or "").strip()
        filename_text = "" if show_all else str(state.filename_query or "").strip()
        observation_text = "" if show_all else str(state.obsid_query or "").strip()
        if filename_text and not observation_text:
            observation_text = obsid_from_filename(filename_text)

        limit = backend.server_catalog_count() if show_all else max(1, int(parse_float(state.result_limit, "Limit")))
        state.status_message = "Searching server catalog..."
        state.dirty("status_message")
        search_kwargs = {
            "object_text": object_text,
            "observation_text": observation_text,
            "center_ra": center_ra,
            "center_dec": center_dec,
            "radius_deg": radius_deg,
            "pointing_margin_deg": 1.0 if CURRENT_SKY_REGION is not None else 0.0,
            "date_start": "" if show_all else str(state.date_start or "").strip(),
            "date_end": "" if show_all else str(state.date_end or "").strip(),
        }
        if balanced_instruments and selected_pairs:
            per_instrument = max(10, int(np.ceil(limit / len(selected_pairs))))
            parts = [
                backend.search_server_catalog(
                    selected_pairs=[pair], limit=per_instrument, **search_kwargs
                )
                for pair in selected_pairs
            ]
            RESULTS = pd.concat(
                [part for part in parts if not part.empty], ignore_index=True
            ) if any(not part.empty for part in parts) else pd.DataFrame()
            if not RESULTS.empty:
                RESULTS = RESULTS.sort_values(
                    ["separation_deg", "mission", "instrument", "observation_id"],
                    na_position="last",
                )
        else:
            RESULTS = backend.search_server_catalog(
                selected_pairs=selected_pairs,
                limit=max(limit, 1000 if filename_text else limit),
                **search_kwargs,
            )
        RESULTS = apply_filename_filter(RESULTS, filename_text).head(limit).reset_index(drop=True)

        records = backend.server_records_from_dataframe(RESULTS)
        labels = [result_label(row) for _, row in RESULTS.iterrows()]
        RESULT_RECORDS_BY_LABEL = {
            label: record
            for label, record in zip(labels, records)
        }

        state.result_options = labels
        state.selected_file = ""
        state.selected_files = []
        state.results_summary = f"{len(RESULTS):,} matching server file(s)"
        state.status_message = state.results_summary
        state.dirty(
            "result_options",
            "selected_file",
            "selected_files",
            "results_summary",
            "status_message",
        )
        refresh_results_table_if_visible()
    except Exception as error:
        state.status_message = str(error)
        state.results_summary = str(error)
        state.dirty("status_message", "results_summary")
        if bool(state.show_result_table):
            update_results_chart(empty_figure(str(error)))


def run_search():
    search_files(show_all=False)


def show_all_files():
    state.show_result_table = True
    state.dirty("show_result_table")
    search_files(show_all=True)


def search_sky_region():
    state.use_sky_filter = True
    state.dirty("use_sky_filter")
    search_files(show_all=False)


def resolve_and_load_region():
    global CURRENT_SKY_REGION, REGION_REQUEST_GENERATION
    REGION_REQUEST_GENERATION += 1
    generation = REGION_REQUEST_GENERATION
    state.region_loading = True
    state.dirty("region_loading")
    try:
        state.status_message = "Resolving sky position..."
        state.dirty("status_message")
        if state.spatial_selection_mode == "rectangle":
            region = backend.parse_sky_rectangle(
                state.rectangle_ra_min,
                state.rectangle_ra_max,
                state.rectangle_dec_min,
                state.rectangle_dec_max,
            )
            cache_hit = False
        else:
            region, cache_hit = backend.parse_sky_region(
                state.coordinate_mode,
                state.coordinate_ra,
                state.coordinate_dec,
                state.radius_deg,
                target_name=state.target_name,
            )
        CURRENT_SKY_REGION = region
        state.center_ra = region.center_ra_deg
        state.center_dec = region.center_dec_deg
        state.radius_deg = backend.selection_catalog_radius_deg(region)
        state.coordinate_ra = (
            f"{region.center_ra_deg:.8f}"
            if getattr(region, "source", "") == "target"
            else state.coordinate_ra
        )
        state.coordinate_dec = (
            f"{region.center_dec_deg:.8f}"
            if getattr(region, "source", "") == "target"
            else state.coordinate_dec
        )
        state.use_sky_filter = True
        if isinstance(region, backend.SkyRectangle):
            state.resolved_region = (
                f"{region.label} | center RA {region.center_ra_deg:.6f} deg | "
                f"center DEC {region.center_dec_deg:.6f} deg"
            )
        else:
            state.resolved_region = (
                f"{region.label} | RA {region.center_ra_deg:.6f} deg | "
                f"DEC {region.center_dec_deg:.6f} deg | radius {region.radius_deg:.4g} deg"
            )
        if cache_hit:
            state.resolved_region += " | cached target"
        state.dirty(
            "center_ra", "center_dec", "radius_deg", "coordinate_ra", "coordinate_dec",
            "use_sky_filter", "resolved_region",
        )
        search_files(show_all=False, balanced_instruments=True)
        if generation != REGION_REQUEST_GENERATION:
            return
        if RESULTS.empty:
            state.status_message = "No catalog observations cover the requested region"
            state.dirty("status_message")
            return
        state.selected_files = list(state.result_options)
        state.dirty("selected_files")
        load_selected_files(records_override=list(RESULT_RECORDS_BY_LABEL.values()))
    except Exception as error:
        state.status_message = f"Could not resolve/load region: {error}"
        state.dirty("status_message")
    finally:
        if generation == REGION_REQUEST_GENERATION:
            state.region_loading = False
            state.dirty("region_loading")


def apply_galactic_center_preset():
    for mission, key in MISSION_KEYS.items():
        state[key] = mission == "suzaku"
    state.object_query = ""
    state.obsid_query = ""
    state.filename_query = ""
    state.center_ra = GC_CENTER_RA
    state.center_dec = GC_CENTER_DEC
    state.radius_deg = GC_RADIUS_DEG
    state.use_sky_filter = True
    state.result_limit = 250
    state.gc_center_fe64 = 6.40
    state.gc_width_fe64 = 0.20
    state.gc_palette_fe64 = "inferno"
    state.gc_palette_pending_fe64 = "inferno"
    state.gc_center_fe67 = 6.70
    state.gc_width_fe67 = 0.20
    state.gc_palette_fe67 = "magma"
    state.gc_palette_pending_fe67 = "magma"
    state.gc_stretch = "log"
    state.gc_brightness = 1.35
    state.gc_contrast = 1.2
    state.gc_smoothing_sigma = 1.2
    state.dirty(
        *MISSION_KEYS.values(),
        "object_query",
        "obsid_query",
        "filename_query",
        "center_ra",
        "center_dec",
        "radius_deg",
        "use_sky_filter",
        "result_limit",
        "gc_center_fe64",
        "gc_width_fe64",
        "gc_palette_fe64",
        "gc_palette_pending_fe64",
        "gc_center_fe67",
        "gc_width_fe67",
        "gc_palette_fe67",
        "gc_palette_pending_fe67",
        "gc_stretch",
        "gc_brightness",
        "gc_contrast",
        "gc_smoothing_sigma",
    )
    search_files(show_all=False)


def read_record_preview(record: backend.EventFile, row_limit: int) -> backend.LoadedObservation:
    if CURRENT_SKY_REGION is not None:
        frame, metadata, events_in_region, _cache_hit = backend.read_region_preview(
            record, CURRENT_SKY_REGION, max_rows=max(backend.DEFAULT_PREVIEW_ROWS, row_limit)
        )
        total_events = events_in_region
    else:
        frame, metadata, total_events, _cache_hit = backend.read_compact_preview(record)
        events_in_region = total_events
    frame = apply_kev_filter(frame, record)
    filtered_events = len(frame)

    if filtered_events > row_limit:
        frame = frame.sample(row_limit, random_state=42).sort_index()
    frame["MISSION"] = record.mission
    frame["INSTRUMENT"] = record.instrument
    frame["OBSERVATION_ID"] = record.observation_id
    if "SOURCE_ROW" not in frame:
        frame["SOURCE_ROW"] = frame.index.to_numpy(dtype=np.uint32)
    frame = add_mission_time_columns(frame, record)

    return backend.LoadedObservation(
        record=record,
        frame=frame.reset_index(drop=True),
        metadata=metadata,
        total_events=total_events,
        events_in_region=events_in_region,
        displayed_events=len(frame),
        minimum_separation_deg=None,
    )


def allocated_record_limits(records, max_points: int) -> list[int]:
    records = list(records)
    pair_counts = {}
    for record in records:
        key = backend.pair_key(record.mission, record.instrument)
        pair_counts[key] = pair_counts.get(key, 0) + 1
    per_pair_limit = max(1, int(max_points / max(len(pair_counts), 1)))
    return [
        max(1, int(per_pair_limit / pair_counts[backend.pair_key(record.mission, record.instrument)]))
        for record in records
    ]


def load_records_cached(records, max_points: int):
    records = list(records)
    limits = allocated_record_limits(records, max_points)
    cache_keys = [
        observation_cache_key(record, row_limit)
        for record, row_limit in zip(records, limits)
    ]
    missing = []
    queued_keys = set()
    for record, cache_key in zip(records, cache_keys):
        if cache_key in LOADED_OBSERVATION_CACHE or cache_key in queued_keys:
            continue
        missing.append((record, cache_key))
        queued_keys.add(cache_key)

    skipped = []
    limits_by_key = dict(zip(cache_keys, limits))
    with ThreadPoolExecutor(max_workers=min(4, max(len(missing), 1))) as executor:
        futures = {
            executor.submit(
                read_record_preview, record, limits_by_key[cache_key]
            ): (record, cache_key)
            for record, cache_key in missing
        }
        for future in as_completed(futures):
            record, cache_key = futures[future]
            try:
                store_memory_observation(cache_key, future.result())
            except Exception as error:
                skipped.append(f"{backend.record_key(record)}: {error}")

    observations = []
    for cache_key in cache_keys:
        observation = memory_observation(cache_key)
        if observation is not None:
            observations.append(observation)
    return backend.SearchResult(
        observations=observations,
        summary=pd.DataFrame(),
        skipped=skipped,
        center_ra=0.0,
        center_dec=0.0,
        radius_deg=None,
    )


def instrument_coverage_text(observations) -> str:
    loaded = {
        backend.pair_key(item.record.mission, item.record.instrument)
        for item in observations
        if item.displayed_events > 0
    }
    parts = []
    for (mission, instrument), key in INSTRUMENT_KEYS.items():
        if not bool(state[key]):
            continue
        pair = backend.pair_key(mission, instrument)
        parts.append(
            f"{mission.upper()} {instrument.upper()}: "
            f"{'loaded' if pair in loaded else 'no events in region'}"
        )
    return " | ".join(parts) or "No instruments selected"


def load_selected_files(records_override=None):
    try:
        records = (
            list(records_override)
            if records_override is not None
            else selected_records_from_state()
        )
        if records_override is None and not records and not RESULTS.empty:
            records = [RESULT_RECORDS_BY_LABEL[result_label(RESULTS.iloc[0])]]
        if not records:
            state.status_message = "Select one or more result files first"
            state.dirty("status_message")
            return

        state.download_ready = False
        state.download_csv_href = ""
        state.download_json_href = ""
        state.dirty(
            "download_ready",
            "download_csv_href",
            "download_json_href",
        )

        max_points = effective_max_points()
        record_keys = selected_record_keys(records)
        filter_label = kev_filter_label()
        record_limits = allocated_record_limits(records, max_points)
        observation_keys = [
            observation_cache_key(record, row_limit)
            for record, row_limit in zip(records, record_limits)
        ]
        region_key = (
            tuple(sorted(CURRENT_SKY_REGION.signature().items()))
            if CURRENT_SKY_REGION is not None
            else None
        )
        figure_key = (record_keys, max_points, kev_filter_signature(), region_key)

        if figure_key in VIEWER_FIGURE_CACHE:
            VIEWER_FIGURE_CACHE.move_to_end(figure_key)
            cached_observations = []
            for key in observation_keys:
                observation = memory_observation(key)
                if observation is not None:
                    cached_observations.append(observation)
            cached_events = sum(
                observation.displayed_events
                for observation in cached_observations
            )
            loaded_labels = sync_selected_files_to_loaded(cached_observations)
            state.loaded_summary = (
                f"Preview | {len(loaded_labels)} file(s), "
                f"{cached_events:,} event rows loaded from memory"
            )
            state.has_loaded_data = bool(cached_observations)
            state.show_pyvista_3d = bool(cached_observations)
            if filter_label:
                state.loaded_summary += f" | {filter_label}"
            state.status_message = state.loaded_summary
            state.instrument_coverage = instrument_coverage_text(cached_observations)
            state.dirty(
                "loaded_summary", "status_message", "has_loaded_data", "show_pyvista_3d",
                "instrument_coverage",
            )
            update_viewer_chart(VIEWER_FIGURE_CACHE[figure_key])
            update_pyvista_scene(
                backend.SearchResult(
                    observations=cached_observations,
                    summary=pd.DataFrame(),
                    skipped=[],
                    center_ra=0.0,
                    center_dec=0.0,
                    radius_deg=None,
                ),
                point_limit=int(float(state.pyvista_point_limit)),
            )
            return

        uncached_count = sum(
            1
            for cache_key in observation_keys
            if cache_key not in LOADED_OBSERVATION_CACHE
        )
        if uncached_count:
            state.status_message = f"Downloading/loading {uncached_count} new file(s)..."
        else:
            state.status_message = f"Preparing {len(records)} cached file preview..."
        state.dirty("status_message")

        result = load_records_cached(records, max_points=max_points)
        events = sum(item.displayed_events for item in result.observations)
        loaded_labels = sync_selected_files_to_loaded(result.observations)
        state.loaded_summary = (
            f"Preview | {len(loaded_labels)} file(s), {events:,} event rows loaded"
        )
        state.has_loaded_data = bool(result.observations)
        state.show_pyvista_3d = bool(result.observations)
        if not uncached_count:
            state.loaded_summary += " from memory"
        if filter_label:
            state.loaded_summary += f" | {filter_label}"
        if result.skipped:
            state.loaded_summary += f" | skipped {len(result.skipped)}"
        state.status_message = state.loaded_summary
        state.instrument_coverage = instrument_coverage_text(result.observations)
        state.dirty(
            "loaded_summary", "status_message", "has_loaded_data", "show_pyvista_3d",
            "instrument_coverage",
        )
        figure = loaded_viewer_figure(result, max_points=max_points)
        store_viewer_figure(figure_key, figure)
        update_viewer_chart(figure)
        update_pyvista_scene(
            result,
            point_limit=int(float(state.pyvista_point_limit)),
        )
        refresh_cache_status()
    except Exception as error:
        state.has_loaded_data = False
        state.status_message = str(error)
        state.loaded_summary = str(error)
        state.dirty("status_message", "loaded_summary", "has_loaded_data")
        update_viewer_chart(empty_figure(str(error)))
        clear_pyvista_scene(str(error), reset_camera=True)


def load_direct_selected_file():
    selected = str(state.selected_file or "").strip()
    if not selected:
        return
    state.selected_files = [selected]
    state.dirty("selected_files")
    load_selected_files()


def export_loaded_data():
    try:
        state.download_ready = False
        state.download_csv_href = ""
        state.download_json_href = ""
        state.dirty("download_ready", "download_csv_href", "download_json_href")

        records = selected_records_from_state()
        if not records:
            state.export_summary = "Select one or more files first"
            state.status_message = state.export_summary
            state.dirty("export_summary", "status_message")
            return

        max_points = effective_max_points()
        result = load_records_cached(records, max_points=max_points)
        frames = []
        for observation in result.observations:
            frame = observation.frame.copy()
            frame["FILE"] = backend.record_label(observation.record)
            frame["OBJECT"] = str(observation.metadata.get("OBJECT", ""))
            frame["DATE_OBS"] = str(observation.metadata.get("DATE-OBS", ""))
            frame["DATE_END"] = str(observation.metadata.get("DATE-END", ""))
            frame["PARQUET_URL"] = str(observation.record.parquet_url or "")
            frame["HEADER_URL"] = str(observation.record.header_url or "")
            frames.append(frame)

        if not frames:
            state.export_summary = "No loaded rows to export"
            state.status_message = state.export_summary
            state.dirty("export_summary", "status_message")
            return

        export_frame = pd.concat(frames, ignore_index=True)
        range_parts = [point_mode_label()]
        filter_label = kev_filter_label()
        if filter_label:
            range_parts.append(filter_label)
        if bool(state.use_sky_filter) and CURRENT_SKY_REGION is not None:
            export_frame = export_frame.loc[
                backend.selection_contains(
                    CURRENT_SKY_REGION,
                    export_frame["RA"].to_numpy(dtype=float),
                    export_frame["DEC"].to_numpy(dtype=float),
                )
            ].copy()
            range_parts.append(CURRENT_SKY_REGION.label)
        range_text = " | ".join(range_parts)

        if export_frame.empty:
            state.export_summary = "No loaded rows inside the current filters"
            state.status_message = state.export_summary
            state.dirty("export_summary", "status_message")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"udon3_selected_events_{timestamp}.csv"
        json_name = f"udon3_selected_events_{timestamp}.json"
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = EXPORT_DIR / csv_name
        json_path = EXPORT_DIR / json_name
        export_frame.to_csv(csv_path, index=False)
        export_frame.to_json(json_path, orient="records", indent=2)

        state.download_csv_name = csv_name
        state.download_json_name = json_name
        state.download_csv_href = f"/download-file/{csv_name}"
        state.download_json_href = f"/download-file/{json_name}"
        state.download_ready = True

        state.export_summary = (
            f"Prepared {len(export_frame):,} rows ({range_text}) for browser download."
            f" Files are served from exports/: {csv_name}, {json_name}"
        )
        state.status_message = state.export_summary
        state.dirty(
            "download_ready",
            "download_csv_name",
            "download_json_name",
            "download_csv_href",
            "download_json_href",
            "export_summary",
            "status_message",
        )
    except Exception as error:
        state.download_ready = False
        state.download_csv_href = ""
        state.download_json_href = ""
        state.export_summary = str(error)
        state.status_message = str(error)
        state.dirty(
            "download_ready",
            "download_csv_href",
            "download_json_href",
            "export_summary",
            "status_message",
        )


def prepare_slice_image_download():
    try:
        if (
            not ACTIVE_SLICE_IMAGE_DATA
            or CURRENT_SEARCH_RESULT is None
            or not CURRENT_SEARCH_RESULT.observations
        ):
            raise ValueError("Load data and select an active slice first")
        data = ACTIVE_SLICE_IMAGE_DATA
        config = data["config"]
        state.slice_download_status = "Computing exact image from source events..."
        state.dirty("slice_download_status")
        exact = backend.exact_energy_image(
            [item.record for item in CURRENT_SEARCH_RESULT.observations],
            config["low"],
            config["high"],
            bins=max(16, int(float(state.slice_image_bins))),
            region=CURRENT_SKY_REGION,
        )
        image = np.asarray(exact["hist"], dtype=np.uint32)
        x_edges = np.asarray(exact["x_edges"], dtype=float)
        y_edges = np.asarray(exact["y_edges"], dtype=float)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        xx, yy = np.meshgrid(x_centers, y_centers)
        pixel_frame = pd.DataFrame(
            {
                "x_index": np.tile(np.arange(len(x_centers)), len(y_centers)),
                "y_index": np.repeat(np.arange(len(y_centers)), len(x_centers)),
                "ra_deg": xx.ravel(),
                "dec_deg": yy.ravel(),
                "event_count": image.ravel(),
                "energy_low_kev": config["low"],
                "energy_high_kev": config["high"],
                "energy_center_kev": config["center"],
                "energy_width_kev": config["width"],
            }
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"udon3_slice_{int(data['index']) + 1}_{timestamp}"
        csv_name = f"{stem}.csv"
        fits_name = f"{stem}.fits"
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        pixel_frame.to_csv(EXPORT_DIR / csv_name, index=False)

        hdu = fits.PrimaryHDU(image)
        header = hdu.header
        header["BUNIT"] = "EVENTS"
        header["CTYPE1"] = "RA---CAR"
        header["CTYPE2"] = "DEC--CAR"
        header["CUNIT1"] = "deg"
        header["CUNIT2"] = "deg"
        header["CRPIX1"] = 1.0
        header["CRPIX2"] = 1.0
        header["CRVAL1"] = float(x_centers[0])
        header["CRVAL2"] = float(y_centers[0])
        header["CDELT1"] = float(np.mean(np.diff(x_centers))) if len(x_centers) > 1 else 1.0
        header["CDELT2"] = float(np.mean(np.diff(y_centers))) if len(y_centers) > 1 else 1.0
        header["E_MIN"] = float(config["low"])
        header["E_MAX"] = float(config["high"])
        header["E_CTR"] = float(config["center"])
        header["E_WIDTH"] = float(config["width"])
        header["NEVENTS"] = int(exact["event_count"])
        header["QUALITY"] = "EXACT"
        header["CAL_VER"] = int(exact["calibration_version"])
        hdu.writeto(EXPORT_DIR / fits_name, overwrite=True)

        state.slice_download_csv_name = csv_name
        state.slice_download_fits_name = fits_name
        state.slice_download_csv_href = f"/download-file/{csv_name}"
        state.slice_download_fits_href = f"/download-file/{fits_name}"
        state.slice_download_ready = True
        state.slice_download_status = (
            f"Exact | prepared {len(pixel_frame):,} pixels and "
            f"{exact['event_count']:,} events for "
            f"{config['low']:.3f}-{config['high']:.3f} keV"
        )
        state.dirty(
            "slice_download_csv_name",
            "slice_download_fits_name",
            "slice_download_csv_href",
            "slice_download_fits_href",
            "slice_download_ready",
            "slice_download_status",
        )
        refresh_cache_status()
    except Exception as error:
        state.slice_download_ready = False
        state.slice_download_status = str(error)
        state.dirty("slice_download_ready", "slice_download_status")


def prepare_profile_download():
    try:
        comparison = comparative_profile_data()
        point_a_ra, point_a_dec = comparison["point_a"]
        point_b_ra, point_b_dec = comparison["point_b"]
        source_keys = "|".join(
            backend.record_key(item.record)
            for item in (
                CURRENT_SEARCH_RESULT.observations
                if CURRENT_SEARCH_RESULT is not None
                else []
            )
        )
        frames = []
        sample_index = np.arange(len(comparison["distance_arcmin"]), dtype=np.int32)
        for profile in comparison["profiles"]:
            config = profile["config"]
            frames.append(
                pd.DataFrame(
                    {
                        "slice_number": int(profile["index"]) + 1,
                        "slice_label": profile["label"],
                        "slice_color": config["color"],
                        "quality": profile["quality"],
                        "energy_low_kev": config["low"],
                        "energy_high_kev": config["high"],
                        "energy_center_kev": config["center"],
                        "energy_width_kev": config["width"],
                        "sample_index": sample_index,
                        "distance_arcmin": comparison["distance_arcmin"],
                        "ra_deg": comparison["ra_deg"],
                        "dec_deg": comparison["dec_deg"],
                        "mean_events_per_pixel": profile["values"],
                        "profile_width_pixels": comparison["width_pixels"],
                        "point_a_ra_deg": point_a_ra,
                        "point_a_dec_deg": point_a_dec,
                        "point_b_ra_deg": point_b_ra,
                        "point_b_dec_deg": point_b_dec,
                        "source_observations": source_keys,
                    }
                )
            )
        export_frame = pd.concat(frames, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"udon3_comparative_profiles_{timestamp}"
        csv_name = f"{stem}.csv"
        json_name = f"{stem}.json"
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        export_frame.to_csv(EXPORT_DIR / csv_name, index=False)
        export_frame.to_json(EXPORT_DIR / json_name, orient="records", indent=2)

        state.profile_download_csv_name = csv_name
        state.profile_download_json_name = json_name
        state.profile_download_csv_href = f"/download-file/{csv_name}"
        state.profile_download_json_href = f"/download-file/{json_name}"
        state.profile_download_ready = True
        state.profile_download_status = (
            f"Prepared {len(export_frame):,} samples from "
            f"{len(comparison['profiles'])} slice profile(s)"
        )
        state.dirty(
            "profile_download_csv_name",
            "profile_download_json_name",
            "profile_download_csv_href",
            "profile_download_json_href",
            "profile_download_ready",
            "profile_download_status",
        )
    except Exception as error:
        state.profile_download_ready = False
        state.profile_download_csv_href = ""
        state.profile_download_json_href = ""
        state.profile_download_status = str(error)
        state.dirty(
            "profile_download_ready",
            "profile_download_csv_href",
            "profile_download_json_href",
            "profile_download_status",
        )


def toggle_result_table():
    state.show_result_table = not bool(state.show_result_table)
    state.dirty("show_result_table")
    refresh_results_table_if_visible()


def live_search_inputs_ready() -> bool:
    query_values = [
        str(state.object_query or "").strip(),
        str(state.obsid_query or "").strip(),
        str(state.filename_query or "").strip(),
    ]
    short_values = [
        value
        for value in query_values
        if 0 < len(value) < LIVE_SEARCH_MIN_CHARS
    ]
    if short_values:
        state.status_message = f"Type at least {LIVE_SEARCH_MIN_CHARS} characters to search"
        state.dirty("status_message")
        return False

    if bool(state.use_sky_filter):
        try:
            parse_float(state.center_ra, "RA")
            parse_float(state.center_dec, "DEC")
            radius = parse_float(state.radius_deg, "Radius")
            if radius <= 0:
                raise ValueError("Radius must be greater than zero")
        except ValueError as error:
            state.status_message = str(error)
            state.dirty("status_message")
            return False

    return True


def run_live_search():
    if live_search_inputs_ready():
        search_files(show_all=False)


server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller
state.trame__title = "UDON3 Server File Search"
state.object_query = ""
state.obsid_query = ""
state.filename_query = ""
state.date_start = ""
state.date_end = ""
state.center_ra = 266.404996
state.center_dec = -28.936172
state.radius_deg = 1.0
state.spatial_selection_mode = "circle"
state.rectangle_ra_min = 265.9
state.rectangle_ra_max = 266.9
state.rectangle_dec_min = -29.4
state.rectangle_dec_max = -28.4
state.coordinate_mode = "degrees"
state.coordinate_ra = "266.404996"
state.coordinate_dec = "-28.936172"
state.target_name = ""
state.resolved_region = "No sky region loaded"
state.instrument_coverage = "All eight instruments selected; resolve a region to check coverage"
state.region_loading = False
state.use_sky_filter = False
state.result_limit = DEFAULT_LIMIT
state.max_points = DEFAULT_MAX_POINTS
state.load_all_points = False
state.use_kev_filter = False
state.kev_min = 0.5
state.kev_max = 10.0
state.channels_per_kev = DEFAULT_CHANNELS_PER_KEV
state.result_options = []
state.selected_file = ""
state.selected_files = []
state.show_result_table = False
state.cache_usage_fraction = 0.0
state.cache_summary = "Reading cache status..."
state.cache_message = ""
catalog_count = backend.server_catalog_count()
state.catalog_summary = (
    f"{catalog_count:,} UDON3 server files indexed"
    if catalog_count
    else "UDON3 server catalog not indexed"
)
state.results_summary = "No search yet"
state.loaded_summary = "No event file loaded"
state.export_summary = "No export yet"
state.download_ready = False
state.download_csv_href = ""
state.download_json_href = ""
state.download_csv_name = "udon3_selected_events.csv"
state.download_json_name = "udon3_selected_events.json"
state.show_pyvista_3d = True
state.has_loaded_data = False
state.pyvista_available = PYVISTA_AVAILABLE
state.pyvista_point_limit = DEFAULT_PYVISTA_POINT_LIMIT
state.pyvista_point_size = DEFAULT_PYVISTA_POINT_SIZE
state.pyvista_display_mode = "points"
state.voxel_spatial_size = 1.0
state.voxel_energy_size = 0.1
state.pyvista_color_mode = "mission"
state.pyvista_colormap = "turbo"
state.pyvista_colormap_pending = "turbo"
state.event_spatial_sigma = 0.0
state.event_energy_sigma = 0.0
state.density_size_strength = 0.8
state.density_opacity_strength = 0.8
state.image_spatial_sigma = 0.0
state.pyvista_status = (
    "Select result files, then load the PyVista 3D view"
    if PYVISTA_AVAILABLE
    else f"PyVista is not available: {PYVISTA_IMPORT_ERROR}"
)
state.pyvista_selected_file = "-"
state.pyvista_selected_object = "-"
state.pyvista_selected_row = "-"
state.pyvista_selected_time = "-"
state.pyvista_selected_mission_datetime = "-"
state.pyvista_selected_pi = "-"
state.pyvista_selected_kev = "-"
state.pyvista_selected_ra = "-"
state.pyvista_selected_dec = "-"
state.pyvista_selected_xy = "-"
state.pyvista_has_selection = False
state.slice_count = 1
state.active_slice_index = 0
state.slice_3d_mode = "all"
state.slice_3d_mode_options = [
    {"title": "All enabled slices", "value": "all"},
    {"title": "Active slice only", "value": "active"},
    {"title": "Full cloud + slices", "value": "cloud"},
]
state.slice_point_limit = DEFAULT_SLICE_POINT_LIMIT
state.slice_image_bins = DEFAULT_SLICE_IMAGE_BINS
state.slice_status_message = "Load data, then move or add keV slices"
state.profile_width_pixels = 3
state.profile_status = "Click point A, then point B on any slice image"
state.profile_download_ready = False
state.profile_download_csv_href = ""
state.profile_download_json_href = ""
state.profile_download_csv_name = "udon3_comparative_profiles.csv"
state.profile_download_json_name = "udon3_comparative_profiles.json"
state.profile_download_status = "Select A and B to prepare profile data"
state.slice_download_ready = False
state.slice_download_csv_href = ""
state.slice_download_fits_href = ""
state.slice_download_csv_name = "udon3_slice_pixels.csv"
state.slice_download_fits_name = "udon3_slice_image.fits"
state.slice_download_status = "No slice image prepared"
state.workspace_tab = "events"
state.gc_center_fe64 = 6.40
state.gc_width_fe64 = 0.20
state.gc_palette_fe64 = "inferno"
state.gc_palette_pending_fe64 = "inferno"
state.gc_status_fe64 = "Load data to build energy map 1"
state.gc_center_fe67 = 6.70
state.gc_width_fe67 = 0.20
state.gc_palette_fe67 = "magma"
state.gc_palette_pending_fe67 = "magma"
state.gc_status_fe67 = "Load data to build energy map 2"
state.gc_stretch = "log"
state.gc_brightness = 1.35
state.gc_contrast = 1.2
state.gc_smoothing_sigma = 1.2
state.gc_image_bins = DEFAULT_GC_IMAGE_BINS
state.rgb_red_center = 1.85
state.rgb_red_width = 0.20
state.rgb_green_center = 2.44
state.rgb_green_width = 0.20
state.rgb_blue_center = 6.40
state.rgb_blue_width = 0.40
state.rgb_red_gain = 1.0
state.rgb_green_gain = 1.0
state.rgb_blue_gain = 1.0
state.rgb_brightness = 1.25
state.rgb_gamma = 1.0
state.show_rgb_plane = True
state.rgb_status = "Load data to build the RGB energy composite"
for index in range(MAX_SLICES):
    state[slice_state_name(index, "enabled")] = index == 0
    state[slice_state_name(index, "center_kev")] = 6.55 + index * 0.25
    state[slice_state_name(index, "width_kev")] = 0.30
    state[slice_state_name(index, "color")] = SLICE_COLORS[index % len(SLICE_COLORS)]
    state[slice_state_name(index, "opacity")] = 0.18
    state[slice_state_name(index, "status")] = (
        "6.400-6.700 keV | no loaded data" if index == 0 else "Inactive"
    )
state.drawer_panels = [
    "missions", "sky", "files", "energy", "load", "slices", "gc", "cache"
]
state.status_message = state.catalog_summary
for mission, key in MISSION_KEYS.items():
    state[key] = True
for key in INSTRUMENT_KEYS.values():
    state[key] = True

ctrl.refresh_catalog = refresh_catalog
ctrl.run_search = run_search
ctrl.search_sky_region = search_sky_region
ctrl.resolve_and_load_region = resolve_and_load_region
ctrl.show_all_files = show_all_files
ctrl.load_selected_files = load_selected_files
ctrl.load_direct_selected_file = load_direct_selected_file
ctrl.select_all_current_results = select_all_current_results
ctrl.load_current_results_in_3d = load_current_results_in_3d
ctrl.export_loaded_data = export_loaded_data
ctrl.toggle_result_table = toggle_result_table
ctrl.reset_pyvista_camera = reset_pyvista_camera
ctrl.rebuild_pyvista_from_current = rebuild_pyvista_from_current
ctrl.add_slice = add_slice
ctrl.remove_slice = remove_slice
ctrl.update_all_slices = update_all_slices
ctrl.update_gc_image = apply_energy_map_settings
ctrl.apply_energy_map_settings = apply_energy_map_settings
ctrl.apply_pyvista_palette = apply_pyvista_palette
ctrl.update_rgb_preview = lambda: update_rgb_composite(exact=False)
ctrl.update_rgb_exact = lambda: update_rgb_composite(exact=True)
ctrl.clear_slice_profile = clear_slice_profile
ctrl.prepare_profile_download = prepare_profile_download
ctrl.prepare_slice_image_download = prepare_slice_image_download
ctrl.refresh_cache_status = refresh_cache_status
ctrl.clear_raw_cache = clear_raw_cache
ctrl.clear_derived_cache = clear_derived_cache
for slice_index in range(MAX_SLICES):
    setattr(
        ctrl,
        f"set_active_slice_{slice_index}",
        lambda index=slice_index: set_active_slice(index),
    )
    setattr(
        ctrl,
        f"slice_image_clicked_{slice_index}",
        lambda coordinates=None, index=slice_index: slice_image_clicked(
            index,
            coordinates,
        ),
    )
    setattr(
        ctrl,
        f"update_exact_slice_{slice_index}",
        lambda *_, index=slice_index: update_exact_slice_image(index),
    )
for palette_mode in ENERGY_MAP_NAMES:
    for palette_key in PALETTE_OPTIONS:
        setattr(
            ctrl,
            f"select_energy_palette_{palette_mode}_{palette_key}",
            lambda mode=palette_mode, key=palette_key: select_energy_palette(mode, key),
        )
for palette_key in PALETTE_OPTIONS:
    setattr(
        ctrl,
        f"select_pyvista_palette_{palette_key}",
        lambda key=palette_key: select_pyvista_palette(key),
    )
ctrl.on_server_bind.add(configure_download_routes)
backend.register_existing_cache()
refresh_cache_status()


@state.change("selected_file")
def selected_file_changed(selected_file=None, **_):
    if selected_file:
        load_direct_selected_file()


@state.change("object_query", "obsid_query", "filename_query", "date_start", "date_end", "result_limit")
def live_text_search_changed(**_):
    run_live_search()


@state.change(*MISSION_KEYS.values())
def live_mission_search_changed(**_):
    global INSTRUMENT_SYNC_ACTIVE
    if INSTRUMENT_SYNC_ACTIVE:
        return
    INSTRUMENT_SYNC_ACTIVE = True
    try:
        for mission, mission_key in MISSION_KEYS.items():
            enabled = bool(state[mission_key])
            for instrument in backend.KNOWN_INSTRUMENTS.get(mission, ()):
                instrument_key = INSTRUMENT_KEYS[(mission, instrument)]
                if bool(state[instrument_key]) != enabled:
                    state[instrument_key] = enabled
    finally:
        INSTRUMENT_SYNC_ACTIVE = False
    run_live_search()


@state.change(*INSTRUMENT_KEYS.values())
def live_instrument_search_changed(**_):
    global INSTRUMENT_SYNC_ACTIVE
    if INSTRUMENT_SYNC_ACTIVE:
        return
    INSTRUMENT_SYNC_ACTIVE = True
    try:
        for mission, mission_key in MISSION_KEYS.items():
            instrument_keys = [
                INSTRUMENT_KEYS[(mission, instrument)]
                for instrument in backend.KNOWN_INSTRUMENTS.get(mission, ())
            ]
            enabled = bool(instrument_keys) and all(
                bool(state[key]) for key in instrument_keys
            )
            if bool(state[mission_key]) != enabled:
                state[mission_key] = enabled
    finally:
        INSTRUMENT_SYNC_ACTIVE = False
    run_live_search()


@state.change("use_sky_filter")
def live_sky_filter_changed(**_):
    run_live_search()


@state.change("center_ra", "center_dec", "radius_deg")
def live_sky_inputs_changed(**_):
    if not bool(state.use_sky_filter):
        state.use_sky_filter = True
        state.dirty("use_sky_filter")
    run_live_search()


@state.change("max_points", "load_all_points")
def point_loading_changed(**_):
    if state.selected_files:
        load_selected_files()


@state.change(
    "pyvista_point_limit",
    "pyvista_point_size",
    "pyvista_display_mode",
    "voxel_spatial_size",
    "voxel_energy_size",
    "pyvista_color_mode",
    "event_spatial_sigma",
    "event_energy_sigma",
    "density_size_strength",
    "density_opacity_strength",
)
def pyvista_controls_changed(**_):
    rebuild_pyvista_from_current()


@state.change(
    "rgb_red_center", "rgb_red_width", "rgb_green_center", "rgb_green_width",
    "rgb_blue_center", "rgb_blue_width", "rgb_red_gain", "rgb_green_gain",
    "rgb_blue_gain", "rgb_brightness", "rgb_gamma", "show_rgb_plane",
    "image_spatial_sigma",
)
def rgb_and_image_controls_changed(**_):
    if CURRENT_SEARCH_RESULT is None:
        return
    update_all_slices(reset_camera=False)
    update_rgb_composite(exact=False)
    update_gc_image()
    pyvista_update_view()


@state.change(
    "slice_count",
    "slice_point_limit",
    "slice_image_bins",
    "slice_3d_mode",
    *SLICE_CHANGE_KEYS,
)
def slice_controls_changed(**_):
    state.slice_download_ready = False
    state.dirty("slice_download_ready")
    if CURRENT_SEARCH_RESULT is not None:
        update_all_slices()
        update_profile_chart()


@state.change(
    "gc_center_fe64",
    "gc_width_fe64",
    "gc_center_fe67",
    "gc_width_fe67",
    "gc_stretch",
    "gc_brightness",
    "gc_contrast",
    "gc_smoothing_sigma",
    "gc_image_bins",
)
def energy_image_controls_changed(**_):
    if not CURRENT_PYVISTA_POINTS.empty:
        update_gc_image()


@state.change("profile_width_pixels")
def profile_width_changed(**_):
    update_profile_chart()


@state.change("use_kev_filter", "kev_min", "kev_max", "channels_per_kev")
def kev_filter_changed(**_):
    state.download_ready = False
    state.download_csv_href = ""
    state.download_json_href = ""
    state.dirty("download_ready", "download_csv_href", "download_json_href")
    try:
        kev_filter_signature()
    except ValueError as error:
        state.status_message = str(error)
        state.dirty("status_message")
        return
    if state.selected_files:
        load_selected_files()


with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("UDON3 Server File Search")
    layout.drawer.width = 420

    with layout.toolbar:
        vuetify.VChip("{{ status_message }}", color="primary", variant="tonal", size="small")
        vuetify.VSpacer()
        vuetify.VTextField(
            label="RA",
            v_model=("center_ra", 266.404996),
            type="number",
            step="0.000001",
            density="compact",
            hide_details=True,
            variant="outlined",
            style="max-width: 128px;",
            classes="ml-2",
        )
        vuetify.VTextField(
            label="DEC",
            v_model=("center_dec", -28.936172),
            type="number",
            step="0.000001",
            density="compact",
            hide_details=True,
            variant="outlined",
            style="max-width: 128px;",
            classes="ml-2",
        )
        vuetify.VTextField(
            label="Radius",
            v_model=("radius_deg", 1.0),
            type="number",
            min=0.001,
            step=0.01,
            density="compact",
            hide_details=True,
            variant="outlined",
            style="max-width: 112px;",
            classes="ml-2",
        )
        vuetify.VBtn(
            "Search RA/DEC",
            prepend_icon="mdi-crosshairs-gps",
            color="primary",
            variant="flat",
            classes="ml-2",
            click=ctrl.search_sky_region,
        )
        vuetify.VBtn(
            "Search",
            prepend_icon="mdi-magnify",
            color="primary",
            variant="tonal",
            classes="ml-2",
            click=ctrl.run_search,
        )
        vuetify.VBtn(
            "Show all",
            prepend_icon="mdi-table-eye",
            color="primary",
            variant="tonal",
            classes="ml-2",
            click=ctrl.show_all_files,
        )

    with layout.drawer:
        with vuetify.VContainer(classes="pa-3"):
            with vuetify.VExpansionPanels(
                v_model=("drawer_panels", ["missions", "sky", "files", "energy", "load"]),
                multiple=True,
                variant="accordion",
            ):
                with vuetify.VExpansionPanel(value="missions"):
                    vuetify.VExpansionPanelTitle("Missions and instruments")
                    with vuetify.VExpansionPanelText():
                        vuetify.VListItemSubtitle("All eight imaging instruments are enabled initially")
                        vuetify.VDivider(classes="my-2")
                        for mission, key in MISSION_KEYS.items():
                            vuetify.VSwitch(
                                label=mission.upper(),
                                v_model=(key, True),
                                color="primary",
                                density="compact",
                                hide_details=True,
                            )
                            with vuetify.VContainer(classes="py-0 pl-5"):
                                for instrument in backend.KNOWN_INSTRUMENTS.get(mission, ()):
                                    vuetify.VCheckbox(
                                        label=instrument.upper(),
                                        v_model=(INSTRUMENT_KEYS[(mission, instrument)], True),
                                        color="primary",
                                        density="compact",
                                        hide_details=True,
                                    )

                with vuetify.VExpansionPanel(value="sky"):
                    vuetify.VExpansionPanelTitle("Sky region")
                    with vuetify.VExpansionPanelText():
                        with vuetify.VBtnToggle(
                            v_model=("spatial_selection_mode", "circle"),
                            mandatory=True,
                            divided=True,
                            color="primary",
                            variant="outlined",
                            classes="w-100 mb-3",
                        ):
                            vuetify.VBtn("Circle", value="circle", size="small")
                            vuetify.VBtn("RA/DEC rectangle", value="rectangle", size="small")
                        with vuetify.VBtnToggle(
                            v_model=("coordinate_mode", "degrees"),
                            mandatory=True,
                            divided=True,
                            color="primary",
                            variant="outlined",
                            classes="w-100 mb-3",
                            v_show="spatial_selection_mode === 'circle'",
                        ):
                            vuetify.VBtn("Degrees", value="degrees", size="small")
                            vuetify.VBtn("HH:MM:SS", value="sexagesimal", size="small")
                            vuetify.VBtn("Target", value="target", size="small")
                        with vuetify.VRow(
                            classes="mt-1",
                            v_show="spatial_selection_mode === 'circle' && coordinate_mode !== 'target'",
                        ):
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="Right ascension",
                                    v_model=("coordinate_ra", "266.404996"),
                                    placeholder="266.405 or 17:45:37.2",
                                    variant="outlined",
                                    density="compact",
                                    hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="Declination",
                                    v_model=("coordinate_dec", "-28.936172"),
                                    placeholder="-28.936 or -28:56:10",
                                    variant="outlined",
                                    density="compact",
                                    hide_details=True,
                                )
                        vuetify.VTextField(
                            label="Target name",
                            v_model=("target_name", ""),
                            placeholder="Cas A, SN 1006, Sagittarius A*",
                            prepend_inner_icon="mdi-telescope",
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            v_show="spatial_selection_mode === 'circle' && coordinate_mode === 'target'",
                        )
                        vuetify.VTextField(
                            label="Display radius",
                            v_model=("radius_deg", 1.0),
                            type="number",
                            min=0.001,
                            step=0.01,
                            suffix="deg",
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            classes="mt-3",
                            v_show="spatial_selection_mode === 'circle'",
                        )
                        with vuetify.VRow(
                            dense=True,
                            v_show="spatial_selection_mode === 'rectangle'",
                        ):
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="RA minimum",
                                    v_model=("rectangle_ra_min", 265.9),
                                    type="number", step="0.000001", suffix="deg",
                                    variant="outlined", density="compact", hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="RA maximum",
                                    v_model=("rectangle_ra_max", 266.9),
                                    type="number", step="0.000001", suffix="deg",
                                    variant="outlined", density="compact", hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="DEC minimum",
                                    v_model=("rectangle_dec_min", -29.4),
                                    type="number", step="0.000001", suffix="deg",
                                    variant="outlined", density="compact", hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="DEC maximum",
                                    v_model=("rectangle_dec_max", -28.4),
                                    type="number", step="0.000001", suffix="deg",
                                    variant="outlined", density="compact", hide_details=True,
                                )
                        vuetify.VListItemSubtitle(
                            "RA minimum greater than maximum crosses 0 degrees.",
                            classes="mt-2 text-caption",
                            v_show="spatial_selection_mode === 'rectangle'",
                        )
                        vuetify.VBtn(
                            "Resolve and load region",
                            prepend_icon="mdi-crosshairs-gps",
                            color="primary",
                            block=True,
                            classes="mt-3",
                            click=ctrl.resolve_and_load_region,
                            loading=("region_loading", False),
                            disabled=("region_loading", False),
                        )
                        vuetify.VListItemSubtitle(
                            "{{ resolved_region }}",
                            classes="mt-2 text-caption",
                        )
                        vuetify.VListItemSubtitle(
                            "{{ instrument_coverage }}",
                            classes="mt-2 text-caption",
                        )

                with vuetify.VExpansionPanel(value="files"):
                    vuetify.VExpansionPanelTitle("File Search")
                    with vuetify.VExpansionPanelText():
                        vuetify.VTextField(
                            label="Object / target name",
                            v_model=("object_query", ""),
                            placeholder="CRAB, Perseus, NGC4507",
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                        )
                        vuetify.VTextField(
                            label="Observation ID",
                            v_model=("obsid_query", ""),
                            placeholder="201067010",
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            classes="mt-3",
                        )
                        vuetify.VTextField(
                            label="Parquet filename",
                            v_model=("filename_query", ""),
                            placeholder="201067010_resolve_events.parquet",
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            classes="mt-3",
                        )
                        with vuetify.VRow(classes="mt-1"):
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="Date start",
                                    v_model=("date_start", ""),
                                    placeholder="YYYY-MM-DD",
                                    variant="outlined",
                                    density="compact",
                                    hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="Date end",
                                    v_model=("date_end", ""),
                                    placeholder="YYYY-MM-DD",
                                    variant="outlined",
                                    density="compact",
                                    hide_details=True,
                                )
                        vuetify.VTextField(
                            label="Result limit",
                            v_model=("result_limit", DEFAULT_LIMIT),
                            type="number",
                            min=1,
                            step=1,
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            classes="mt-3",
                        )
                        vuetify.VBtn(
                            "Search selected missions",
                            prepend_icon="mdi-magnify",
                            color="primary",
                            block=True,
                            classes="mt-3",
                            click=ctrl.run_search,
                        )
                        vuetify.VBtn(
                            "Show all selected missions",
                            prepend_icon="mdi-table-eye",
                            color="primary",
                            variant="tonal",
                            block=True,
                            classes="mt-2",
                            click=ctrl.show_all_files,
                        )

                with vuetify.VExpansionPanel(value="energy"):
                    vuetify.VExpansionPanelTitle("Energy / keV Filter")
                    with vuetify.VExpansionPanelText():
                        vuetify.VSwitch(
                            label="Use keV filter",
                            v_model=("use_kev_filter", False),
                            color="primary",
                            density="compact",
                            hide_details=True,
                        )
                        with vuetify.VRow(classes="mt-1"):
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="keV min",
                                    v_model=("kev_min", 0.5),
                                    type="number",
                                    min=0,
                                    step=0.01,
                                    variant="outlined",
                                    density="compact",
                                    hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="keV max",
                                    v_model=("kev_max", 10.0),
                                    type="number",
                                    min=0,
                                    step=0.01,
                                    variant="outlined",
                                    density="compact",
                                    hide_details=True,
                                )
                        vuetify.VTextField(
                            label="Fallback PI channels per keV",
                            v_model=("channels_per_kev", DEFAULT_CHANNELS_PER_KEV),
                            type="number",
                            min=1,
                            step=1,
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            classes="mt-3",
                        )
                        vuetify.VBtn(
                            "Apply keV filter",
                            prepend_icon="mdi-tune-variant",
                            color="primary",
                            variant="tonal",
                            block=True,
                            classes="mt-3",
                            click=ctrl.load_selected_files,
                        )

                with vuetify.VExpansionPanel(value="slices"):
                    vuetify.VExpansionPanelTitle("Energy Slices")
                    with vuetify.VExpansionPanelText():
                        vuetify.VListItemSubtitle("{{ slice_status_message }}")
                        with vuetify.VRow(classes="mt-1"):
                            with vuetify.VCol(cols=6):
                                vuetify.VBtn(
                                    "Add slice",
                                    prepend_icon="mdi-plus",
                                    color="primary",
                                    variant="flat",
                                    block=True,
                                    click=ctrl.add_slice,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VBtn(
                                    "Remove slice",
                                    prepend_icon="mdi-minus",
                                    color="primary",
                                    variant="tonal",
                                    block=True,
                                    click=ctrl.remove_slice,
                                )
                        with vuetify.VRow(classes="mt-1"):
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="Total 3D slice point limit",
                                    v_model=("slice_point_limit", DEFAULT_SLICE_POINT_LIMIT),
                                    type="number",
                                    min=100,
                                    step=1000,
                                    variant="outlined",
                                    density="compact",
                                    hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="2D image bins",
                                    v_model=("slice_image_bins", DEFAULT_SLICE_IMAGE_BINS),
                                    type="number",
                                    min=16,
                                    max=400,
                                    step=8,
                                    variant="outlined",
                                    density="compact",
                                    hide_details=True,
                                )
                        vuetify.VSelect(
                            label="3D slice display",
                            v_model=("slice_3d_mode", "all"),
                            items=("slice_3d_mode_options",),
                            item_title="title",
                            item_value="value",
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            classes="mt-2",
                        )
                        for index in range(MAX_SLICES):
                            with vuetify.VSheet(
                                v_show=f"slice_count > {index}",
                                classes="mt-3 pa-2 border rounded",
                            ):
                                with vuetify.VRow(dense=True, classes="align-center"):
                                    with vuetify.VCol(cols=6):
                                        vuetify.VSwitch(
                                            label=f"Slice {index + 1}",
                                            v_model=(slice_state_name(index, "enabled"), index == 0),
                                            color="primary",
                                            density="compact",
                                            hide_details=True,
                                        )
                                    with vuetify.VCol(cols=2):
                                        with vuetify.VBtn(
                                            icon=True,
                                            size="small",
                                            title=f"Focus slice {index + 1}",
                                            color="primary",
                                            variant="tonal",
                                            click=getattr(ctrl, f"set_active_slice_{index}"),
                                        ):
                                            vuetify.VIcon(
                                                "mdi-eye",
                                                v_show=f"active_slice_index === {index}",
                                            )
                                            vuetify.VIcon(
                                                "mdi-eye-outline",
                                                v_show=f"active_slice_index !== {index}",
                                            )
                                    with vuetify.VCol(cols=4):
                                        vuetify.VTextField(
                                            label="Color",
                                            v_model=(slice_state_name(index, "color"), SLICE_COLORS[index % len(SLICE_COLORS)]),
                                            type="color",
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                                vuetify.VSlider(
                                    label="Center keV",
                                    v_model=(slice_state_name(index, "center_kev"), 6.55 + index * 0.25),
                                    min=0,
                                    max=30,
                                    step=0.01,
                                    thumb_label=True,
                                    hide_details=True,
                                    classes="mt-1",
                                    end=getattr(ctrl, f"update_exact_slice_{index}"),
                                )
                                vuetify.VSlider(
                                    label="Width keV",
                                    v_model=(slice_state_name(index, "width_kev"), 0.30),
                                    min=0.01,
                                    max=5,
                                    step=0.01,
                                    thumb_label=True,
                                    hide_details=True,
                                    classes="mt-1",
                                    end=getattr(ctrl, f"update_exact_slice_{index}"),
                                )
                                vuetify.VSlider(
                                    label="Slab opacity",
                                    v_model=(slice_state_name(index, "opacity"), 0.18),
                                    min=0,
                                    max=0.6,
                                    step=0.02,
                                    thumb_label=True,
                                    hide_details=True,
                                    classes="mt-1",
                                )
                                vuetify.VListItemSubtitle(
                                    f"{{{{ {slice_state_name(index, 'status')} }}}}",
                                    classes="text-caption",
                                )

                with vuetify.VExpansionPanel(value="gc"):
                    vuetify.VExpansionPanelTitle("Energy Images")
                    with vuetify.VExpansionPanelText():
                        vuetify.VListItemSubtitle(
                            "Configure RGB energy bands and independent RA/DEC maps"
                        )
                        vuetify.VLabel(
                            "RGB ENERGY COMPOSITE",
                            classes="text-caption font-weight-bold mt-3",
                        )
                        for color_name, center, width, swatch in (
                            ("red", 1.85, 0.20, "#d71920"),
                            ("green", 2.44, 0.20, "#16a34a"),
                            ("blue", 6.40, 0.40, "#2563eb"),
                        ):
                            with vuetify.VRow(dense=True, classes="mt-1 align-center"):
                                with vuetify.VCol(cols=1):
                                    vuetify.VIcon("mdi-circle", color=swatch, size="small")
                                with vuetify.VCol(cols=5):
                                    vuetify.VTextField(
                                        label=f"{color_name.title()} center",
                                        v_model=(f"rgb_{color_name}_center", center),
                                        type="number", min=0, max=30, step=0.01,
                                        suffix="keV", variant="outlined", density="compact",
                                        hide_details=True,
                                    )
                                with vuetify.VCol(cols=6):
                                    vuetify.VTextField(
                                        label="Width",
                                        v_model=(f"rgb_{color_name}_width", width),
                                        type="number", min=0.001, max=10, step=0.01,
                                        suffix="keV", variant="outlined", density="compact",
                                        hide_details=True,
                                    )
                        vuetify.VSlider(
                            label="RGB brightness",
                            v_model=("rgb_brightness", 1.25), min=0.1, max=4,
                            step=0.05, thumb_label=True, hide_details=True, classes="mt-2",
                        )
                        vuetify.VSlider(
                            label="RGB gamma",
                            v_model=("rgb_gamma", 1.0), min=0.2, max=3,
                            step=0.05, thumb_label=True, hide_details=True, classes="mt-2",
                        )
                        vuetify.VSwitch(
                            label="Show RGB image plane in 3D",
                            v_model=("show_rgb_plane", True), color="primary",
                            density="compact", hide_details=True,
                        )
                        vuetify.VBtn(
                            "Compute exact RGB image",
                            prepend_icon="mdi-image-filter-hdr", color="primary",
                            variant="tonal", block=True, classes="mt-2",
                            click=ctrl.update_rgb_exact,
                        )
                        vuetify.VListItemSubtitle("{{ rgb_status }}", classes="mt-2 text-caption")
                        vuetify.VDivider(classes="my-3")
                        for mode, number, center in (
                            ("fe64", 1, 6.40),
                            ("fe67", 2, 6.70),
                        ):
                            vuetify.VLabel(
                                f"ENERGY MAP {number}",
                                classes="text-caption font-weight-bold mt-4",
                            )
                            with vuetify.VRow(classes="mt-1", dense=True):
                                with vuetify.VCol(cols=6):
                                    vuetify.VTextField(
                                        label="Center (keV)",
                                        v_model=(f"gc_center_{mode}", center),
                                        type="number",
                                        min=0,
                                        max=30,
                                        step=0.01,
                                        variant="outlined",
                                        density="compact",
                                        hide_details=True,
                                    )
                                with vuetify.VCol(cols=6):
                                    vuetify.VTextField(
                                        label="Width (keV)",
                                        v_model=(f"gc_width_{mode}", 0.20),
                                        type="number",
                                        min=0.001,
                                        max=10,
                                        step=0.01,
                                        variant="outlined",
                                        density="compact",
                                        hide_details=True,
                                    )
                            vuetify.VLabel("Color map", classes="text-caption mt-1")
                            with vuetify.VRow(dense=True, classes="mt-1 flex-nowrap"):
                                for palette_key, palette_config in PALETTE_OPTIONS.items():
                                    with vuetify.VCol(classes="pa-1"):
                                        with vuetify.VBtn(
                                            icon=True,
                                            size="small",
                                            title=palette_config["label"],
                                            variant="outlined",
                                            style=(
                                                f"background:{palette_config['preview']};"
                                                "min-width:38px;width:38px;height:34px;"
                                                "border:2px solid #ffffff;"
                                                "box-shadow:0 0 0 1px #64748b;"
                                            ),
                                            click=getattr(
                                                ctrl,
                                                f"select_energy_palette_{mode}_{palette_key}",
                                            ),
                                        ):
                                            vuetify.VIcon(
                                                "mdi-check-circle",
                                                color="white",
                                                size="small",
                                                v_show=(
                                                    f"gc_palette_pending_{mode} === "
                                                    f"'{palette_key}'"
                                                ),
                                            )
                            vuetify.VListItemSubtitle(
                                f"Selected palette: {{{{ gc_palette_pending_{mode} }}}}",
                                classes="text-caption",
                            )
                            vuetify.VListItemSubtitle(
                                f"{{{{ gc_status_{mode} }}}}",
                                classes="text-caption mt-1",
                            )
                        vuetify.VSelect(
                            label="Stretch",
                            v_model=("gc_stretch", "log"),
                            items=["log", "sqrt", "linear"],
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            classes="mt-3",
                        )
                        vuetify.VSlider(
                            label="Brightness",
                            v_model=("gc_brightness", 1.35),
                            min=0.1,
                            max=4.0,
                            step=0.05,
                            thumb_label=True,
                            hide_details=True,
                            classes="mt-3",
                        )
                        vuetify.VSlider(
                            label="Contrast",
                            v_model=("gc_contrast", 1.2),
                            min=0.1,
                            max=4.0,
                            step=0.05,
                            thumb_label=True,
                            hide_details=True,
                            classes="mt-3",
                        )
                        vuetify.VSlider(
                            label="Legacy map smoothing (pixels)",
                            v_model=("gc_smoothing_sigma", 1.2),
                            min=0,
                            max=8,
                            step=0.1,
                            thumb_label=True,
                            hide_details=True,
                            classes="mt-3",
                        )
                        vuetify.VSlider(
                            label="Image spatial smoothing (arcmin)",
                            v_model=("image_spatial_sigma", 0.0),
                            min=0, max=10, step=0.1, thumb_label=True,
                            hide_details=True, classes="mt-3",
                        )
                        vuetify.VSlider(
                            label="Image bins",
                            v_model=("gc_image_bins", DEFAULT_GC_IMAGE_BINS),
                            min=64,
                            max=500,
                            step=8,
                            thumb_label=True,
                            hide_details=True,
                            classes="mt-3",
                        )
                        vuetify.VAlert(
                            "The maps use the currently loaded events. Keep the main keV load filter off, or make it wide enough to include both image ranges.",
                            type="info",
                            variant="tonal",
                            density="compact",
                            classes="mt-3 text-caption",
                        )

                with vuetify.VExpansionPanel(value="catalog"):
                    vuetify.VExpansionPanelTitle("Server Catalog")
                    with vuetify.VExpansionPanelText():
                        vuetify.VListItemSubtitle("{{ catalog_summary }}")
                        vuetify.VBtn(
                            "Refresh catalog",
                            prepend_icon="mdi-cloud-sync",
                            color="primary",
                            variant="tonal",
                            block=True,
                            classes="mt-3",
                            click=ctrl.refresh_catalog,
                        )

                with vuetify.VExpansionPanel(value="cache"):
                    vuetify.VExpansionPanelTitle("Local Cache")
                    with vuetify.VExpansionPanelText():
                        vuetify.VListItemSubtitle("{{ cache_summary }}")
                        vuetify.VProgressLinear(
                            model_value=("cache_usage_fraction", 0.0),
                            max=1,
                            color="primary",
                            height=7,
                            rounded=False,
                            classes="mt-2",
                        )
                        vuetify.VListItemSubtitle(
                            "{{ cache_message }}",
                            classes="text-caption mt-2",
                        )
                        with vuetify.VRow(dense=True, classes="mt-2"):
                            with vuetify.VCol(cols=4):
                                with vuetify.VBtn(
                                    icon=True,
                                    title="Refresh cache status",
                                    variant="tonal",
                                    color="primary",
                                    block=True,
                                    click=ctrl.refresh_cache_status,
                                ):
                                    vuetify.VIcon("mdi-refresh")
                            with vuetify.VCol(cols=4):
                                with vuetify.VBtn(
                                    icon=True,
                                    title="Clear downloaded raw Parquet files",
                                    variant="tonal",
                                    color="primary",
                                    block=True,
                                    click=ctrl.clear_raw_cache,
                                ):
                                    vuetify.VIcon("mdi-database-remove-outline")
                            with vuetify.VCol(cols=4):
                                with vuetify.VBtn(
                                    icon=True,
                                    title="Clear previews and exact images",
                                    variant="tonal",
                                    color="primary",
                                    block=True,
                                    click=ctrl.clear_derived_cache,
                                ):
                                    vuetify.VIcon("mdi-image-remove-outline")

                with vuetify.VExpansionPanel(value="load"):
                    vuetify.VExpansionPanelTitle("Select / Load / Download")
                    with vuetify.VExpansionPanelText():
                        vuetify.VSelect(
                            label="Click one file to visualize",
                            v_model=("selected_file", ""),
                            items=("result_options", []),
                            prepend_inner_icon="mdi-cursor-pointer",
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                        )
                        vuetify.VSelect(
                            label="Batch files",
                            v_model=("selected_files", []),
                            items=("result_options", []),
                            multiple=True,
                            chips=True,
                            closable_chips=True,
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            classes="mt-3",
                        )
                        vuetify.VBtn(
                            "Select all current results",
                            prepend_icon="mdi-select-all",
                            color="primary",
                            variant="tonal",
                            block=True,
                            classes="mt-3",
                            click=ctrl.select_all_current_results,
                        )
                        vuetify.VBtn(
                            "Load current search in 3D",
                            prepend_icon="mdi-cube-scan",
                            color="primary",
                            variant="flat",
                            block=True,
                            classes="mt-2",
                            click=ctrl.load_current_results_in_3d,
                        )
                        vuetify.VSwitch(
                            label="Load all points",
                            v_model=("load_all_points", False),
                            color="primary",
                            density="compact",
                            hide_details=True,
                            classes="mt-3",
                        )
                        vuetify.VTextField(
                            label="Point limit",
                            v_model=("max_points", DEFAULT_MAX_POINTS),
                            type="number",
                            min=100,
                            step=1000,
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            classes="mt-3",
                            v_show="!load_all_points",
                        )
                        vuetify.VDivider(classes="my-3")
                        vuetify.VSwitch(
                            label="Show PyVista 3D",
                            v_model=("show_pyvista_3d", True),
                            color="primary",
                            density="compact",
                            hide_details=True,
                        )
                        with vuetify.VBtnToggle(
                            v_model=("pyvista_display_mode", "points"),
                            mandatory=True,
                            divided=True,
                            color="primary",
                            variant="outlined",
                            classes="w-100 mt-3",
                            v_show=("show_pyvista_3d",),
                        ):
                            vuetify.VBtn("Points", value="points", size="small")
                            vuetify.VBtn("Voxels", value="voxels", size="small")
                        vuetify.VSelect(
                            label="3D color mode",
                            v_model=("pyvista_color_mode", "mission"),
                            items=[
                                {"title": "Mission", "value": "mission"},
                                {"title": "Energy palette", "value": "pi"},
                                {"title": "RGB energy bands", "value": "rgb"},
                            ],
                            item_title="title",
                            item_value="value",
                            variant="outlined",
                            density="compact",
                            hide_details=True,
                            classes="mt-3",
                            v_show=("show_pyvista_3d",),
                        )
                        with vuetify.VRow(
                            dense=True,
                            classes="mt-2",
                            v_show="show_pyvista_3d && pyvista_display_mode === 'voxels'",
                        ):
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="Spatial cell",
                                    v_model=("voxel_spatial_size", 1.0),
                                    type="number", min=0.01, step=0.1, suffix="arcmin",
                                    variant="outlined", density="compact", hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    label="Energy cell",
                                    v_model=("voxel_energy_size", 0.1),
                                    type="number", min=0.001, step=0.01, suffix="keV",
                                    variant="outlined", density="compact", hide_details=True,
                                )
                        with vuetify.VContainer(
                            classes="pa-0 mt-3",
                            v_show="show_pyvista_3d && pyvista_color_mode === 'pi'",
                        ):
                            vuetify.VLabel("3D energy color map", classes="text-caption")
                            with vuetify.VRow(dense=True, classes="mt-1 flex-nowrap"):
                                for palette_key, palette_config in PALETTE_OPTIONS.items():
                                    with vuetify.VCol(classes="pa-1"):
                                        with vuetify.VBtn(
                                            icon=True,
                                            size="small",
                                            title=palette_config["label"],
                                            variant="outlined",
                                            style=(
                                                f"background:{palette_config['preview']};"
                                                "min-width:38px;width:38px;height:34px;"
                                                "border:2px solid #ffffff;"
                                                "box-shadow:0 0 0 1px #64748b;"
                                            ),
                                            click=getattr(
                                                ctrl,
                                                f"select_pyvista_palette_{palette_key}",
                                            ),
                                        ):
                                            vuetify.VIcon(
                                                "mdi-check-circle",
                                                color="white",
                                                size="small",
                                                v_show=(
                                                    "pyvista_colormap_pending === "
                                                    f"'{palette_key}'"
                                                ),
                                            )
                            vuetify.VListItemSubtitle(
                                "Selected palette: {{ pyvista_colormap_pending }}",
                                classes="text-caption",
                            )
                        with vuetify.VRow(classes="mt-1", v_show=("show_pyvista_3d",)):
                            with vuetify.VCol(cols=7):
                                vuetify.VListItemSubtitle(
                                    "3D uses the loaded files and point setting above.",
                                    classes="text-caption mt-2",
                                )
                            with vuetify.VCol(cols=5):
                                vuetify.VTextField(
                                    label="3D point size",
                                    v_model=("pyvista_point_size", DEFAULT_PYVISTA_POINT_SIZE),
                                    type="number",
                                    min=1,
                                    max=14,
                                    step=1,
                                    variant="outlined",
                                    density="compact",
                                    hide_details=True,
                                )
                        vuetify.VSlider(
                            label="Event spatial smoothing (arcmin)",
                            v_model=("event_spatial_sigma", 0.0),
                            min=0, max=10, step=0.1, thumb_label=True,
                            hide_details=True, classes="mt-2",
                            v_show=("show_pyvista_3d",),
                        )
                        vuetify.VSlider(
                            label="Event energy smoothing (keV)",
                            v_model=("event_energy_sigma", 0.0),
                            min=0, max=2, step=0.02, thumb_label=True,
                            hide_details=True, classes="mt-2",
                            v_show=("show_pyvista_3d",),
                        )
                        vuetify.VSlider(
                            label="Density point size",
                            v_model=("density_size_strength", 0.8),
                            min=0, max=2, step=0.1, thumb_label=True,
                            hide_details=True, classes="mt-2",
                            v_show="show_pyvista_3d && (event_spatial_sigma > 0 || event_energy_sigma > 0)",
                        )
                        vuetify.VSlider(
                            label="Density opacity",
                            v_model=("density_opacity_strength", 0.8),
                            min=0, max=2, step=0.1, thumb_label=True,
                            hide_details=True, classes="mt-2",
                            v_show="show_pyvista_3d && (event_spatial_sigma > 0 || event_energy_sigma > 0)",
                        )
                        vuetify.VBtn(
                            "Reset 3D camera",
                            prepend_icon="mdi-cube-scan",
                            color="primary",
                            variant="tonal",
                            block=True,
                            classes="mt-2",
                            click=ctrl.reset_pyvista_camera,
                            v_show=("show_pyvista_3d",),
                        )
                        vuetify.VBtn(
                            "Load and visualize",
                            prepend_icon="mdi-chart-scatter-plot",
                            color="primary",
                            variant="tonal",
                            block=True,
                            classes="mt-3",
                            click=ctrl.load_selected_files,
                        )
                        vuetify.VListItemSubtitle("{{ loaded_summary }}", classes="mt-2")
                        vuetify.VBtn(
                            "Prepare downloads",
                            prepend_icon="mdi-download",
                            color="primary",
                            variant="flat",
                            block=True,
                            classes="mt-3",
                            click=ctrl.export_loaded_data,
                        )
                        vuetify.VBtn(
                            "Download CSV",
                            prepend_icon="mdi-file-delimited",
                            color="primary",
                            variant="tonal",
                            block=True,
                            classes="mt-2",
                            href=("download_csv_href", ""),
                            download=("download_csv_name", "udon3_selected_events.csv"),
                            rel="noopener",
                            v_show=("download_ready",),
                        )
                        vuetify.VBtn(
                            "Download JSON",
                            prepend_icon="mdi-code-json",
                            color="primary",
                            variant="tonal",
                            block=True,
                            classes="mt-2",
                            href=("download_json_href", ""),
                            download=("download_json_name", "udon3_selected_events.json"),
                            rel="noopener",
                            v_show=("download_ready",),
                        )
                        vuetify.VListItemSubtitle("{{ export_summary }}", classes="mt-2")

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-3",
            style="height: 100%; overflow-y: auto; background: #f8fafc;",
        ):
            with vuetify.VRow(dense=True):
                with vuetify.VCol(cols=12):
                    with vuetify.VCard(variant="outlined"):
                        vuetify.VCardTitle("UDON3 server files")
                        vuetify.VCardSubtitle("{{ results_summary }}")
                        with vuetify.VCardText(classes="pb-0"):
                            vuetify.VSelect(
                                label="Click one server file to visualize",
                                v_model=("selected_file", ""),
                                items=("result_options", []),
                                prepend_inner_icon="mdi-cursor-pointer",
                                variant="outlined",
                                density="compact",
                                hide_details=True,
                            )
                            with vuetify.VRow(classes="mt-1", dense=True):
                                with vuetify.VCol(cols=12, md=6):
                                    vuetify.VBtn(
                                        "Select all current results",
                                        prepend_icon="mdi-select-all",
                                        color="primary",
                                        variant="tonal",
                                        block=True,
                                        click=ctrl.select_all_current_results,
                                    )
                                with vuetify.VCol(cols=12, md=6):
                                    vuetify.VBtn(
                                        "Load current search in 3D",
                                        prepend_icon="mdi-cube-scan",
                                        color="primary",
                                        variant="flat",
                                        block=True,
                                        click=ctrl.load_current_results_in_3d,
                                    )
                            vuetify.VBtn(
                                "{{ show_result_table ? 'Hide file list' : 'Show file list' }}",
                                prepend_icon="mdi-format-list-bulleted",
                                color="primary",
                                variant="tonal",
                                classes="mt-3",
                                click=ctrl.toggle_result_table,
                            )
                        RESULTS_CHART = trame_plotly.Figure(
                            figure=empty_figure("Search or show all server files"),
                            display_logo=False,
                            display_mode_bar=True,
                            responsive=True,
                            v_show=("show_result_table",),
                            style="height: auto; min-height: 520px; width: 100%;",
                        )
                with vuetify.VCol(cols=12, v_show=("has_loaded_data",)):
                    with vuetify.VTabs(
                        v_model=("workspace_tab", "events"),
                        color="primary",
                        density="compact",
                    ):
                        vuetify.VTab("Events", value="events")
                        vuetify.VTab("Energy maps", value="maps")
                        vuetify.VTab("3D and slice", value="slice")
                with vuetify.VCol(cols=12):
                    with vuetify.VCard(
                        variant="outlined",
                        v_show="has_loaded_data && workspace_tab === 'events'",
                    ):
                        vuetify.VCardTitle("Loaded event preview")
                        vuetify.VCardSubtitle("{{ loaded_summary }}")
                        VIEWER_CHART = trame_plotly.Figure(
                            figure=empty_figure("Select result files, then load and visualize"),
                            display_logo=False,
                            display_mode_bar=True,
                            responsive=True,
                            style="height: 48vh; width: 100%;",
                        )
                with vuetify.VCol(cols=12):
                    with vuetify.VCard(
                        variant="outlined",
                        v_show="has_loaded_data && workspace_tab === 'maps'",
                    ):
                        vuetify.VCardTitle("Energy-band images")
                        vuetify.VCardSubtitle(
                            "Compare two independently selected keV ranges"
                        )
                        with vuetify.VCardText(classes="pb-0"):
                            vuetify.VAlert(
                                "Event-density images from the loaded parquet files. These are not exposure-corrected publication products.",
                                type="info",
                                variant="tonal",
                                density="compact",
                            )
                        with vuetify.VRow(dense=True):
                            with vuetify.VCol(cols=12):
                                RGB_IMAGE_CHART = trame_plotly.Figure(
                                    figure=gc_empty_figure("Load data for RGB energy composite"),
                                    state_variable_name="rgb_image_figure",
                                    display_logo=False,
                                    display_mode_bar=True,
                                    responsive=True,
                                    style="height:430px; width:100%;",
                                )
                        with vuetify.VRow(dense=True):
                            with vuetify.VCol(cols=12, md=6):
                                GC_IMAGE_CHARTS["fe64"] = trame_plotly.Figure(
                                    figure=gc_empty_figure("Load data for energy map 1"),
                                    state_variable_name=GC_IMAGE_STATE_KEYS["fe64"],
                                    display_logo=False,
                                    display_mode_bar=True,
                                    responsive=True,
                                    style="height: 430px; width: 100%;",
                                )
                            with vuetify.VCol(cols=12, md=6):
                                GC_IMAGE_CHARTS["fe67"] = trame_plotly.Figure(
                                    figure=gc_empty_figure("Load data for energy map 2"),
                                    state_variable_name=GC_IMAGE_STATE_KEYS["fe67"],
                                    display_logo=False,
                                    display_mode_bar=True,
                                    responsive=True,
                                    style="height: 430px; width: 100%;",
                                )
                with vuetify.VCol(cols=12):
                    with vuetify.VCard(
                        variant="outlined",
                        v_show=(
                            "has_loaded_data && show_pyvista_3d && "
                            "workspace_tab === 'slice'"
                        ),
                    ):
                        vuetify.VCardTitle("PyVista 3D event view")
                        vuetify.VCardSubtitle("{{ pyvista_status }}")
                        if PYVISTA_AVAILABLE and PYVISTA_PLOTTER is not None:
                            with vuetify.VRow(no_gutters=True):
                                with vuetify.VCol(cols=12, md=6):
                                    with vuetify.VCardText(classes="pa-0"):
                                        PYVISTA_VIEW = vtk_widgets.VtkRemoteView(
                                            PYVISTA_PLOTTER.ren_win,
                                            interactive_ratio=1,
                                            still_ratio=1,
                                            style="height: 72vh; width: 100%;",
                                        )
                                        ctrl.pyvista_view_update = PYVISTA_VIEW.update
                                        ctrl.pyvista_view_reset_camera = PYVISTA_VIEW.reset_camera
                                with vuetify.VCol(
                                    cols=12,
                                    md=6,
                                    classes="pa-2",
                                    style=(
                                        "height:72vh; overflow-y:auto; "
                                        "border-left:1px solid #d8e0e8;"
                                    ),
                                ):
                                    vuetify.VLabel(
                                        "ENERGY SLICE IMAGES",
                                        classes="text-caption font-weight-bold",
                                    )
                                    vuetify.VListItemSubtitle(
                                        "Click A and B on any image; the same sky line is shared by every slice",
                                        classes="mb-1",
                                    )
                                    with vuetify.VRow(dense=True):
                                        for index in range(MAX_SLICES):
                                            with vuetify.VCol(
                                                cols=12,
                                                lg=6,
                                                v_show=(
                                                    f"slice_count > {index} && "
                                                    f"{slice_state_name(index, 'enabled')}"
                                                ),
                                            ):
                                                with vuetify.VSheet(
                                                    classes="pa-1 border rounded",
                                                ):
                                                    with vuetify.VRow(
                                                        dense=True,
                                                        classes="align-center px-1",
                                                    ):
                                                        with vuetify.VCol(cols=7):
                                                            vuetify.VLabel(
                                                                f"Slice {index + 1}",
                                                                classes=(
                                                                    "text-caption "
                                                                    "font-weight-bold"
                                                                ),
                                                            )
                                                        with vuetify.VCol(cols=5):
                                                            vuetify.VChip(
                                                                "Active",
                                                                color="primary",
                                                                size="x-small",
                                                                variant="tonal",
                                                                v_show=(
                                                                    "active_slice_index === "
                                                                    f"{index}"
                                                                ),
                                                            )
                                                    SLICE_IMAGE_CHARTS[index] = (
                                                        trame_plotly.Figure(
                                                            figure=slice_empty_figure(
                                                                f"Load data for slice {index + 1}"
                                                            ),
                                                            state_variable_name=(
                                                                f"slice_image_figure_{index}"
                                                            ),
                                                            display_logo=False,
                                                            display_mode_bar=True,
                                                            responsive=True,
                                                            click=(
                                                                getattr(
                                                                    ctrl,
                                                                    f"slice_image_clicked_{index}",
                                                                ),
                                                                "[[$event.points[0].x, "
                                                                "$event.points[0].y]]",
                                                            ),
                                                            style=(
                                                                "height:280px; "
                                                                "width:100%;"
                                                            ),
                                                        )
                                                    )
                                                    vuetify.VBtn(
                                                        "Focus in 3D",
                                                        prepend_icon="mdi-cube-scan",
                                                        color="primary",
                                                        variant="text",
                                                        size="small",
                                                        block=True,
                                                        click=getattr(
                                                            ctrl,
                                                            f"set_active_slice_{index}",
                                                        ),
                                                    )
                                    vuetify.VDivider(classes="my-3")
                                    vuetify.VLabel(
                                        "COMPARATIVE SLICE PROFILES",
                                        classes="text-caption font-weight-bold",
                                    )
                                    vuetify.VListItemSubtitle(
                                        "{{ profile_status }}",
                                        classes="mb-1",
                                    )
                                    with vuetify.VRow(dense=True, classes="align-center mt-1"):
                                        with vuetify.VCol(cols=8):
                                            vuetify.VSlider(
                                                label="Profile width (pixels)",
                                                v_model=("profile_width_pixels", 3),
                                                min=1,
                                                max=21,
                                                step=2,
                                                thumb_label=True,
                                                hide_details=True,
                                            )
                                        with vuetify.VCol(cols=4):
                                            with vuetify.VBtn(
                                                icon=True,
                                                title="Clear profile",
                                                variant="tonal",
                                                color="primary",
                                                click=ctrl.clear_slice_profile,
                                            ):
                                                vuetify.VIcon("mdi-close")
                                    SLICE_PROFILE_CHART = trame_plotly.Figure(
                                        figure=profile_empty_figure(),
                                        state_variable_name="slice_profile_figure",
                                        display_logo=False,
                                        display_mode_bar=True,
                                        responsive=True,
                                        style="height: 310px; width: 100%;",
                                    )
                                    vuetify.VBtn(
                                        "Prepare profile data",
                                        prepend_icon="mdi-chart-line",
                                        color="primary",
                                        variant="flat",
                                        block=True,
                                        click=ctrl.prepare_profile_download,
                                    )
                                    with vuetify.VRow(
                                        dense=True,
                                        classes="mt-1",
                                        v_show=("profile_download_ready",),
                                    ):
                                        with vuetify.VCol(cols=6):
                                            vuetify.VBtn(
                                                "Profile CSV",
                                                prepend_icon="mdi-file-delimited",
                                                block=True,
                                                variant="outlined",
                                                href=("profile_download_csv_href", ""),
                                                download=(
                                                    "profile_download_csv_name",
                                                    "profiles.csv",
                                                ),
                                            )
                                        with vuetify.VCol(cols=6):
                                            vuetify.VBtn(
                                                "Profile JSON",
                                                prepend_icon="mdi-code-json",
                                                block=True,
                                                variant="outlined",
                                                href=("profile_download_json_href", ""),
                                                download=(
                                                    "profile_download_json_name",
                                                    "profiles.json",
                                                ),
                                            )
                                    vuetify.VListItemSubtitle(
                                        "{{ profile_download_status }}",
                                        classes="text-caption mt-1",
                                    )
                                    vuetify.VDivider(classes="my-3")
                                    vuetify.VBtn(
                                        "Prepare pixel data",
                                        prepend_icon="mdi-database-export-outline",
                                        color="primary",
                                        variant="tonal",
                                        block=True,
                                        click=ctrl.prepare_slice_image_download,
                                    )
                                    with vuetify.VRow(
                                        dense=True,
                                        classes="mt-1",
                                        v_show=("slice_download_ready",),
                                    ):
                                        with vuetify.VCol(cols=6):
                                            vuetify.VBtn(
                                                "Pixel CSV",
                                                prepend_icon="mdi-file-delimited",
                                                block=True,
                                                variant="outlined",
                                                href=("slice_download_csv_href", ""),
                                                download=("slice_download_csv_name", "slice.csv"),
                                            )
                                        with vuetify.VCol(cols=6):
                                            vuetify.VBtn(
                                                "Image FITS",
                                                prepend_icon="mdi-image-filter-hdr",
                                                block=True,
                                                variant="outlined",
                                                href=("slice_download_fits_href", ""),
                                                download=("slice_download_fits_name", "slice.fits"),
                                            )
                                    vuetify.VListItemSubtitle(
                                        "{{ slice_download_status }}",
                                        classes="text-caption mt-1",
                                    )
                            with vuetify.VCardText(classes="pb-0"):
                                vuetify.VAlert(
                                    "{{ pyvista_status }}",
                                    type="info",
                                    variant="tonal",
                                    density="compact",
                                )
                                vuetify.VListItemSubtitle(
                                    "Click a 3D point to display its file, coordinates, time, PI, and keV.",
                                    classes="mt-2",
                                    v_show="!pyvista_has_selection",
                                )
                            with vuetify.VCardText(
                                classes="pt-3",
                                v_show=("pyvista_has_selection",),
                            ):
                                with vuetify.VRow(dense=True):
                                    with vuetify.VCol(cols=12, md=4):
                                        vuetify.VTextField(
                                            label="Selected file",
                                            v_model=("pyvista_selected_file", "-"),
                                            readonly=True,
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                                    with vuetify.VCol(cols=6, md=2):
                                        vuetify.VTextField(
                                            label="Row",
                                            v_model=("pyvista_selected_row", "-"),
                                            readonly=True,
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                                    with vuetify.VCol(cols=6, md=2):
                                        vuetify.VTextField(
                                            label="PI",
                                            v_model=("pyvista_selected_pi", "-"),
                                            readonly=True,
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                                    with vuetify.VCol(cols=6, md=2):
                                        vuetify.VTextField(
                                            label="RA",
                                            v_model=("pyvista_selected_ra", "-"),
                                            readonly=True,
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                                    with vuetify.VCol(cols=6, md=2):
                                        vuetify.VTextField(
                                            label="DEC",
                                            v_model=("pyvista_selected_dec", "-"),
                                            readonly=True,
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                                with vuetify.VRow(dense=True, classes="mt-1"):
                                    with vuetify.VCol(cols=12, md=4):
                                        vuetify.VTextField(
                                            label="Object",
                                            v_model=("pyvista_selected_object", "-"),
                                            readonly=True,
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                                    with vuetify.VCol(cols=12, md=3):
                                        vuetify.VTextField(
                                            label="TIME",
                                            v_model=("pyvista_selected_time", "-"),
                                            readonly=True,
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                                    with vuetify.VCol(cols=12, md=3):
                                        vuetify.VTextField(
                                            label="Mission datetime",
                                            v_model=("pyvista_selected_mission_datetime", "-"),
                                            readonly=True,
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                                    with vuetify.VCol(cols=6, md=2):
                                        vuetify.VTextField(
                                            label="keV",
                                            v_model=("pyvista_selected_kev", "-"),
                                            readonly=True,
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                                    with vuetify.VCol(cols=6, md=3):
                                        vuetify.VTextField(
                                            label="Detector X/Y",
                                            v_model=("pyvista_selected_xy", "-"),
                                            readonly=True,
                                            variant="outlined",
                                            density="compact",
                                            hide_details=True,
                                        )
                        else:
                            with vuetify.VCardText():
                                vuetify.VAlert(
                                    "{{ pyvista_status }}",
                                    type="warning",
                                    variant="tonal",
                                    density="compact",
                                )
    layout.footer.hide()


if backend.server_catalog_exists():
    show_all_files()


if __name__ == "__main__":
    server.start()
