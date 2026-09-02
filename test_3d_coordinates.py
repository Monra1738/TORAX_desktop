"""Standalone real-data check for absolute RA, DEC, and energy in PyVista.

This intentionally does not use the desktop viewer's scene transform.  The
coordinates passed to PyVista are the exact values read from the cached parquet
files, making this a small diagnostic for the 3D axes themselves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "var" / "data_cache" / "_products" / "region_preview"

# Edit these two global values to choose the exact visible 3D energy axis.
# Command-line --energy-min/--energy-max values override them when supplied.
GLOBAL_ENERGY_MIN_KEV = 3.10
GLOBAL_ENERGY_MAX_KEV = 4.89


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display cached event files using absolute RA/DEC/keV coordinates."
    )
    parser.add_argument("--file", type=Path, help="Use one specific preview parquet file")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--target-ra", type=float, default=350.8584)
    parser.add_argument("--target-dec", type=float, default=58.8113)
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=1.0,
        help="Include cached previews whose median RA and DEC are this close to the target",
    )
    parser.add_argument("--energy-min", type=float, default=GLOBAL_ENERGY_MIN_KEV)
    parser.add_argument("--energy-max", type=float, default=GLOBAL_ENERGY_MAX_KEV)
    parser.add_argument("--max-points", type=int, default=180_000)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load and validate coordinates without opening a window",
    )
    return parser.parse_args()


def candidate_files(options: argparse.Namespace) -> list[Path]:
    if options.file is not None:
        path = options.file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]
    cache = options.cache.expanduser().resolve()
    paths = sorted(cache.glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No cached preview parquet files found below {cache}")
    return paths


def load_events(options: argparse.Namespace) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    accepted: list[Path] = []
    for path in candidate_files(options):
        try:
            frame = pd.read_parquet(path, columns=["RA", "DEC", "KEV"])
        except Exception:
            continue
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            continue
        if options.file is None:
            median_ra = float(frame["RA"].median())
            median_dec = float(frame["DEC"].median())
            if (
                abs(median_ra - options.target_ra) > options.target_tolerance
                or abs(median_dec - options.target_dec) > options.target_tolerance
            ):
                continue
        frame = frame.loc[
            frame["KEV"].between(options.energy_min, options.energy_max, inclusive="both")
        ]
        if not frame.empty:
            frames.append(frame)
            accepted.append(path)
    if not frames:
        raise RuntimeError("No finite RA/DEC/KEV rows matched the requested target and energy band")
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > options.max_points:
        indices = np.linspace(0, len(combined) - 1, options.max_points, dtype=np.int64)
        combined = combined.iloc[indices].reset_index(drop=True)
    return combined, accepted


def exact_bounds(frame: pd.DataFrame) -> tuple[float, float, float, float, float, float]:
    return (
        float(frame["RA"].min()),
        float(frame["RA"].max()),
        float(frame["DEC"].min()),
        float(frame["DEC"].max()),
        float(frame["KEV"].min()),
        float(frame["KEV"].max()),
    )


def responsive_axis_bounds(
    frame: pd.DataFrame, energy_min: float, energy_max: float
) -> tuple[float, float, float, float, float, float]:
    """Use data-driven sky limits and the exact requested global energy limits."""
    bounds = exact_bounds(frame)
    low, high = sorted((float(energy_min), float(energy_max)))
    return bounds[0], bounds[1], bounds[2], bounds[3], low, high


def show_absolute_coordinates(
    frame: pd.DataFrame,
    paths: list[Path],
    energy_min: float,
    energy_max: float,
) -> None:
    # These are the actual scientific values.  There is deliberately no local
    # -10…10 offset transform anywhere in this standalone test.
    points = frame[["RA", "DEC", "KEV"]].to_numpy(dtype=np.float64, copy=True)
    cloud = pv.PolyData(points)
    cloud["Energy (keV)"] = points[:, 2]
    data_bounds = exact_bounds(frame)
    axis_bounds = responsive_axis_bounds(frame, energy_min, energy_max)

    plotter = pv.Plotter(window_size=(1280, 860))
    plotter.set_background("#eef2f7")
    plotter.add_points(
        cloud,
        scalars="Energy (keV)",
        cmap="turbo_r",
        point_size=3.0,
        opacity=0.80,
        render_points_as_spheres=False,
        scalar_bar_args={"title": "Energy (keV)", "fmt": "%.4f"},
    )
    plotter.show_bounds(
        mesh=cloud,
        bounds=axis_bounds,
        grid="front",
        location="outer",
        all_edges=True,
        xtitle="Right Ascension (deg)",
        ytitle="Declination (deg)",
        ztitle="Energy (keV)",
        fmt="%.4f",
        font_size=11,
    )
    plotter.add_text(
        "ABSOLUTE COORDINATES (no scene offset)\n"
        f"RA {axis_bounds[0]:.6f} to {axis_bounds[1]:.6f} deg\n"
        f"DEC {axis_bounds[2]:.6f} to {axis_bounds[3]:.6f} deg\n"
        f"Energy axis {axis_bounds[4]:.6f} to {axis_bounds[5]:.6f} keV\n"
        f"Event energy {data_bounds[4]:.6f} to {data_bounds[5]:.6f} keV\n"
        f"{len(frame):,} displayed events from {len(paths)} files",
        position="upper_left",
        font_size=10,
        color="#18212b",
    )
    plotter.view_isometric()
    plotter.reset_camera(bounds=axis_bounds)
    plotter.show(title="UDON3 absolute-coordinate 3D test")


def main() -> None:
    options = arguments()
    if options.energy_min > options.energy_max:
        options.energy_min, options.energy_max = options.energy_max, options.energy_min
    frame, paths = load_events(options)
    bounds = exact_bounds(frame)
    axis_bounds = responsive_axis_bounds(frame, options.energy_min, options.energy_max)
    print(f"Loaded {len(frame):,} displayed events from {len(paths)} cached files")
    print(f"RA:     {bounds[0]:.6f} to {bounds[1]:.6f} deg")
    print(f"DEC:    {bounds[2]:.6f} to {bounds[3]:.6f} deg")
    print(f"Energy: {bounds[4]:.6f} to {bounds[5]:.6f} keV")
    print(f"3D axis: {axis_bounds[4]:.6f} to {axis_bounds[5]:.6f} keV")
    print(f"First file: {paths[0]}")
    if not options.check_only:
        show_absolute_coordinates(frame, paths, options.energy_min, options.energy_max)


if __name__ == "__main__":
    main()
