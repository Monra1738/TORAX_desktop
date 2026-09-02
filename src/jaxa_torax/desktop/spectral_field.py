from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None
from jaxa_torax.desktop.science_views import energy_gradient_colors, local_spectrum_points


def build_spectral_field(plotter, observations, visible_keys, center_ra, center_dec, actors, coordinates, set_visible):
    if pv is None:
        return
    current = set()
    for obs in observations:
        key = f"{obs.record.mission}/{obs.record.instrument}/{obs.record.observation_id}"
        if key not in visible_keys:
            continue
        summary = local_spectrum_points(obs.frame, center_ra, center_dec, 1.0, 20_000)
        if summary.empty:
            continue
        current.add(key)
        points = coordinates(summary.rename(columns={"MEAN_KEV": "KEV"}), center_ra, center_dec)
        cloud = pv.PolyData(points)
        cloud["RGB"] = energy_gradient_colors(summary["MEAN_KEV"].to_numpy(float))
        old = actors.get(key)
        if old is not None:
            plotter.remove_actor(old, render=False)
        actors[key] = plotter.add_points(
            cloud, scalars="RGB", rgb=True, point_size=float(
                np.clip(3.0 + 2.5 * np.log1p(summary["COUNT"].to_numpy(float)), 3.0, 16.0).mean()
            ), render_points_as_spheres=True, name=f"spectral-field::{key}", reset_camera=False
        )
        set_visible(actors[key], True)
    for key in set(actors) - current:
        plotter.remove_actor(actors.pop(key), render=False)
