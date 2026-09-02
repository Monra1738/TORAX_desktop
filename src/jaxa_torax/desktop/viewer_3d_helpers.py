from __future__ import annotations

import numpy as np

from jaxa_torax.desktop.science_views import EnergySceneTransform, SkyViewport
from jaxa_torax.desktop.theme import PLOT_TEXT


def exclude_actor_from_bounds(actor):
    """Keep a visual guide from expanding the scientific data axes."""
    try:
        actor.use_bounds = False
    except Exception:
        try:
            actor.SetUseBounds(False)
        except Exception:
            pass
    return actor


def balanced_scene_transform(
    center_ra_deg,
    center_dec_deg,
    radius_deg,
    energy_band,
    depth_aspect=1.0,
):
    """Balance the displayed energy depth against the sky footprint.

    VTK receives compact local display coordinates, like the web viewer, while
    the cube axes continue to show the exact physical RA, DEC and keV ranges.
    A depth aspect of one makes the selected energy span as deep as the sky
    region is wide in arcminutes.
    """
    low, high = sorted(map(float, energy_band))
    sky_span_arcmin = max(2.0 * abs(float(radius_deg)) * 60.0, 1e-3)
    energy_span_kev = max(high - low, 1e-9)
    aspect = float(np.clip(float(depth_aspect), 0.25, 4.0))
    energy_scale = sky_span_arcmin * aspect / energy_span_kev
    return EnergySceneTransform(
        center_ra_deg=float(center_ra_deg),
        center_dec_deg=float(center_dec_deg),
        reference_kev=0.5 * (low + high),
        energy_scale=energy_scale,
        absolute_coordinates=False,
    )


def transformed_voxel_geometry(hist, edges, transform):
    energy_centers = 0.5 * (edges[2][:-1] + edges[2][1:])
    if transform is None:
        return hist, tuple(float(np.mean(np.diff(axis))) for axis in edges), tuple(float(axis[0]) for axis in edges), energy_centers, energy_centers
    if transform.absolute_coordinates:
        x = 0.5 * (edges[0][:-1] + edges[0][1:])
        y = 0.5 * (edges[1][:-1] + edges[1][1:])
    else:
        cosine = max(abs(np.cos(np.deg2rad(transform.center_dec_deg))), 0.02)
        hist = hist[::-1, :, :]
        x = -(0.5 * (edges[0][:-1] + edges[0][1:]) - transform.center_ra_deg) * cosine * 60.0
        y = (0.5 * (edges[1][:-1] + edges[1][1:]) - transform.center_dec_deg) * 60.0
    z = transform.energy_to_scene(energy_centers)
    spacing = tuple(abs(float(np.mean(np.diff(axis)))) if len(axis) > 1 else 1.0 for axis in (x, y, z))
    origin = tuple(float(np.min(axis) - step / 2) for axis, step in zip((x, y, z), spacing))
    return hist, spacing, origin, z, energy_centers

def reference_plane(viewer, region, reference_kev, visible):
    if viewer._reference_actor is not None:
        viewer._plotter.remove_actor(viewer._reference_actor, render=False)
        viewer._reference_actor = None
    if not visible:
        return
    import pyvista as pv
    transform = viewer._scene_transform
    z = 0.0 if transform is None else float(transform.energy_to_scene(reference_kev))
    if transform is not None and transform.absolute_coordinates:
        viewport = SkyViewport.from_region(region)
        center = (transform.center_ra_deg, transform.center_dec_deg, z)
        i_size = max(viewport.ra_max_deg - viewport.ra_min_deg, 1e-4)
        j_size = max(viewport.dec_max_deg - viewport.dec_min_deg, 1e-4)
    else:
        center = (0.0, 0.0, z)
        i_size = max(2 * float(region.radius_deg) * 60.0, 1e-4)
        j_size = i_size
    plane = pv.Plane(center=center, direction=(0, 0, 1), i_size=i_size, j_size=j_size)
    viewer._reference_actor = exclude_actor_from_bounds(viewer._plotter.add_mesh(
        plane, color="#f59e0b", opacity=0.14, lighting=False,
        name="energy-reference-plane", reset_camera=False))


def _remove_actor(viewer, name):
    actor = getattr(viewer, name, None)
    if actor is not None:
        viewer._plotter.remove_actor(actor, render=False)
    setattr(viewer, name, None)


def scientific_axis_bounds(region, transform, energy_band):
    """Return scene bounds plus physical labels matching the 2D sky image and active energy band."""
    viewport = SkyViewport.from_region(region)
    low, high = sorted(map(float, energy_band))
    if transform.absolute_coordinates:
        scene = (
            viewport.ra_min_deg, viewport.ra_max_deg,
            viewport.dec_min_deg, viewport.dec_max_deg,
            low, high,
        )
    else:
        half = float(region.radius_deg) * 60.0
        z_low, z_high = sorted(map(float, transform.energy_to_scene((low, high))))
        scene = (-half, half, -half, half, z_low, z_high)
    return {
        "scene": scene,
        "ra": (viewport.ra_max_deg, viewport.ra_min_deg),
        "dec": (viewport.dec_min_deg, viewport.dec_max_deg),
        "energy": (low, high),
    }


def _set_cube_axes(viewer, region, energy_band, show_axes, show_values):
    ranges = scientific_axis_bounds(region, viewer._scene_transform, energy_band)
    current_band = tuple(ranges["energy"])
    previous_band = getattr(viewer, "_energy_axis_band", None)
    viewer._energy_axis_band = current_band
    actor = None
    if previous_band != current_band:
        try:
            viewer._plotter.remove_bounds_axes()
            actor = viewer._plotter.show_bounds(
                bounds=ranges["scene"],
                axes_ranges=(*ranges["ra"], *ranges["dec"], *ranges["energy"]),
                grid="front",
                location="outer",
                color=PLOT_TEXT,
                xtitle="Right Ascension (deg)",
                ytitle="Declination (deg)",
                ztitle="Energy (keV)",
                n_xlabels=3,
                n_ylabels=3,
                n_zlabels=3,
                show_xlabels=bool(show_values),
                show_ylabels=bool(show_values),
                show_zlabels=bool(show_values),
                font_size=9,
                all_edges=True,
                render=False,
            )
        except Exception:
            actor = None
    try:
        actor = actor or viewer._plotter.renderer.cube_axes_actor
        # Use PyVista's properties rather than the raw VTK setters.  PyVista
        # stores explicit tick-label strings and only regenerates them from
        # these property setters; raw SetXAxisRange calls leave the old local
        # scene labels (for example -10…10 arcmin) visible.
        for name, value in (
            ("bounds", ranges["scene"]),
            ("x_label_format", "%.4f"),
            ("y_label_format", "%.4f"),
            ("z_label_format", "%.4f" if ranges["energy"][1] - ranges["energy"][0] < 1 else "%.2f"),
            ("x_axis_range", ranges["ra"]),
            ("y_axis_range", ranges["dec"]),
            ("z_axis_range", ranges["energy"]),
            # Three labels per side remain legible for the narrow desktop
            # viewport while preserving the exact physical endpoint values.
            ("n_xlabels", 3),
            ("n_ylabels", 3),
            ("n_zlabels", 3),
            ("x_label_visibility", bool(show_values)),
            ("y_label_visibility", bool(show_values)),
            ("z_label_visibility", bool(show_values)),
        ):
            try:
                setattr(actor, name, value)
            except Exception:
                pass
        # Keep the scientific labels bright, but push the cage and grid behind
        # the data as in the vtk.js viewer.  A white grid overwhelms sparse
        # translucent voxels, especially in the split layout.
        for getter in (
            "GetXAxesGridlinesProperty",
            "GetYAxesGridlinesProperty",
            "GetZAxesGridlinesProperty",
        ):
            try:
                prop = getattr(actor, getter)()
                prop.SetColor(0.22, 0.30, 0.42)
                prop.SetOpacity(0.52)
            except Exception:
                pass
        for getter in (
            "GetXAxesLinesProperty",
            "GetYAxesLinesProperty",
            "GetZAxesLinesProperty",
        ):
            try:
                prop = getattr(actor, getter)()
                prop.SetColor(0.45, 0.60, 0.85)
                prop.SetOpacity(0.82)
            except Exception:
                pass
        actor.Modified()
        actor.SetVisibility(bool(show_axes))
    except Exception:
        pass
    if previous_band != current_band:
        try:
            viewer._plotter.reset_camera(bounds=ranges["scene"])
        except Exception:
            pass


def sync_scene_guides(viewer, region, energy_band, show_grid, show_window, show_axes, show_values):
    """Draw low-cost XY and active-energy guides without changing voxel data."""
    if not viewer.available or region is None:
        return
    import pyvista as pv

    transform = viewer._scene_transform
    if transform is not None and transform.absolute_coordinates:
        viewport = SkyViewport.from_region(region)
        center_x, center_y = transform.center_ra_deg, transform.center_dec_deg
        half_x = max(viewport.ra_max_deg - viewport.ra_min_deg, 1e-4) / 2.0
        half_y = max(viewport.dec_max_deg - viewport.dec_min_deg, 1e-4) / 2.0
    else:
        center_x = center_y = 0.0
        half_x = half_y = max(float(region.radius_deg) * 60.0, 1e-4)
    low, high = sorted(map(float, energy_band))
    grid_z = low if transform is None else float(transform.energy_to_scene(low))
    _remove_actor(viewer, "_backdrop_actor")
    _remove_actor(viewer, "_grid_actor")
    if show_grid:
        plane = pv.Plane(
            center=(center_x, center_y, grid_z), direction=(0, 0, 1),
            i_size=2 * half_x, j_size=2 * half_y,
            i_resolution=8, j_resolution=8,
        )
        viewer._backdrop_actor = exclude_actor_from_bounds(viewer._plotter.add_mesh(
            plane, color="#0b1525", opacity=0.46, lighting=False,
            name="xy-backdrop", reset_camera=False,
        ))
        viewer._grid_actor = exclude_actor_from_bounds(viewer._plotter.add_mesh(
            plane, style="wireframe", color="#42526a", opacity=0.42,
            lighting=False, name="xy-reference-grid", reset_camera=False,
        ))
    _remove_actor(viewer, "_slice_window_actor")
    if show_window and energy_band is not None:
        z_low = 0.0 if transform is None else float(transform.energy_to_scene(low))
        z_high = 0.0 if transform is None else float(transform.energy_to_scene(high))
        bounds = (center_x - half_x, center_x + half_x, center_y - half_y, center_y + half_y, min(z_low, z_high), max(z_low, z_high))
        viewer._slice_window_actor = exclude_actor_from_bounds(viewer._plotter.add_mesh(
            pv.Box(bounds=bounds), style="wireframe", color="#7dd3fc", opacity=0.58,
            line_width=1.1, lighting=False, name="active-energy-window", reset_camera=False,
        ))
    # Apply the physical ranges last. Adding a guide actor can otherwise make
    # VTK regenerate the cube axes from its geometry (0 keV grid / 6.7 keV
    # reference plane), replacing the selected slice range just before render.
    _set_cube_axes(viewer, region, energy_band, show_axes, show_values)
    try:
        viewer._plotter.renderer.reset_camera_clipping_range()
    except Exception:
        pass


def apply_camera_preset(viewer, preset):
    if not viewer.available:
        return
    method = {"top": "view_xy", "east": "view_yz", "north": "view_xz"}.get(preset, "view_isometric")
    try:
        getattr(viewer.plotter, method)()
        viewer.plotter.render()
    except Exception:
        pass


def set_voxel_actor_appearance(actor, opacity, show_edges):
    try:
        prop = actor.GetProperty()
        prop.SetOpacity(float(opacity))
        prop.SetEdgeVisibility(bool(show_edges))
    except Exception:
        pass
