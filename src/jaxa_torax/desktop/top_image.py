from __future__ import annotations

import logging

import numpy as np
import pyvista as pv

from jaxa_torax.desktop.science_views import scalar_to_rgb
from jaxa_torax.desktop.viewer_3d_helpers import exclude_actor_from_bounds

LOGGER = logging.getLogger(__name__)


def scene_image_plane_geometry(product, transform):
    """Map a 2D RA/DEC image rectangle into the same scene coordinates as events."""
    x_edges = np.asarray(product.x_edges[[0, -1]], dtype=float)
    y_edges = np.asarray(product.y_edges[[0, -1]], dtype=float)
    if transform is None:
        x_scene, y_scene = x_edges, y_edges
        z_scene = float(product.high_kev or 0.0) + 1.0e-3
    elif transform.absolute_coordinates:
        x_scene, y_scene = x_edges, y_edges
        z_scene = float(product.high_kev or transform.reference_kev) + 1.0e-3
    else:
        cosine = max(abs(np.cos(np.deg2rad(transform.center_dec_deg))), 0.02)
        x_scene = -(x_edges - transform.center_ra_deg) * cosine * 60.0
        y_scene = (y_edges - transform.center_dec_deg) * 60.0
        z_scene = float(transform.energy_to_scene(product.high_kev or transform.reference_kev)) + 1.0e-3
    bounds = (float(x_scene.min()), float(x_scene.max()), float(y_scene.min()), float(y_scene.max()))
    return (0.5 * (bounds[0] + bounds[1]), 0.5 * (bounds[2] + bounds[3]), z_scene), bounds


def build_top_image(view, product, region, mode, opacity, palette, stretch, brightness, contrast):
    if mode == "off" or product is None:
        if view._top_image_actor is not None:
            view._set_actor_visible(view._top_image_actor, False)
        return
    values = scalar_to_rgb(product.values, palette, stretch, brightness, contrast)
    texture = pv.Texture(np.asarray(values, dtype=np.uint8))
    center, bounds = scene_image_plane_geometry(product, view._scene_transform)
    LOGGER.debug("3D image plane raw RA/DEC=%s/%s, scene bounds=%s", product.x_edges[[0, -1]], product.y_edges[[0, -1]], bounds)
    plane = pv.Plane(
        center=center,
        direction=(0, 0, 1), i_size=max(bounds[1] - bounds[0], 1e-4),
        j_size=max(bounds[3] - bounds[2], 1e-4),
    )
    if view._top_image_actor is not None:
        view._plotter.remove_actor(view._top_image_actor, render=False)
    view._top_image_actor = exclude_actor_from_bounds(view._plotter.add_mesh(
        plane, texture=texture, opacity=float(opacity), lighting=False,
        name="top-image", reset_camera=False,
    ))
