import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxa_udon3.desktop.science_views import EnergySceneTransform, SkyViewport
from jaxa_udon3.desktop.top_image import scene_image_plane_geometry
from jaxa_udon3.desktop.viewer_3d_helpers import (
    _set_cube_axes,
    balanced_scene_transform,
    scientific_axis_bounds,
    sync_scene_guides,
)


class TopImageCoordinateTests(unittest.TestCase):
    def test_balanced_transform_fills_cube_but_keeps_physical_axis_labels(self):
        region = SimpleNamespace(
            center_ra_deg=350.8584,
            center_dec_deg=58.8113,
            radius_deg=10.0 / 60.0,
        )
        band = (1.86, 2.36)
        transform = balanced_scene_transform(
            region.center_ra_deg,
            region.center_dec_deg,
            region.radius_deg,
            band,
        )

        # A 20 arcmin sky box and the selected 0.5 keV band both occupy 20
        # display units, matching the balanced-volume approach in the JS view.
        self.assertFalse(transform.absolute_coordinates)
        self.assertAlmostEqual(transform.reference_kev, 2.11)
        self.assertAlmostEqual(transform.energy_scale, 40.0)
        np.testing.assert_allclose(transform.energy_to_scene(band), (-10.0, 10.0))
        axes = scientific_axis_bounds(region, transform, band)
        np.testing.assert_allclose(
            axes["scene"], (-10.0, 10.0, -10.0, 10.0, -10.0, 10.0)
        )
        self.assertEqual(axes["energy"], band)
        self.assertAlmostEqual(axes["ra"][0], SkyViewport.from_region(region).ra_max_deg)
        self.assertAlmostEqual(axes["ra"][1], SkyViewport.from_region(region).ra_min_deg)

    def test_depth_aspect_is_a_bounded_relative_multiplier(self):
        base = balanced_scene_transform(10.0, 20.0, 0.25, (4.1, 5.4), 1.0)
        double = balanced_scene_transform(10.0, 20.0, 0.25, (4.1, 5.4), 2.0)
        restored_legacy = balanced_scene_transform(10.0, 20.0, 0.25, (4.1, 5.4), 100.0)
        self.assertAlmostEqual(double.energy_scale, 2.0 * base.energy_scale)
        self.assertAlmostEqual(restored_legacy.energy_scale, 4.0 * base.energy_scale)

    def test_scene_plane_bounds_equal_the_2d_image_bounds_after_transform(self):
        product = SimpleNamespace(
            x_edges=np.asarray([287.60, 288.00]),
            y_edges=np.asarray([8.85, 9.35]),
            high_kev=6.7025,
        )
        transform = EnergySceneTransform(287.80, 9.10, 6.7000, 100.0)
        center, bounds = scene_image_plane_geometry(product, transform)
        cosine = np.cos(np.deg2rad(9.10))
        expected_x = sorted((-(287.60 - 287.80) * cosine * 60.0, -(288.00 - 287.80) * cosine * 60.0))
        expected_y = sorted(((8.85 - 9.10) * 60.0, (9.35 - 9.10) * 60.0))
        np.testing.assert_allclose(bounds, [*expected_x, *expected_y])
        self.assertLess(max(abs(value) for value in bounds), 20.0)
        self.assertAlmostEqual(center[2], 0.251, places=6)

    def test_3d_axis_labels_match_2d_sky_bounds_and_active_energy_filter(self):
        region = SimpleNamespace(center_ra_deg=287.80, center_dec_deg=9.10, radius_deg=0.25)
        transform = EnergySceneTransform(287.80, 9.10, 6.7000, 100.0)
        axes = scientific_axis_bounds(region, transform, (0.0, 30.0))
        image = SkyViewport.from_region(region)
        self.assertEqual(axes["ra"], (image.ra_max_deg, image.ra_min_deg))
        self.assertEqual(axes["dec"], image.y_range)
        self.assertEqual(axes["energy"], (0.0, 30.0))
        self.assertEqual(axes["scene"][:4], (-15.0, 15.0, -15.0, 15.0))
        narrow = scientific_axis_bounds(region, transform, (6.6975, 6.7025))
        self.assertEqual(narrow["energy"], (6.6975, 6.7025))
        np.testing.assert_allclose(narrow["scene"][4:], (-0.25, 0.25))

    def test_cube_axis_actor_uses_physical_labels_not_local_scene_offsets(self):
        region = SimpleNamespace(center_ra_deg=287.80, center_dec_deg=9.10, radius_deg=0.25)
        axis = pv.CubeAxesActor(pv.Camera())
        viewer = SimpleNamespace(
            _plotter=SimpleNamespace(renderer=SimpleNamespace(cube_axes_actor=axis)),
            _scene_transform=EnergySceneTransform(287.80, 9.10, 6.7000, 100.0),
        )

        _set_cube_axes(viewer, region, (6.6975, 6.7025), True, True)

        expected = scientific_axis_bounds(
            region, viewer._scene_transform, (6.6975, 6.7025)
        )
        self.assertEqual(axis.x_axis_range, expected["ra"])
        self.assertEqual(axis.y_axis_range, expected["dec"])
        self.assertEqual(axis.z_axis_range, expected["energy"])
        self.assertEqual(axis.x_labels[0], f"{expected['ra'][0]:.4f}")
        self.assertEqual(axis.x_labels[-1], f"{expected['ra'][1]:.4f}")
        self.assertEqual(axis.y_labels[0], f"{expected['dec'][0]:.4f}")
        self.assertEqual(axis.y_labels[-1], f"{expected['dec'][1]:.4f}")
        self.assertEqual(axis.z_labels[0], f"{expected['energy'][0]:.4f}")
        self.assertEqual(axis.z_labels[-1], f"{expected['energy'][1]:.4f}")
        np.testing.assert_allclose(
            axis.GetXAxesGridlinesProperty().GetColor(), (0.22, 0.30, 0.42)
        )
        np.testing.assert_allclose(
            axis.GetZAxesLinesProperty().GetColor(), (0.45, 0.60, 0.85)
        )

    def test_zero_grid_and_reference_guides_cannot_restore_zero_to_reference_bounds(self):
        class Plotter:
            def __init__(self):
                self.renderer = SimpleNamespace(
                    cube_axes_actor=pv.CubeAxesActor(pv.Camera()),
                    reset_camera_clipping_range=lambda: None,
                )
                self.camera_bounds = []

            def add_mesh(self, *_args, **_kwargs):
                return pv.Actor()

            def remove_actor(self, *_args, **_kwargs):
                pass

            def reset_camera(self, bounds=None):
                self.camera_bounds.append(tuple(bounds))

        region = SimpleNamespace(
            center_ra_deg=350.8584,
            center_dec_deg=58.8113,
            radius_deg=10.0 / 60.0,
        )
        plotter = Plotter()
        viewer = SimpleNamespace(
            available=True,
            _plotter=plotter,
            _scene_transform=EnergySceneTransform(
                350.8584, 58.8113, 6.7, 1.0, absolute_coordinates=True
            ),
            _energy_axis_band=(0.0, 6.7),
            _grid_actor=None,
            _slice_window_actor=None,
        )

        sync_scene_guides(viewer, region, (1.86, 2.36), True, True, True, True)

        self.assertEqual(viewer._energy_axis_band, (1.86, 2.36))
        self.assertEqual(plotter.renderer.cube_axes_actor.z_axis_range, (1.86, 2.36))
        self.assertEqual(plotter.camera_bounds[-1][4:], (1.86, 2.36))
        self.assertFalse(viewer._grid_actor.use_bounds)
        self.assertFalse(viewer._slice_window_actor.use_bounds)

        sync_scene_guides(viewer, region, (4.1, 5.4), True, True, True, True)
        self.assertEqual(viewer._energy_axis_band, (4.1, 5.4))
        self.assertEqual(plotter.renderer.cube_axes_actor.z_axis_range, (4.1, 5.4))
        self.assertEqual(plotter.camera_bounds[-1][4:], (4.1, 5.4))


if __name__ == "__main__":
    unittest.main()
