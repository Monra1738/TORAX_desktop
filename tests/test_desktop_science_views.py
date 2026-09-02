import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxa_torax.desktop import science_views as views
from jaxa_torax.desktop.viewer_3d import ThreeDView


class DesktopScienceViewTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "RA": [359.9, 0.1, 10.0, 10.1, 10.2],
                "DEC": [0.0, 0.0, 20.0, 20.1, 20.2],
                "KEV": [0.5, 2.0, 4.0, 6.0, 10.0],
                "MISSION": ["asca", "suzaku", "hitomi", "xrism", "xrism"],
            }
        )

    def test_energy_gradient_runs_red_low_to_blue_high(self):
        colors = views.energy_gradient_colors(np.asarray([0.3, 12.0]))
        self.assertGreater(colors[0, 0], colors[0, 2])
        self.assertGreater(colors[1, 2], colors[1, 0])

    def test_rectangle_filter_handles_ra_zero_wrap(self):
        selected = views.filter_rectangle(self.frame, 350.0, 5.0, -1.0, 1.0)
        self.assertEqual(len(selected), 2)

    def test_spectrum_has_independent_smoothing(self):
        raw = views.spectrum(self.frame, bins=32, low=0.3, high=12.0)
        smooth = views.spectrum(
            self.frame, bins=32, low=0.3, high=12.0, smoothing_sigma_bins=1.5
        )
        self.assertEqual(len(raw.counts), 32)
        self.assertEqual(len(smooth.counts), 32)
        np.testing.assert_array_equal(raw.counts, smooth.counts)
        self.assertIsNone(raw.smoothed_counts)
        self.assertIsNotNone(smooth.smoothed_counts)
        self.assertFalse(np.array_equal(smooth.counts, smooth.smoothed_counts))
        self.assertAlmostEqual(float(smooth.counts.sum()), len(self.frame))

    def test_rgb_product_uses_shared_grid(self):
        class Region:
            center_ra_deg = 10.0
            center_dec_deg = 20.0
            radius_deg = 1.0

        rgb, x_edges, y_edges, counts = views.rgb_image(
            self.frame,
            [(0.3, 1.0), (1.0, 5.0), (5.0, 12.0)],
            bins=24,
            center_ra=10.0,
            region=Region(),
        )
        self.assertEqual(rgb.shape, (24, 24, 3))
        self.assertEqual(len(x_edges), 25)
        self.assertEqual(len(y_edges), 25)
        self.assertEqual(len(counts), 3)

    def test_rgb_event_colors_follow_independent_channel_controls(self):
        colors = views.rgb_event_colors(
            np.asarray([1.65, 1.85, 2.435]),
            centers=(1.65, 1.85, 2.435),
            widths=(0.20, 0.20, 0.17),
        )
        self.assertGreater(colors[0, 0], colors[0, 1])
        self.assertGreater(colors[1, 1], colors[1, 2])
        self.assertGreater(colors[2, 2], colors[2, 0])

    def test_voxel_histogram_is_three_dimensional(self):
        product = views.voxel_histogram(
            self.frame.loc[self.frame.RA.between(9.0, 11.0)],
            10.0,
            20.0,
            spatial_voxel_arcmin=0.5,
            energy_voxel_kev=0.5,
        )
        self.assertIsNotNone(product)
        histogram, edges = product
        self.assertEqual(histogram.ndim, 3)
        self.assertEqual(len(edges), 3)

    def test_3d_coordinates_are_absolute_ra_dec_degrees(self):
        frame = pd.DataFrame({
            "RA": [359.9, 0.1], "DEC": [-42.0, -41.9], "KEV": [1.0, 6.0]
        })
        points = ThreeDView._coordinates(frame, center_ra=359.95, center_dec=-42.0)
        np.testing.assert_allclose(points[:, 0], [359.9, 360.1])
        np.testing.assert_allclose(points[:, 1], [-42.0, -41.9])
        np.testing.assert_allclose(points[:, 2], [1.0, 6.0])

    def test_energy_scene_transform_centers_and_exaggerates_energy_only(self):
        transform = views.EnergySceneTransform(10.0, 20.0, 6.700, 100.0)
        frame = pd.DataFrame({"RA": [10.0, 10.0], "DEC": [20.0, 20.0], "KEV": [6.6975, 6.7025]})
        points = transform.coordinates(frame)
        np.testing.assert_allclose(points[:, :2], 0.0)
        np.testing.assert_allclose(points[:, 2], [-0.25, 0.25])
        np.testing.assert_allclose(transform.energy_to_scene([6.700]), [0.0])

    def test_absolute_scene_transform_matches_image_coordinates(self):
        transform = views.EnergySceneTransform(
            350.8584, 58.8113, 6.700, 100.0, absolute_coordinates=True
        )
        frame = pd.DataFrame({
            "RA": [350.60, 351.10], "DEC": [58.65, 58.95], "KEV": [2.10, 5.89]
        })
        points = transform.coordinates(frame)
        np.testing.assert_allclose(points, frame[["RA", "DEC", "KEV"]].to_numpy())
        np.testing.assert_allclose(transform.energy_to_scene([2.10, 5.89]), [2.10, 5.89])

    def test_local_spectrum_points_preserve_spatial_energy_variation(self):
        points = views.local_spectrum_points(
            self.frame, center_ra=10.0, center_dec=20.0, spatial_bin_arcmin=8.0
        )
        self.assertGreaterEqual(len(points), 2)
        self.assertTrue({"RA", "DEC", "COUNT", "MEAN_KEV", "ENERGY_STD"} <= set(points))
        self.assertTrue(np.all(points["COUNT"].to_numpy() >= 1))
        self.assertGreater(points["MEAN_KEV"].max(), points["MEAN_KEV"].min())

    def test_scalar_palettes_keep_zero_count_pixels_black(self):
        counts = np.asarray([[0.0, 1.0, 5.0], [0.0, 25.0, 100.0]])
        for palette in views.SCALAR_PALETTES:
            with self.subTest(palette=palette):
                rgb = views.scalar_to_rgb(counts, palette=palette)
                self.assertTrue(np.array_equal(rgb[counts == 0], np.zeros((2, 3), dtype=np.uint8)))
                self.assertTrue(np.any(rgb[counts > 0] > 0))

    def test_no_colormap_is_neutral_grayscale(self):
        counts = np.asarray([[0.0, 1.0, 100.0]])
        rgb = views.scalar_to_rgb(counts, palette="none", stretch="linear")
        self.assertTrue(np.array_equal(rgb[0, 0], np.zeros(3, dtype=np.uint8)))
        self.assertTrue(np.array_equal(rgb[0, 1], np.repeat(rgb[0, 1, 0], 3)))
        self.assertTrue(np.array_equal(rgb[0, 2], np.repeat(rgb[0, 2, 0], 3)))
        self.assertLess(int(rgb[0, 1, 0]), int(rgb[0, 2, 0]))

    def test_scalar_display_is_finite_and_clipped_for_sparse_extreme_images(self):
        cases = (
            np.zeros((3, 3)),
            np.asarray([[0.0, 1.0], [0.0, 0.0]]),
            np.asarray([[0.0, 1.0e-12], [1.0, 1.0e30]]),
        )
        for stretch in views.SCALAR_STRETCHES:
            for counts in cases:
                with self.subTest(stretch=stretch, counts=counts.tolist()):
                    display = views.scalar_display_values(counts, stretch, 3.0, 3.0)
                    self.assertTrue(np.all(np.isfinite(display)))
                    self.assertTrue(np.all((display >= 0.0) & (display <= 1.0)))
                    self.assertTrue(np.all(display[counts == 0] == 0.0))

    def test_scalar_display_is_monotonic_before_palette_lookup(self):
        counts = np.asarray([0.0, 1.0, 2.0, 10.0, 100.0])
        for stretch in views.SCALAR_STRETCHES:
            with self.subTest(stretch=stretch):
                display = views.scalar_display_values(counts, stretch, 1.0, 1.0)
                self.assertTrue(np.all(np.diff(display) >= 0.0))

    def test_scalar_display_is_independent_of_energy_histogram_cache(self):
        frame = self.frame.copy()
        original = views.energy_image
        calls = 0

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        # Palette/display operations work on the returned raw product and
        # therefore cannot call the histogram generator again.
        views.energy_image = counted
        try:
            product = views.energy_image(frame, 1.0, 8.0, bins=24)
            for palette in views.SCALAR_PALETTES:
                for stretch in views.SCALAR_STRETCHES:
                    views.scalar_to_rgb(product.values, palette, stretch, 1.2, 0.9)
            self.assertEqual(calls, 1)
        finally:
            views.energy_image = original

    def test_four_slice_display_changes_do_not_recompute_other_slice_histograms(self):
        original = views.energy_image
        calls = 0

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        views.energy_image = counted
        try:
            bands = ((0.3, 1.0), (1.0, 3.0), (3.0, 7.0), (7.0, 12.0))
            products = [views.energy_image(self.frame, *band, bins=24) for band in bands]
            self.assertEqual(calls, 4)
            for palette in ("gray", "inferno", "viridis", "rainbow"):
                for product in products:
                    views.scalar_to_rgb(product.values, palette=palette, stretch="log")
            self.assertEqual(calls, 4)
            products[1] = views.energy_image(self.frame, 1.2, 3.2, bins=24)
            self.assertEqual(calls, 5)
        finally:
            views.energy_image = original


if __name__ == "__main__":
    unittest.main()
