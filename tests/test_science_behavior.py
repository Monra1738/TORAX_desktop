import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxa_udon3.infrastructure import science as backend


class ScienceBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.db_path = self.root / "catalog.duckdb"
        self.original_product_dir = backend.PRODUCT_CACHE_DIR
        backend.PRODUCT_CACHE_DIR = self.root / "products"
        observation_dir = self.root / "xrism" / "synthetic"
        observation_dir.mkdir(parents=True)
        parquet_path = observation_dir / "synthetic_resolve_events.parquet"
        header_path = observation_dir / "synthetic_resolve_hdr.json"
        frame = pd.DataFrame(
            {
                "TIME": np.arange(10, dtype=float),
                "PI": np.arange(1000, 11000, 1000, dtype=np.int32),
                "X": np.asarray([50, 51, 49, 52, 48, 55, 45, 70, 30, 90], dtype=np.int16),
                "Y": np.asarray([50, 50, 50, 51, 49, 55, 45, 70, 30, 90], dtype=np.int16),
            }
        )
        frame.to_parquet(parquet_path, index=False)
        metadata = {
            "TCTYP_X": "RA---TAN",
            "TCTYP_Y": "DEC--TAN",
            "TCRPX_X": 50.0,
            "TCRPX_Y": 50.0,
            "TCRVL_X": 10.0,
            "TCRVL_Y": 20.0,
            "TCDLT_X": -0.01,
            "TCDLT_Y": 0.01,
            "TCUNI_X": "deg",
            "TCUNI_Y": "deg",
            "OBJECT": "SYNTHETIC",
        }
        header_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.record = backend.EventFile(
            "xrism", "resolve", "synthetic", parquet_path, header_path
        )

    def tearDown(self):
        backend.PRODUCT_CACHE_DIR = self.original_product_dir
        self.temporary.cleanup()

    def test_hitomi_hxi_pi_conversion_is_0p1_kev_per_channel(self):
        record = backend.EventFile(
            "hitomi", "hxi", "synthetic", self.root / "x.parquet", self.root / "x.json"
        )
        self.assertAlmostEqual(backend.pi_to_kev_factor(record), 0.1, places=12)

    def test_degree_and_sexagesimal_inputs_match(self):
        degrees, _ = backend.parse_sky_region(
            "degrees", 10.0, 20.0, 0.1, db_path=self.db_path
        )
        sexagesimal, _ = backend.parse_sky_region(
            "sexagesimal", "00:40:00", "+20:00:00", 0.1, db_path=self.db_path
        )
        self.assertAlmostEqual(degrees.center_ra_deg, sexagesimal.center_ra_deg, places=7)
        self.assertAlmostEqual(degrees.center_dec_deg, sexagesimal.center_dec_deg, places=7)

    def test_j2000_decimal_degree_input_is_preserved(self):
        region, _ = backend.parse_sky_region(
            "degrees", "225.592100", "-42.096900", 0.2, db_path=self.db_path
        )
        self.assertAlmostEqual(region.center_ra_deg, 225.592100, places=7)
        self.assertAlmostEqual(region.center_dec_deg, -42.096900, places=7)
        self.assertEqual(region.source, "degrees")

    def test_displayed_ra_dec_label_is_reused_as_coordinates(self):
        region, cache_hit = backend.parse_sky_region(
            "target", "", "", 0.2,
            target_name="RA 350.858400, DEC 58.811300",
            db_path=self.db_path,
            resolver=lambda _name: self.fail("coordinate label sent to Sesame"),
        )
        self.assertFalse(cache_hit)
        self.assertAlmostEqual(region.center_ra_deg, 350.8584, places=7)
        self.assertAlmostEqual(region.center_dec_deg, 58.8113, places=7)
        self.assertEqual(region.source, "degrees")

    def test_target_resolution_is_cached(self):
        calls = []

        def resolver(name):
            calls.append(name)
            return SkyCoord(10.0, 20.0, unit="deg")

        first, first_hit = backend.parse_sky_region(
            "target", "", "", 0.1, target_name="Test Target",
            db_path=self.db_path, resolver=resolver,
        )
        second, second_hit = backend.parse_sky_region(
            "target", "", "", 0.1, target_name="  test   target ",
            db_path=self.db_path, resolver=lambda _name: self.fail("resolver called twice"),
        )
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(calls, ["Test Target"])
        self.assertEqual(first.signature(), second.signature())

    def test_cas_a_aliases_share_a_canonical_label_and_cache_entry(self):
        calls = []

        def resolver(name):
            calls.append(name)
            return SkyCoord(350.8584, 58.8113, unit="deg")

        first, first_hit = backend.parse_sky_region(
            "target", "", "", 0.1, target_name="CAS A", db_path=self.db_path, resolver=resolver,
        )
        second, second_hit = backend.parse_sky_region(
            "target", "", "", 0.1, target_name="cas a", db_path=self.db_path,
            resolver=lambda _name: self.fail("resolver called twice"),
        )
        self.assertEqual((first.label, second.label), ("Cas A", "Cas A"))
        self.assertEqual(calls, ["Cas A"])
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)

    def test_cas_shorthand_resolves_as_cas_a(self):
        calls = []

        def resolver(name):
            calls.append(name)
            return SkyCoord(350.8584, 58.8113, unit="deg")

        region, cache_hit = backend.parse_sky_region(
            "target", "", "", 0.1, target_name="Cas", db_path=self.db_path,
            resolver=resolver,
        )

        self.assertFalse(cache_hit)
        self.assertEqual(calls, ["Cas A"])
        self.assertEqual(region.label, "Cas A")

    def test_catalog_object_search_ignores_case_and_separators(self):
        con = duckdb.connect(str(self.db_path))
        try:
            con.execute(
                """
                CREATE TABLE server_files (
                    mission TEXT, instrument TEXT, observation_id TEXT, object TEXT
                )
                """
            )
            con.execute(
                """
                INSERT INTO server_files VALUES
                    ('xrism', 'resolve', '1', 'CAS_A_C1O'),
                    ('hitomi', 'sxs', '2', 'Cas A'),
                    ('asca', 'gis', '3', 'Cassiopeia-A')
                """
            )
        finally:
            con.close()

        result = backend.search_server_catalog(object_text="cas a", db_path=self.db_path)

        self.assertEqual(result.observation_id.tolist(), ["2", "1"])

    def test_region_preview_filters_events_and_reuses_cache(self):
        region = backend.SkyRegion(10.0, 20.0, 0.06)
        first, _metadata, count, cache_hit = backend.read_region_preview(
            self.record, region, max_rows=3, db_path=self.db_path
        )
        second, _metadata, second_count, second_hit = backend.read_region_preview(
            self.record, region, max_rows=3, db_path=self.db_path
        )
        separations = backend.angular_separation_deg(
            first.RA, first.DEC, region.center_ra_deg, region.center_dec_deg
        )
        self.assertLessEqual(len(first), 3)
        self.assertGreater(count, 0)
        self.assertTrue(np.all(separations <= region.radius_deg + 1e-8))
        self.assertFalse(cache_hit)
        self.assertTrue(second_hit)
        self.assertEqual(count, second_count)
        pd.testing.assert_frame_equal(first, second)

    def test_rectangle_selection_supports_ra_wrap_and_wcs_preview(self):
        wrapped = backend.parse_sky_rectangle(350.0, 10.0, -5.0, 5.0)
        mask = backend.selection_contains(
            wrapped,
            np.asarray([355.0, 5.0, 180.0]),
            np.asarray([0.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(mask, np.asarray([True, True, False]))

        rectangle = backend.parse_sky_rectangle(9.96, 10.04, 19.96, 20.04)
        first, _metadata, count, cache_hit = backend.read_region_preview(
            self.record, rectangle, max_rows=10, db_path=self.db_path
        )
        second, _metadata, second_count, second_hit = backend.read_region_preview(
            self.record, rectangle, max_rows=10, db_path=self.db_path
        )
        self.assertGreater(count, 0)
        self.assertTrue(
            np.all(backend.selection_contains(rectangle, first.RA, first.DEC))
        )
        self.assertFalse(cache_hit)
        self.assertTrue(second_hit)
        self.assertEqual(count, second_count)
        pd.testing.assert_frame_equal(first, second)

    def test_exact_rectangle_image_uses_fixed_bounds(self):
        rectangle = backend.parse_sky_rectangle(9.95, 10.05, 19.95, 20.05)
        image = backend.exact_energy_image(
            [self.record], 0.4, 5.5, bins=24, db_path=self.db_path, region=rectangle
        )
        self.assertEqual(image["hist"].shape, (24, 24))
        self.assertAlmostEqual(image["x_edges"][0], 9.95)
        self.assertAlmostEqual(image["x_edges"][-1], 10.05)
        self.assertAlmostEqual(image["y_edges"][0], 19.95)
        self.assertAlmostEqual(image["y_edges"][-1], 20.05)

    def test_exact_region_and_rgb_images_share_grid(self):
        region = backend.SkyRegion(10.0, 20.0, 0.15)
        image = backend.exact_energy_image(
            [self.record], 0.4, 5.5, bins=32, db_path=self.db_path, region=region
        )
        config = backend.RGBBandConfig(0.75, 0.5, 1.50, 0.5, 2.50, 0.5)
        rgb = backend.exact_rgb_image(
            [self.record], config, bins=32, db_path=self.db_path, region=region
        )
        self.assertEqual(image["hist"].shape, (32, 32))
        self.assertEqual(rgb["channels"].shape, (32, 32, 3))
        self.assertEqual(len(rgb["event_counts"]), 3)
        self.assertTrue(np.all(np.isfinite(rgb["channels"])))

    def test_exact_all_events_image_counts_every_spatially_matching_event(self):
        region = backend.SkyRegion(10.0, 20.0, 0.15)
        all_events = backend.exact_all_events_image(
            [self.record], bins=32, db_path=self.db_path, region=region
        )
        wide_band = backend.exact_energy_image(
            [self.record], 0.0, 1000.0, bins=32, db_path=self.db_path, region=region
        )
        self.assertTrue(all_events["exact"])
        self.assertEqual(all_events["event_count"], int(all_events["hist"].sum()))
        self.assertEqual(all_events["event_count"], wide_band["event_count"])
        np.testing.assert_array_equal(all_events["hist"], wide_band["hist"])

    def test_rgb_bands_allow_independently_positioned_channels(self):
        config = backend.RGBBandConfig(6.4, 0.4, 1.85, 0.2, 2.44, 0.2)
        self.assertEqual(len(config.bands()), 3)


if __name__ == "__main__":
    unittest.main()
