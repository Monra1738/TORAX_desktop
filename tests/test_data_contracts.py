"""Offline data-contract and scientific-product regression tests."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from jaxa_udon3.desktop.science_views import energy_image, rgb_image, spectrum, voxel_histogram
from jaxa_udon3.domain import EventFile, SkyRegion
from jaxa_udon3.infrastructure import cache_repository, event_sources, science


class DataContractTests(unittest.TestCase):
    def _record(self, root: Path, mission="xrism", instrument="resolve", obsid="000001"):
        directory = root / mission / f"{obsid}-{instrument}"
        directory.mkdir(parents=True, exist_ok=True)
        parquet = directory / f"{obsid}_{instrument}_events.parquet"
        header = directory / f"{obsid}_{instrument}_hdr.json"
        frame = pd.DataFrame({
            "TIME": np.arange(8, dtype=float),
            "PI": np.asarray([100, 200, 400, 800, 1200, 2400, 4800, 9600], dtype=np.int32),
            "X": np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int16),
            "Y": np.asarray([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int16),
        })
        frame.to_parquet(parquet, index=False)
        header.write_text(
            '{"TCTYP_X":"RA---TAN","TCTYP_Y":"DEC--TAN",'
            '"TCRPX_X":1,"TCRPX_Y":1,"TCRVL_X":10,"TCRVL_Y":20,'
            '"TCDLT_X":-0.01,"TCDLT_Y":0.01}',
            encoding="utf-8",
        )
        return EventFile(mission, instrument, obsid, parquet, header)

    def test_cache_reads_share_duckdb_configuration_with_active_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "cache.duckdb"
            science.ensure_storage_schema(db_path)
            connection = science.duckdb.connect(str(db_path))
            try:
                item = self._record(Path(directory), "asca", "gis", "writer-open")
                self.assertIsNone(
                    cache_repository.cached_server_header(item, db_path=db_path)
                )
            finally:
                connection.close()

    def test_five_hundred_concurrent_metadata_cycles_share_one_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "cache.duckdb"
            metadata = {
                "TCTYP_X": "RA---TAN",
                "TCTYP_Y": "DEC--TAN",
                "TCRPX_X": 1,
                "TCRPX_Y": 1,
                "TCRVL_X": 10,
                "TCRVL_Y": 20,
                "TCDLT_X": -0.01,
                "TCDLT_Y": 0.01,
            }
            records = [
                EventFile(
                    "asca",
                    "gis",
                    f"stress-{index:03d}",
                    root / f"stress-{index:03d}.parquet",
                    root / f"stress-{index:03d}.json",
                )
                for index in range(500)
            ]

            def cycle(item):
                cache_repository.store_server_header(item, metadata, db_path=db_path)
                return cache_repository.cached_server_header(item, db_path=db_path)

            with ThreadPoolExecutor(max_workers=3) as pool:
                results = list(pool.map(cycle, records))
            self.assertEqual(len(results), 500)
            self.assertTrue(all(item["TCTYP_X"] == "RA---TAN" for item in results))

    def test_discovery_requires_matching_header_and_preserves_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = self._record(root, "asca", "gis", "57021000")
            records = event_sources.discover_event_files(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(science.record_key(records[0]), "asca/gis/57021000")
            record.header_path.unlink()
            self.assertEqual(event_sources.discover_event_files(root), [])
            self.assertTrue(record.parquet_path.exists())

    def test_local_event_read_validates_required_columns_and_projects_sky(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self._record(Path(tmp))
            frame, metadata = event_sources.read_events_with_sky(record)
            self.assertEqual(len(frame), 8)
            self.assertTrue({"RA", "DEC", "MISSION", "INSTRUMENT", "OBSERVATION_ID"} <= set(frame))
            self.assertTrue(np.all(np.isfinite(frame[["RA", "DEC"]].to_numpy())))
            self.assertEqual(metadata["TCTYP_X"], "RA---TAN")

    def test_preview_calibration_is_record_specific_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for instrument, expected in (
                ("resolve", 0.0005),
                ("xtend", 0.006),
                ("gis", 1 / 84.9),
            ):
                record = self._record(
                    root, "xrism" if instrument != "gis" else "asca", instrument, instrument
                )
                self.assertAlmostEqual(science.pi_to_kev_factor(record), expected)
                frame, total = science.read_preview_source(record, max_rows=3)
                self.assertEqual(total, 8)
                self.assertLessEqual(len(frame), 3)
                self.assertTrue(set(science.REQUIRED_COLUMNS) <= set(frame))

    def test_asca_gis_factor_detects_256_and_1024_channel_parquets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eight_bit = self._record(root, "asca", "gis", "eight-bit")
            frame = pd.read_parquet(eight_bit.parquet_path)
            frame["PI"] = np.asarray([3, 12, 31, 64, 95, 127, 191, 255], dtype=np.int32)
            frame.to_parquet(eight_bit.parquet_path, index=False)
            ten_bit = self._record(root, "asca", "gis", "ten-bit")
            frame = pd.read_parquet(ten_bit.parquet_path)
            frame["PI"] = np.asarray([17, 64, 128, 255, 384, 512, 768, 1023], dtype=np.int32)
            frame.to_parquet(ten_bit.parquet_path, index=False)
            self.assertAlmostEqual(science.pi_to_kev_factor(eight_bit), 4.0 / 84.9)
            self.assertAlmostEqual(science.pi_to_kev_factor(ten_bit), 1.0 / 84.9)

    def test_negative_pi_sentinels_are_filtered_before_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = self._record(root)
            frame = pd.read_parquet(record.parquet_path)
            frame.loc[0, "PI"] = -1000
            frame.to_parquet(record.parquet_path, index=False)
            preview, total = science.read_preview_source(record, max_rows=100)
            self.assertEqual(total, 7)
            self.assertTrue(np.all(preview["PI"].to_numpy() >= 0))

    def test_products_have_consistent_grid_shapes_and_finite_values(self):
        frame = pd.DataFrame({
            "RA": [9.9, 10.0, 10.1, 10.2],
            "DEC": [19.9, 20.0, 20.1, 20.2],
            "KEV": [1.0, 2.0, 4.0, 6.0],
        })
        image = energy_image(frame, 1.0, 6.0, bins=16)
        rgb, x_edges, y_edges, counts = rgb_image(
            frame, [(1.0, 2.0), (2.0, 4.0), (4.0, 6.0)],
            bins=16, center_ra=10.0, region=SkyRegion(10.0, 20.0, 1.0),
        )
        spec = spectrum(frame, bins=12, low=1.0, high=6.0)
        voxels = voxel_histogram(frame, 10.0, 20.0, 1.0, 1.0)
        self.assertEqual(image.values.shape, (16, 16))
        self.assertEqual(rgb.shape, (16, 16, 3))
        self.assertEqual((len(x_edges), len(y_edges)), (17, 17))
        self.assertEqual(len(counts), 3)
        self.assertEqual(len(spec.counts), 16)  # product enforces a minimum of 16 bins
        self.assertTrue(np.all(np.isfinite(image.values)))
        self.assertIsNotNone(voxels)
        self.assertEqual(voxels[0].ndim, 3)

    def test_invalid_science_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            science.parse_sky_region("degrees", "10", "91", 0.1)
        with self.assertRaises(ValueError):
            science.parse_sky_rectangle("10", "10", "0", "1")
        with self.assertRaises(ValueError):
            science.RGBBandConfig(1.0, 0.0, 2.0, 1.0, 3.0, 1.0).bands()


if __name__ == "__main__":
    unittest.main()
