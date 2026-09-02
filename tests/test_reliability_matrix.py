"""Independent regression scenarios for numerical and large-workspace safety."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jaxa_udon3.desktop.science_views import filter_rectangle, rgb_event_colors
from jaxa_udon3.desktop.viewer_3d import ThreeDView
from jaxa_udon3.domain import EventFile
from jaxa_udon3.infrastructure import science as backend


class _Observation:
    def __init__(self, index, rows):
        self.record = type("Record", (), {"mission": "xrism", "instrument": "resolve", "observation_id": str(index)})()
        self.frame = pd.DataFrame({"KEV": np.zeros(rows)})


class ReliabilityMatrixTests(unittest.TestCase):
    def test_workspace_size_matrix(self):
        with tempfile.TemporaryDirectory() as root:
            db_path = Path(root) / "matrix.duckdb"
            for count in (1, 2, 5, 10, 25, 50, 75, 100, 125, 143):
                records = [
                    {
                        "record_key": f"asca/gis/{index}", "visible": index % 3 != 0,
                        "record": {
                            "mission": "asca", "instrument": "gis", "observation_id": str(index),
                            "parquet_path": f"/cache/{index}.parquet", "header_path": f"/cache/{index}.json",
                        },
                    }
                    for index in range(count)
                ]
                snapshot = {
                    "workspace_id": f"matrix-{count}", "target_name": "Cas A",
                    "region": {"center_ra_deg": 350.8584, "center_dec_deg": 58.8113, "radius_deg": 0.2},
                    "state": {"energy_band": [4.0, 6.0]}, "observations": records,
                    "slices": [], "rois": {},
                }
                backend.save_workspace(snapshot, db_path)
                restored = backend.load_workspace(snapshot["workspace_id"], db_path)
                self.assertEqual(len(restored["observations"]), count)
                self.assertEqual(len({item["record_key"] for item in restored["observations"]}), count)


def _add_pi_factor_tests():
    values = {
        ("xrism", "resolve"): 0.0005, ("xrism", "xtend"): 0.006,
        ("hitomi", "sxs"): 0.0005, ("hitomi", "sxi"): 0.006, ("hitomi", "hxi"): 0.1,
        ("suzaku", "xis"): 0.00365,
        ("asca", "sis"): 0.0146,
        ("asca", "gis"): 1.0 / 84.9,
    }
    for (mission, instrument), expected in values.items():
        def test(self, mission=mission, instrument=instrument, expected=expected):
            record = EventFile(mission, instrument, "test", Path("x.parquet"), Path("x.json"))
            self.assertAlmostEqual(backend.pi_to_kev_factor(record), expected, places=5)
        setattr(ReliabilityMatrixTests, f"test_pi_factor_{mission}_{instrument}", test)


def _add_coordinate_label_tests():
    examples = (
        ("RA 350.858400, DEC 58.811300", 350.8584, 58.8113),
        ("ra=0,dec=-0.5", 0.0, -0.5),
        ("RA: 12.5 DEC: +33.2", 12.5, 33.2),
        (" RA 359.999°, DEC -89.9° ", 359.999, -89.9),
        ("RA 1.0 deg DEC 2.0 deg", 1.0, 2.0),
        ("RA +5, DEC +6", 5.0, 6.0),
        ("RA 180,DEC 0", 180.0, 0.0),
        ("ra: 270.25, dec: -45.25", 270.25, -45.25),
        ("RA 30 DEC +15", 30.0, 15.0),
        ("RA 1.25°, DEC -1.25°", 1.25, -1.25),
        ("RA 72.4, DEC 23.7", 72.4, 23.7),
        ("RA 350.8584, DEC 58.8113", 350.8584, 58.8113),
    )
    for index, (label, ra, dec) in enumerate(examples):
        def test(self, label=label, ra=ra, dec=dec):
            region, _ = backend.parse_sky_region("target", "", "", 0.1, target_name=label)
            self.assertAlmostEqual(region.center_ra_deg, ra % 360.0, places=6)
            self.assertAlmostEqual(region.center_dec_deg, dec, places=6)
        setattr(ReliabilityMatrixTests, f"test_coordinate_label_{index:02d}", test)


def _add_rgb_tests():
    scenarios = (
        (1.65, 1.0, 1.0), (1.85, 1.0, 1.0), (2.435, 1.0, 1.0),
        (1.65, 0.5, 1.0), (1.85, 1.5, 1.0), (2.435, 2.0, 1.0),
        (1.65, 1.0, 0.5), (1.85, 1.0, 1.5), (2.435, 1.0, 2.0),
        (1.65, 0.0, 1.0), (1.85, 1.0, 0.7), (2.435, 1.0, 1.3),
        (1.65, 1.8, 0.8), (1.85, 0.8, 1.8), (2.435, 1.2, 1.2),
    )
    for index, (energy, brightness, gamma) in enumerate(scenarios):
        def test(self, energy=energy, brightness=brightness, gamma=gamma):
            colors = rgb_event_colors(
                np.asarray([energy]), (1.65, 1.85, 2.435), (0.2, 0.2, 0.17),
                brightness=brightness, gamma=gamma,
            )
            self.assertEqual(colors.shape, (1, 3))
            self.assertTrue(np.all(np.isfinite(colors)))
            self.assertTrue(np.all((colors >= 0) & (colors <= 255)))
        setattr(ReliabilityMatrixTests, f"test_rgb_control_scenario_{index:02d}", test)


def _add_budget_tests():
    scenarios = ((1, 1), (2, 5), (3, 10), (4, 100), (8, 250), (16, 500), (32, 1_000),
                 (50, 5_000), (75, 10_000), (100, 25_000), (125, 80_000), (143, 160_000),
                 (143, 143), (143, 500), (143, 250_000), (7, 3_000), (11, 10_001), (37, 77_777))
    for index, (count, total) in enumerate(scenarios):
        def test(self, count=count, total=total):
            observations = [_Observation(item, item % 17 + 1) for item in range(count)]
            keys = {f"xrism/resolve/{item}" for item in range(count)}
            budgets = ThreeDView.display_budgets(observations, keys, total)
            self.assertEqual(len(budgets), count)
            self.assertLessEqual(sum(budgets.values()), max(count, total))
            self.assertTrue(all(value >= 1 for value in budgets.values()))
        setattr(ReliabilityMatrixTests, f"test_global_budget_scenario_{index:02d}", test)


def _add_roi_tests():
    frame = pd.DataFrame(
        {"RA": [359.9, 0.1, 10.0, 20.0, 180.0], "DEC": [-2.0, 0.0, 2.0, 5.0, -0.5]}
    )
    scenarios = (
        (350, 5, -3, 1, 2), (0, 15, -1, 3, 2), (5, 25, 0, 6, 2),
        (350, 5, -3, -1, 1), (170, 190, -1, 0, 1), (359, 360, -3, 1, 1),
        (359.95, 0.05, -3, 1, 0), (15, 25, 4, 6, 1), (15, 25, -3, 1, 0),
        (0, 360, -3, 6, 5), (350, 20, -3, 3, 3),
    )
    for index, (ra_min, ra_max, dec_min, dec_max, expected) in enumerate(scenarios):
        def test(self, ra_min=ra_min, ra_max=ra_max, dec_min=dec_min, dec_max=dec_max, expected=expected):
            self.assertEqual(len(filter_rectangle(frame, ra_min, ra_max, dec_min, dec_max)), expected)
        setattr(ReliabilityMatrixTests, f"test_roi_scenario_{index:02d}", test)


_add_pi_factor_tests()
_add_coordinate_label_tests()
_add_rgb_tests()
_add_budget_tests()
_add_roi_tests()


if __name__ == "__main__":
    unittest.main()
