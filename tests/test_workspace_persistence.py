import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jaxa_udon3.desktop.state import DesktopState, EnergySlice
from jaxa_udon3.desktop.viewer_3d import ThreeDView
from jaxa_udon3.desktop.workspace_persistence import WorkspacePersistenceMixin
from jaxa_udon3.infrastructure import science as backend


def snapshot(workspace_id="cas-a", target="Cas A", records=2):
    region = {
        "center_ra_deg": 350.8584,
        "center_dec_deg": 58.8113,
        "radius_deg": 20 / 60,
        "label": "Cas A",
        "source": "degrees",
    }
    observations = []
    for index in range(records):
        observation_id = f"{129000 + index:09d}"
        observations.append(
            {
                "record_key": f"xrism/resolve/{observation_id}",
                "visible": index % 2 == 0,
                "record": {
                    "mission": "xrism", "instrument": "resolve", "observation_id": observation_id,
                    "parquet_path": f"/cache/{observation_id}.parquet",
                    "header_path": f"/cache/{observation_id}.json",
                    "parquet_url": f"https://example.test/{observation_id}.parquet",
                    "header_url": f"https://example.test/{observation_id}.json", "source": "remote",
                },
            }
        )
    return {
        "workspace_id": workspace_id,
        "target_name": target,
        "region": region,
        "state": {"energy_band": [4.0, 6.0], "rgb_centers": [1.65, 1.85, 2.435]},
        "observations": observations,
        "slices": [{"uid": "slice-1", "low_kev": 1.55, "high_kev": 1.75, "visible": True}],
        "rois": {"primary": [350.80, 350.90, 58.77, 58.86]},
    }


class WorkspacePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "workspace.duckdb"

    def tearDown(self):
        self.temporary.cleanup()

    def test_workspace_round_trip_keeps_scientific_state(self):
        backend.save_workspace(snapshot(), self.db_path)
        restored = backend.load_workspace("cas-a", self.db_path)
        self.assertEqual(restored["target_name"], "Cas A")
        self.assertEqual(restored["state"]["energy_band"], [4.0, 6.0])
        self.assertEqual(restored["rois"]["primary"], [350.80, 350.90, 58.77, 58.86])
        self.assertEqual([item["visible"] for item in restored["observations"]], [True, False])

    def test_workspace_round_trip_keeps_scalar_display_settings(self):
        data = snapshot()
        data["state"].update({
            "image_palette": "inferno",
            "image_stretch": "sqrt",
            "image_brightness": 1.25,
            "image_contrast": 1.10,
        })
        backend.save_workspace(data, self.db_path)
        restored = backend.load_workspace("cas-a", self.db_path)
        self.assertEqual(restored["state"]["image_palette"], "inferno")
        self.assertEqual(restored["state"]["image_stretch"], "sqrt")
        self.assertEqual(restored["state"]["image_brightness"], 1.25)
        self.assertEqual(restored["state"]["image_contrast"], 1.10)

    def test_workspace_round_trip_keeps_scale_viewport_zoom_and_queue_references(self):
        data = snapshot()
        failed = {
            "record_key": "xrism/resolve/failed",
            "message": "source unavailable",
            "record": data["observations"][0]["record"],
        }
        pending = {
            "record_key": "xrism/resolve/pending",
            "record": data["observations"][1]["record"],
        }
        data["state"].update({
            "spectrum_scale": "log_y",
            "sky_viewport": {"ra_min_deg": 350.2, "ra_max_deg": 351.5},
            "two_d_zoom": {"energy": [350.5, 351.0, 58.7, 58.9]},
            "failed_observations": [failed],
            "pending_observations": [pending],
        })
        backend.save_workspace(data, self.db_path)
        restored = backend.load_workspace("cas-a", self.db_path)["state"]
        self.assertEqual(restored["spectrum_scale"], "log_y")
        self.assertEqual(restored["two_d_zoom"]["energy"], [350.5, 351.0, 58.7, 58.9])
        self.assertEqual(restored["failed_observations"][0]["message"], "source unavailable")
        self.assertEqual(restored["pending_observations"][0]["record_key"], "xrism/resolve/pending")

    def test_old_workspace_uses_safe_scalar_display_defaults(self):
        class RestoreHarness:
            state = DesktopState()

        harness = RestoreHarness()
        WorkspacePersistenceMixin._apply_restored_state(
            harness, {"energy_band": [4.0, 6.0]}, []
        )
        self.assertEqual(harness.state.image_palette, "gray")
        self.assertEqual(harness.state.image_stretch, "log")
        self.assertEqual(harness.state.image_brightness, 1.0)
        self.assertEqual(harness.state.image_contrast, 1.0)
        self.assertFalse(harness.state.filter_3d_by_energy)
        self.assertFalse(harness.state.filter_2d_by_energy)

    def test_workspace_restore_keeps_spectrum_and_independent_filter_controls(self):
        class RestoreHarness:
            state = DesktopState()

        harness = RestoreHarness()
        WorkspacePersistenceMixin._apply_restored_state(
            harness,
            {
                "spectrum_smooth_visible": False,
                "energy_scan_speed_hz": 8,
                "filter_3d_by_energy": True,
                "filter_2d_by_energy": False,
            },
            [],
        )
        self.assertFalse(harness.state.spectrum_smooth_visible)
        self.assertEqual(harness.state.energy_scan_speed_hz, 8)
        self.assertTrue(harness.state.filter_3d_by_energy)
        self.assertFalse(harness.state.filter_2d_by_energy)

    def test_latest_workspace_is_restored_automatically(self):
        backend.save_workspace(snapshot("cas-a"), self.db_path)
        backend.save_workspace(snapshot("sn1006", "SN1006"), self.db_path)
        self.assertEqual(backend.load_active_workspace(self.db_path)["workspace_id"], "sn1006")

    def test_resaving_workspace_replaces_rows_without_duplicates(self):
        backend.save_workspace(snapshot(records=50), self.db_path)
        backend.save_workspace(snapshot(records=143), self.db_path)
        restored = backend.load_workspace("cas-a", self.db_path)
        self.assertEqual(len(restored["observations"]), 143)
        self.assertEqual(len({item["record_key"] for item in restored["observations"]}), 143)

    def test_workspaces_remain_separate_by_target(self):
        backend.save_workspace(snapshot("cas-a", "Cas A", 50), self.db_path)
        backend.save_workspace(snapshot("sn1006", "SN1006", 143), self.db_path)
        self.assertEqual(len(backend.load_workspace("cas-a", self.db_path)["observations"]), 50)
        self.assertEqual(len(backend.load_workspace("sn1006", self.db_path)["observations"]), 143)
        self.assertEqual([item["target_name"] for item in backend.list_workspaces(self.db_path)], ["SN1006", "Cas A"])

    def test_new_slice_does_not_reuse_restored_slice_id(self):
        state = DesktopState(slices=[EnergySlice(1.55, 1.75, uid="slice-1")])
        self.assertNotEqual(state.add_slice(1.75, 1.95).uid, "slice-1")

    def test_143_observation_budgets_respect_global_limit(self):
        class Observation:
            def __init__(self, index):
                self.record = type("Record", (), {"mission": "asca", "instrument": "gis", "observation_id": str(index)})()
                self.frame = pd.DataFrame({"KEV": np.zeros(index % 9 + 1)})

        observations = [Observation(index) for index in range(143)]
        keys = {f"asca/gis/{index}" for index in range(143)}
        budgets = ThreeDView.display_budgets(observations, keys, 160_000)
        self.assertEqual(len(budgets), 143)
        self.assertLessEqual(sum(budgets.values()), 160_000)
        self.assertTrue(all(value > 0 for value in budgets.values()))


if __name__ == "__main__":
    unittest.main()
