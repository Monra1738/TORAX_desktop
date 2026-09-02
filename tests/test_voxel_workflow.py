import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtWidgets import QApplication

from jaxa_udon3.desktop.inspector import InspectorPanel
from jaxa_udon3.desktop.main_window import MainWindow
from jaxa_udon3.desktop.state import DesktopState, EnergySlice
from jaxa_udon3.desktop.viewer_3d import enabled_slice_point_uids
from jaxa_udon3.desktop.voxel_workflow import resolved_voxel_energy_band


class VoxelWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_selected_slice_drives_voxels_even_when_its_plane_is_hidden(self):
        state = DesktopState(energy_band=(6.68, 6.72))
        item = EnergySlice(6.6975, 6.7025, visible=False, show_plane=False, uid="fe-line")
        state.slices = [item]
        state.selected_slice_uid = item.uid
        self.assertEqual(resolved_voxel_energy_band(state), (6.6975, 6.7025))

    def test_global_and_all_energy_sources_are_explicit(self):
        state = DesktopState(energy_band=(6.68, 6.72))
        state.voxel_energy_source = "global_band"
        self.assertEqual(resolved_voxel_energy_band(state), (6.68, 6.72))
        state.voxel_energy_source = "all_energies"
        state.loaded_observations = [
            SimpleNamespace(frame=pd.DataFrame({"KEV": [1.2, 8.4]})),
            SimpleNamespace(frame=pd.DataFrame({"KEV": [0.8, 9.1]})),
        ]
        self.assertEqual(resolved_voxel_energy_band(state), (0.8, 9.1))

    def test_native_controls_expose_linking_and_overlay_choices(self):
        panel = InspectorPanel()
        self.assertEqual(panel.voxel_energy_source.currentData(), "selected_slice")
        self.assertTrue(panel.show_slice_window.isChecked())
        self.assertTrue(panel.show_grid_backdrop.isChecked())
        self.assertEqual(panel.camera_preset.currentData(), "isometric")
        self.assertIn("Active slice", panel.voxel_energy_source.currentText())
        self.assertTrue(hasattr(panel, "slice_color"))
        panel.voxel_energy_source.setCurrentIndex(panel.voxel_energy_source.findData("all_energies"))
        self.assertEqual(panel.voxel_energy_source.currentData(), "all_energies")
        panel.close()

    def test_content_modes_enable_only_the_expected_colored_slice_points(self):
        active = EnergySlice(2.0, 3.0, uid="active", visible=False, color="#ff0000")
        second = EnergySlice(4.0, 5.0, uid="second", visible=True, color="#0000ff")
        slices = [active, second]

        self.assertEqual(enabled_slice_point_uids(slices, "all", "active"), set())
        self.assertEqual(enabled_slice_point_uids(slices, "planes", "active"), set())
        self.assertEqual(enabled_slice_point_uids(slices, "active", "active"), {"active"})
        self.assertEqual(enabled_slice_point_uids(slices, "all_active", "active"), {"active"})
        self.assertEqual(enabled_slice_point_uids(slices, "multiple", "active"), {"second"})

    def test_global_slider_always_schedules_a_3d_refresh(self):
        class Spin:
            def blockSignals(self, _blocked):
                pass

            def setValue(self, _value):
                pass

        scheduled = []
        harness = SimpleNamespace(
            state=DesktopState(energy_band=(2.0, 6.0), spectrum_linked=False),
            inspector=SimpleNamespace(
                energy_low=Spin(), energy_high=Spin(), _sync_energy_sliders=lambda *_: None
            ),
            analysis=SimpleNamespace(spectrum=SimpleNamespace(set_band=lambda *_: None)),
            _invalidate_exact=lambda: None,
            _schedule_viewer=lambda: scheduled.append("3d"),
            _schedule_analysis=lambda *_: None,
            _workspace_changed=lambda: None,
        )
        MainWindow._set_energy_band(harness, 3.0, 4.0)
        self.assertEqual(harness.state.energy_band, (3.0, 4.0))
        self.assertEqual(scheduled, ["3d"])

    def test_slice_color_change_updates_the_model_and_schedules_render(self):
        item = EnergySlice(2.0, 3.0, uid="active", color="#00ff00")
        scheduled = []
        harness = SimpleNamespace(
            state=DesktopState(slices=[item], selected_slice_uid=item.uid),
            left_panel=SimpleNamespace(update_slice=lambda *_args, **_kwargs: None),
            inspector=SimpleNamespace(_set_slice_color_button=lambda *_: None),
            _schedule_viewer=lambda: scheduled.append("3d"),
            _schedule_analysis=lambda *kinds: scheduled.extend(kinds),
            _workspace_changed=lambda: None,
        )
        MainWindow._set_slice_color(harness, "active", "#ff00ff")
        self.assertEqual(item.color, "#ff00ff")
        self.assertEqual(scheduled, ["3d", "slices"])

if __name__ == "__main__":
    unittest.main()
