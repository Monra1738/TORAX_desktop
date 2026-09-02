"""Headless Qt checks for the controls used in the manual verification guide."""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from jaxa_torax.desktop.analysis import AnalysisDock
from jaxa_torax.desktop.inspector import InspectorPanel
from jaxa_torax.desktop.panels import (
    DataLayersPanel,
    ObservationBrowserDialog,
    SearchDialog,
)
from jaxa_torax.desktop.state import EnergySlice

APP = QApplication.instance() or QApplication([])


class GuiVerificationTests(unittest.TestCase):
    def test_slice_profile_draws_one_curve_for_every_visible_slice_image(self):
        analysis = AnalysisDock()
        slices = [
            EnergySlice(1.0 + index, 1.5 + index, uid=f"slice-{index}")
            for index in range(5)
        ]
        products = [
            SimpleNamespace(values=np.full((4, 6), index + 1.0))
            for index in range(5)
        ]

        analysis.profile.set_products(products, slices, selected_uid="slice-2")

        self.assertEqual(len(analysis.profile.curves), 5)
        self.assertEqual(analysis.profile.summary.text(), "5 visible slice profiles")
        np.testing.assert_allclose(
            analysis.profile.curves[4].getData()[1], np.full(6, 20.0)
        )
        self.assertEqual(analysis.tabs.tabText(2), "Slice profiles")
        analysis.close()

    def test_search_dialog_exposes_all_coordinate_modes_and_quick_targets(self):
        dialog = SearchDialog()
        modes = {dialog.mode.itemData(index) for index in range(dialog.mode.count())}
        targets = {dialog.quick_target.itemData(index) for index in range(dialog.quick_target.count())}
        self.assertEqual(modes, {"target", "degrees", "sexagesimal"})
        self.assertTrue({"casa", "sn1006"} <= targets)
        dialog.close()

    def test_data_layers_exposes_all_eight_instruments_and_render_modes(self):
        dialog = SearchDialog()
        self.assertEqual(len(dialog.instrument_checks), 8)
        panel = DataLayersPanel()
        self.assertEqual(
            {panel.events_button.text(), panel.density_button.text(), panel.voxels_button.text()},
            {"Events", "Density", "Voxels"},
        )
        dialog.close()
        panel.close()

    def test_inspector_energy_rgb_image_and_spectrum_controls_exist(self):
        panel = InspectorPanel()
        self.assertGreater(panel.energy_low_slider.maximum(), panel.energy_low_slider.minimum())
        self.assertGreater(panel.energy_high_slider.maximum(), panel.energy_high_slider.minimum())
        analysis = AnalysisDock()
        self.assertEqual(analysis.spectrum.scale.count(), 3)
        self.assertTrue(analysis.spectrum.show_smooth.isChecked())
        self.assertTrue(analysis.spectrum.scan_lock_width.isChecked())
        self.assertEqual(analysis.spectrum.scan_speed.count(), 4)
        self.assertFalse(panel.filter_3d_by_energy.isChecked())
        self.assertFalse(panel.filter_2d_by_energy.isChecked())
        self.assertEqual(len(panel.rgb_controls), 3)
        self.assertTrue(hasattr(panel, "exact_all_events"))
        self.assertTrue(hasattr(panel, "exact_all_events_energy"))
        self.assertTrue(hasattr(panel, "image_palette"))
        self.assertTrue(hasattr(panel, "image_stretch"))
        analysis.close()
        panel.close()

    def test_observation_browser_load_all_emits_every_available_search_result(self):
        frame = pd.DataFrame([
            {"mission": "asca", "instrument": "gis", "observation_id": "one"},
            {"mission": "asca", "instrument": "sis", "observation_id": "one"},
            {"mission": "xrism", "instrument": "xtend", "observation_id": "two"},
        ])
        dialog = ObservationBrowserDialog(frame, {"asca/gis/one"})
        emitted = []
        dialog.add_requested.connect(emitted.append)
        dialog._emit_add_all()
        self.assertEqual(
            emitted,
            [["asca/sis/one", "xrism/xtend/two"]],
        )
        dialog.close()


if __name__ == "__main__":
    unittest.main()
