import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtWidgets import QApplication

from jaxa_udon3.desktop.analysis import SpectrumWidget
from jaxa_udon3.desktop.data_controller import load_previews
from jaxa_udon3.desktop.inspector import InspectorPanel
from jaxa_udon3.desktop.science_views import SpectrumProduct
from jaxa_udon3.desktop.state import EnergySlice
from jaxa_udon3.infrastructure import science as backend


class NativeEnergyControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_energy_sliders_update_precise_fields_and_emit_one_valid_band(self):
        panel = InspectorPanel()
        received = []
        panel.energy_changed.connect(lambda low, high: received.append((low, high)))
        panel.energy_low_slider.setValue(325)
        self.assertEqual(panel.energy_low.value(), 3.25)
        self.assertGreater(panel.energy_high.value(), panel.energy_low.value())
        self.assertEqual(received[-1], (3.25, panel.energy_high.value()))
        panel.energy_high.setValue(7.5)
        self.assertEqual(panel.energy_high_slider.value(), 750)

    def test_energy_filter_scopes_are_independent_and_default_to_all_events(self):
        panel = InspectorPanel()
        received = []
        panel.energy_filter_scope_changed.connect(
            lambda filter_3d, filter_2d: received.append((filter_3d, filter_2d))
        )
        self.assertFalse(panel.filter_3d_by_energy.isChecked())
        self.assertFalse(panel.filter_2d_by_energy.isChecked())
        self.assertTrue(panel.spectrum_linked.isChecked())
        panel.filter_3d_by_energy.setChecked(True)
        self.assertEqual(received[-1], (True, False))
        panel.filter_2d_by_energy.setChecked(True)
        self.assertEqual(received[-1], (True, True))
        panel.close()

    def test_spectrum_link_control_emits_and_can_restore_all_event_mode(self):
        panel = InspectorPanel()
        received = []
        panel.spectrum_link_changed.connect(received.append)
        panel.spectrum_linked.setChecked(False)
        self.assertEqual(received, [False])
        panel.spectrum_linked.setChecked(True)
        self.assertEqual(received, [False, True])
        panel.close()

    def test_energy_voxel_control_supports_five_electron_volt_bins(self):
        panel = InspectorPanel()
        self.assertAlmostEqual(panel.voxel_energy.minimum(), 0.005)
        panel.voxel_energy.setValue(0.005)
        self.assertAlmostEqual(panel.voxel_energy.value(), 0.005)
        panel.close()

    def test_energy_scan_moves_and_ping_pongs_without_changing_band_width(self):
        spectrum = SpectrumWidget()
        product = SpectrumProduct(
            x=np.linspace(0.5, 9.5, 10),
            counts=np.arange(10, dtype=float),
            edges=np.linspace(0.0, 10.0, 11),
            smoothed_counts=np.linspace(0.0, 9.0, 10),
        )
        received = []
        spectrum.band_changed.connect(lambda low, high: received.append((low, high)))
        spectrum.set_spectrum(product)
        spectrum.set_band(2.0, 4.0)
        spectrum._move_band(1)
        self.assertAlmostEqual(received[-1][1] - received[-1][0], 2.0)
        self.assertGreater(received[-1][0], 2.0)
        spectrum.set_band(8.0, 10.0)
        spectrum._scan_direction = 1
        spectrum._scan_tick()
        self.assertEqual(spectrum._scan_direction, -1)
        spectrum._scan_tick()
        self.assertLess(received[-1][1], 10.0)
        spectrum._toggle_scan(True)
        self.assertTrue(spectrum._scan_timer.isActive())
        spectrum.stop_scan()
        self.assertFalse(spectrum._scan_timer.isActive())
        spectrum.close()

    def test_slice_region_signal_preserves_string_uid(self):
        spectrum = SpectrumWidget()
        received = []
        spectrum.slice_changed.connect(lambda uid, low, high: received.append((uid, low, high)))
        spectrum.set_slices([EnergySlice(1.0, 2.0, uid="slice-real")])
        spectrum._slice_regions["slice-real"].setRegion((1.2, 2.2))
        self.app.processEvents()
        self.assertEqual(received[-1][0], "slice-real")

    def test_preview_loader_attaches_provenance_for_3d_mission_colours(self):
        record = backend.EventFile(
            "xrism", "resolve", "001", Path("events.parquet"), Path("header.json")
        )
        frame = pd.DataFrame({"RA": [1.0], "DEC": [2.0], "PI": [2000], "KEV": [1.0]})
        with patch(
            "jaxa_udon3.desktop.data_controller.backend.read_region_preview",
            return_value=(frame, {}, 1, False),
        ):
            payload = load_previews([record], object(), max_points=1_000)
        loaded = payload.observations[0].frame
        self.assertEqual(loaded.loc[0, "MISSION"], "xrism")
        self.assertEqual(loaded.loc[0, "INSTRUMENT"], "resolve")
        self.assertEqual(loaded.loc[0, "OBSERVATION_ID"], "001")
