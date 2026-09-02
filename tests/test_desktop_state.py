import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxa_udon3.desktop.science_views import combine_frames
from jaxa_udon3.desktop.state import DesktopState


class DummyRecord:
    def __init__(self, mission, instrument, obsid):
        self.mission = mission
        self.instrument = instrument
        self.observation_id = obsid


class DummyObservation:
    def __init__(self, mission, instrument, obsid, energy):
        self.record = DummyRecord(mission, instrument, obsid)
        self.frame = pd.DataFrame({"RA": [10.0], "DEC": [20.0], "KEV": [energy]})
        self.events_in_region = 1


class DesktopStateTests(unittest.TestCase):
    def test_hidden_observation_is_not_unloaded(self):
        state = DesktopState()
        obs = DummyObservation("xrism", "resolve", "1", 4.0)
        key = state.record_key(obs)
        state.observation_cache[key] = obs
        state.loaded_observations = [obs]
        state.combined_frame = combine_frames([obs])
        state.visible_record_keys = {key}
        self.assertEqual(len(state.displayed_frame()), 1)
        state.visible_record_keys.remove(key)
        self.assertEqual(len(state.displayed_frame()), 0)
        self.assertIs(state.observation_cache[key], obs)

    def test_global_energy_filter_is_separate_from_multiple_slices(self):
        state = DesktopState(energy_band=(4.0, 8.0))
        first = state.add_slice(4.0, 5.0)
        second = state.add_slice(6.2, 6.7)
        self.assertEqual(state.energy_band, (4.0, 8.0))
        self.assertEqual(len(state.slices), 2)
        self.assertNotEqual(first.uid, second.uid)
        self.assertEqual(state.selected_slice_uid, second.uid)

    def test_combined_frame_uses_compact_categorical_labels(self):
        observations = [
            DummyObservation("xrism", "resolve", "1", 4.0),
            DummyObservation("suzaku", "xis", "2", 6.0),
        ]
        frame = combine_frames(observations)
        self.assertEqual(str(frame["MISSION"].dtype), "category")
        self.assertEqual(str(frame["RECORD_KEY"].dtype), "category")

    def test_displayed_frame_cache_reuses_same_selection(self):
        state = DesktopState()
        observations = [
            DummyObservation("xrism", "resolve", "1", 4.0),
            DummyObservation("suzaku", "xis", "2", 6.0),
        ]
        state.loaded_observations = observations
        state.combined_frame = combine_frames(observations)
        state.visible_record_keys = {state.record_key(item) for item in observations}
        first = state.displayed_frame()
        second = state.displayed_frame()
        self.assertIs(first, second)
        state.visible_record_keys.remove(state.record_key(observations[1]))
        third = state.displayed_frame()
        self.assertEqual(len(third), 1)
        self.assertIsNot(third, first)

    def test_auto_image_quality_smooths_sparse_bands_more(self):
        state = DesktopState(auto_image_quality=True)
        self.assertGreater(state.effective_image_smoothing(50), state.effective_image_smoothing(50_000))

    def test_two_roi_spectrum_frames_are_independent(self):
        state = DesktopState()
        observations = [
            DummyObservation("xrism", "resolve", "1", 4.0),
            DummyObservation("suzaku", "xis", "2", 6.0),
        ]
        observations[0].frame["RA"], observations[0].frame["DEC"] = [10.0], [20.0]
        observations[1].frame["RA"], observations[1].frame["DEC"] = [11.0], [21.0]
        state.loaded_observations = observations
        state.combined_frame = combine_frames(observations)
        state.visible_record_keys = {state.record_key(item) for item in observations}
        first = state.spectrum_frame_for((9.9, 10.1, 19.9, 20.1))
        second = state.spectrum_frame_for((10.9, 11.1, 20.9, 21.1))
        self.assertEqual(first["OBSERVATION_ID"].iloc[0], "1")
        self.assertEqual(second["OBSERVATION_ID"].iloc[0], "2")

    def test_cas_a_reference_bands_match_task(self):
        state = DesktopState()
        created = state.add_cas_a_reference_slices()
        self.assertEqual(
            [(round(item.low_kev, 2), round(item.high_kev, 2)) for item in created],
            [(1.55, 1.75), (1.75, 1.95), (2.35, 2.52), (3.93, 6.23)],
        )

    def test_slice_membership_is_inclusive_and_cached(self):
        state = DesktopState()
        first = DummyObservation("xrism", "resolve", "1", 1.60)
        first.frame = pd.DataFrame({
            "RA": [10, 11, 12], "DEC": [20, 21, 22], "KEV": [1.60, 1.70, 1.80],
        })
        state.loaded_observations = [first]
        state.combined_frame = combine_frames([first])
        state.visible_record_keys = {state.record_key(first)}
        item = state.add_slice(1.55, 1.75)
        membership = state.slice_event_indices(item)
        self.assertEqual(membership[state.record_key(first)].tolist(), [0, 1])
        self.assertIs(state.slice_event_indices(item), membership)


if __name__ == "__main__":
    unittest.main()
