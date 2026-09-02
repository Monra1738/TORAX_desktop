import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxa_torax.desktop.analysis import SpectrumWidget
from jaxa_torax.desktop.data_controller import balanced_row_limits, exact_energy
from jaxa_torax.desktop.main_refresh import RefreshMixin
from jaxa_torax.desktop.observation_session import (
    ObservationLoadSession,
    is_transient_preview_error,
    load_observation_preview_resilient,
)
from jaxa_torax.desktop.panels import ObservationBrowserDialog
from jaxa_torax.desktop.science_views import (
    SkyViewport,
    SpectrumProduct,
    energy_image,
    energy_to_plot_x,
    plot_x_to_energy,
)
from jaxa_torax.desktop.state import DesktopState, EnergySlice
from jaxa_torax.desktop.viewers import ImagePlot, SkyView
from jaxa_torax.desktop.window_actions import WindowActionsMixin
from jaxa_torax.desktop.workspace_persistence import WorkspacePersistenceMixin
from jaxa_torax.domain import EventFile, SkyRegion
from jaxa_torax.infrastructure import event_sources, previews

APP = QApplication.instance() or QApplication([])


def record(mission, instrument, observation_id):
    return EventFile(
        mission, instrument, observation_id,
        Path(f"/{observation_id}.parquet"), Path(f"/{observation_id}.json"),
    )


class ImmediatePool:
    def start(self, worker):
        worker.run()


class DeferredPool:
    def __init__(self):
        self.workers = []

    def start(self, worker):
        self.workers.append(worker)


class ProgressiveLoadingTests(unittest.TestCase):
    def test_transient_cache_conflict_is_retried_automatically(self):
        item = record("asca", "gis", "retry")
        attempts = []
        sleeps = []
        notices = []

        def preview(*_args):
            attempts.append(True)
            if len(attempts) < 3:
                raise RuntimeError(
                    "Can't open a connection to same database file with a different "
                    "configuration than existing connections"
                )
            return "loaded"

        with patch(
            "jaxa_torax.desktop.observation_session.load_observation_preview",
            side_effect=preview,
        ):
            result = load_observation_preview_resilient(
                item,
                object(),
                250,
                retry_callback=lambda *args: notices.append(args),
                sleep=lambda delay: sleeps.append(delay),
            )
        self.assertEqual(result, "loaded")
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(notices), 2)
        self.assertEqual(sleeps, [0.2, 0.6])

    def test_only_transient_failures_are_classified_for_retry(self):
        self.assertTrue(is_transient_preview_error(TimeoutError("timed out")))
        self.assertTrue(is_transient_preview_error(RuntimeError("database is locked")))
        self.assertFalse(is_transient_preview_error(ValueError("Missing WCS values")))
        self.assertFalse(is_transient_preview_error(FileNotFoundError("no source URL")))

    def test_five_hundred_record_queue_completes_without_dropping_items(self):
        pool = DeferredPool()
        records = [record("asca", "gis", f"stress-{index:03d}") for index in range(500)]
        session = ObservationLoadSession(pool, records, object())
        finished = []
        failures = []
        session.observation_failed.connect(lambda *_args: failures.append(_args))
        session.session_finished.connect(
            lambda _sid, keys, failed: finished.append((keys, failed))
        )
        with patch(
            "jaxa_torax.desktop.observation_session.load_observation_preview",
            side_effect=lambda item, *_args: SimpleNamespace(record=item),
        ):
            session.start()
            worker_index = 0
            while worker_index < len(pool.workers):
                pool.workers[worker_index].run()
                worker_index += 1
        self.assertEqual(session.completed_count, 500)
        self.assertEqual(len(session.successful_keys), 500)
        self.assertEqual(len(pool.workers), 500)
        self.assertFalse(failures)
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0][1])

    def test_hierarchical_preview_budget_has_hard_caps_for_143_records(self):
        records = (
            [record("xrism", "resolve", f"r{i}") for i in range(50)]
            + [record("xrism", "xtend", f"x{i}") for i in range(43)]
            + [record("asca", "gis", f"a{i}") for i in range(50)]
        )
        limits = balanced_row_limits(records, 600_000)
        self.assertEqual(len(limits), 143)
        self.assertLessEqual(sum(limits), 600_000)
        self.assertTrue(all(250 <= value <= 15_000 for value in limits))
        self.assertAlmostEqual(sum(limits[:93]), sum(limits[93:]), delta=150)

    def test_session_publishes_successes_and_isolates_one_failure(self):
        records = [record("xrism", "resolve", value) for value in ("one", "bad", "three")]
        loaded, failed, progress, finished = [], [], [], []

        def preview(item, _region, max_rows):
            if item.observation_id == "bad":
                raise OSError("calibrated source unavailable")
            frame = pd.DataFrame({"RA": [1.0], "DEC": [2.0], "KEV": [3.0]})
            return frame, {"state": "cached"}, 1, True

        session = ObservationLoadSession(ImmediatePool(), records, object())
        session.observation_loaded.connect(lambda _sid, obs: loaded.append(obs.record.observation_id))
        session.observation_failed.connect(lambda _sid, key, message: failed.append((key, message)))
        session.progress_changed.connect(lambda _sid, done, total, _key: progress.append((done, total)))
        session.session_finished.connect(lambda _sid, keys, failures: finished.append((keys, failures)))
        with patch("jaxa_torax.desktop.data_controller.backend.read_region_preview", side_effect=preview):
            session.start()
        self.assertEqual(loaded, ["one", "three"])
        self.assertEqual(len(failed), 1)
        self.assertEqual(progress[-1], (3, 3))
        self.assertEqual(len(finished), 1)

    def test_cancelled_session_rejects_late_worker_results(self):
        pool = DeferredPool()
        session = ObservationLoadSession(pool, [record("asca", "gis", "late")], object())
        loaded = []
        session.observation_loaded.connect(lambda *_: loaded.append(True))
        with patch(
            "jaxa_torax.desktop.data_controller.backend.read_region_preview",
            return_value=(pd.DataFrame({"RA": [1], "DEC": [2], "KEV": [3]}), {}, 1, True),
        ):
            session.start()
            session.cancel()
            pool.workers[0].run()
        self.assertEqual(loaded, [])

    def test_session_starts_no_more_than_three_observations_concurrently(self):
        pool = DeferredPool()
        records = [record("asca", "gis", str(index)) for index in range(50)]
        session = ObservationLoadSession(pool, records, object())
        session.start()
        self.assertEqual(len(pool.workers), 3)

    def test_exact_energy_reports_per_record_progress_and_keeps_successes(self):
        records = [record("asca", "gis", value) for value in ("one", "bad", "three")]
        progress = []

        def product(items, low, high, bins, region):
            if items[0].observation_id == "bad":
                raise OSError("source unavailable")
            value = 1 if items[0].observation_id == "one" else 2
            return {
                "hist": np.full((2, 2), value),
                "x_edges": np.asarray([0.0, 0.5, 1.0]),
                "y_edges": np.asarray([0.0, 0.5, 1.0]),
                "event_count": value,
                "low_kev": low, "high_kev": high,
                "source_keys": [items[0].observation_id],
                "exact": True, "cache_hit": True,
            }

        with patch(
            "jaxa_torax.desktop.data_controller.backend.exact_energy_image",
            side_effect=product,
        ):
            result = exact_energy(
                records, object(), 1.0, 2.0, 2,
                progress_callback=lambda done, total, key: progress.append((done, total, key)),
            )
        self.assertEqual(progress[-1][:2], (3, 3))
        self.assertEqual(result["event_count"], 3)
        self.assertTrue(np.all(result["hist"] == 3))
        self.assertEqual(list(result["failures"]), ["asca/gis/bad"])

    def test_browser_sorting_keeps_identity_and_excludes_loaded_and_loading(self):
        frame = pd.DataFrame({
            "mission": ["xrism"] * 4, "instrument": ["resolve"] * 4,
            "observation_id": ["3", "1", "4", "2"], "object": ["x"] * 4,
        })
        dialog = ObservationBrowserDialog(
            frame, {"xrism/resolve/1"}, queued_keys={"xrism/resolve/2"}
        )
        dialog.table.sortItems(3)
        dialog._select_all_not_loaded()
        self.assertEqual(set(dialog.selected_keys()), {"xrism/resolve/3", "xrism/resolve/4"})
        dialog.close()

    def test_stale_remote_record_is_relocated_to_current_cache(self):
        stale = EventFile(
            "xrism", "resolve", "300000001",
            Path("/removed/legacy/cache/events.parquet"),
            Path("/removed/legacy/cache/header.json"),
            "https://example.test/events.parquet", "https://example.test/header.json",
            "remote",
        )
        normalized = event_sources.normalized_cache_record(stale)
        expected = event_sources.remote_cache_paths("xrism", "resolve", "300000001")
        self.assertEqual((normalized.parquet_path, normalized.header_path), expected)
        self.assertEqual(normalized.parquet_url, stale.parquet_url)

    def test_changed_preview_budget_reuses_compatible_cached_region(self):
        item = record("asca", "gis", "cached")
        region = SkyRegion(10.0, 20.0, 1.0)
        frame = pd.DataFrame({
            "TIME": np.arange(400, dtype=float),
            "PI": np.arange(400, dtype=np.int32),
            "X": np.arange(400, dtype=np.int16),
            "Y": np.arange(400, dtype=np.int16),
            "SOURCE_ROW": np.arange(400, dtype=np.uint32),
            "RA": np.full(400, 10.0, dtype=np.float32),
            "DEC": np.full(400, 20.0, dtype=np.float32),
            "KEV": np.ones(400, dtype=np.float32),
        })
        signature = {
            "version": previews.PREVIEW_VERSION,
            "calibration": previews.CALIBRATION_VERSION,
            "record": previews.record_key(item),
            "rows": 400,
            "region": region.signature(),
            "factor": previews.pi_to_kev_factor(item),
            "wcs": previews.wcs_metadata_signature({}),
            "events_in_region": 12_345,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cached.parquet"
            frame.to_parquet(path, index=False)
            candidate = {"path": str(path), "metadata": signature}
            with (
                patch.object(previews, "get_record_header", return_value={}),
                patch.object(previews, "cache_entry", return_value=None),
                patch.object(previews, "cache_entries_for_observation", return_value=[candidate]),
                patch.object(previews, "_read_region_candidates", side_effect=AssertionError("source read")),
            ):
                result, _metadata, total, cache_hit = previews.read_region_preview(
                    item, region, max_rows=250, db_path=Path(directory) / "cache.duckdb"
                )
        self.assertEqual(len(result), 250)
        self.assertEqual(total, 12_345)
        self.assertTrue(cache_hit)


class SpectrumScaleTests(unittest.TestCase):
    def test_energy_coordinate_adapter_round_trips(self):
        energy = np.asarray([0.3, 2.0, 12.0])
        for scale in ("linear", "log_y", "log_log"):
            np.testing.assert_allclose(
                plot_x_to_energy(energy_to_plot_x(energy, scale), scale), energy
            )
        self.assertTrue(np.isnan(energy_to_plot_x(0.0, "log_log")))

    def test_scale_switch_preserves_canonical_band_and_slice_and_omits_zero_bins(self):
        widget = SpectrumWidget()
        product = SpectrumProduct(
            np.asarray([0.5, 1.5, 2.5, 3.5]),
            np.asarray([0.0, 4.0, 0.0, 16.0]),
            np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        )
        item = EnergySlice(1.0, 3.0, uid="slice-test")
        widget.set_spectrum(product)
        widget.set_band(1.0, 3.0)
        widget.set_slices([item])
        widget.set_scale("log_log")
        np.testing.assert_allclose(widget.region.getRegion(), [0.0, np.log10(3.0)])
        np.testing.assert_allclose(
            widget._slice_regions[item.uid].getRegion(), [0.0, np.log10(3.0)]
        )
        _x, y = widget.curve.getData()
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(y > 0.0))
        widget.set_scale("linear")
        np.testing.assert_allclose(widget.region.getRegion(), [1.0, 3.0])
        widget.close()

    def test_large_linear_count_range_does_not_overflow_when_entering_log_y(self):
        widget = SpectrumWidget()
        product = SpectrumProduct(
            np.asarray([1.0, 2.0]), np.asarray([1.0, 1.0e8]),
            np.asarray([0.5, 1.5, 2.5]),
        )
        widget.set_spectrum(product)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            widget.set_scale("log_y")
            APP.processEvents()
        self.assertFalse(any("overflow" in str(item.message).lower() for item in caught))
        widget.close()

    def test_scale_persistence_change_does_not_recompute_spectrum(self):
        class Harness:
            state = DesktopState()

            def _workspace_changed(self):
                self.saved = True

            def _refresh_spectrum(self):
                raise AssertionError("display-only scale change recomputed science")

        harness = Harness()
        WorkspacePersistenceMixin._set_spectrum_scale(harness, "log_y")
        self.assertEqual(harness.state.spectrum_scale, "log_y")
        self.assertTrue(harness.saved)

    def test_exact_energy_survives_a_normal_display_refresh(self):
        data = {
            "hist": np.ones((2, 2)),
            "x_edges": np.asarray([9.0, 10.0, 11.0]),
            "y_edges": np.asarray([19.0, 20.0, 21.0]),
            "event_count": 4,
            "low_kev": 1.0,
            "high_kev": 2.0,
            "cache_hit": True,
        }

        class Harness:
            state = DesktopState(region=SkyRegion(10.0, 20.0, 1.0))
            _exact_energy_data = data
            _exact_energy_product = staticmethod(WindowActionsMixin._exact_energy_product)

            def _display_energy_product(self, product):
                self.displayed = product

            def _energy_product(self, *_args):
                raise AssertionError("exact product was replaced by a preview")

        harness = Harness()
        harness.state.energy_image_exact = True
        harness.state.filter_2d_by_energy = True
        RefreshMixin._refresh_energy(harness)
        self.assertTrue(harness.displayed.exact)
        self.assertEqual(harness.displayed.count, 4)

    def test_all_event_defaults_and_independent_energy_filter_scopes(self):
        class Harness:
            state = DesktopState(region=SkyRegion(10.0, 20.0, 1.0))
            _exact_energy_data = None

            def _energy_product(self, _frame, low, high):
                self.requested_band = (low, high)
                return SimpleNamespace()

            def _display_energy_product(self, _product):
                pass

        harness = Harness()
        harness.state.combined_frame = pd.DataFrame({"KEV": [0.5, 2.0, 9.0]})
        # The linked spectrum behavior is the default.  Explicitly disable it
        # here to verify the opt-in all-event/independent-filter mode.
        harness.state.spectrum_linked = False
        self.assertEqual(RefreshMixin._effective_3d_energy_band(harness), (0.5, 9.0))
        RefreshMixin._refresh_energy(harness)
        self.assertEqual(harness.requested_band, (0.5, 9.0))

        harness.state.filter_3d_by_energy = True
        harness.state.filter_2d_by_energy = True
        self.assertEqual(
            RefreshMixin._effective_3d_energy_band(harness), harness.state.energy_band
        )
        RefreshMixin._refresh_energy(harness)
        self.assertEqual(harness.requested_band, harness.state.energy_band)

    def test_linked_spectrum_band_drives_both_3d_and_2d_by_default(self):
        class Harness:
            state = DesktopState(region=SkyRegion(10.0, 20.0, 1.0))
            _exact_energy_data = None

            def _energy_product(self, _frame, low, high):
                self.requested_band = (low, high)
                return SimpleNamespace()

            def _display_energy_product(self, _product):
                pass

        harness = Harness()
        harness.state.combined_frame = pd.DataFrame({"KEV": [0.5, 2.0, 9.0]})
        self.assertTrue(harness.state.spectrum_linked)
        self.assertEqual(
            RefreshMixin._effective_3d_energy_band(harness), harness.state.energy_band
        )
        RefreshMixin._refresh_energy(harness)
        self.assertEqual(harness.requested_band, harness.state.energy_band)


class FixedViewportTests(unittest.TestCase):
    def test_each_3d_mode_uses_its_explicit_energy_source(self):
        class Harness(RefreshMixin):
            state = DesktopState(energy_band=(2.0, 6.0))

        harness = Harness()
        item = EnergySlice(6.6975, 6.7025, uid="active")
        harness.state.slices = [item]
        harness.state.selected_slice_uid = item.uid

        harness.state.render_mode = "events"
        harness.state.voxel_energy_source = "selected_slice"
        harness.state.content_mode = "all"
        self.assertEqual(harness._effective_3d_display_band(), (2.0, 6.0))

        harness.state.content_mode = "active"
        self.assertEqual(harness._effective_3d_display_band(), (6.6975, 6.7025))

        harness.state.render_mode = "voxels"
        harness.state.voxel_energy_source = "selected_slice"
        self.assertEqual(harness._effective_3d_display_band(), (6.6975, 6.7025))
        harness.state.voxel_energy_source = "global_band"
        self.assertEqual(harness._effective_3d_display_band(), (2.0, 6.0))

    def test_refresh_passes_same_correct_band_to_points_and_energy_axis(self):
        class Viewer:
            available = False

            def __init__(self):
                self.event_band = None

            def set_scene_transform(self, *_args):
                pass

            def sync_event_actors(self, *args, **_kwargs):
                self.event_band = args[4]

            def sync_slice_points(self, *_args, **_kwargs):
                pass

            def set_content_mode(self, *_args, **_kwargs):
                pass

            def sync_slices(self, *_args, **_kwargs):
                pass

            def sync_reference_plane(self, *_args, **_kwargs):
                pass

        class Harness(RefreshMixin):
            def __init__(self):
                self.state = DesktopState(
                    region=SkyRegion(350.35, 58.80, 0.25),
                    energy_band=(2.0, 6.0),
                )
                self.state.loaded_observations = [SimpleNamespace(frame=pd.DataFrame())]
                self.workspace = SimpleNamespace(three_d=Viewer())

        harness = Harness()
        item = EnergySlice(3.15, 3.45, uid="active")
        harness.state.slices = [item]
        harness.state.selected_slice_uid = item.uid
        guide_bands = []

        with patch(
            "jaxa_torax.desktop.main_refresh.sync_scene_guides",
            side_effect=lambda _viewer, _region, band, *_args: guide_bands.append(band),
        ):
            harness.state.content_mode = "active"
            harness._refresh_3d()
            self.assertEqual(harness.workspace.three_d.event_band, (3.15, 3.45))
            self.assertEqual(guide_bands[-1], (3.15, 3.45))

            harness.state.content_mode = "all"
            harness._refresh_3d()
            self.assertEqual(harness.workspace.three_d.event_band, (2.0, 6.0))
            self.assertEqual(guide_bands[-1], (2.0, 6.0))

    def test_programmatic_roi_sync_does_not_reemit_rectangle_change(self):
        viewport = SkyViewport.from_region(SkyRegion(10.0, 20.0, 1.0))
        rectangle = (9.5, 10.5, 19.5, 20.5)
        for widget_type in (SkyView, ImagePlot):
            with self.subTest(widget=widget_type.__name__):
                widget = widget_type()
                widget.set_viewport(viewport)
                widget.roi.setVisible(True)
                emitted = []
                widget.rectangle_changed.connect(emitted.append)

                widget.set_roi(rectangle)
                APP.processEvents()
                self.assertEqual(emitted, [])

                widget._roi_finished()
                APP.processEvents()
                self.assertEqual(emitted, [rectangle])
                widget.close()

    def test_circular_and_ra_wrap_bounds_are_continuous_and_clamped(self):
        region = SkyRegion(359.9, 60.0, 1.0)
        viewport = SkyViewport.from_region(region)
        self.assertAlmostEqual(viewport.ra_min_deg, 357.9, places=6)
        self.assertAlmostEqual(viewport.ra_max_deg, 361.9, places=6)
        self.assertEqual(viewport.normalize_ra(360.25), 0.25)
        rectangle = viewport.clamp_rectangle((359.8, 0.2, 59.5, 60.5))
        self.assertGreater(rectangle[1], 360.0)
        self.assertGreaterEqual(rectangle[0], viewport.ra_min_deg)
        self.assertLessEqual(rectangle[1], viewport.ra_max_deg)

    def test_empty_energy_band_keeps_full_fixed_grid(self):
        region = SkyRegion(10.0, 20.0, 1.0)
        product = energy_image(pd.DataFrame(), 4.0, 6.0, bins=32, region=region)
        viewport = SkyViewport.from_region(region)
        self.assertEqual(product.values.shape, (32, 32))
        self.assertEqual((product.x_edges[0], product.x_edges[-1]), viewport.x_range)
        self.assertEqual((product.y_edges[0], product.y_edges[-1]), viewport.y_range)

    def test_image_pan_and_zoom_out_are_limited_to_workspace(self):
        viewport = SkyViewport.from_region(SkyRegion(10.0, 20.0, 1.0))
        view = ImagePlot()
        view.set_viewport(viewport)
        view.plot.getPlotItem().vb.setRange(
            xRange=(-100.0, 100.0), yRange=(-100.0, 100.0), padding=0
        )
        APP.processEvents()
        x_range, y_range = view.plot.viewRange()
        self.assertGreaterEqual(x_range[0], viewport.ra_min_deg - 1e-8)
        self.assertLessEqual(x_range[1], viewport.ra_max_deg + 1e-8)
        self.assertGreaterEqual(y_range[0], viewport.dec_min_deg - 1e-8)
        self.assertLessEqual(y_range[1], viewport.dec_max_deg + 1e-8)
        view.close()

    def test_legacy_log_flag_migrates_and_invalid_roi_is_clamped(self):
        class Harness:
            state = DesktopState(region=SkyRegion(10.0, 20.0, 1.0))

        harness = Harness()
        WorkspacePersistenceMixin._apply_restored_state(
            harness, {"spectrum_log_log": True, "energy_display_scale": 100.0}, []
        )
        self.assertEqual(harness.state.spectrum_scale, "log_log")
        self.assertEqual(harness.state.energy_display_scale, 1.0)
        clamped = harness.state.clamp_rectangle((0.0, 30.0, -90.0, 90.0))
        self.assertEqual(clamped, harness.state.sky_viewport.bounds)

    def test_workspace_snapshot_places_failed_and_pending_references_in_state_json(self):
        item = record("xrism", "resolve", "pending")

        class Harness:
            state = DesktopState(region=SkyRegion(10.0, 20.0, 1.0))
            workspace = SimpleNamespace(
                three_d=SimpleNamespace(camera_state=lambda: None)
            )
            _region_payload = WorkspacePersistenceMixin._region_payload
            _record_payload = staticmethod(WorkspacePersistenceMixin._record_payload)

        harness = Harness()
        harness.state.search_records = {"xrism/resolve/pending": item}
        harness.state.pending_observation_keys = ["xrism/resolve/pending"]
        harness.state.failed_observations = {"xrism/resolve/pending": "failed once"}
        saved = WorkspacePersistenceMixin._workspace_snapshot(harness)["state"]
        self.assertEqual(saved["pending_observations"][0]["record_key"], "xrism/resolve/pending")
        self.assertEqual(saved["failed_observations"][0]["message"], "failed once")


if __name__ == "__main__":
    unittest.main()
