"""Main-window controller for progressive observation load sessions."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QMessageBox

from jaxa_torax.desktop.data_controller import balanced_row_limits
from jaxa_torax.desktop.observation_session import ObservationLoadSession
from jaxa_torax.desktop.science_views import combine_frames


class ObservationLoadingMixin:
    def load_catalog_keys(self, keys: list[str]):
        queued = set(self.state.pending_observation_keys)
        records, seen = [], set()
        for key in keys:
            if key in seen or key in queued or key in self.state.observation_cache:
                continue
            seen.add(key)
            if key in self.state.search_records:
                records.append(self.state.search_records[key])
        if not records:
            self._update_status("Selected observations are already loaded")
            return
        catalog_order = {key: index for index, key in enumerate(self.state.search_records)}
        records.sort(key=lambda item: catalog_order.get(
            f"{item.mission}/{item.instrument}/{item.observation_id}", len(catalog_order)
        ))
        self._start_load_session(records)

    def _start_load_session(self, records):
        if self.state.region is None:
            return
        if self._load_session is not None:
            if self._load_session.region_signature == self._region_signature(self.state.region):
                pending = [
                    record for record in self._load_session.records
                    if f"{record.mission}/{record.instrument}/{record.observation_id}"
                    not in self.state.observation_cache
                ]
                records = pending + list(records)
            self._cancel_load_session()
        all_records = [obs.record for obs in self.state.loaded_observations] + list(records)
        limits = balanced_row_limits(all_records, self.state.combined_preview_maximum)
        limit_by_key = {
            f"{record.mission}/{record.instrument}/{record.observation_id}": limit
            for record, limit in zip(all_records, limits)
        }
        for observation in self.state.loaded_observations:
            key = self.state.record_key(observation)
            limit = limit_by_key[key]
            if len(observation.frame) > limit:
                indexes = np.linspace(0, len(observation.frame) - 1, limit, dtype=int)
                observation.frame = observation.frame.iloc[indexes].reset_index(drop=True)
                observation.displayed_events = len(observation.frame)
        self.state.combined_frame = combine_frames(self.state.loaded_observations)
        current = [self.state.record_key(obs) for obs in self.state.loaded_observations]
        queued = [f"{r.mission}/{r.instrument}/{r.observation_id}" for r in records]
        self._load_order = {key: index for index, key in enumerate(current + queued)}
        session = ObservationLoadSession(
            self.thread_pool, records, self.state.region,
            workspace_id=self.state.workspace_id,
            region_signature=self._region_signature(self.state.region),
            existing_keys=self.state.observation_cache,
            combined_maximum=max(0, self.state.combined_preview_maximum - sum(
                len(obs.frame) for obs in self.state.loaded_observations
            )),
            parent=self,
        )
        self._load_session = session
        self.state.pending_observation_keys = queued
        self._load_refresh_count = 0
        session.observation_loaded.connect(self._observation_loaded)
        session.observation_failed.connect(self._observation_failed)
        session.observation_retrying.connect(self._observation_retrying)
        session.progress_changed.connect(self._load_progress_changed)
        session.session_finished.connect(self._load_session_finished)
        session.session_cancelled.connect(self._load_session_cancelled)
        self.progress.setRange(0, session.total_count)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_load_button.setVisible(True)
        self.view_failures_button.setVisible(False)
        self.retry_failures_button.setVisible(False)
        self._update_status(f"Loading 0 / {session.total_count:,}")
        session.start()

    def _is_current_session(self, session_id: str) -> bool:
        return (
            self._load_session is not None
            and self._load_session.session_id == session_id
            and self._load_session.region_signature == self._region_signature(self.state.region)
        )

    def _observation_loaded(self, session_id: str, observation):
        if not self._is_current_session(session_id):
            return
        key = self.state.record_key(observation)
        if key in self.state.observation_cache:
            return
        self.state.observation_cache[key] = observation
        self.state.loaded_observations.append(observation)
        self.state.loaded_observations.sort(
            key=lambda item: self._load_order.get(self.state.record_key(item), len(self._load_order))
        )
        if self._pending_restore_visibility is None or key in self._pending_restore_visibility:
            self.state.visible_record_keys.add(key)
        self.state.failed_observations.pop(key, None)
        self.left_panel.set_observations(self.state.loaded_observations, self.state.visible_record_keys)
        if self.state.render_mode != "voxels" and self.state.layout_mode != "2d":
            self._publish_progressive_actor(observation)
        self._load_refresh_count += 1
        if self._load_refresh_count >= 5:
            self._flush_progressive_load_refresh()
        elif not self._load_refresh_timer.isActive():
            self._load_refresh_timer.start(250)

    def _publish_progressive_actor(self, observation):
        visible_count = max(1, len(self.state.visible_record_keys))
        density = (
            max(self.state.density_size_strength, self.state.density_opacity_strength)
            if self.state.render_mode == "density" else 0.0
        )
        self.workspace.three_d.publish_observation_actor(
            observation,
            self.state.region.center_ra_deg, self.state.region.center_dec_deg,
            self._effective_3d_energy_band(), self.state.event_color_mode,
            self.state.point_size, self.state.point_opacity,
            (self.state.rgb_centers, self.state.rgb_widths, self.state.rgb_gains,
             self.state.rgb_brightness, self.state.rgb_gamma),
            density, self.state.interactive_point_budget // visible_count,
            reset_camera=len(self.state.loaded_observations) == 1,
        )

    def _observation_failed(self, session_id: str, key: str, message: str):
        if self._is_current_session(session_id):
            self.state.failed_observations[key] = message

    def _observation_retrying(
        self,
        session_id: str,
        key: str,
        attempt: int,
        maximum: int,
        message: str,
    ):
        if self._is_current_session(session_id):
            summary = str(message).replace("\n", " ").strip()
            if len(summary) > 90:
                summary = summary[:87] + "…"
            self._update_status(
                f"Transient load problem — retrying {key} "
                f"({attempt}/{maximum})  │  {summary}"
            )

    def _load_progress_changed(self, session_id: str, completed: int, total: int, current: str):
        if self._is_current_session(session_id):
            self.progress.setValue(completed)
            self._update_status(f"Loading {completed:,} / {total:,}  │  {current}")

    def _flush_progressive_load_refresh(self):
        self._load_refresh_count = 0
        self.state.combined_frame = combine_frames(self.state.loaded_observations)
        self.state.clear_derived_caches()
        self._preview_products.clear()
        self._invalidate_exact()
        self._schedule_viewer()
        self._schedule_analysis("spectrum", "2d", "slices", "summary")
        self._workspace_changed()

    def _load_session_finished(self, session_id: str, successful_keys, failures):
        if not self._is_current_session(session_id):
            return
        session = self._load_session
        self._flush_progressive_load_refresh()
        self.state.pending_observation_keys = []
        self.state.failed_observations.update(
            {key: value["message"] for key, value in failures.items()}
        )
        self.cancel_load_button.setVisible(False)
        self.view_failures_button.setVisible(bool(self.state.failed_observations))
        self.retry_failures_button.setVisible(bool(self.state.failed_observations))
        self.progress.setValue(session.total_count)
        self.progress.setVisible(False)
        self._load_session = None
        self._refresh_after_data_change(reset_camera=False)
        failed = f"  │  {len(failures):,} failed" if failures else ""
        self._update_status(f"Loaded {len(successful_keys):,} / {session.total_count:,}{failed}")
        if self._restoring_workspace:
            self._finish_progressive_restore()
        self._workspace_changed()

    def _cancel_load_session(self):
        if self._load_session is not None:
            self._load_session.cancel()

    def _load_session_cancelled(self, session_id: str):
        if self._load_session is None or self._load_session.session_id != session_id:
            return
        self._load_refresh_timer.stop()
        self._flush_progressive_load_refresh()
        self.state.pending_observation_keys = []
        self._load_session = None
        self.cancel_load_button.setVisible(False)
        self.progress.setVisible(self._busy_count > 0)
        self._update_status("Observation loading cancelled; completed observations were kept")

    def _view_load_failures(self):
        if not self.state.failed_observations:
            return
        rows = []
        for key, message in self.state.failed_observations.items():
            record = self.state.search_records.get(key)
            path = getattr(record, "parquet_path", None)
            cache_state = "cached source present" if path is not None and path.exists() else "source not cached"
            rows.append(
                f"{key}\n  {message}\n  source/cache: "
                f"{getattr(record, 'source', 'unknown')}; {cache_state}"
            )
        box = QMessageBox(self)
        box.setWindowTitle("Observation load failures")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"{len(rows)} observation(s) failed to load.")
        box.setDetailedText("\n\n".join(rows))
        box.exec()

    def _retry_failed_observations(self):
        records = [
            self.state.search_records[key] for key in self.state.failed_observations
            if key in self.state.search_records and key not in self.state.observation_cache
        ]
        if records:
            self._start_load_session(records)

    def _two_d_view_changed(self, product: str, view):
        if self.state.sky_viewport is not None:
            self.state.two_d_zoom[str(product)] = self.state.sky_viewport.clamp_view(view)
            self._workspace_changed()
