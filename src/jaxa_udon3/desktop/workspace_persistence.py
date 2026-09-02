"""Autosave and restore for durable native desktop workspaces."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction

from jaxa_udon3.desktop.science_views import normalize_spectrum_scale
from jaxa_udon3.desktop.state import EnergySlice
from jaxa_udon3.infrastructure import science as backend


class WorkspacePersistenceMixin:
    """Mixin for MainWindow; keeps persistence out of interactive UI methods."""

    def _initialize_workspace_persistence(self):
        self._workspace_save_timer = QTimer(self)
        self._workspace_save_timer.setSingleShot(True)
        self._workspace_save_timer.timeout.connect(self._save_workspace_now)
        self._restoring_workspace = False
        self._pending_restore_visibility: set[str] | None = None
        self._pending_camera_state = None
        self.analysis.spectrum_scale_changed.connect(self._set_spectrum_scale)
        self.inspector.comparison_roi_toggled.connect(self._comparison_roi_toggled)
        self.inspector.comparison_roi_changed.connect(self._set_comparison_rectangle)
        self._workspace_menu = self.menuBar().addMenu("Workspaces")
        self._workspace_menu.aboutToShow.connect(self._rebuild_workspace_menu)
        QTimer.singleShot(0, self._restore_last_workspace)

    def _rebuild_workspace_menu(self):
        self._workspace_menu.clear()
        try:
            workspaces = backend.list_workspaces()
        except Exception as error:
            self._workspace_menu.addAction(f"Workspace list unavailable: {error}").setEnabled(False)
            return
        if not workspaces:
            self._workspace_menu.addAction("No saved workspaces").setEnabled(False)
            return
        for item in workspaces:
            region = item["region"]
            label = (
                f"{item['target_name']}  —  RA {float(region['center_ra_deg']):.3f}, "
                f"DEC {float(region['center_dec_deg']):+.3f}"
            )
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(item["workspace_id"] == self.state.workspace_id)
            action.triggered.connect(lambda _checked=False, key=item["workspace_id"]: self._open_workspace(key))
            self._workspace_menu.addAction(action)

    def _open_workspace(self, workspace_id: str):
        if str(workspace_id) == self.state.workspace_id:
            return
        self._save_workspace_now()
        snapshot = backend.load_workspace(workspace_id)
        if snapshot is None:
            self._update_status("Saved workspace is no longer available")
            return
        self._clear_loaded_observations()
        self._restore_workspace(snapshot)

    def _workspace_changed(self):
        if not self._restoring_workspace:
            self._workspace_save_timer.start(500)

    def _region_payload(self):
        region = self.state.region
        if region is None:
            return None
        return {
            "center_ra_deg": float(region.center_ra_deg),
            "center_dec_deg": float(region.center_dec_deg),
            "radius_deg": float(region.radius_deg),
            "label": str(getattr(region, "label", "")),
            "source": str(getattr(region, "source", "degrees")),
        }

    def _workspace_snapshot(self):
        region = self._region_payload()
        if region is None:
            return None
        workspace_id = self.state.workspace_id or backend.compact_hash({"region": region})
        self.state.workspace_id = workspace_id
        state = {
            "energy_band": list(self.state.energy_band),
            "selected_slice_uid": self.state.selected_slice_uid,
            "layout_mode": self.state.layout_mode,
            "two_d_product": self.state.two_d_product,
            "render_mode": self.state.render_mode,
            "content_mode": self.state.content_mode,
            "global_detail_mode": self.state.global_detail_mode,
            "global_custom_count": self.state.global_custom_count,
            "observation_detail_overrides": self.state.observation_detail_overrides,
            "top_image_mode": self.state.top_image_mode,
            "top_image_source": self.state.top_image_source,
            "top_image_opacity": self.state.top_image_opacity,
            "preview_rows_per_observation": self.state.preview_rows_per_observation,
            "interactive_point_budget": self.state.interactive_point_budget,
            "image_bins": self.state.image_bins,
            "spectrum_bins": self.state.spectrum_bins,
            "spectrum_smoothing_bins": self.state.spectrum_smoothing_bins,
            "spectrum_smooth_visible": self.state.spectrum_smooth_visible,
            "spectrum_scale": self.state.spectrum_scale,
            "auto_image_quality": self.state.auto_image_quality,
            "auto_spectrum_binning": self.state.auto_spectrum_binning,
            "energy_scan_speed_hz": self.state.energy_scan_speed_hz,
            "filter_3d_by_energy": self.state.filter_3d_by_energy,
            "filter_2d_by_energy": self.state.filter_2d_by_energy,
            "spectrum_linked": self.state.spectrum_linked,
            "energy_reference_kev": self.state.energy_reference_kev,
            "energy_display_scale": self.state.energy_display_scale,
            "show_coordinate_triad": self.state.show_coordinate_triad,
            "show_energy_reference_plane": self.state.show_energy_reference_plane,
            "w49b_centroid_surface": self.state.w49b_centroid_surface,
            "show_slice_planes": self.state.show_slice_planes,
            "show_grid_backdrop": self.state.show_grid_backdrop,
            "show_coordinate_values": self.state.show_coordinate_values,
            "show_slice_window": self.state.show_slice_window,
            "camera_preset": self.state.camera_preset,
            "event_color_mode": self.state.event_color_mode,
            "point_size": self.state.point_size,
            "point_opacity": self.state.point_opacity,
            "event_spatial_smoothing_arcmin": self.state.event_spatial_smoothing_arcmin,
            "event_energy_smoothing_kev": self.state.event_energy_smoothing_kev,
            "density_size_strength": self.state.density_size_strength,
            "density_opacity_strength": self.state.density_opacity_strength,
            "image_smoothing_pixels": self.state.image_smoothing_pixels,
            "image_palette": self.state.image_palette,
            "image_stretch": self.state.image_stretch,
            "image_brightness": self.state.image_brightness,
            "image_contrast": self.state.image_contrast,
            "rgb_centers": list(self.state.rgb_centers),
            "rgb_widths": list(self.state.rgb_widths),
            "rgb_gains": list(self.state.rgb_gains),
            "rgb_brightness": self.state.rgb_brightness,
            "rgb_gamma": self.state.rgb_gamma,
            "spatial_voxel_arcmin": self.state.spatial_voxel_arcmin,
            "energy_voxel_kev": self.state.energy_voxel_kev,
            "voxel_spatial_smoothing_arcmin": self.state.voxel_spatial_smoothing_arcmin,
            "voxel_energy_smoothing_kev": self.state.voxel_energy_smoothing_kev,
            "voxel_threshold_fraction": self.state.voxel_threshold_fraction,
            "voxel_energy_source": self.state.voxel_energy_source,
            "voxel_opacity": self.state.voxel_opacity,
            "voxel_show_edges": self.state.voxel_show_edges,
            "sky_viewport": self.state.sky_viewport.to_payload(),
            "two_d_zoom": {key: list(value) for key, value in self.state.two_d_zoom.items()},
            "camera": self.workspace.three_d.camera_state(),
        }
        observations = []
        for observation in self.state.loaded_observations:
            record = observation.record
            key = self.state.record_key(observation)
            observations.append(
                {
                    "record_key": key,
                    "visible": key in self.state.visible_record_keys,
                    "record": {
                        "mission": record.mission,
                        "instrument": record.instrument,
                        "observation_id": record.observation_id,
                        "parquet_path": str(record.parquet_path),
                        "header_path": str(record.header_path),
                        "parquet_url": record.parquet_url,
                        "header_url": record.header_url,
                        "source": record.source,
                    },
                }
            )
        slices = [
            {
                "uid": item.uid,
                "low_kev": item.low_kev,
                "high_kev": item.high_kev,
                "label": item.label,
                "color": item.color,
                "opacity": item.opacity,
                "visible": item.visible,
                "show_plane": item.show_plane,
                "show_points": item.show_points,
            }
            for item in self.state.slices
        ]
        failed_items = [
            {
                "record_key": key,
                "message": message,
                "record": self._record_payload(self.state.search_records[key]),
            }
            for key, message in self.state.failed_observations.items()
            if key in self.state.search_records
        ]
        pending_items = [
            {
                "record_key": key,
                "record": self._record_payload(self.state.search_records[key]),
            }
            for key in self.state.pending_observation_keys
            if key in self.state.search_records
        ]
        # The repository persists the state JSON and normalized completed rows.
        # Keep unfinished references inside state so schema-v1 databases retain them.
        state["failed_observations"] = failed_items
        state["pending_observations"] = pending_items
        return {
            "workspace_id": workspace_id,
            "target_name": self.state.target_name,
            "region": region,
            "state": state,
            "observations": observations,
            "slices": slices,
            "rois": {
                "primary": self.state.spatial_rectangle,
                "comparison": self.state.comparison_rectangle,
            },
            "failed_observations": failed_items,
            "pending_observations": pending_items,
        }

    @staticmethod
    def _record_payload(record):
        return {
            "mission": record.mission,
            "instrument": record.instrument,
            "observation_id": record.observation_id,
            "parquet_path": str(record.parquet_path),
            "header_path": str(record.header_path),
            "parquet_url": record.parquet_url,
            "header_url": record.header_url,
            "source": record.source,
        }

    def _save_workspace_now(self):
        snapshot = self._workspace_snapshot()
        if snapshot is None:
            return
        try:
            backend.save_workspace(snapshot)
        except Exception as error:
            self._update_status(f"Workspace autosave failed: {error}")

    def _restore_last_workspace(self):
        try:
            snapshot = backend.load_active_workspace()
        except Exception as error:
            self._update_status(f"Workspace restore unavailable: {error}")
            return
        if snapshot is not None:
            self._restore_workspace(snapshot)

    def _restore_workspace(self, snapshot: dict):
        self.analysis.spectrum.stop_scan()
        if int(snapshot.get("schema_version", 0)) > backend.WORKSPACE_SCHEMA_VERSION:
            self._update_status("Saved workspace is newer than this version of UDON3")
            return
        region_data = snapshot.get("region") or {}
        try:
            region = backend.SkyRegion(**region_data)
        except (TypeError, ValueError) as error:
            self._update_status(f"Saved workspace has an invalid region: {error}")
            return
        self._restoring_workspace = True
        self.state.workspace_id = str(snapshot["workspace_id"])
        self.state.region = region
        self.state.target_name = str(snapshot.get("target_name") or region.label or "Sky region")
        saved_state = dict(snapshot.get("state") or {})
        self._apply_restored_state(saved_state, list(snapshot.get("slices") or []))
        roi = dict(snapshot.get("rois") or {}).get("primary")
        self.state.spatial_rectangle = self.state.clamp_rectangle(tuple(map(float, roi))) if roi else None
        comparison = dict(snapshot.get("rois") or {}).get("comparison")
        self.state.comparison_rectangle = self.state.clamp_rectangle(tuple(map(float, comparison))) if comparison else None
        self.global_search.setText(self.state.target_name)
        self.left_panel.set_target(self.state.target_name, region)
        self.region_badge.setText(
            f"RA {region.center_ra_deg:.5f}°   DEC {region.center_dec_deg:+.5f}°   "
            f"r={region.radius_deg * 60:.2f}′"
        )
        self._sync_restored_controls()
        self.left_panel.set_slices(self.state.slices, self.state.selected_slice_uid)
        self.analysis.spectrum.set_slices(self.state.slices)
        self._pending_camera_state = dict(snapshot.get("state") or {}).get("camera")
        records = [self._record_from_snapshot(item.get("record", {})) for item in snapshot.get("observations", [])]
        records = [record for record in records if record is not None]
        self._pending_restore_visibility = {
            str(item["record_key"])
            for item in snapshot.get("observations", [])
            if bool(item.get("visible", True))
        }
        failure_items = list(snapshot.get("failed_observations") or saved_state.get("failed_observations") or [])
        pending_items = list(snapshot.get("pending_observations") or saved_state.get("pending_observations") or [])
        self._pending_restore_visibility.update(
            str(item["record_key"]) for item in pending_items if item.get("record_key")
        )
        pending_records = [self._record_from_snapshot(item.get("record", {})) for item in pending_items]
        pending_records = [record for record in pending_records if record is not None]
        reference_records = records + pending_records + [
            record for record in (
                self._record_from_snapshot(item.get("record", {})) for item in failure_items
            ) if record is not None
        ]
        self.state.search_records = {backend.record_key(record): record for record in reference_records}
        self.state.failed_observations = {
            str(item.get("record_key")): str(item.get("message") or "Previous load failed")
            for item in failure_items if item.get("record_key")
        }
        records.extend(
            record for record in pending_records
            if backend.record_key(record) not in {backend.record_key(item) for item in records}
        )
        if not records:
            self._restoring_workspace = False
            self.view_failures_button.setVisible(bool(self.state.failed_observations))
            self.retry_failures_button.setVisible(bool(self.state.failed_observations))
            self._refresh_dataset_summary()
            self._update_status(f"Restored workspace: {self.state.target_name}")
            return
        self._start_load_session(records)

    @staticmethod
    def _record_from_snapshot(data: dict):
        required = ("mission", "instrument", "observation_id", "parquet_path", "header_path")
        if not all(str(data.get(key, "")) for key in required):
            return None
        record = backend.EventFile(
            str(data["mission"]),
            str(data["instrument"]),
            str(data["observation_id"]),
            Path(str(data["parquet_path"])),
            Path(str(data["header_path"])),
            data.get("parquet_url") or None,
            data.get("header_url") or None,
            str(data.get("source") or "remote"),
        )
        return backend.normalized_cache_record(record)

    def _finish_progressive_restore(self):
        self._restoring_workspace = False
        self._pending_restore_visibility = None
        if self._pending_camera_state is not None:
            self.workspace.three_d.restore_camera_state(self._pending_camera_state)
            self._pending_camera_state = None
        self._update_status(f"Restored workspace: {self.state.target_name}")

    def _apply_restored_state(self, values: dict, slices: list[dict]):
        for name in (
            "preview_rows_per_observation", "interactive_point_budget", "image_bins",
            "spectrum_bins", "spectrum_smoothing_bins", "auto_image_quality",
            "spectrum_smooth_visible", "auto_spectrum_binning", "energy_scan_speed_hz",
            "filter_3d_by_energy", "filter_2d_by_energy", "spectrum_linked", "layout_mode", "two_d_product", "render_mode",
            "energy_reference_kev", "energy_display_scale", "show_coordinate_triad", "show_energy_reference_plane",
            "w49b_centroid_surface",
            "show_slice_planes", "show_grid_backdrop", "show_coordinate_values", "show_slice_window", "camera_preset",
            "event_color_mode", "point_size", "point_opacity", "event_spatial_smoothing_arcmin",
            "event_energy_smoothing_kev", "density_size_strength", "density_opacity_strength",
            "content_mode", "global_detail_mode", "global_custom_count",
            "observation_detail_overrides", "top_image_mode", "top_image_source",
            "top_image_opacity",
            "image_smoothing_pixels", "image_palette", "image_stretch", "image_brightness",
            "image_contrast", "rgb_brightness", "rgb_gamma", "spatial_voxel_arcmin",
            "energy_voxel_kev", "voxel_spatial_smoothing_arcmin", "voxel_energy_smoothing_kev",
            "voxel_threshold_fraction", "voxel_energy_source", "voxel_opacity", "voxel_show_edges",
        ):
            if name in values:
                setattr(self.state, name, values[name])
        # Versions before the balanced 3D transform used large absolute
        # multipliers (the W49B preset stored 100×).  Those values no longer
        # describe a useful relative aspect, so migrate them to the automatic
        # balanced default instead of restoring a distorted scene.
        restored_depth = float(self.state.energy_display_scale)
        self.state.energy_display_scale = (
            restored_depth if 0.25 <= restored_depth <= 4.0 else 1.0
        )
        for name in ("energy_band", "rgb_centers", "rgb_widths", "rgb_gains"):
            if name in values:
                setattr(self.state, name, tuple(map(float, values[name])))
        self.state.spectrum_scale = normalize_spectrum_scale(
            values.get("spectrum_scale", "log_log" if values.get("spectrum_log_log") else "linear")
        )
        if self.state.render_mode == "spectral":
            self.state.render_mode = "events"
        zooms = dict(values.get("two_d_zoom") or {})
        self.state.two_d_zoom = {
            str(key): self.state.sky_viewport.clamp_view(tuple(map(float, value)))
            for key, value in zooms.items() if len(value) == 4 and self.state.sky_viewport is not None
        }
        self.state.selected_slice_uid = values.get("selected_slice_uid")
        self.state.slices = [EnergySlice(**item) for item in slices]
        if self.state.selected_slice() is not None:
            self.state.selected_slice_uid = self.state.selected_slice().uid

    def _sync_restored_controls(self):
        self.workspace.set_layout_mode(self.state.layout_mode)
        self.workspace.set_two_d_product(self.state.two_d_product)
        if self.state.two_d_product == "rgb":
            self.inspector.set_page("rgb")
        elif self.state.two_d_product == "slice" and self.state.selected_slice() is not None:
            self.inspector.set_page("slice")
        elif self.state.two_d_product == "energy":
            self.inspector.set_page("energy")
        self.workspace.set_sky_viewport(self.state.sky_viewport, self.state.two_d_zoom)
        self.left_panel.set_render_mode(self.state.render_mode)
        self.inspector.set_display_state(self.state)
        self.inspector.set_energy_state(self.state.energy_band, self.state.auto_image_quality, self.state.image_smoothing_pixels)
        self.inspector.set_energy_filter_state(
            self.state.filter_3d_by_energy, self.state.filter_2d_by_energy
        )
        self.inspector.set_spectrum_link_state(self.state.spectrum_linked)
        self.inspector.set_scalar_display_state(self.state)
        self.inspector.set_rgb_state(self.state)
        self.inspector.set_voxel_state(self.state)
        self.analysis.spectrum.set_scale(self.state.spectrum_scale)
        self.analysis.spectrum.set_smooth_visible(self.state.spectrum_smooth_visible)
        self.analysis.spectrum.set_scan_speed(self.state.energy_scan_speed_hz)
        self.workspace.set_roi(self.state.spatial_rectangle)
        self.inspector.set_roi(self.state.spatial_rectangle)
        self.inspector.set_comparison_roi(self.state.comparison_rectangle)
        self.inspector.roi_enabled.blockSignals(True)
        self.inspector.roi_enabled.setChecked(self.state.spatial_rectangle is not None)
        self.inspector.roi_enabled.blockSignals(False)
        self.workspace.roi_button.blockSignals(True)
        self.workspace.roi_button.setChecked(self.state.spatial_rectangle is not None)
        self.workspace.roi_button.blockSignals(False)
        self.workspace._roi_toggled(self.state.spatial_rectangle is not None)
        self.workspace.set_roi(self.state.spatial_rectangle)
        self.inspector.roi_compare_enabled.blockSignals(True)
        self.inspector.roi_compare_enabled.setChecked(self.state.comparison_rectangle is not None)
        self.inspector.roi_compare_enabled.blockSignals(False)

    def _set_spectrum_scale(self, scale: str):
        self.state.spectrum_scale = normalize_spectrum_scale(scale)
        self._workspace_changed()

    def _set_comparison_rectangle(self, rectangle):
        self.state.comparison_rectangle = self.state.clamp_rectangle(tuple(map(float, rectangle)))
        self.inspector.set_comparison_roi(self.state.comparison_rectangle)
        self.state.clear_derived_caches()
        self._schedule_analysis("spectrum")
        self._workspace_changed()

    def _comparison_roi_toggled(self, enabled: bool):
        if not enabled:
            self.state.comparison_rectangle = None
            self.state.clear_derived_caches()
            self._schedule_analysis("spectrum")
        else:
            self._set_comparison_rectangle((
                self.inspector.roi_b_ra_min.value(), self.inspector.roi_b_ra_max.value(),
                self.inspector.roi_b_dec_min.value(), self.inspector.roi_b_dec_max.value(),
            ))
        self._workspace_changed()

    def closeEvent(self, event):
        self.analysis.spectrum.stop_scan()
        if hasattr(self, "_workspace_save_timer"):
            self._workspace_save_timer.stop()
            if not getattr(self, "_skip_workspace_save_on_close", False):
                self._save_workspace_now()
        self._cancel_load_session()
        super().closeEvent(event)
