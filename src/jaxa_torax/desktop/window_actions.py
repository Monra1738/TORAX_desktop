from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QProcess, QThreadPool
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from jaxa_torax.desktop.app_reset import clear_application_storage
from jaxa_torax.desktop.data_controller import (
    exact_all_events,
    exact_energy,
    exact_rgb,
    export_preview_frame,
)
from jaxa_torax.desktop.science_views import ImageProduct, rgb_from_channels
from jaxa_torax.desktop.screenshot import save_widget_screenshot, show_screenshot_preview


class WindowActionsMixin:
    """Exact-product, export, screenshot and presentation actions."""

    def _real_records(self):
        return [obs.record for obs in self.state.visible_observations()]

    def reset_application(self):
        answer = QMessageBox.warning(
            self,
            "Reset TORAX completely",
            "This removes downloaded cache, saved workspaces and logs, then restarts "
            "TORAX as a new application. Exports and local source data are preserved.",
            QMessageBox.Reset | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Reset:
            return

        self.analysis.spectrum.stop_scan()
        self._viewer_timer.stop()
        self._analysis_timer.stop()
        self._load_refresh_timer.stop()
        self._cancel_load_session()
        pool = QThreadPool.globalInstance()
        pool.clear()
        if not pool.waitForDone(10_000):
            QMessageBox.critical(
                self, "Reset postponed", "A data operation is still running. Try again shortly."
            )
            return

        self._skip_workspace_save_on_close = True
        if hasattr(self, "_workspace_save_timer"):
            self._workspace_save_timer.stop()
        self._clear_loaded_observations()
        try:
            clear_application_storage()
        except Exception as error:
            self._skip_workspace_save_on_close = False
            QMessageBox.critical(self, "Reset failed", str(error))
            return

        started, _process_id = QProcess.startDetached(
            sys.executable,
            ["-m", "jaxa_torax.desktop.app"],
            str(Path.cwd()),
        )
        if not started:
            QMessageBox.information(
                self, "Reset complete", "The app is clean. Close it and launch TORAX again."
            )
        QApplication.quit()

    def compute_exact_energy(self):
        records = self._real_records()
        if not records:
            QMessageBox.information(self, "Exact image", "Load one or more DARTS observations first.")
            return
        # An exact energy request is defined for the selected band, so make the
        # 2D scope explicit instead of silently replacing the all-energy preview.
        self.state.filter_2d_by_energy = True
        self.inspector.set_energy_filter_state(
            self.state.filter_3d_by_energy, self.state.filter_2d_by_energy
        )
        low, high = self.state.energy_band
        self._run(
            exact_energy, records, self.state.region, low, high, self.state.image_bins,
            status="Computing exact energy image…", on_result=self._exact_energy_done,
            report_progress=True,
        )

    def _exact_energy_done(self, data):
        product = self._exact_energy_product(data)
        self.state.energy_image_exact = True
        self.state.energy_image_exact_scope = "band"
        self.state.energy_image_cache_hit = bool(data.get("cache_hit"))
        self._exact_energy_data = data
        self.state.two_d_product = "energy"
        self.workspace.set_two_d_product("energy")
        if self.state.layout_mode == "3d":
            self.state.layout_mode = "split"
            self.workspace.set_layout_mode("split")
        self.inspector.set_page("energy")
        self._display_energy_product(product)
        failures = len(data.get("failures", {}))
        suffix = f"; {failures} source(s) skipped" if failures else ""
        self._update_status(f"Exact energy image ready and displayed{suffix}")
        self._workspace_changed()

    def compute_exact_all_events(self):
        records = self._real_records()
        if not records:
            QMessageBox.information(
                self, "All events", "Load one or more DARTS observations first."
            )
            return
        self.state.filter_2d_by_energy = False
        self.inspector.set_energy_filter_state(
            self.state.filter_3d_by_energy, self.state.filter_2d_by_energy
        )
        self._run(
            exact_all_events,
            records,
            self.state.region,
            self.state.image_bins,
            status="Compressing every matching parquet event into RA/DEC…",
            on_result=self._exact_all_events_done,
            report_progress=True,
        )

    def _exact_all_events_done(self, data):
        product = self._exact_energy_product(data)
        self.state.energy_image_exact = True
        self.state.energy_image_exact_scope = "all_events"
        self.state.energy_image_cache_hit = bool(data.get("cache_hit"))
        self._exact_energy_data = data
        self.state.two_d_product = "energy"
        self.workspace.set_two_d_product("energy")
        if self.state.layout_mode == "3d":
            self.state.layout_mode = "split"
            self.workspace.set_layout_mode("split")
        self.inspector.set_page("energy")
        self._display_energy_product(product)
        failures = len(data.get("failures", {}))
        suffix = f"; {failures} source(s) skipped" if failures else ""
        self._update_status(
            f"Exact all-event image ready: {int(data['event_count']):,} events{suffix}"
        )
        self._workspace_changed()

    @staticmethod
    def _exact_energy_product(data):
        return ImageProduct(
            np.asarray(data["hist"], float), np.asarray(data["x_edges"], float),
            np.asarray(data["y_edges"], float), int(data["event_count"]),
            float(data["low_kev"]), float(data["high_kev"]), True,
            bool(data.get("cache_hit")),
        )

    def compute_exact_rgb(self):
        records = self._real_records()
        if not records:
            QMessageBox.information(self, "Exact RGB", "Load one or more DARTS observations first.")
            return
        self._run(
            exact_rgb, records, self.state.region, self.state.rgb_centers, self.state.rgb_widths,
            self.state.image_bins, status="Computing exact RGB composite…",
            on_result=self._exact_rgb_done,
        )

    def _exact_rgb_done(self, data):
        rgb = rgb_from_channels(
            np.asarray(data["channels"], float), self.state.rgb_gains,
            self.state.rgb_brightness, self.state.rgb_gamma,
        )
        self.state.rgb_image_exact = True
        self.state.rgb_image_cache_hit = bool(data.get("cache_hit"))
        self._exact_rgb_data = data
        self.state.two_d_product = "rgb"
        self.workspace.set_two_d_product("rgb")
        if self.state.layout_mode == "3d":
            self.state.layout_mode = "split"
            self.workspace.set_layout_mode("split")
        self.inspector.set_page("rgb")
        self._display_rgb_product(rgb, data["x_edges"], data["y_edges"], data["event_counts"])
        self._update_status("Exact RGB composite ready and displayed")
        self._workspace_changed()

    def export_preview(self):
        frame = self.state.displayed_frame()
        if frame.empty:
            QMessageBox.information(self, "Export", "Load data before exporting a preview.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export visible preview", "torax_visible_events.csv",
            "CSV (*.csv);;Parquet (*.parquet)",
        )
        if not path:
            return
        try:
            export_preview_frame(frame, path)
            self._update_status(f"Exported {len(frame):,} preview rows to {path}")
        except Exception as error:
            QMessageBox.critical(self, "Export failed", str(error))

    def save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save viewer screenshot", "torax_view.png", "PNG image (*.png)"
        )
        if not path:
            return
        try:
            saved = save_widget_screenshot(self.workspace, path)
            self._update_status(f"Screenshot saved to {saved} ({saved.stat().st_size:,} bytes)")
            show_screenshot_preview(self, saved)
        except Exception as error:
            QMessageBox.critical(self, "Screenshot failed", str(error))

    def toggle_viewer_fullscreen(self):
        self._viewer_fullscreen = not self._viewer_fullscreen
        for dock in (self.left_dock, self.right_dock, self.bottom_dock):
            dock.setVisible(not self._viewer_fullscreen)
        self.main_toolbar.setVisible(not self._viewer_fullscreen)
        self.menuBar().setVisible(not self._viewer_fullscreen)
        self.statusBar().setVisible(not self._viewer_fullscreen)

    def _show_help(self):
        QMessageBox.about(
            self,
            "About TORAX",
            "TORAX — DARTS X-ray Event Explorer\n\n"
            "Native desktop application using PySide6, PyVista/VTK and PyQtGraph.\n"
            "RA increases toward the left in astronomical 2D views.\n\n"
            "Loaded observations remain cached in memory; visibility toggles do not reload parquet data.",
        )
