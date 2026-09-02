from __future__ import annotations

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QToolBar,
    QWidget,
)

from jaxa_udon3.desktop.analysis import AnalysisDock
from jaxa_udon3.desktop.background_work import BackgroundWorkMixin
from jaxa_udon3.desktop.data_controller import search_catalog
from jaxa_udon3.desktop.inspector import InspectorPanel
from jaxa_udon3.desktop.main_refresh import RefreshMixin
from jaxa_udon3.desktop.observation_loading import ObservationLoadingMixin
from jaxa_udon3.desktop.panels import DataLayersPanel, ObservationBrowserDialog, SearchDialog
from jaxa_udon3.desktop.science_views import combine_frames
from jaxa_udon3.desktop.state import DesktopState
from jaxa_udon3.desktop.theme import APP_QSS, JAXA_BLUE_BRIGHT
from jaxa_udon3.desktop.viewer_3d_helpers import apply_camera_preset
from jaxa_udon3.desktop.viewers import WorkspaceWidget
from jaxa_udon3.desktop.window_actions import WindowActionsMixin
from jaxa_udon3.desktop.workspace_persistence import WorkspacePersistenceMixin


class MainWindow(BackgroundWorkMixin, ObservationLoadingMixin, WorkspacePersistenceMixin, RefreshMixin, WindowActionsMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UDON3 — DARTS X-ray Event Explorer")
        self.resize(1640, 1000)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(APP_QSS)
        self.state = DesktopState()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(max(2, min(4, self.thread_pool.maxThreadCount())))
        self._exact_energy_data = None
        self._exact_rgb_data = None
        self._busy_count = 0
        self._viewer_fullscreen = False
        self._search_results_dialog = None
        self._pending_refresh: set[str] = set()
        self._load_session = None
        self._load_order: dict[str, int] = {}
        self._load_refresh_count = 0
        self._preview_products: dict[tuple, object] = {}
        self._viewer_timer = QTimer(self)
        self._viewer_timer.setSingleShot(True)
        self._viewer_timer.timeout.connect(self._refresh_3d)
        self._analysis_timer = QTimer(self)
        self._analysis_timer.setSingleShot(True)
        self._analysis_timer.timeout.connect(self._refresh_analysis_products)
        self._load_refresh_timer = QTimer(self)
        self._load_refresh_timer.setSingleShot(True)
        self._load_refresh_timer.timeout.connect(self._flush_progressive_load_refresh)
        self._build_topbar()
        self._build_workspace()
        self._build_statusbar()
        self._build_menus()
        self._connect_signals()
        self.workspace.set_layout_mode("split")
        self.workspace.set_two_d_product("energy")
        self.inspector.set_page("energy")
        self._update_status("Ready — search DARTS observations")
        self._initialize_workspace_persistence()
    def _build_topbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setMinimumHeight(54)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        container = QFrame()
        container.setObjectName("topBar")
        row = QHBoxLayout(container)
        row.setContentsMargins(14, 6, 14, 6)
        brand = QLabel("JAXA  UDON3")
        brand.setObjectName("appTitle")
        brand.setStyleSheet(f"color:{JAXA_BLUE_BRIGHT};")
        row.addWidget(brand)
        subtitle = QLabel("DARTS X-ray Event Explorer")
        subtitle.setObjectName("subtitle")
        row.addWidget(subtitle)
        row.addSpacing(24)
        self.global_search = QLineEdit()
        self.global_search.setPlaceholderText("Search target name (e.g. Cas A, SN 1006)…")
        self.global_search.setMinimumWidth(300)
        self.global_search.setMaximumWidth(460)
        self.global_search.returnPressed.connect(self._search_from_topbar)
        row.addWidget(self.global_search)
        self.region_badge = QLabel("No fixed search region")
        self.region_badge.setObjectName("muted")
        row.addWidget(self.region_badge, 1)
        row.addStretch(1)
        for text, callback in (
            ("Search / Add", self.open_search),
            ("Export", self.export_preview),
            ("Reset app", self.reset_application),
            ("Help", self._show_help),
        ):
            button = QPushButton(text)
            if text == "Search / Add":
                button.setObjectName("primary")
            elif text == "Reset app":
                button.setObjectName("danger")
            button.clicked.connect(callback)
            row.addWidget(button)
        toolbar.addWidget(container)
        self.main_toolbar = toolbar
    def _dock(self, title: str, widget: QWidget, area, width=None, height=None):
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower().replace(' ', '_')}")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        if width:
            dock.setMinimumWidth(width)
        if height:
            dock.setMinimumHeight(height)
        return dock
    def _build_workspace(self):
        self.workspace = WorkspaceWidget()
        self.setCentralWidget(self.workspace)
        self.left_panel = DataLayersPanel()
        self.inspector = InspectorPanel()
        self.analysis = AnalysisDock()
        self.left_dock = self._dock("Data & Layers", self.left_panel, Qt.LeftDockWidgetArea, 265)
        self.right_dock = self._dock("Inspector", self.inspector, Qt.RightDockWidgetArea, 305)
        self.bottom_dock = self._dock("Analysis", self.analysis, Qt.BottomDockWidgetArea, height=245)
        self.resizeDocks([self.left_dock, self.right_dock], [280, 320], Qt.Horizontal)
        self.resizeDocks([self.bottom_dock], [270], Qt.Vertical)
    def _build_statusbar(self):
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.status_text = QLabel("Ready")
        self.status_text.setObjectName("muted")
        self.status_metrics = QLabel("0 loaded  │  0 visible  │  — keV")
        self.status_metrics.setObjectName("muted")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(150)
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.cancel_load_button = QPushButton("Cancel")
        self.cancel_load_button.clicked.connect(self._cancel_load_session)
        self.cancel_load_button.setVisible(False)
        self.view_failures_button = QPushButton("View failures")
        self.view_failures_button.clicked.connect(self._view_load_failures)
        self.view_failures_button.setVisible(False)
        self.retry_failures_button = QPushButton("Retry failed")
        self.retry_failures_button.clicked.connect(self._retry_failed_observations)
        self.retry_failures_button.setVisible(False)
        bar.addWidget(self.status_text, 1)
        bar.addPermanentWidget(self.cancel_load_button)
        bar.addPermanentWidget(self.view_failures_button)
        bar.addPermanentWidget(self.retry_failures_button)
        bar.addPermanentWidget(self.progress)
        bar.addPermanentWidget(self.status_metrics)
    def _build_menus(self):
        file_menu = self.menuBar().addMenu("File")
        for text, callback, shortcut in (
            ("Search / add observations…", self.open_search, "Ctrl+K"),
            ("Export visible preview…", self.export_preview, "Ctrl+E"),
            ("Save viewer screenshot…", self.save_screenshot, "Ctrl+Shift+S"),
        ):
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(callback)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.left_dock.toggleViewAction())
        view_menu.addAction(self.right_dock.toggleViewAction())
        view_menu.addAction(self.bottom_dock.toggleViewAction())
    def _connect_signals(self):
        self.left_panel.search_clicked.connect(self.open_search)
        self.left_panel.visibility_changed.connect(self._set_visibility)
        self.left_panel.visibility_many_changed.connect(self._set_visibility_many)
        self.left_panel.observation_selected.connect(self._observation_selected)
        self.left_panel.render_mode_changed.connect(self._set_render_mode)
        self.left_panel.slice_add_requested.connect(self._add_slice)
        self.left_panel.slice_preset_requested.connect(self._add_slice_preset)
        self.left_panel.slice_selected.connect(self._select_slice)
        self.left_panel.slice_visibility_changed.connect(self._set_slice_visibility)
        self.left_panel.slice_remove_requested.connect(self._remove_slice)
        self.workspace.event_selected.connect(self._select_event)
        self.workspace.rectangle_changed.connect(self._set_rectangle)
        self.workspace.layout_changed.connect(self._layout_changed)
        self.workspace.product_changed.connect(self._product_changed)
        self.workspace.screenshot_requested.connect(self.save_screenshot)
        self.workspace.fullscreen_requested.connect(self.toggle_viewer_fullscreen)
        self.workspace.view_changed.connect(self._two_d_view_changed)
        self.analysis.band_changed.connect(self._set_energy_band)
        self.analysis.slice_changed.connect(self._move_slice)
        self.analysis.spectrum_settings_changed.connect(self._spectrum_settings_changed)
        self.analysis.spectrum_smooth_visibility_changed.connect(
            self._spectrum_smooth_visibility_changed
        )
        self.analysis.energy_scan_speed_changed.connect(self._energy_scan_speed_changed)
        self.analysis.tab_changed.connect(self._analysis_tab_changed)
        self.inspector.display_changed.connect(self._display_controls_changed)
        self.inspector.slice_points_changed.connect(self._set_slice_points)
        self.inspector.top_image_changed.connect(self._top_image_changed)
        self.inspector.image_quality_changed.connect(self._image_quality_changed)
        self.inspector.scalar_display_changed.connect(self._scalar_display_changed)
        self.inspector.energy_changed.connect(self._set_energy_band_from_inspector)
        self.inspector.energy_filter_scope_changed.connect(self._energy_filter_scope_changed)
        self.inspector.spectrum_link_changed.connect(self._spectrum_link_changed)
        self.inspector.rgb_changed.connect(self._rgb_controls_changed)
        self.inspector.voxel_changed.connect(self._voxel_controls_changed)
        self.inspector.energy_geometry_changed.connect(self._energy_geometry_changed)
        self.inspector.w49b_preset_requested.connect(self._apply_w49b_preset)
        self.inspector.roi_toggled.connect(self._roi_toggled)
        self.inspector.roi_changed.connect(self._set_rectangle)
        self.inspector.slice_changed.connect(self._edit_slice)
        self.inspector.slice_color_changed.connect(self._set_slice_color)
        self.inspector.observation_remove_requested.connect(self._remove_observation)
        self.inspector.exact_energy_requested.connect(self.compute_exact_energy)
        self.inspector.exact_all_events_requested.connect(self.compute_exact_all_events)
        self.inspector.exact_rgb_requested.connect(self.compute_exact_rgb)
    def open_search(self):
        radius = self.state.region.radius_deg * 60.0 if self.state.region is not None else 10.0
        dialog = SearchDialog(self, target=self.global_search.text().strip(), radius_arcmin=radius, region=self.state.region)
        dialog.search_requested.connect(self._start_search)
        dialog.exec()
    def _search_from_topbar(self):
        name = self.global_search.text().strip()
        if not name:
            self.open_search()
            return
        radius = self.state.region.radius_deg * 60.0 if self.state.region is not None else 10.0
        self._start_search({
            "mode": "target", "target_name": name, "ra_value": "", "dec_value": "",
            "radius_arcmin": radius,
            "selected_pairs": [
                "xrism/resolve", "xrism/xtend", "hitomi/sxs", "hitomi/sxi", "hitomi/hxi",
                "suzaku/xis", "asca/gis", "asca/sis",
            ],
            "object_text": "", "observation_text": "", "date_start": "", "date_end": "",
            "limit": 500,
        })
    def _start_search(self, payload: dict):
        self.analysis.spectrum.stop_scan()
        self._run(
            search_catalog, status="Resolving target and searching DARTS catalog…",
            on_result=self._search_done, **payload,
        )
    def _region_signature(self, region):
        if region is None:
            return None
        return (
            round(float(region.center_ra_deg), 5), round(float(region.center_dec_deg), 5),
            round(float(region.radius_deg), 5),
        )
    def _search_done(self, result):
        changed_region = (
            self.state.region is not None
            and self._region_signature(self.state.region) != self._region_signature(result.region)
        )
        if changed_region:
            self._cancel_load_session()
            self.state.two_d_zoom.clear()
        if changed_region:
            self._save_workspace_now()
            self._clear_loaded_observations()
            self._update_status("New target region selected; previous workspace observations cleared")
        self.state.region = result.region
        self.workspace.set_sky_viewport(self.state.sky_viewport, self.state.two_d_zoom)
        self.state.region_cache_hit = result.region_cache_hit
        self.state.target_name = result.target_label
        if changed_region or not self.state.workspace_id:
            self.state.workspace_id = ""
        self.state.search_frame = result.frame
        self.state.search_records = result.records_by_key
        self.global_search.setText(result.target_label)
        self.left_panel.set_target(result.target_label, result.region, result.region_cache_hit)
        self.region_badge.setText(
            f"RA {result.region.center_ra_deg:.5f}°   DEC {result.region.center_dec_deg:+.5f}°   "
            f"r={result.region.radius_deg * 60:.2f}′"
        )
        dialog = ObservationBrowserDialog(
            result.frame, set(self.state.observation_cache), parent=self,
            queued_keys=self.state.pending_observation_keys,
        )
        dialog.add_requested.connect(self.load_catalog_keys)
        self._search_results_dialog = dialog
        dialog.exec()
        self._refresh_dataset_summary()
        if self._load_session is None:
            self._update_status(f"Found {len(result.frame):,} observations around {result.target_label}")
        self._workspace_changed()
    def _invalidate_exact(self):
        self.state.energy_image_exact = False
        self.state.energy_image_exact_scope = "band"
        self.state.rgb_image_exact = False
        self._exact_energy_data = None
        self._exact_rgb_data = None
    def _set_visibility(self, key: str, visible: bool):
        if visible:
            self.state.visible_record_keys.add(key)
        else:
            self.state.visible_record_keys.discard(key)
        self.state.clear_derived_caches()
        self._invalidate_exact()
        self.workspace.three_d.set_record_visibility(key, visible, self.state.render_mode)
        self.left_panel.sync_visibility_states(self.state.visible_record_keys)
        if visible:
            self._schedule_viewer()
        self._schedule_analysis("spectrum", "2d", "slices")
        self._refresh_dataset_summary()
        self._workspace_changed()
    def _set_visibility_many(self, keys, visible: bool):
        keys = [str(key) for key in keys if str(key) in self.state.observation_cache]
        if not keys:
            return
        if visible:
            self.state.visible_record_keys.update(keys)
        else:
            self.state.visible_record_keys.difference_update(keys)
        self.state.clear_derived_caches()
        self._invalidate_exact()
        self.workspace.three_d.set_records_visibility(
            keys, visible, self.state.render_mode, render=True
        )
        self.left_panel.sync_visibility_states(self.state.visible_record_keys)
        if visible:
            self._schedule_viewer()
        self._schedule_analysis("spectrum", "2d", "slices")
        self._refresh_dataset_summary()
        self._workspace_changed()
    def _set_render_mode(self, mode: str):
        if mode == "spectral":
            mode = "events"
        self.state.render_mode = mode
        self.inspector.set_page("voxels" if mode == "voxels" else "events")
        status = {"voxels": "Building voxel preview…", "density": "Showing density event preview"}
        self._update_status(status.get(mode, "Showing event preview"))
        self._schedule_viewer()
        self._workspace_changed()
    def _layout_changed(self, mode: str):
        self.state.layout_mode = mode
        if mode != "2d":
            self._refresh_3d(reset_camera=False)
        if mode != "3d":
            self._refresh_current_2d()
        self._workspace_changed()
    def _product_changed(self, product: str):
        self.state.two_d_product = product
        if product == "rgb":
            self.inspector.set_page("rgb")
        elif product in ("energy", "slice"):
            self.inspector.set_page("slice" if product == "slice" and self.state.selected_slice() else "energy")
        self._refresh_current_2d()
        self._workspace_changed()
    def _set_rectangle(self, rectangle):
        self.state.spatial_rectangle = self.state.clamp_rectangle(tuple(map(float, rectangle)))
        self.state.clear_derived_caches()
        self.inspector.set_roi(self.state.spatial_rectangle)
        self.workspace.set_roi(self.state.spatial_rectangle)
        self._schedule_analysis("spectrum")
        self._workspace_changed()
    def _roi_toggled(self, enabled: bool):
        self.workspace.roi_button.blockSignals(True)
        self.workspace.roi_button.setChecked(enabled)
        self.workspace.roi_button.blockSignals(False)
        self.workspace._roi_toggled(enabled)
        if not enabled:
            self.state.spatial_rectangle = None
            self.state.clear_derived_caches()
            self._schedule_analysis("spectrum")
        self._workspace_changed()
    def _spectrum_settings_changed(self, bins: int, smoothing: float, automatic: bool):
        self.state.spectrum_bins = int(bins)
        self.state.spectrum_smoothing_bins = float(smoothing)
        self.state.auto_spectrum_binning = bool(automatic)
        self._refresh_spectrum()
        self._workspace_changed()
    def _spectrum_smooth_visibility_changed(self, visible: bool):
        self.state.spectrum_smooth_visible = bool(visible)
        self._workspace_changed()
    def _energy_scan_speed_changed(self, speed: int):
        self.state.energy_scan_speed_hz = max(1, int(speed))
        self._workspace_changed()
    def _analysis_tab_changed(self, key: str):
        if key in ("images", "profile"):
            self._refresh_slice_products(force_all=True)
        elif key == "spectrum":
            self._refresh_spectrum()
    def _set_energy_band(self, low: float, high: float):
        low, high = sorted((float(low), float(high)))
        if high <= low:
            return
        self.state.energy_band = (low, high)
        self._invalidate_exact()
        for widget, value in ((self.inspector.energy_low, low), (self.inspector.energy_high, high)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.inspector._sync_energy_sliders(low, high)
        self.analysis.spectrum.set_band(low, high)
        # The global band always drives All events and the 3D scan. Active-slice
        # and voxel modes cheaply no-op here when their own source is selected.
        self._schedule_viewer()
        if self.state.spectrum_linked or self.state.filter_2d_by_energy:
            self._schedule_analysis("2d")
        self._schedule_analysis("summary")
        self._workspace_changed()
    def _set_energy_band_from_inspector(self, low: float, high: float):
        self.analysis.spectrum.stop_scan()
        if self.analysis.spectrum.scan_lock_width.isChecked():
            old_low, old_high = self.state.energy_band
            width = old_high - old_low
            if abs(float(low) - old_low) >= abs(float(high) - old_high):
                high = float(low) + width
            else:
                low = float(high) - width
            limits = self.state.energy_range()
            if limits is not None:
                minimum, maximum = limits
                width = min(width, max(0.001, maximum - minimum))
                if low < minimum:
                    low, high = minimum, minimum + width
                if high > maximum:
                    low, high = maximum - width, maximum
        self._set_energy_band(low, high)
    def _energy_filter_scope_changed(self, filter_3d: bool, filter_2d: bool):
        changed_3d = self.state.filter_3d_by_energy != bool(filter_3d)
        changed_2d = self.state.filter_2d_by_energy != bool(filter_2d)
        self.state.filter_3d_by_energy, self.state.filter_2d_by_energy = bool(filter_3d), bool(filter_2d)
        if changed_3d:
            self._schedule_viewer()
        if changed_2d:
            self._schedule_analysis("2d")
        self._schedule_analysis("summary")
        self._workspace_changed()
    def _display_controls_changed(self):
        self.state.event_color_mode = self.inspector.color_mode.currentData()
        self.state.content_mode = str(self.inspector.content_mode.currentData() or "all")
        self.state.show_slice_planes = self.inspector.show_slice_planes.isChecked()
        self.state.show_coordinate_triad = self.inspector.show_coordinate_triad.isChecked()
        self.state.show_grid_backdrop = self.inspector.show_grid_backdrop.isChecked()
        self.state.show_coordinate_values = self.inspector.show_coordinate_values.isChecked()
        self.state.show_slice_window = self.inspector.show_slice_window.isChecked()
        preset = str(self.inspector.camera_preset.currentData() or "isometric")
        if preset != self.state.camera_preset:
            self.state.camera_preset = preset
            apply_camera_preset(self.workspace.three_d, preset)
        self.state.point_size = self.inspector.point_size.value()
        self.state.point_opacity = self.inspector.opacity.value()
        self.state.event_spatial_smoothing_arcmin = self.inspector.event_spatial.value()
        self.state.event_energy_smoothing_kev = self.inspector.event_energy.value()
        self.state.density_size_strength = self.inspector.density_size.value()
        self.state.density_opacity_strength = self.inspector.density_opacity.value()
        self._schedule_viewer()
        if self.state.two_d_product == "sky":
            self._schedule_analysis("2d")
        self._workspace_changed()
    def _energy_geometry_changed(self, reference, scale, triad, reference_plane):
        self.state.energy_reference_kev = float(reference)
        self.state.energy_display_scale = float(scale)
        self.state.show_coordinate_triad = bool(triad)
        self.state.show_energy_reference_plane = bool(reference_plane)
        self._schedule_viewer()
        self._workspace_changed()
    def _apply_w49b_preset(self):
        self.state.energy_reference_kev = 6.700
        self.state.energy_display_scale = 1.0
        self.state.show_coordinate_triad = True
        self.state.show_energy_reference_plane = True
        self.state.w49b_centroid_surface = True
        self.global_search.setText("W49B")
        self.state.energy_band = (6.685, 6.715)
        self.state.slices = []
        self.state.add_slice(6.6975, 6.7000, label="W49B blue shift · 6.6975–6.7000 keV", color="#278cff")
        self.state.add_slice(6.7000, 6.7025, label="W49B red shift · 6.7000–6.7025 keV", color="#ff5b67")
        self.state.add_slice(6.6995, 6.7005, label="Fe XXV He-α reference · 6.700 keV", color="#ffd166", opacity=0.28)
        self.left_panel.set_slices(self.state.slices, self.state.selected_slice_uid)
        self.analysis.spectrum.set_slices(self.state.slices)
        self.inspector.set_display_state(self.state)
        self.inspector.set_energy_state(self.state.energy_band, self.state.auto_image_quality, self.state.image_smoothing_pixels)
        self._schedule_viewer()
        self._workspace_changed()
        self._update_status("W49B preset ready: search W49B, load Resolve 300055010/300056010, then choose Voxels")
    def _top_image_changed(self, mode: str, opacity: float, source: str):
        self.state.top_image_mode = str(mode)
        self.state.top_image_opacity = float(opacity)
        self.state.top_image_source = str(source)
        self._refresh_3d(reset_camera=False)
        self._workspace_changed()
    def _image_quality_changed(self):
        self.state.image_smoothing_pixels = self.inspector.image_smoothing.value()
        self.state.auto_image_quality = self.inspector.auto_image_quality.isChecked()
        self._schedule_analysis("2d", "slices")
        self._workspace_changed()
    def _rgb_controls_changed(self):
        centers = tuple(trio[0].value() for trio in self.inspector.rgb_controls)
        widths = tuple(trio[1].value() for trio in self.inspector.rgb_controls)
        bands_changed = centers != self.state.rgb_centers or widths != self.state.rgb_widths
        self.state.rgb_centers = centers
        self.state.rgb_widths = widths
        self.state.rgb_gains = tuple(trio[2].value() for trio in self.inspector.rgb_controls)
        self.state.rgb_brightness = self.inspector.rgb_brightness.value()
        self.state.rgb_gamma = self.inspector.rgb_gamma.value()
        if not self.state.auto_image_quality:
            self.state.image_smoothing_pixels = self.inspector.rgb_smoothing.value()
        if bands_changed:
            self.state.rgb_image_exact = False
            self._exact_rgb_data = None
        if self.state.event_color_mode == "rgb":
            self._schedule_viewer()
        if self.state.two_d_product == "rgb" or (
            self.state.two_d_product == "sky" and self.state.event_color_mode == "rgb"
        ):
            self._schedule_analysis("2d")
        self._workspace_changed()
    def _voxel_controls_changed(self):
        self.state.spatial_voxel_arcmin = self.inspector.voxel_spatial.value()
        self.state.energy_voxel_kev = self.inspector.voxel_energy.value()
        self.state.voxel_spatial_smoothing_arcmin = self.inspector.voxel_smooth_spatial.value()
        self.state.voxel_energy_smoothing_kev = self.inspector.voxel_smooth_energy.value()
        self.state.voxel_threshold_fraction = self.inspector.voxel_threshold.value()
        self.state.voxel_energy_source = str(self.inspector.voxel_energy_source.currentData() or "selected_slice")
        self.state.voxel_opacity = self.inspector.voxel_opacity.value()
        self.state.voxel_show_edges = self.inspector.voxel_edges.isChecked()
        if self.state.render_mode == "voxels":
            # Coalesce rapid spin-box/slider changes into one voxel rebuild.
            # A voxel rebuild is much heavier than changing event styling.
            self._schedule_viewer(delay_ms=180)
        self._workspace_changed()
    def _add_slice(self):
        low, high = self.state.energy_band
        if high - low > 1.0:
            center = 0.5 * (low + high)
            low, high = center - 0.25, center + 0.25
        item = self.state.add_slice(low, high)
        self.left_panel.set_slices(self.state.slices, item.uid)
        self.analysis.spectrum.set_slices(self.state.slices)
        self.inspector.set_slice(item)
        self.workspace.set_two_d_product("slice")
        self.state.two_d_product = "slice"
        self._schedule_viewer()
        self._schedule_analysis("slices", "2d")
        self._workspace_changed()
    def _add_slice_preset(self, preset: str):
        if preset != "casa":
            return
        created = self.state.add_cas_a_reference_slices()
        if not created:
            return
        self.left_panel.set_slices(self.state.slices, self.state.selected_slice_uid)
        self.analysis.spectrum.set_slices(self.state.slices)
        self.inspector.set_slice(self.state.selected_slice())
        self.analysis.tabs.setCurrentWidget(self.analysis.images)
        self._schedule_viewer()
        self._refresh_slice_products(force_all=True)
        self._update_status("Added the four Cas A reference energy bands from Ken's task")
        self._workspace_changed()
    def _select_slice(self, uid: str):
        self.state.selected_slice_uid = uid
        item = self.state.selected_slice()
        self.inspector.set_slice(item)
        self.workspace.set_two_d_product("slice")
        self.state.two_d_product = "slice"
        self._refresh_selected_slice()
        if self.analysis.tabs.currentWidget() is self.analysis.profile:
            self._refresh_slice_products(force_all=True)
        self._schedule_viewer()
        self._workspace_changed()
    def _set_slice_visibility(self, uid: str, visible: bool):
        for item in self.state.slices:
            if item.uid == uid:
                item.visible = bool(visible)
                break
        self.analysis.spectrum.set_slices(self.state.slices)
        self._schedule_viewer()
        self._schedule_analysis("slices", "2d")
        self._workspace_changed()
    def _set_slice_points(self, uid: str, visible: bool):
        for item in self.state.slices:
            if item.uid == uid:
                item.show_points = bool(visible)
                break
        self._schedule_viewer()
        self._workspace_changed()
    def _move_slice(self, uid: str, low: float, high: float):
        """Fast path used while dragging a slice on the spectrum."""
        low, high = sorted((float(low), float(high)))
        for item in self.state.slices:
            if item.uid != uid:
                continue
            item.low_kev, item.high_kev = low, high
            self.state.selected_slice_uid = uid
            self.left_panel.update_slice(item, select=True)
            break
        self._schedule_viewer()
        self._schedule_analysis("2d", "slices")
        self._workspace_changed()
    def _edit_slice(self, uid: str, low: float, high: float, opacity, show_plane):
        for item in self.state.slices:
            if item.uid == uid:
                item.low_kev, item.high_kev = sorted((float(low), float(high)))
                if opacity is not None:
                    item.opacity = float(opacity)
                if show_plane is not None:
                    item.show_plane = bool(show_plane)
                self.state.selected_slice_uid = uid
                self.inspector.set_slice(item)
                break
        if self.state.selected_slice() is not None:
            self.left_panel.update_slice(self.state.selected_slice(), select=True)
        self.analysis.spectrum.set_slices(self.state.slices)
        self._schedule_viewer()
        self._schedule_analysis("slices", "2d")
        self._workspace_changed()
    def _set_slice_color(self, uid: str, color: str):
        for item in self.state.slices:
            if item.uid == uid:
                item.color = str(color)
                self.left_panel.update_slice(item, select=True)
                self.inspector._set_slice_color_button(item.color)
                break
        self._schedule_viewer()
        self._schedule_analysis("slices")
        self._workspace_changed()
    def _remove_slice(self, uid: str):
        self.state.remove_slice(uid)
        self.left_panel.set_slices(self.state.slices, self.state.selected_slice_uid)
        self.analysis.spectrum.set_slices(self.state.slices)
        self._schedule_viewer()
        self._schedule_analysis("slices", "2d")
        self._workspace_changed()
    def _select_event(self, event):
        self.state.selected_event = event
        self.inspector.set_selected_event(event)
    def _observation_selected(self, key: str):
        obs = self.state.observation_cache.get(key)
        if obs is not None:
            self.state.selected_observation_key = key
            self.inspector.set_observation(key, obs)
    def _remove_observation(self, key: str):
        if key not in self.state.observation_cache:
            return
        self.workspace.three_d.remove_record(key)
        self.state.observation_cache.pop(key, None)
        self.state.visible_record_keys.discard(key)
        self.state.loaded_observations = [
            obs for obs in self.state.loaded_observations if self.state.record_key(obs) != key
        ]
        self.state.combined_frame = combine_frames(self.state.loaded_observations)
        self.state.clear_derived_caches()
        self.left_panel.set_observations(self.state.loaded_observations, self.state.visible_record_keys)
        self._invalidate_exact()
        self._refresh_after_data_change(reset_camera=False)
        self._workspace_changed()
