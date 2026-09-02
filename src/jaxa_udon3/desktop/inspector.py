from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from jaxa_udon3.desktop.energy_controls import ENERGY_SLIDER_SCALE, build_energy_page
from jaxa_udon3.desktop.scalar_display_widgets import sync_scalar_display_controls
from jaxa_udon3.desktop.voxel_controls import build_voxel_page, sync_overlay_controls


def _dspin(low, high, value, step=0.05, decimals=3, suffix=""):
    widget = QDoubleSpinBox()
    widget.setRange(low, high)
    widget.setValue(value)
    widget.setSingleStep(step)
    widget.setDecimals(decimals)
    if suffix:
        widget.setSuffix(suffix)
    return widget
def _page(title: str):
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(9, 9, 9, 9)
    label = QLabel(title)
    label.setObjectName("panelTitle")
    layout.addWidget(label)
    return page, layout
class InspectorPanel(QWidget):
    display_changed = Signal()
    image_quality_changed = Signal()
    scalar_display_changed, energy_changed = Signal(), Signal(float, float)
    energy_filter_scope_changed = Signal(bool, bool)
    spectrum_link_changed = Signal(bool)
    rgb_changed, voxel_changed = Signal(), Signal()
    energy_geometry_changed, w49b_preset_requested = Signal(float, float, bool, bool), Signal()
    roi_toggled = Signal(bool)
    roi_changed = Signal(object)
    comparison_roi_toggled = Signal(bool)
    comparison_roi_changed = Signal(object)
    slice_changed = Signal(str, float, float, float, bool)
    slice_color_changed = Signal(str, str)
    slice_points_changed = Signal(str, bool)
    observation_remove_requested = Signal(str)
    exact_energy_requested = Signal()
    exact_all_events_requested = Signal()
    exact_rgb_requested = Signal()
    top_image_changed = Signal(str, float, str)
    PAGE_DATASET = 0
    PAGE_EVENTS = 1
    PAGE_ENERGY = 2
    PAGE_RGB = 3
    PAGE_VOXELS = 4
    PAGE_SLICE = 5
    PAGE_EVENT = 6
    PAGE_OBSERVATION = 7
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        header = QFrame()
        header.setObjectName("dockTitle")
        row = QHBoxLayout(header)
        row.setContentsMargins(9, 6, 9, 6)
        title = QLabel("INSPECTOR")
        title.setObjectName("sectionTitle")
        row.addWidget(title)
        row.addStretch(1)
        root.addWidget(header)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self._build_dataset()
        self._build_events()
        self._build_energy()
        self._build_rgb()
        self._build_voxels()
        self._build_slice()
        self._build_selected_event()
        self._build_observation()
    def _wrap(self, page):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)
    @staticmethod
    def _subheading(text):
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label
    def _build_dataset(self):
        page, box = _page("DATASET")
        self.dataset_target = QLabel("No target")
        self.dataset_target.setObjectName("value")
        self.dataset_region = QLabel("—")
        self.dataset_events = QLabel("0")
        self.dataset_energy = QLabel("—")
        form = QFormLayout()
        form.addRow("Target", self.dataset_target)
        form.addRow("Fixed search region", self.dataset_region)
        form.addRow("Visible preview", self.dataset_events)
        form.addRow("Energy coverage", self.dataset_energy)
        box.addLayout(form)
        note = QLabel(
            "The search centre/radius stay fixed while ROI, energy filters, slices and visibility change."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        box.addWidget(note)
        self.exact_all_events = QPushButton("All events → exact 2D image")
        self.exact_all_events.setObjectName("primary")
        self.exact_all_events.setToolTip(
            "Use every event from every visible matching parquet file in the fixed sky "
            "region (excluding invalid negative-PI sentinels). Processing is out-of-core; "
            "the rotatable 3D view remains safely sampled."
        )
        self.exact_all_events.clicked.connect(self.exact_all_events_requested)
        box.addWidget(self.exact_all_events)
        all_events_note = QLabel(
            "This is the full-data test. It compresses all energies into RA/DEC and does not "
            "limit the calculation to the interactive preview."
        )
        all_events_note.setObjectName("muted")
        all_events_note.setWordWrap(True)
        box.addWidget(all_events_note)
        box.addStretch(1)
        self._wrap(page)
    def _build_events(self):
        page, box = _page("EVENT DISPLAY")
        form = QFormLayout()
        self.color_mode = QComboBox()
        self.color_mode.addItem("Energy gradient", "energy")
        self.color_mode.addItem("Mission", "mission")
        self.color_mode.addItem("RGB energy bands", "rgb")
        self.content_mode = QComboBox()
        for label, key in (
            ("All events", "all"),
            ("Active slice", "active"),
            ("All + active slice", "all_active"),
            ("Multiple slices", "multiple"),
            ("Slice planes only", "planes"),
        ):
            self.content_mode.addItem(label, key)
        self.point_size = _dspin(1.0, 16.0, 4.0, 0.5, 1, " px")
        self.opacity = _dspin(0.05, 1.0, 0.82, 0.05, 2)
        form.addRow("Color by", self.color_mode)
        form.addRow("3D content", self.content_mode)
        self.show_slice_planes = QCheckBox("Show energy-slice planes")
        self.show_slice_planes.setChecked(True)
        form.addRow("Slice overlays", self.show_slice_planes)
        self.show_coordinate_triad = QCheckBox("Show coordinate axes")
        self.show_coordinate_triad.setChecked(True)
        self.show_grid_backdrop = QCheckBox("Show XY reference grid")
        self.show_grid_backdrop.setChecked(True)
        self.show_coordinate_values = QCheckBox("Show coordinate values")
        self.show_coordinate_values.setChecked(True)
        self.show_slice_window = QCheckBox("Show active slice window")
        self.show_slice_window.setChecked(True)
        form.addRow("Coordinate axes", self.show_coordinate_triad)
        form.addRow("Reference grid", self.show_grid_backdrop)
        form.addRow("Coordinate values", self.show_coordinate_values)
        form.addRow("Slice window", self.show_slice_window)
        self.camera_preset = QComboBox()
        for label, key in (("Isometric", "isometric"), ("Top: XY", "top"), ("East: YZ", "east"), ("North: XZ", "north")):
            self.camera_preset.addItem(label, key)
        form.addRow("Camera", self.camera_preset)
        self.top_image_mode = QComboBox()
        for label, key in (("Off", "off"), ("Top plane", "plane"), ("Top inset", "inset")):
            self.top_image_mode.addItem(label, key)
        self.top_image_opacity = QSlider(Qt.Horizontal)
        self.top_image_opacity.setRange(0, 100)
        self.top_image_opacity.setValue(75)
        form.addRow("Top image", self.top_image_mode)
        form.addRow("Top opacity", self.top_image_opacity)
        form.addRow("Point size", self.point_size)
        form.addRow("Opacity", self.opacity)
        box.addLayout(form)
        box.addWidget(self._subheading("DISPLAY DENSITY"))
        smooth = QFormLayout()
        self.event_spatial = _dspin(0.0, 10.0, 0.0, 0.1, 2, " arcmin")
        self.event_energy = _dspin(0.0, 2.0, 0.0, 0.02, 3, " keV")
        self.density_size = _dspin(0.0, 2.0, 0.7, 0.1, 2)
        self.density_opacity = _dspin(0.0, 2.0, 0.7, 0.1, 2)
        smooth.addRow("Spatial sigma", self.event_spatial)
        smooth.addRow("Energy sigma", self.event_energy)
        smooth.addRow("Density size", self.density_size)
        smooth.addRow("Density opacity", self.density_opacity)
        box.addLayout(smooth)
        box.addWidget(self._subheading("SPECTRUM SPATIAL ROI"))
        self.roi_enabled = QCheckBox("Use RA/DEC rectangle for spectrum")
        box.addWidget(self.roi_enabled)
        roi_form = QFormLayout()
        self.roi_ra_min = _dspin(0.0, 360.0, 350.80, 0.005, 5, "°")
        self.roi_ra_max = _dspin(0.0, 360.0, 350.90, 0.005, 5, "°")
        self.roi_dec_min = _dspin(-90.0, 90.0, 58.77, 0.005, 5, "°")
        self.roi_dec_max = _dspin(-90.0, 90.0, 58.86, 0.005, 5, "°")
        for label, widget in (
            ("RA min", self.roi_ra_min), ("RA max", self.roi_ra_max),
            ("DEC min", self.roi_dec_min), ("DEC max", self.roi_dec_max),
        ):
            roi_form.addRow(label, widget)
        box.addLayout(roi_form)
        self.roi_enabled.toggled.connect(self.roi_toggled)
        for widget in (self.roi_ra_min, self.roi_ra_max, self.roi_dec_min, self.roi_dec_max):
            widget.editingFinished.connect(self._roi_emit)
        self.roi_compare_enabled = QCheckBox("Compare with ROI B")
        box.addWidget(self.roi_compare_enabled)
        comparison_form = QFormLayout()
        self.roi_b_ra_min = _dspin(0.0, 360.0, 350.80, 0.005, 5, "°")
        self.roi_b_ra_max = _dspin(0.0, 360.0, 350.90, 0.005, 5, "°")
        self.roi_b_dec_min = _dspin(-90.0, 90.0, 58.77, 0.005, 5, "°")
        self.roi_b_dec_max = _dspin(-90.0, 90.0, 58.86, 0.005, 5, "°")
        for label, widget in (
            ("B RA min", self.roi_b_ra_min), ("B RA max", self.roi_b_ra_max),
            ("B DEC min", self.roi_b_dec_min), ("B DEC max", self.roi_b_dec_max),
        ):
            comparison_form.addRow(label, widget)
        box.addLayout(comparison_form)
        self.roi_compare_enabled.toggled.connect(self.comparison_roi_toggled)
        for widget in (self.roi_b_ra_min, self.roi_b_ra_max, self.roi_b_dec_min, self.roi_b_dec_max):
            widget.editingFinished.connect(self._comparison_roi_emit)
        box.addStretch(1)
        box.addWidget(self._subheading("3D ENERGY GEOMETRY"))
        geometry = QFormLayout()
        self.energy_reference = _dspin(0.0, 1000.0, 6.70, 0.001, 3, " keV")
        self.energy_display_scale = _dspin(0.25, 4.0, 1.0, 0.1, 2, "×")
        self.energy_display_scale.setToolTip(
            "Relative energy-axis depth. 1× automatically balances the selected "
            "energy band with the sky width."
        )
        geometry.addRow("Reference plane", self.energy_reference)
        geometry.addRow("Depth aspect", self.energy_display_scale)
        box.addLayout(geometry)
        self.show_energy_reference_plane = QCheckBox("Show reference energy plane")
        self.show_energy_reference_plane.setChecked(True)
        box.addWidget(self.show_energy_reference_plane)
        w49b = QPushButton("Use W49B Fe XXV quick-look")
        w49b.setToolTip("6.700 keV reference with ±2.5 eV blue/red comparison planes")
        w49b.clicked.connect(self.w49b_preset_requested)
        box.addWidget(w49b)
        for widget in (self.energy_reference, self.energy_display_scale):
            widget.valueChanged.connect(self._energy_geometry_emit)
        self.show_energy_reference_plane.toggled.connect(self._energy_geometry_emit)
        self.color_mode.currentIndexChanged.connect(lambda *_: self.display_changed.emit())
        self.content_mode.currentIndexChanged.connect(lambda *_: self.display_changed.emit())
        self.show_slice_planes.toggled.connect(lambda *_: self.display_changed.emit())
        for widget in (self.show_coordinate_triad, self.show_grid_backdrop, self.show_coordinate_values, self.show_slice_window):
            widget.toggled.connect(lambda *_: self.display_changed.emit())
        self.camera_preset.currentIndexChanged.connect(lambda *_: self.display_changed.emit())
        self.top_image_mode.currentIndexChanged.connect(lambda *_: self._top_image_emit())
        self.top_image_opacity.valueChanged.connect(lambda *_: self._top_image_emit())
        for widget in (
            self.point_size, self.opacity, self.event_spatial, self.event_energy,
            self.density_size, self.density_opacity,
        ):
            widget.valueChanged.connect(lambda *_: self.display_changed.emit())
        self._wrap(page)
    def _build_energy(self):
        build_energy_page(self, _page, _dspin)
    def _energy_geometry_emit(self, *_):
        self.energy_geometry_changed.emit(
            self.energy_reference.value(), self.energy_display_scale.value(),
            True, self.show_energy_reference_plane.isChecked(),
        )
    def _build_rgb(self):
        page, box = _page("RGB COMPOSITE")
        self.rgb_controls = []
        for name, center, width in (
            ("RED", 1.85, 0.20), ("GREEN", 2.44, 0.20), ("BLUE", 6.40, 0.40),
        ):
            box.addWidget(self._subheading(name))
            form = QFormLayout()
            center_widget = _dspin(0.0, 1000.0, center, 0.01, 3, " keV")
            width_widget = _dspin(0.001, 100.0, width, 0.01, 3, " keV")
            gain_widget = _dspin(0.0, 10.0, 1.0, 0.05, 2)
            form.addRow("Center", center_widget)
            form.addRow("Width", width_widget)
            form.addRow("Gain", gain_widget)
            box.addLayout(form)
            self.rgb_controls.append((center_widget, width_widget, gain_widget))
        form = QFormLayout()
        self.rgb_brightness = _dspin(0.05, 5.0, 1.1, 0.05, 2)
        self.rgb_gamma = _dspin(0.1, 4.0, 1.0, 0.05, 2)
        self.rgb_smoothing = _dspin(0.0, 20.0, 0.8, 0.1, 2, " px")
        self.rgb_smoothing.setEnabled(False)
        form.addRow("Brightness", self.rgb_brightness)
        form.addRow("Gamma", self.rgb_gamma)
        form.addRow("Custom smoothing", self.rgb_smoothing)
        box.addLayout(form)
        self.rgb_ranges = QLabel("")
        self.rgb_ranges.setObjectName("muted")
        self.rgb_ranges.setWordWrap(True)
        box.addWidget(self.rgb_ranges)
        self.rgb_quality = QLabel("PREVIEW")
        self.rgb_quality.setObjectName("qualityBadge")
        box.addWidget(self.rgb_quality)
        cas_a = QPushButton("Use Cas A ASCA RGB")
        cas_a.setToolTip(
            "R: 1.55–1.75 keV continuum; G: 1.75–1.95 keV Si He α; "
            "B: 2.35–2.52 keV S He α."
        )
        cas_a.clicked.connect(self.apply_cas_a_rgb_preset)
        box.addWidget(cas_a)
        exact = QPushButton("Compute exact RGB image")
        exact.setObjectName("primary")
        exact.clicked.connect(self.exact_rgb_requested)
        box.addWidget(exact)
        box.addStretch(1)
        for trio in self.rgb_controls:
            for widget in trio:
                widget.valueChanged.connect(self._rgb_emit)
        for widget in (self.rgb_brightness, self.rgb_gamma, self.rgb_smoothing):
            widget.valueChanged.connect(self._rgb_emit)
        self._rgb_emit()
        self._wrap(page)
    def _build_voxels(self):
        build_voxel_page(self, _page, _dspin)
    def _build_slice(self):
        page, box = _page("SELECTED ENERGY SLICE")
        self.slice_uid = ""
        self.slice_low = _dspin(0.0, 1000.0, 2.0, 0.05, 3, " keV")
        self.slice_high = _dspin(0.001, 1000.0, 3.0, 0.05, 3, " keV")
        self.slice_opacity = _dspin(0.05, 1.0, 0.62, 0.05, 2)
        self.slice_color = QPushButton("Choose slice color")
        self.slice_color.clicked.connect(self._choose_slice_color)
        self.slice_plane = QCheckBox("Show plane in 3D")
        self.slice_points = QCheckBox("Show points in 3D")
        self.slice_plane.setChecked(True)
        self.slice_points.setChecked(True)
        form = QFormLayout()
        form.addRow("Lower", self.slice_low)
        form.addRow("Upper", self.slice_high)
        form.addRow("Plane opacity", self.slice_opacity)
        form.addRow("Slice color", self.slice_color)
        box.addLayout(form)
        box.addWidget(self.slice_plane)
        box.addWidget(self.slice_points)
        note = QLabel("Move this band here or drag its colored region directly on the spectrum.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        box.addWidget(note)
        box.addStretch(1)
        for widget in (self.slice_low, self.slice_high, self.slice_opacity):
            widget.valueChanged.connect(self._slice_emit)
        self.slice_plane.toggled.connect(self._slice_emit)
        self.slice_points.toggled.connect(
            lambda checked: self.slice_points_changed.emit(self.slice_uid, bool(checked))
        )
        self._wrap(page)
    def _build_selected_event(self):
        page, box = _page("SELECTED EVENT")
        self.event_fields = {}
        form = QFormLayout()
        for name in ("Mission", "Instrument", "Observation", "Energy", "RA", "DEC", "PI", "Time"):
            label = QLabel("—")
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.event_fields[name] = label
            form.addRow(name, label)
        box.addLayout(form)
        box.addStretch(1)
        self._wrap(page)
    def _build_observation(self):
        page, box = _page("OBSERVATION")
        self.observation_key = ""
        self.observation_fields = {}
        form = QFormLayout()
        for name in ("Mission", "Instrument", "Observation", "Events in region", "Preview rows"):
            label = QLabel("—")
            self.observation_fields[name] = label
            form.addRow(name, label)
        box.addLayout(form)
        remove = QPushButton("Remove observation from workspace")
        remove.setObjectName("danger")
        remove.clicked.connect(lambda: self.observation_remove_requested.emit(self.observation_key))
        box.addWidget(remove)
        box.addStretch(1)
        self._wrap(page)
    def _auto_quality_toggled(self, enabled: bool):
        self.image_smoothing.setEnabled(not enabled)
        if hasattr(self, "rgb_smoothing"):
            self.rgb_smoothing.setEnabled(not enabled)
        self.image_quality_changed.emit()
    def _energy_emit(self):
        low, high = self.energy_low.value(), self.energy_high.value()
        if high > low:
            self._sync_energy_sliders(low, high)
            self.energy_changed.emit(low, high)
    def _sync_energy_sliders(self, low: float, high: float):
        for slider, value in ((self.energy_low_slider, low), (self.energy_high_slider, high)):
            slider.blockSignals(True)
            slider.setValue(min(slider.maximum(), max(slider.minimum(), round(value * ENERGY_SLIDER_SCALE))))
            slider.blockSignals(False)
    def _energy_slider_emit(self, *_):
        low = self.energy_low_slider.value() / ENERGY_SLIDER_SCALE
        high = self.energy_high_slider.value() / ENERGY_SLIDER_SCALE
        if high <= low:
            if self.sender() is self.energy_low_slider:
                low = max(0.0, high - 1.0 / ENERGY_SLIDER_SCALE)
            else:
                high = min(self.energy_high_slider.maximum() / ENERGY_SLIDER_SCALE, low + 1.0 / ENERGY_SLIDER_SCALE)
        self._sync_energy_sliders(low, high)
        for widget, value in ((self.energy_low, low), (self.energy_high, high)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.energy_changed.emit(low, high)
    def _energy_filter_scope_emit(self, *_):
        self.energy_filter_scope_changed.emit(
            self.filter_3d_by_energy.isChecked(),
            self.filter_2d_by_energy.isChecked(),
        )
    def _spectrum_link_emit(self, enabled: bool):
        self.spectrum_link_changed.emit(bool(enabled))
    def _scalar_display_emit(self, *_):
        self.scalar_display_changed.emit()
    def _top_image_emit(self):
        self.top_image_changed.emit(
            str(self.top_image_mode.currentData() or "off"),
            self.top_image_opacity.value() / 100.0,
            "global",
        )
    def _roi_emit(self):
        if not self.roi_enabled.isChecked():
            return
        rectangle = (
            self.roi_ra_min.value(), self.roi_ra_max.value(),
            self.roi_dec_min.value(), self.roi_dec_max.value(),
        )
        if rectangle[3] > rectangle[2] and rectangle[1] != rectangle[0]:
            self.roi_changed.emit(rectangle)
    def _comparison_roi_emit(self):
        if not self.roi_compare_enabled.isChecked():
            return
        rectangle = (
            self.roi_b_ra_min.value(), self.roi_b_ra_max.value(),
            self.roi_b_dec_min.value(), self.roi_b_dec_max.value(),
        )
        if rectangle[3] > rectangle[2] and rectangle[1] != rectangle[0]:
            self.comparison_roi_changed.emit(rectangle)
    def _rgb_emit(self, *_):
        centers = [trio[0].value() for trio in self.rgb_controls]
        widths = [trio[1].value() for trio in self.rgb_controls]
        bands = [(c - w / 2, c + w / 2) for c, w in zip(centers, widths)]
        self.rgb_ranges.setText(
            f"R {bands[0][0]:.2f}–{bands[0][1]:.2f}  •  "
            f"G {bands[1][0]:.2f}–{bands[1][1]:.2f}  •  "
            f"B {bands[2][0]:.2f}–{bands[2][1]:.2f} keV"
        )
        self.rgb_changed.emit()
    def apply_cas_a_rgb_preset(self):
        values = ((1.65, 0.20), (1.85, 0.20), (2.435, 0.17))
        for (center_widget, width_widget, _gain_widget), (center, width) in zip(
            self.rgb_controls, values
        ):
            center_widget.blockSignals(True)
            width_widget.blockSignals(True)
            center_widget.setValue(center)
            width_widget.setValue(width)
            center_widget.blockSignals(False)
            width_widget.blockSignals(False)
        self._rgb_emit()
    def _slice_emit(self, *_):
        if not self.slice_uid:
            return
        low, high = self.slice_low.value(), self.slice_high.value()
        if high > low:
            self.slice_changed.emit(
                self.slice_uid, low, high, self.slice_opacity.value(), self.slice_plane.isChecked()
            )
    def _choose_slice_color(self):
        if not self.slice_uid:
            return
        color = QColorDialog.getColor(parent=self, title="Choose active slice color")
        if color.isValid():
            self._set_slice_color_button(color.name())
            self.slice_color_changed.emit(self.slice_uid, color.name())

    def _set_slice_color_button(self, color: str):
        self.slice_color.setStyleSheet(
            f"background-color: {color}; color: white; font-weight: 700;"
        )
    def set_roi(self, rectangle):
        if not rectangle:
            return
        for widget, value in zip(
            (self.roi_ra_min, self.roi_ra_max, self.roi_dec_min, self.roi_dec_max), rectangle
        ):
            widget.blockSignals(True)
            widget.setValue(float(value) % 360.0 if widget in (self.roi_ra_min, self.roi_ra_max) else float(value))
            widget.blockSignals(False)
    def set_comparison_roi(self, rectangle):
        if not rectangle:
            return
        for widget, value in zip(
            (self.roi_b_ra_min, self.roi_b_ra_max, self.roi_b_dec_min, self.roi_b_dec_max), rectangle
        ):
            widget.blockSignals(True)
            widget.setValue(float(value) % 360.0 if widget in (self.roi_b_ra_min, self.roi_b_ra_max) else float(value))
            widget.blockSignals(False)
    @staticmethod
    def _restore_widget(widget, value):
        widget.blockSignals(True)
        widget.setValue(value)
        widget.blockSignals(False)
    def set_display_state(self, state):
        index = self.color_mode.findData(state.event_color_mode)
        self.color_mode.blockSignals(True)
        self.color_mode.setCurrentIndex(max(0, index))
        self.color_mode.blockSignals(False)
        content_index = self.content_mode.findData(getattr(state, "content_mode", "all"))
        self.content_mode.blockSignals(True)
        self.content_mode.setCurrentIndex(max(0, content_index))
        self.content_mode.blockSignals(False)
        self.top_image_mode.blockSignals(True)
        self.top_image_mode.setCurrentIndex(max(0, self.top_image_mode.findData(
            getattr(state, "top_image_mode", "off")
        )))
        self.top_image_mode.blockSignals(False)
        self.top_image_opacity.blockSignals(True)
        self.top_image_opacity.setValue(round(float(getattr(state, "top_image_opacity", 0.75)) * 100))
        self.top_image_opacity.blockSignals(False)
        for widget, value in (
            (self.point_size, state.point_size), (self.opacity, state.point_opacity),
            (self.event_spatial, state.event_spatial_smoothing_arcmin),
            (self.event_energy, state.event_energy_smoothing_kev),
            (self.density_size, state.density_size_strength),
            (self.density_opacity, state.density_opacity_strength),
        ):
            self._restore_widget(widget, value)
        self.show_slice_planes.blockSignals(True)
        self.show_slice_planes.setChecked(getattr(state, "show_slice_planes", True))
        self.show_slice_planes.blockSignals(False)
        sync_overlay_controls(self, state)
        for widget, value in (
            (self.energy_reference, getattr(state, "energy_reference_kev", 6.70)),
            (self.energy_display_scale, getattr(state, "energy_display_scale", 1.0)),
        ):
            self._restore_widget(widget, value)
        self.show_energy_reference_plane.blockSignals(True)
        self.show_energy_reference_plane.setChecked(getattr(state, "show_energy_reference_plane", True))
        self.show_energy_reference_plane.blockSignals(False)
    def set_energy_state(self, band, automatic: bool, smoothing: float):
        self._restore_widget(self.energy_low, band[0])
        self._restore_widget(self.energy_high, band[1])
        self._sync_energy_sliders(*band)
        self.auto_image_quality.blockSignals(True)
        self.auto_image_quality.setChecked(bool(automatic))
        self.auto_image_quality.blockSignals(False)
        self.image_smoothing.setEnabled(not automatic)
        self.rgb_smoothing.setEnabled(not automatic)
        self._restore_widget(self.image_smoothing, smoothing)
        self._restore_widget(self.rgb_smoothing, smoothing)
    def set_energy_filter_state(self, filter_3d: bool, filter_2d: bool):
        for widget, value in (
            (self.filter_3d_by_energy, filter_3d),
            (self.filter_2d_by_energy, filter_2d),
        ):
            widget.blockSignals(True)
            widget.setChecked(bool(value))
            widget.blockSignals(False)
    def set_spectrum_link_state(self, linked: bool):
        self.spectrum_linked.blockSignals(True)
        self.spectrum_linked.setChecked(bool(linked))
        self.spectrum_linked.blockSignals(False)
    def set_scalar_display_state(self, state):
        sync_scalar_display_controls(
            self.image_palette, self.image_stretch, self.image_brightness,
            self.image_contrast, state,
        )
    def set_rgb_state(self, state):
        for trio, center, width, gain in zip(
            self.rgb_controls, state.rgb_centers, state.rgb_widths, state.rgb_gains
        ):
            for widget, value in zip(trio, (center, width, gain)):
                self._restore_widget(widget, value)
        self._restore_widget(self.rgb_brightness, state.rgb_brightness)
        self._restore_widget(self.rgb_gamma, state.rgb_gamma)
        self._rgb_emit()
    def set_voxel_state(self, state):
        for widget, value in (
            (self.voxel_spatial, state.spatial_voxel_arcmin),
            (self.voxel_energy, state.energy_voxel_kev),
            (self.voxel_smooth_spatial, state.voxel_spatial_smoothing_arcmin),
            (self.voxel_smooth_energy, state.voxel_energy_smoothing_kev),
            (self.voxel_threshold, state.voxel_threshold_fraction),
            (self.voxel_opacity, getattr(state, "voxel_opacity", 0.82)),
        ):
            self._restore_widget(widget, value)
        self.voxel_energy_source.blockSignals(True)
        self.voxel_energy_source.setCurrentIndex(max(0, self.voxel_energy_source.findData(getattr(state, "voxel_energy_source", "selected_slice"))))
        self.voxel_energy_source.blockSignals(False)
        self.voxel_edges.blockSignals(True)
        self.voxel_edges.setChecked(getattr(state, "voxel_show_edges", False))
        self.voxel_edges.blockSignals(False)
    def set_page(self, key: str):
        page = {
            "dataset": self.PAGE_DATASET, "events": self.PAGE_EVENTS, "energy": self.PAGE_ENERGY,
            "rgb": self.PAGE_RGB, "voxels": self.PAGE_VOXELS, "slice": self.PAGE_SLICE,
            "event": self.PAGE_EVENT, "observation": self.PAGE_OBSERVATION,
        }.get(key, self.PAGE_DATASET)
        self.stack.setCurrentIndex(page)
    def set_dataset(self, target: str, region, rows: int, energy_range):
        self.dataset_target.setText(target or "Sky region")
        self.dataset_region.setText("—" if region is None else f"{region.radius_deg * 60:.2f} arcmin")
        self.dataset_events.setText(f"{rows:,}")
        self.dataset_energy.setText(
            "—" if not energy_range else f"{energy_range[0]:.2f}–{energy_range[1]:.2f} keV"
        )
    def set_slice(self, item):
        if item is None:
            self.slice_uid = ""
            return
        self.slice_uid = item.uid
        for widget, value in (
            (self.slice_low, item.low_kev), (self.slice_high, item.high_kev),
            (self.slice_opacity, item.opacity),
        ):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)
        self.slice_plane.blockSignals(True)
        self.slice_plane.setChecked(bool(item.show_plane))
        self.slice_plane.blockSignals(False)
        self.slice_points.blockSignals(True)
        self.slice_points.setChecked(bool(item.show_points))
        self.slice_points.blockSignals(False)
        self._set_slice_color_button(str(item.color))
        self.set_page("slice")
    def set_selected_event(self, event: dict | None):
        if not event:
            return
        values = {
            "Mission": event.get("MISSION", "—"), "Instrument": event.get("INSTRUMENT", "—"),
            "Observation": event.get("OBSERVATION_ID", "—"),
            "Energy": f"{float(event.get('KEV', float('nan'))):.4f} keV",
            "RA": f"{float(event.get('RA', float('nan'))):.6f}°",
            "DEC": f"{float(event.get('DEC', float('nan'))):+.6f}°",
            "PI": str(event.get("PI", "—")), "Time": str(event.get("TIME", "—")),
        }
        for key, value in values.items():
            self.event_fields[key].setText(value)
        self.set_page("event")
    def set_observation(self, key: str, obs):
        self.observation_key = key
        values = {
            "Mission": str(obs.record.mission).upper(),
            "Instrument": str(obs.record.instrument).upper(),
            "Observation": str(obs.record.observation_id),
            "Events in region": f"{obs.events_in_region:,}",
            "Preview rows": f"{len(obs.frame):,}",
        }
        for name, value in values.items():
            self.observation_fields[name].setText(value)
        self.set_page("observation")
