from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from jaxa_torax.desktop.science_views import (
    ImageProduct,
    energy_to_plot_x,
    normalize_spectrum_scale,
    plot_x_to_energy,
)
from jaxa_torax.desktop.theme import PLOT_BACKGROUND, PLOT_TEXT
from jaxa_torax.desktop.viewers import ImagePlot


class SpectrumWidget(QWidget):
    band_changed = Signal(float, float)
    slice_changed = Signal(str, float, float)
    settings_changed = Signal(int, float, bool)
    scale_changed = Signal(str)
    smooth_visibility_changed = Signal(bool)
    scan_speed_changed = Signal(int)

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(5, 5, 5, 5)
        header = QHBoxLayout()
        self.selection_label = QLabel("All visible events")
        self.selection_label.setObjectName("muted")
        header.addWidget(self.selection_label)
        header.addStretch(1)

        self.auto_bins = QCheckBox("Auto bins")
        self.auto_bins.setChecked(True)
        self.auto_bins.toggled.connect(self._auto_toggled)
        header.addWidget(self.auto_bins)
        self.bins = QComboBox()
        for value in (120, 180, 260, 360, 520):
            self.bins.addItem(str(value), value)
        self.bins.setCurrentText("260")
        self.bins.setEnabled(False)
        self.bins.currentIndexChanged.connect(self._settings_emit)
        header.addWidget(self.bins)

        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(0.0, 12.0)
        self.smoothing.setSingleStep(0.25)
        self.smoothing.setDecimals(2)
        self.smoothing.setValue(1.25)
        self.smoothing.setSuffix(" bins")
        self.smoothing.setToolTip("Optional display smoothing. Spectrum binning is preferred for science.")
        self.smoothing.valueChanged.connect(self._settings_emit)
        header.addWidget(QLabel("Smooth"))
        header.addWidget(self.smoothing)

        self.show_smooth = QCheckBox("Smooth curve")
        self.show_smooth.setChecked(True)
        self.show_smooth.toggled.connect(self._smooth_visibility_toggled)
        header.addWidget(self.show_smooth)

        header.addWidget(QLabel("Scale"))
        self.scale = QComboBox()
        self.scale.addItem("Linear", "linear")
        self.scale.addItem("Log Y", "log_y")
        self.scale.addItem("Log–Log", "log_log")
        self.scale.currentIndexChanged.connect(self._scale_selected)
        header.addWidget(self.scale)
        self.reset_zoom = QPushButton("Reset zoom")
        self.reset_zoom.clicked.connect(self._fit_spectrum_range)
        header.addWidget(self.reset_zoom)
        root.addLayout(header)

        scan = QHBoxLayout()
        scan.addWidget(QLabel("Energy scan"))
        self.scan_previous = QPushButton("◀ Previous")
        self.scan_lock_width = QCheckBox("Lock width")
        self.scan_lock_width.setChecked(True)
        self.scan_lock_width.setToolTip(
            "Keep the current band width when a boundary is changed in the Energy inspector."
        )
        self.scan_play = QPushButton("▶ Auto Scan")
        self.scan_play.setCheckable(True)
        self.scan_next = QPushButton("Next ▶")
        self.scan_speed = QComboBox()
        for value in (1, 2, 4, 8):
            self.scan_speed.addItem(f"{value}×/s", value)
        self.scan_speed.setCurrentIndex(self.scan_speed.findData(4))
        self.scan_previous.clicked.connect(lambda: self._shift_band(-1))
        self.scan_next.clicked.connect(lambda: self._shift_band(1))
        self.scan_play.clicked.connect(self._toggle_scan)
        self.scan_speed.currentIndexChanged.connect(self._scan_speed_selected)
        for widget in (
            self.scan_previous, self.scan_lock_width, self.scan_play,
            self.scan_next, self.scan_speed,
        ):
            scan.addWidget(widget)
        scan.addStretch(1)
        root.addLayout(scan)

        self.plot = pg.PlotWidget(background=PLOT_BACKGROUND)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Energy", units="keV", color=PLOT_TEXT)
        self.plot.setLabel("left", "Counts", color=PLOT_TEXT)
        self.curve = self.plot.plot(pen=pg.mkPen("#6786a3", width=1.0))
        self.smooth_curve = self.plot.plot(pen=pg.mkPen("#1677d2", width=2.3))
        self.comparison_curve = self.plot.plot(pen=pg.mkPen("#a98954", width=1.0))
        self.comparison_smooth_curve = self.plot.plot(pen=pg.mkPen("#d99a2b", width=2.0))
        self.region = pg.LinearRegionItem(
            values=(2.0, 6.0),
            orientation="vertical",
            brush=pg.mkBrush(22, 119, 210, 42),
            pen=pg.mkPen("#1677d2", width=1.5),
        )
        self.plot.addItem(self.region)
        # Live motion is cheap because MainWindow debounces the expensive linked refreshes.
        self.region.sigRegionChanged.connect(self._emit_region)
        root.addWidget(self.plot)
        self._block = False
        self._slice_regions: dict[str, pg.LinearRegionItem] = {}
        self._slice_block = False
        self._scale = "linear"
        self._product = None
        self._comparison = None
        self._canonical_band = (2.0, 6.0)
        self._canonical_slices = []
        self._positive_floor = 1e-6
        self._scan_direction = 1
        self._scan_timer = QTimer(self)
        self._scan_timer.timeout.connect(self._scan_tick)
        self._apply_scan_speed()

    def set_spectrum(self, product, rectangle=None, comparison=None, comparison_rectangle=None):
        self._product = product
        self._comparison = comparison
        positive_edges = np.asarray(product.edges, dtype=float)
        positive_edges = positive_edges[np.isfinite(positive_edges) & (positive_edges > 0)]
        if positive_edges.size:
            self._positive_floor = float(positive_edges.min())
        self._set_curves(self.curve, self.smooth_curve, product)
        self._set_curves(self.comparison_curve, self.comparison_smooth_curve, comparison)
        self._fit_spectrum_range()
        if rectangle and comparison_rectangle:
            self.selection_label.setText("ROI A (blue) and ROI B (orange)")
        elif rectangle:
            self.selection_label.setText(
                f"ROI: RA {rectangle[0]:.4f}–{rectangle[1]:.4f}°, "
                f"DEC {rectangle[2]:+.4f}–{rectangle[3]:+.4f}°"
            )
        else:
            self.selection_label.setText("All visible observations")

    def set_effective_bins(self, bins: int, automatic: bool):
        self.auto_bins.blockSignals(True)
        self.auto_bins.setChecked(bool(automatic))
        self.auto_bins.blockSignals(False)
        self.bins.setEnabled(not automatic)
        if automatic:
            index = self.bins.findData(int(bins))
            if index >= 0:
                self.bins.blockSignals(True)
                self.bins.setCurrentIndex(index)
                self.bins.blockSignals(False)

    def set_band(self, low: float, high: float):
        wanted = (float(low), float(high))
        self._canonical_band = wanted
        plotted = tuple(energy_to_plot_x(max(value, self._positive_floor), self._scale) for value in wanted)
        current = tuple(sorted(map(float, self.region.getRegion())))
        if abs(current[0] - plotted[0]) < 1e-6 and abs(current[1] - plotted[1]) < 1e-6:
            return
        self._block = True
        self.region.setRegion(plotted)
        self._block = False

    def set_slices(self, slices):
        self._canonical_slices = list(slices)
        self._slice_block = True
        wanted = {item.uid for item in slices}
        for uid in list(self._slice_regions):
            if uid not in wanted:
                region = self._slice_regions.pop(uid)
                self.plot.removeItem(region)
        for item in slices:
            region = self._slice_regions.get(item.uid)
            if region is None:
                color = pg.mkColor(item.color)
                region = pg.LinearRegionItem(
                    values=(0.0, 1.0),
                    orientation="vertical",
                    brush=pg.mkBrush(color.red(), color.green(), color.blue(), 28),
                    pen=pg.mkPen(color, width=1.1),
                    movable=True,
                )
                region.setZValue(5)
                # The Qt signal supplies the region as a positional argument.
                # Capture our UID explicitly and ignore that argument; the old
                # lambda treated it as the UID and emitted conversion warnings.
                region.sigRegionChanged.connect(
                    lambda *_args, uid=item.uid, r=region: self._emit_slice(uid, r)
                )
                self.plot.addItem(region)
                self._slice_regions[item.uid] = region
            region.setRegion(tuple(
                energy_to_plot_x(max(value, self._positive_floor), self._scale)
                for value in (item.low_kev, item.high_kev)
            ))
            region.setVisible(bool(item.visible))
        self._slice_block = False

    def _emit_region(self):
        if self._block:
            return
        self.stop_scan()
        low, high = sorted(
            plot_x_to_energy(value, self._scale) for value in self.region.getRegion()
        )
        self.band_changed.emit(low, high)

    def _emit_slice(self, uid: str, region):
        if self._slice_block:
            return
        low, high = sorted(
            plot_x_to_energy(value, self._scale) for value in region.getRegion()
        )
        self.slice_changed.emit(uid, low, high)

    def _auto_toggled(self, enabled: bool):
        self.bins.setEnabled(not enabled)
        self._settings_emit()

    def _settings_emit(self, *_):
        self.settings_changed.emit(
            int(self.bins.currentData()),
            float(self.smoothing.value()),
            bool(self.auto_bins.isChecked()),
        )

    def _set_curves(self, raw_curve, smooth_curve, product):
        if product is None:
            raw_curve.setData([], [])
            smooth_curve.setData([], [])
            return
        edges = np.asarray(product.edges, dtype=float)
        raw = np.asarray(product.counts, dtype=float)
        x = np.asarray(product.x, dtype=float)
        mask = np.isfinite(raw)
        if self._scale in ("log_y", "log_log"):
            mask &= raw > 0.0
        if self._scale == "log_log":
            mask &= x > 0.0
        # Step mode cannot omit isolated bins, so use centers for logarithmic
        # scales and exact bin edges for the normal linear histogram.
        if self._scale == "linear" and len(edges) == len(raw) + 1:
            raw_curve.setData(edges, raw, stepMode="center", connect="finite")
        else:
            raw_curve.setData(x[mask], raw[mask], stepMode=False, connect="finite")
        smooth = product.smoothed_counts
        if smooth is None or not self.show_smooth.isChecked():
            smooth_curve.setData([], [])
            return
        smooth = np.asarray(smooth, dtype=float)
        smooth_mask = np.isfinite(x) & np.isfinite(smooth)
        if self._scale in ("log_y", "log_log"):
            smooth_mask &= smooth > 0.0
        if self._scale == "log_log":
            smooth_mask &= x > 0.0
        smooth_curve.setData(x[smooth_mask], smooth[smooth_mask], connect="finite")

    def _smooth_visibility_toggled(self, visible: bool):
        self._set_curves(self.curve, self.smooth_curve, self._product)
        self._set_curves(
            self.comparison_curve, self.comparison_smooth_curve, self._comparison
        )
        self.smooth_visibility_changed.emit(bool(visible))

    def _energy_limits(self):
        if self._product is None:
            return None
        edges = np.asarray(self._product.edges, dtype=float)
        edges = edges[np.isfinite(edges)]
        if edges.size < 2:
            return None
        return float(edges.min()), float(edges.max())

    def _shift_band(self, direction: int):
        self.stop_scan()
        self._move_band(int(direction))

    def _move_band(self, direction: int):
        limits = self._energy_limits()
        if limits is None:
            return
        minimum, maximum = limits
        low, high = self._canonical_band
        width = max(0.001, high - low)
        step = max(0.05, width * 0.1)
        new_low, new_high = low + direction * step, high + direction * step
        if new_high >= maximum:
            new_high = maximum
            new_low = max(minimum, maximum - width)
            self._scan_direction = -1
        elif new_low <= minimum:
            new_low = minimum
            new_high = min(maximum, minimum + width)
            self._scan_direction = 1
        self.set_band(new_low, new_high)
        self.band_changed.emit(new_low, new_high)

    def _toggle_scan(self, playing: bool):
        if playing and self._energy_limits() is not None:
            self.scan_play.setText("Ⅱ Pause")
            self._scan_timer.start()
        else:
            self.stop_scan()

    def stop_scan(self):
        self._scan_timer.stop()
        self.scan_play.blockSignals(True)
        self.scan_play.setChecked(False)
        self.scan_play.blockSignals(False)
        self.scan_play.setText("▶ Auto Scan")

    def _scan_tick(self):
        self._move_band(self._scan_direction)

    def _apply_scan_speed(self):
        speed = max(1, int(self.scan_speed.currentData() or 4))
        self._scan_timer.setInterval(round(1000 / speed))

    def _scan_speed_selected(self):
        self._apply_scan_speed()
        self.scan_speed_changed.emit(int(self.scan_speed.currentData() or 4))

    def set_scan_speed(self, speed: int):
        index = self.scan_speed.findData(int(speed))
        self.scan_speed.blockSignals(True)
        self.scan_speed.setCurrentIndex(index if index >= 0 else self.scan_speed.findData(4))
        self.scan_speed.blockSignals(False)
        self._apply_scan_speed()

    def set_smooth_visible(self, visible: bool):
        self.show_smooth.blockSignals(True)
        self.show_smooth.setChecked(bool(visible))
        self.show_smooth.blockSignals(False)
        self._smooth_visibility_toggled(bool(visible))

    def _fit_spectrum_range(self):
        if self._product is None:
            return
        edges = np.asarray(self._product.edges, dtype=float)
        edges = edges[np.isfinite(edges)]
        if self._scale == "log_log":
            edges = edges[edges > 0.0]
        if edges.size:
            self.plot.setXRange(
                float(energy_to_plot_x(edges.min(), self._scale)),
                float(energy_to_plot_x(edges.max(), self._scale)), padding=0.01,
            )
        values = []
        for product in (self._product, self._comparison):
            if product is None:
                continue
            series = [product.counts]
            if self.show_smooth.isChecked() and product.smoothed_counts is not None:
                series.append(product.smoothed_counts)
            for counts in series:
                y = np.asarray(counts, dtype=float)
                y = y[
                    np.isfinite(y)
                    & ((y > 0.0) if self._scale != "linear" else np.ones(y.shape, dtype=bool))
                ]
                if y.size:
                    values.append(y)
        if not values:
            self.plot.setYRange(0.0, 1.0, padding=0.05)
            return
        y = np.concatenate(values)
        if self._scale in ("log_y", "log_log"):
            lo, hi = np.log10(y.min()), np.log10(y.max())
            if hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = min(0.0, float(y.min())), float(y.max())
            if hi <= lo:
                hi = lo + 1.0
        self.plot.setYRange(float(lo), float(hi), padding=0.05)

    def _scale_selected(self):
        self.set_scale(str(self.scale.currentData()))
        self.scale_changed.emit(self._scale)

    def set_scale(self, scale: str):
        scale = normalize_spectrum_scale(scale)
        if scale == self._scale and self.scale.currentData() == scale:
            self._fit_spectrum_range()
            return
        self._scale = scale
        index = self.scale.findData(scale)
        if index >= 0 and index != self.scale.currentIndex():
            self.scale.blockSignals(True)
            self.scale.setCurrentIndex(index)
            self.scale.blockSignals(False)
        # Avoid one paint with the previous (possibly very large linear-count)
        # range interpreted as logarithmic exponents by AxisItem.
        self.plot.getPlotItem().vb.setRange(
            xRange=(-1.0, 2.0), yRange=(0.0, 1.0), padding=0,
            disableAutoRange=True,
        )
        self.plot.setLogMode(x=scale == "log_log", y=scale in ("log_y", "log_log"))
        self.plot.setLabel(
            "bottom", "Energy" + (" (log)" if scale == "log_log" else ""),
            units="keV", color=PLOT_TEXT,
        )
        self.plot.setLabel(
            "left", "Counts" + (" (log)" if scale in ("log_y", "log_log") else ""),
            color=PLOT_TEXT,
        )
        self._set_curves(self.curve, self.smooth_curve, self._product)
        self._set_curves(
            self.comparison_curve, self.comparison_smooth_curve, self._comparison
        )
        self.set_band(*self._canonical_band)
        self.set_slices(self._canonical_slices)
        self._fit_spectrum_range()

    def set_log_log(self, enabled: bool):
        """Compatibility entry point for pre-scale workspaces and callers."""
        self.set_scale("log_log" if enabled else "linear")


class ImagesCompareWidget(QWidget):
    """All active energy slices shown next to one another, as requested for Cas A comparison."""

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(3, 3, 3, 3)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.row = QHBoxLayout(self.content)
        self.row.setContentsMargins(2, 2, 2, 2)
        self.row.setSpacing(6)
        self.row.addStretch(1)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll)
        self.views: list[ImagePlot] = []
        self._viewport = None

    def _ensure_views(self, count: int):
        while len(self.views) < count:
            view = ImagePlot()
            view.set_viewport(self._viewport)
            view.setMinimumWidth(285)
            self.views.append(view)
            self.row.insertWidget(self.row.count() - 1, view, 1)
        for index, view in enumerate(self.views):
            view.setVisible(index < count)

    def set_products(self, products: list[ImageProduct], labels: list[str] | None = None, scalar_settings=None):
        self._ensure_views(len(products))
        for index, product in enumerate(products):
            view = self.views[index]
            label = labels[index] if labels and index < len(labels) else (
                f"{product.low_kev:.2f}–{product.high_kev:.2f} keV"
            )
            title = f"{label}  •  {product.count:,} events"
            view.set_scalar(product.values, product.x_edges, product.y_edges, title, **(scalar_settings or {}))

    def set_viewport(self, viewport):
        self._viewport = viewport
        for view in self.views:
            view.set_viewport(viewport)

    def refresh_scalar_display(self, **settings):
        for view in self.views:
            if view.isVisible():
                view.refresh_scalar_display(**settings)


class SliceProfileWidget(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(5, 5, 5, 5)
        self.plot = pg.PlotWidget(background=PLOT_BACKGROUND)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "RA image pixel (summed over DEC)", color=PLOT_TEXT)
        self.plot.setLabel("left", "Integrated counts", color=PLOT_TEXT)
        self.legend = self.plot.addLegend(offset=(8, 8))
        self.curves = []
        self.summary = QLabel("No visible slice profiles")
        self.summary.setObjectName("muted")
        root.addWidget(self.summary)
        root.addWidget(self.plot)

    def set_products(self, products, slices, selected_uid=None):
        self.plot.clear()
        self.legend.clear()
        self.curves = []
        for product, item in zip(products, slices):
            values = np.asarray(product.values, dtype=float)
            if values.size == 0:
                continue
            profile = np.nan_to_num(np.sum(values, axis=0), copy=False)
            selected = item.uid == selected_uid
            curve = self.plot.plot(
                np.arange(profile.size),
                profile,
                pen=pg.mkPen(item.color, width=2.6 if selected else 1.5),
                name=item.title,
            )
            self.curves.append(curve)
        count = len(self.curves)
        self.summary.setText(
            f"{count} visible slice profile{'s' if count != 1 else ''}"
            if count else "No visible slice profiles"
        )

    def set_image(self, values):
        """Compatibility helper for callers that only have one image array."""
        product = type("ProfileProduct", (), {"values": values})()
        item = type(
            "ProfileSlice",
            (),
            {"uid": "selected", "color": "#38bdf8", "title": "Selected slice"},
        )()
        self.set_products([product], [item], "selected")


class AnalysisDock(QWidget):
    band_changed = Signal(float, float)
    tab_changed = Signal(str)
    slice_changed = Signal(str, float, float)
    spectrum_settings_changed = Signal(int, float, bool)
    spectrum_scale_changed = Signal(str)
    spectrum_smooth_visibility_changed = Signal(bool)
    energy_scan_speed_changed = Signal(int)

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.spectrum = SpectrumWidget()
        self.images = ImagesCompareWidget()
        self.profile = SliceProfileWidget()
        self.tabs.addTab(self.spectrum, "Spectrum")
        self.tabs.addTab(self.images, "Slice images")
        self.tabs.addTab(self.profile, "Slice profiles")
        root.addWidget(self.tabs)
        self.spectrum.band_changed.connect(self.band_changed)
        self.spectrum.slice_changed.connect(self.slice_changed)
        self.spectrum.settings_changed.connect(self.spectrum_settings_changed)
        self.spectrum.scale_changed.connect(self.spectrum_scale_changed)
        self.spectrum.smooth_visibility_changed.connect(
            self.spectrum_smooth_visibility_changed
        )
        self.spectrum.scan_speed_changed.connect(self.energy_scan_speed_changed)
        self.tabs.currentChanged.connect(self._tab_changed)

    def _tab_changed(self, _index: int):
        current = self.tabs.currentWidget()
        if current is self.images:
            self.tab_changed.emit("images")
        elif current is self.profile:
            self.tab_changed.emit("profile")
        else:
            self.tab_changed.emit("spectrum")
