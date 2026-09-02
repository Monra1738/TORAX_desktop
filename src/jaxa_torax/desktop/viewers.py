from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from jaxa_torax.desktop.science_views import (
    SkyViewport,
    energy_gradient_colors,
    mission_event_colors,
    scalar_display_values,
    scalar_to_rgb,
    wrapped_ra,
)
from jaxa_torax.desktop.theme import (
    PLOT_BACKGROUND,
    PLOT_TEXT,
)


def _brushes(rgb: np.ndarray):
    return [pg.mkBrush(int(r), int(g), int(b), 205) for r, g, b in np.asarray(rgb)]


def _sample_frame(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    index = np.linspace(0, len(frame) - 1, max(1, int(limit)), dtype=int)
    return frame.iloc[index]


class NormalizedRAAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [f"{(float(value) % 360.0):g}" for value in values]


def _fixed_view(plot, viewport: SkyViewport, view=None):
    x0, x1, y0, y1 = viewport.bounds
    vb = plot.getPlotItem().vb
    vb.setLimits(
        xMin=x0, xMax=x1, yMin=y0, yMax=y1,
        maxXRange=x1 - x0, maxYRange=y1 - y0,
    )
    wanted = viewport.clamp_view(view)
    vb.setRange(xRange=wanted[:2], yRange=wanted[2:], padding=0, disableAutoRange=True)


class SkyView(QWidget):
    event_selected = Signal(object)
    rectangle_changed = Signal(object)
    view_changed = Signal(object)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(
            background=PLOT_BACKGROUND,
            axisItems={"bottom": NormalizedRAAxis(orientation="bottom")},
        )
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Right Ascension", units="deg", color=PLOT_TEXT)
        self.plot.setLabel("left", "Declination", units="deg", color=PLOT_TEXT)
        self.plot.getPlotItem().invertX(True)
        self.scatter = pg.ScatterPlotItem(size=4, pxMode=True)
        self.plot.addItem(self.scatter)
        self.scatter.sigClicked.connect(self._point_clicked)
        self.roi = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("#c58b1a", width=1.6))
        self.roi.setZValue(20)
        self.roi.setVisible(False)
        self.roi.sigRegionChangeFinished.connect(self._roi_finished)
        self.plot.addItem(self.roi)
        layout.addWidget(self.plot)
        self._frame = pd.DataFrame()
        self._center_ra = None
        self._viewport = None
        self.plot.getPlotItem().vb.sigRangeChanged.connect(self._view_changed)

    def set_viewport(self, viewport: SkyViewport | None, view=None):
        self._viewport = viewport
        self._center_ra = None if viewport is None else viewport.center_ra_deg
        if viewport is not None:
            x0, x1, y0, y1 = viewport.bounds
            self.roi.maxBounds = QRectF(x0, y0, x1 - x0, y1 - y0)
            _fixed_view(self.plot, viewport, view)

    def set_frame(self, frame: pd.DataFrame, center_ra: float | None, color_mode="energy", rgb=None):
        self._frame = frame.reset_index(drop=True)
        self._center_ra = center_ra
        if frame.empty:
            self.scatter.setData([], [])
            return
        # RGB colors are calculated in the positional order of this filtered
        # frame.  Reset its index before sampling so a prior pandas filter does
        # not turn those positions into stale combined-frame indexes.
        local = _sample_frame(frame.reset_index(drop=True), 60_000)
        ra = local["RA"].to_numpy(float)
        if center_ra is not None:
            ra = wrapped_ra(ra, center_ra)
        dec = local["DEC"].to_numpy(float)
        if color_mode == "mission":
            colors = mission_event_colors(local["MISSION"].astype(str).to_numpy())
        elif color_mode == "rgb" and rgb is not None and len(rgb) == len(frame):
            colors = np.asarray(rgb)[local.index.to_numpy()]
        else:
            colors = energy_gradient_colors(local["KEV"].to_numpy(float))
        self._frame = local.reset_index(drop=True)
        self.scatter.setData(x=ra, y=dec, brush=_brushes(colors), pen=None, data=list(range(len(local))))

    def set_roi_enabled(self, enabled: bool):
        self.roi.setVisible(enabled)
        if enabled and not self._frame.empty:
            ra = self._frame["RA"].to_numpy(float)
            if self._center_ra is not None:
                ra = wrapped_ra(ra, self._center_ra)
            dec = self._frame["DEC"].to_numpy(float)
            x0, x1 = np.nanpercentile(ra, [30, 70])
            y0, y1 = np.nanpercentile(dec, [30, 70])
            self.set_roi((float(x0), float(x1), float(y0), float(y1)))
        elif enabled and self._viewport is not None:
            x0, x1, y0, y1 = self._viewport.bounds
            self.set_roi((
                x0 + 0.3 * (x1 - x0), x0 + 0.7 * (x1 - x0),
                y0 + 0.3 * (y1 - y0), y0 + 0.7 * (y1 - y0),
            ))

    def _point_clicked(self, _plot, points, _event):
        if points and not self._frame.empty:
            idx = int(points[0].data())
            if 0 <= idx < len(self._frame):
                self.event_selected.emit(self._frame.iloc[idx].to_dict())

    def _roi_finished(self):
        if not self.roi.isVisible():
            return
        pos, size = self.roi.pos(), self.roi.size()
        ra0, ra1 = sorted((float(pos.x()), float(pos.x() + size.x())))
        dec0, dec1 = sorted((float(pos.y()), float(pos.y() + size.y())))
        rectangle = (ra0, ra1, dec0, dec1)
        if self._viewport is not None:
            rectangle = self._viewport.clamp_rectangle(rectangle)
            self.set_roi(rectangle)
        self.rectangle_changed.emit(rectangle)

    def set_roi(self, rectangle):
        if rectangle is None:
            return
        if self._viewport is not None:
            rectangle = self._viewport.clamp_rectangle(rectangle)
        a, b, c, d = rectangle
        # Synchronizing this ROI from another view must not emit another
        # rectangle_changed event.  That event would immediately synchronize
        # all views again and recurse until Python's recursion limit.
        was_blocked = self.roi.blockSignals(True)
        try:
            self.roi.setPos((a, c))
            self.roi.setSize((b - a, d - c))
        finally:
            self.roi.blockSignals(was_blocked)

    def _view_changed(self, *_):
        x, y = self.plot.viewRange()
        self.view_changed.emit((x[0], x[1], y[0], y[1]))

    def reset(self):
        if self._viewport is not None:
            _fixed_view(self.plot, self._viewport)
        else:
            self.plot.autoRange()


class ImagePlot(QWidget):
    rectangle_changed = Signal(object)
    view_changed = Signal(object)

    def __init__(self, title="Energy image"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(
            background=PLOT_BACKGROUND,
            axisItems={"bottom": NormalizedRAAxis(orientation="bottom")},
        )
        self.plot.getPlotItem().invertX(True)  # astronomical convention: RA increases left.
        self.plot.showGrid(x=True, y=True, alpha=0.20)
        self.plot.setLabel("bottom", "Right Ascension", units="deg", color=PLOT_TEXT)
        self.plot.setLabel("left", "Declination", units="deg", color=PLOT_TEXT)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.label = pg.TextItem(title, color=PLOT_TEXT, anchor=(0, 0))
        self.plot.addItem(self.label)
        self.hover_label = pg.TextItem("", color=PLOT_TEXT, anchor=(0, 1))
        self.hover_label.setZValue(40)
        self.plot.addItem(self.hover_label)
        self.roi = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("#c58b1a", width=1.6))
        self.roi.setVisible(False)
        self.roi.setZValue(30)
        self.roi.sigRegionChangeFinished.connect(self._roi_finished)
        self.plot.addItem(self.roi)
        layout.addWidget(self.plot)
        self._bounds = None
        self._scalar_values = None
        self._scalar_display = None
        self._scalar_settings = ("gray", "log", 1.0, 1.0)
        self._viewport = None
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.getPlotItem().vb.sigRangeChanged.connect(self._view_changed)

    def set_viewport(self, viewport: SkyViewport | None, view=None):
        self._viewport = viewport
        if viewport is not None:
            x0, x1, y0, y1 = viewport.bounds
            self.roi.maxBounds = QRectF(x0, y0, x1 - x0, y1 - y0)
            _fixed_view(self.plot, viewport, view)

    def set_scalar(
        self, values, x_edges, y_edges, title="Energy image", *,
        palette="gray", stretch="log", brightness=1.0, contrast=1.0,
    ):
        """Store raw counts once; display settings may remap without histograms."""
        self._scalar_values = np.asarray(values, dtype=float)
        self._scalar_settings = (str(palette), str(stretch), float(brightness), float(contrast))
        self._set_scalar_image(x_edges, y_edges, title)

    def refresh_scalar_display(self, *, palette, stretch, brightness, contrast):
        """Cheap LUT-only scalar refresh; keeps bounds, ROI, and raw counts intact."""
        if self._scalar_values is None or self._bounds is None:
            return
        self._scalar_settings = (str(palette), str(stretch), float(brightness), float(contrast))
        self._scalar_display = scalar_display_values(
            self._scalar_values, stretch, brightness, contrast
        )
        self.image.setImage(
            scalar_to_rgb(self._scalar_values, palette, stretch, brightness, contrast),
            autoLevels=False,
        )

    def _set_scalar_image(self, x_edges, y_edges, title):
        palette, stretch, brightness, contrast = self._scalar_settings
        self._scalar_display = scalar_display_values(
            self._scalar_values, stretch, brightness, contrast
        )
        self.set_rgb(
            scalar_to_rgb(self._scalar_values, palette, stretch, brightness, contrast),
            x_edges, y_edges, title,
        )

    def set_rgb(self, rgb, x_edges, y_edges, title="RGB composite"):
        if rgb is None or np.asarray(rgb).size == 0:
            self.image.clear()
            return
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.asarray(np.clip(arr, 0, 1) * 255, dtype=np.uint8)
        x0, x1 = float(x_edges[0]), float(x_edges[-1])
        y0, y1 = float(y_edges[0]), float(y_edges[-1])
        self._bounds = (x0, x1, y0, y1)
        self.image.setImage(arr, autoLevels=False)
        self.image.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        self.label.setText(title)
        self.label.setPos(max(x0, x1), min(y0, y1))
        if self._viewport is None:
            self.plot.setXRange(min(x0, x1), max(x0, x1), padding=0)
            self.plot.setYRange(min(y0, y1), max(y0, y1), padding=0)

    def _mouse_moved(self, point):
        """Expose raw counts and post-transform intensity without changing data."""
        if self._scalar_values is None or self._scalar_display is None or self._bounds is None:
            return
        position = self.plot.getPlotItem().vb.mapSceneToView(point)
        x0, x1, y0, y1 = self._bounds
        if not (min(x0, x1) <= position.x() <= max(x0, x1) and min(y0, y1) <= position.y() <= max(y0, y1)):
            self.hover_label.setText("")
            return
        rows, columns = self._scalar_values.shape[:2]
        column = min(columns - 1, max(0, int((position.x() - x0) / max(x1 - x0, 1e-12) * columns)))
        row = min(rows - 1, max(0, int((position.y() - y0) / max(y1 - y0, 1e-12) * rows)))
        raw = float(self._scalar_values[row, column])
        display = float(self._scalar_display[row, column])
        self.hover_label.setText(
            f"RA {(position.x() % 360.0):.5f}°  DEC {position.y():+.5f}°\n"
            f"Events {raw:.3g}  Display {display:.3f}"
        )
        self.hover_label.setPos(position.x(), position.y())

    def set_roi_enabled(self, enabled: bool):
        self.roi.setVisible(enabled)
        if enabled and self._bounds:
            x0, x1, y0, y1 = self._bounds
            self.set_roi((
                x0 + 0.3 * (x1 - x0), x0 + 0.7 * (x1 - x0),
                y0 + 0.3 * (y1 - y0), y0 + 0.7 * (y1 - y0),
            ))

    def _roi_finished(self):
        if self.roi.isVisible():
            pos, size = self.roi.pos(), self.roi.size()
            ra0, ra1 = sorted((float(pos.x()), float(pos.x() + size.x())))
            dec0, dec1 = sorted((float(pos.y()), float(pos.y() + size.y())))
            rectangle = (ra0, ra1, dec0, dec1)
            if self._viewport is not None:
                rectangle = self._viewport.clamp_rectangle(rectangle)
                self.set_roi(rectangle)
            self.rectangle_changed.emit(rectangle)

    def set_roi(self, rectangle):
        if rectangle is None:
            return
        if self._viewport is not None:
            rectangle = self._viewport.clamp_rectangle(rectangle)
        a, b, c, d = rectangle
        was_blocked = self.roi.blockSignals(True)
        try:
            self.roi.setPos((a, c))
            self.roi.setSize((b - a, d - c))
        finally:
            self.roi.blockSignals(was_blocked)

    def _view_changed(self, *_):
        x, y = self.plot.viewRange()
        self.view_changed.emit((x[0], x[1], y[0], y[1]))

    def reset(self):
        if self._viewport is not None:
            _fixed_view(self.plot, self._viewport)
        else:
            self.plot.autoRange()


from jaxa_torax.desktop.viewer_3d import ThreeDView


class WorkspaceWidget(QWidget):
    event_selected = Signal(object)
    rectangle_changed = Signal(object)
    layout_changed = Signal(str)
    product_changed = Signal(str)
    screenshot_requested = Signal()
    fullscreen_requested = Signal()
    view_changed = Signal(str, object)

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        bar = QFrame()
        bar.setObjectName("workspaceBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(7, 5, 7, 5)
        row.addWidget(QLabel("Layout"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("3D + 2D", "split")
        self.layout_combo.addItem("3D only", "3d")
        self.layout_combo.addItem("2D only", "2d")
        self.layout_combo.currentIndexChanged.connect(self._layout_emit)
        row.addWidget(self.layout_combo)
        row.addSpacing(10)
        row.addWidget(QLabel("2D product"))
        self.product_combo = QComboBox()
        self.product_combo.addItem("Energy band", "energy")
        self.product_combo.addItem("Selected slice", "slice")
        self.product_combo.addItem("RGB composite", "rgb")
        self.product_combo.addItem("Event sky", "sky")
        self.product_combo.currentIndexChanged.connect(self._product_emit)
        row.addWidget(self.product_combo)
        row.addStretch(1)
        self.roi_button = QPushButton("ROI")
        self.roi_button.setCheckable(True)
        self.roi_button.toggled.connect(self._roi_toggled)
        row.addWidget(self.roi_button)
        for label, callback in (
            ("Reset", self.reset_active), ("Fit", self.fit_active),
            ("⛶", lambda: self.fullscreen_requested.emit()),
            ("📷", lambda: self.screenshot_requested.emit()),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        root.addWidget(bar)

        self.splitter = QSplitter(Qt.Horizontal)
        self.three_d = ThreeDView()
        self.two_d_stack = QStackedWidget()
        self.energy = ImagePlot("Energy band image")
        self.slice = ImagePlot("Selected energy slice")
        self.rgb = ImagePlot("RGB energy composite")
        self.sky = SkyView()
        for widget in (self.energy, self.slice, self.rgb, self.sky):
            self.two_d_stack.addWidget(widget)
        self.splitter.addWidget(self.three_d)
        self.splitter.addWidget(self.two_d_stack)
        self.splitter.setSizes([900, 520])
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)
        root.addWidget(self.splitter, 1)

        self.three_d.event_selected.connect(self.event_selected)
        self.sky.event_selected.connect(self.event_selected)
        self.sky.rectangle_changed.connect(self.rectangle_changed)
        self.energy.rectangle_changed.connect(self.rectangle_changed)
        self.slice.rectangle_changed.connect(self.rectangle_changed)
        self.rgb.rectangle_changed.connect(self.rectangle_changed)
        for key, widget in (("energy", self.energy), ("slice", self.slice), ("rgb", self.rgb), ("sky", self.sky)):
            widget.view_changed.connect(lambda view, key=key: self.view_changed.emit(key, view))

    def set_sky_viewport(self, viewport: SkyViewport | None, zooms=None):
        zooms = dict(zooms or {})
        for key, widget in (("energy", self.energy), ("slice", self.slice), ("rgb", self.rgb), ("sky", self.sky)):
            widget.set_viewport(viewport, zooms.get(key))

    def set_roi(self, rectangle):
        for widget in (self.sky, self.energy, self.slice, self.rgb):
            widget.set_roi(rectangle)

    def set_layout_mode(self, mode: str):
        self.three_d.setVisible(mode in ("split", "3d"))
        self.two_d_stack.setVisible(mode in ("split", "2d"))
        if mode == "split":
            self.splitter.setSizes([900, 520])
        index = self.layout_combo.findData(mode)
        if index >= 0 and index != self.layout_combo.currentIndex():
            self.layout_combo.blockSignals(True)
            self.layout_combo.setCurrentIndex(index)
            self.layout_combo.blockSignals(False)

    def set_two_d_product(self, product: str):
        mapping = {"energy": 0, "slice": 1, "rgb": 2, "sky": 3}
        if product in mapping:
            self.two_d_stack.setCurrentIndex(mapping[product])
        index = self.product_combo.findData(product)
        if index >= 0 and index != self.product_combo.currentIndex():
            self.product_combo.blockSignals(True)
            self.product_combo.setCurrentIndex(index)
            self.product_combo.blockSignals(False)

    def _layout_emit(self):
        mode = str(self.layout_combo.currentData())
        self.set_layout_mode(mode)
        self.layout_changed.emit(mode)

    def _product_emit(self):
        product = str(self.product_combo.currentData())
        self.set_two_d_product(product)
        self.product_changed.emit(product)

    def _roi_toggled(self, enabled: bool):
        for widget in (self.sky, self.energy, self.slice, self.rgb):
            widget.set_roi_enabled(enabled)

    def reset_active(self):
        if self.three_d.isVisible():
            self.three_d.reset()
        if self.two_d_stack.isVisible():
            widget = self.two_d_stack.currentWidget()
            if hasattr(widget, "reset"):
                widget.reset()

    def fit_active(self):
        self.reset_active()
