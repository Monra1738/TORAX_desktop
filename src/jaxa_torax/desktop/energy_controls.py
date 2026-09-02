"""Compact native controls for the global scientific energy band."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QPushButton, QSlider

from jaxa_torax.desktop.scalar_display_widgets import add_scalar_palette_items

ENERGY_SLIDER_SCALE = 100
ENERGY_SLIDER_MAX_KEV = 100.0


def build_energy_page(panel, page_factory, spinbox_factory):
    page, box = page_factory("GLOBAL ENERGY FILTER")
    panel.energy_low = spinbox_factory(0.0, 1000.0, 2.0, 0.05, 3, " keV")
    panel.energy_high = spinbox_factory(0.001, 1000.0, 6.0, 0.05, 3, " keV")
    panel.energy_low_slider = QSlider(Qt.Horizontal)
    panel.energy_high_slider = QSlider(Qt.Horizontal)
    for slider, value in ((panel.energy_low_slider, 2.0), (panel.energy_high_slider, 6.0)):
        slider.setRange(0, int(ENERGY_SLIDER_MAX_KEV * ENERGY_SLIDER_SCALE))
        slider.setValue(round(value * ENERGY_SLIDER_SCALE))
        slider.setToolTip("Move the current energy-band boundary (0–100 keV)")
    panel.auto_image_quality = QCheckBox("Auto image quality")
    panel.auto_image_quality.setChecked(True)
    panel.auto_image_quality.setToolTip("Automatically choose smoothing from active-band events.")
    panel.image_smoothing = spinbox_factory(0.0, 20.0, 0.8, 0.1, 2, " px")
    panel.image_smoothing.setEnabled(False)
    form = QFormLayout()
    form.addRow("Lower", panel.energy_low)
    form.addRow("Upper", panel.energy_high)
    form.addRow("Lower slider", panel.energy_low_slider)
    form.addRow("Upper slider", panel.energy_high_slider)
    form.addRow("Image quality", panel.auto_image_quality)
    form.addRow("Custom smoothing", panel.image_smoothing)
    box.addLayout(form)
    box.addWidget(panel._subheading("BAND APPLICATION"))
    panel.spectrum_linked = QCheckBox("Link spectrum band to 3D + 2D image")
    panel.spectrum_linked.setChecked(True)
    panel.spectrum_linked.setToolTip(
        "When enabled, moving the spectrum band immediately updates the 3D points/voxels "
        "and the 2D energy image. Turn it off to control each view independently."
    )
    box.addWidget(panel.spectrum_linked)
    panel.filter_3d_by_energy = QCheckBox("Filter 3D points / voxels by selected band")
    panel.filter_2d_by_energy = QCheckBox("Filter 2D energy image by selected band")
    panel.filter_3d_by_energy.setChecked(False)
    panel.filter_2d_by_energy.setChecked(False)
    panel.filter_3d_by_energy.setToolTip(
        "Off shows every loaded preview event in 3D, subject to the safe point budget."
    )
    panel.filter_2d_by_energy.setToolTip(
        "Off compresses all loaded preview energies into the RA/DEC image."
    )
    box.addWidget(panel.filter_3d_by_energy)
    box.addWidget(panel.filter_2d_by_energy)
    box.addWidget(panel._subheading("IMAGE DISPLAY"))
    display = QFormLayout()
    panel.image_stretch = QComboBox()
    for label, key in (("Linear", "linear"), ("Square root", "sqrt"), ("Log", "log")):
        panel.image_stretch.addItem(label, key)
    panel.image_stretch.setCurrentIndex(2)
    panel.image_palette = QComboBox()
    add_scalar_palette_items(panel.image_palette)
    panel.image_brightness = QSlider(Qt.Horizontal)
    panel.image_contrast = QSlider(Qt.Horizontal)
    for slider, label in ((panel.image_brightness, "brightness"), (panel.image_contrast, "contrast")):
        slider.setRange(25, 300)
        slider.setValue(100)
        slider.setToolTip(f"Display-only scalar image {label}")
    display.addRow("Stretch", panel.image_stretch)
    display.addRow("Brightness", panel.image_brightness)
    display.addRow("Contrast", panel.image_contrast)
    display.addRow("Palette", panel.image_palette)
    box.addLayout(display)
    note = QLabel(
        "With the link enabled, dragging or scanning the spectrum updates both views. "
        "The 3D Energy (Z) axis always follows the active 3D energy range exactly. "
        "Disable it to show all events or apply the two filters independently."
    )
    note.setObjectName("muted")
    note.setWordWrap(True)
    box.addWidget(note)
    panel.energy_quality = QLabel("PREVIEW")
    panel.energy_quality.setObjectName("qualityBadge")
    box.addWidget(panel.energy_quality)
    exact = QPushButton("Compute exact energy image")
    exact.setObjectName("primary")
    exact.clicked.connect(panel.exact_energy_requested)
    box.addWidget(exact)
    panel.exact_all_events_energy = QPushButton("All events → exact 2D image")
    panel.exact_all_events_energy.setObjectName("primary")
    panel.exact_all_events_energy.setToolTip(
        "Stream every valid event from every visible parquet into the RA/DEC image. "
        "The 3D view remains a bounded interactive preview."
    )
    panel.exact_all_events_energy.clicked.connect(panel.exact_all_events_requested)
    box.addWidget(panel.exact_all_events_energy)
    box.addStretch(1)
    panel.energy_low.valueChanged.connect(panel._energy_emit)
    panel.energy_high.valueChanged.connect(panel._energy_emit)
    panel.energy_low_slider.valueChanged.connect(panel._energy_slider_emit)
    panel.energy_high_slider.valueChanged.connect(panel._energy_slider_emit)
    panel.filter_3d_by_energy.toggled.connect(panel._energy_filter_scope_emit)
    panel.filter_2d_by_energy.toggled.connect(panel._energy_filter_scope_emit)
    panel.spectrum_linked.toggled.connect(panel._spectrum_link_emit)
    panel.image_smoothing.valueChanged.connect(lambda *_: panel.image_quality_changed.emit())
    panel.auto_image_quality.toggled.connect(panel._auto_quality_toggled)
    panel.image_stretch.currentIndexChanged.connect(panel._scalar_display_emit)
    panel.image_palette.currentIndexChanged.connect(panel._scalar_display_emit)
    panel.image_brightness.valueChanged.connect(panel._scalar_display_emit)
    panel.image_contrast.valueChanged.connect(panel._scalar_display_emit)
    panel._wrap(page)
