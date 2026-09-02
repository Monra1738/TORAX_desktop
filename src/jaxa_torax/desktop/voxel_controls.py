from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel


def build_voxel_page(panel, page_factory, dspin):
    """Build compact voxel controls while keeping InspectorPanel focused on wiring."""
    page, box = page_factory("VOXEL DISPLAY")
    form = QFormLayout()
    panel.voxel_energy_source = QComboBox()
    panel.voxel_energy_source.addItem("Active slice (voxels)", "selected_slice")
    panel.voxel_energy_source.addItem("Global energy band", "global_band")
    panel.voxel_energy_source.addItem("All loaded energies", "all_energies")
    panel.voxel_spatial = dspin(0.05, 20.0, 0.55, 0.05, 2, " arcmin")
    panel.voxel_energy = dspin(0.005, 5.0, 0.22, 0.005, 3, " keV")
    panel.voxel_energy.setToolTip("Energy voxel width; minimum 0.005 keV (5 eV) for XRISM/Resolve")
    panel.voxel_smooth_spatial = dspin(0.0, 20.0, 1.0, 0.05, 2, " σ")
    panel.voxel_smooth_energy = dspin(0.0, 5.0, 1.0, 0.02, 3, " σ")
    panel.voxel_threshold = dspin(0.0, 1.0, 0.03, 0.01, 2)
    panel.voxel_opacity = dspin(0.05, 1.0, 0.72, 0.05, 2)
    panel.voxel_edges = QCheckBox("Draw individual voxel-cell edges")
    panel.voxel_edges.setChecked(False)
    for label, widget in (
        ("Energy source", panel.voxel_energy_source), ("Sky cell size", panel.voxel_spatial),
        ("Energy cell size", panel.voxel_energy), ("Sky smoothing", panel.voxel_smooth_spatial),
        ("Energy smoothing", panel.voxel_smooth_energy), ("Density cutoff", panel.voxel_threshold),
        ("Voxel opacity", panel.voxel_opacity), ("Cell edges", panel.voxel_edges),
    ):
        form.addRow(label, widget)
    box.addLayout(form)
    note = QLabel(
        "Each voxel is one RA × DEC × energy histogram cell. Smoothing changes "
        "displayed density, the cutoff removes low-density cells, and the active "
        "energy source controls both voxel data and the Z-axis."
    )
    note.setObjectName("muted")
    note.setWordWrap(True)
    box.addWidget(note)
    box.addStretch(1)
    for widget in (panel.voxel_spatial, panel.voxel_energy, panel.voxel_smooth_spatial, panel.voxel_smooth_energy, panel.voxel_threshold, panel.voxel_opacity):
        widget.valueChanged.connect(lambda *_: panel.voxel_changed.emit())
    panel.voxel_energy_source.currentIndexChanged.connect(lambda *_: panel.voxel_changed.emit())
    panel.voxel_edges.toggled.connect(lambda *_: panel.voxel_changed.emit())
    panel._wrap(page)


def sync_overlay_controls(panel, state):
    for widget, value in (
        (panel.show_coordinate_triad, getattr(state, "show_coordinate_triad", True)),
        (panel.show_grid_backdrop, getattr(state, "show_grid_backdrop", True)),
        (panel.show_coordinate_values, getattr(state, "show_coordinate_values", True)),
        (panel.show_slice_window, getattr(state, "show_slice_window", True)),
    ):
        widget.blockSignals(True)
        widget.setChecked(value)
        widget.blockSignals(False)
    panel.camera_preset.blockSignals(True)
    panel.camera_preset.setCurrentIndex(max(0, panel.camera_preset.findData(getattr(state, "camera_preset", "isometric"))))
    panel.camera_preset.blockSignals(False)
