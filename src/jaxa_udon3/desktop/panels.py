from __future__ import annotations

from collections import defaultdict

import pandas as pd
from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from jaxa_udon3.desktop.data_controller import INSTRUMENTS


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label
def _spin(low, high, value, step=0.01, decimals=3, suffix="") -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(low, high)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    widget.setValue(value)
    if suffix:
        widget.setSuffix(suffix)
    return widget
class SearchDialog(QDialog):
    search_requested = Signal(dict)
    def __init__(
        self,
        parent=None,
        selected_pairs=None,
        target="",
        radius_arcmin=10.0,
        region=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Search / add DARTS observations")
        self.resize(600, 650)
        root = QVBoxLayout(self)
        title = QLabel("Search observations")
        title.setObjectName("appTitle")
        root.addWidget(title)
        subtitle = QLabel(
            "The resolved sky centre and radius become the fixed workspace region until you search again."
        )
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)
        quick_form = QFormLayout()
        self.quick_target = QComboBox()
        self.quick_target.addItem("Custom", "custom")
        self.quick_target.addItem("Cas A", "casa")
        self.quick_target.addItem("SN1006", "sn1006")
        self.quick_target.addItem("Galactic Center", "gc")
        self.quick_target.currentIndexChanged.connect(self._quick_target_selected)
        quick_form.addRow("Quick target", self.quick_target)
        root.addLayout(quick_form)
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Target name (Sesame)", "target")
        self.mode.addItem("RA / DEC degrees", "degrees")
        self.mode.addItem("RA / DEC sexagesimal", "sexagesimal")
        form.addRow("Input mode", self.mode)
        root.addLayout(form)
        self.input_stack = QStackedWidget()
        target_page = QWidget()
        target_form = QFormLayout(target_page)
        self.target_name = QLineEdit(target or "Cas A")
        self.target_name.setPlaceholderText("Cas A, SN 1006, Sgr A* …")
        target_form.addRow("Target", self.target_name)
        self.input_stack.addWidget(target_page)
        deg_page = QWidget()
        deg_form = QFormLayout(deg_page)
        self.ra_deg = QLineEdit("350.8500")
        self.dec_deg = QLineEdit("58.8150")
        deg_form.addRow("RA (deg)", self.ra_deg)
        deg_form.addRow("DEC (deg)", self.dec_deg)
        self.input_stack.addWidget(deg_page)
        sex_page = QWidget()
        sex_form = QFormLayout(sex_page)
        self.ra_sex = QLineEdit("23:23:24")
        self.dec_sex = QLineEdit("+58:48:54")
        sex_form.addRow("RA (HH:MM:SS)", self.ra_sex)
        sex_form.addRow("DEC (DD:MM:SS)", self.dec_sex)
        self.input_stack.addWidget(sex_page)
        root.addWidget(self.input_stack)
        self.mode.currentIndexChanged.connect(self.input_stack.setCurrentIndex)
        # Reopening Search / Add is normally an append operation.  Prefill the
        # fixed region as degrees so the displayed "RA …, DEC …" label is not
        # accidentally resolved as a Sesame object name.
        if region is not None:
            self.ra_deg.setText(f"{float(region.center_ra_deg):.6f}")
            self.dec_deg.setText(f"{float(region.center_dec_deg):.6f}")
            self.mode.setCurrentIndex(self.mode.findData("degrees"))
        region_form = QFormLayout()
        self.radius = _spin(0.05, 10800, radius_arcmin, 0.5, 2, " arcmin")
        region_form.addRow("Search radius", self.radius)
        root.addLayout(region_form)
        root.addWidget(section_label("MISSIONS & INSTRUMENTS"))
        instrument_box = QFrame()
        instrument_layout = QVBoxLayout(instrument_box)
        instrument_layout.setContentsMargins(4, 2, 4, 2)
        self.instrument_checks = {}
        selected = set(selected_pairs or [f"{m}/{i}" for m, i in INSTRUMENTS])
        grouped = defaultdict(list)
        for mission, instrument in INSTRUMENTS:
            grouped[mission].append(instrument)
        for mission, instruments in grouped.items():
            row = QHBoxLayout()
            mission_label = QLabel(mission.upper())
            mission_label.setMinimumWidth(70)
            mission_label.setStyleSheet("font-weight:700;")
            row.addWidget(mission_label)
            for instrument in instruments:
                pair = f"{mission}/{instrument}"
                check = QCheckBox(instrument.upper())
                check.setChecked(pair in selected)
                self.instrument_checks[pair] = check
                row.addWidget(check)
            row.addStretch(1)
            instrument_layout.addLayout(row)
        root.addWidget(instrument_box)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced search  ▸")
        self.advanced_toggle.setCheckable(True)
        root.addWidget(self.advanced_toggle)
        self.advanced = QFrame()
        advanced_form = QFormLayout(self.advanced)
        self.object_text = QLineEdit()
        self.obsid_text = QLineEdit()
        self.date_start = QLineEdit()
        self.date_end = QLineEdit()
        self.limit = QSpinBox()
        self.limit.setRange(10, 3000)
        self.limit.setValue(500)
        advanced_form.addRow("Object contains", self.object_text)
        advanced_form.addRow("Observation ID", self.obsid_text)
        advanced_form.addRow("Date start", self.date_start)
        advanced_form.addRow("Date end", self.date_end)
        advanced_form.addRow("Result limit", self.limit)
        self.advanced.setVisible(False)
        root.addWidget(self.advanced)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        root.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        search = QPushButton("Search observations")
        search.setObjectName("primary")
        buttons.addButton(search, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        search.clicked.connect(self._emit_search)
        root.addWidget(buttons)
    def _quick_target_selected(self, _index: int):
        preset = str(self.quick_target.currentData() or "custom")
        if preset == "casa":
            self.mode.setCurrentIndex(self.mode.findData("degrees"))
            self.ra_deg.setText("350.8584")
            self.dec_deg.setText("58.8113")
            self.target_name.setText("Cas A")
        elif preset == "sn1006":
            self.mode.setCurrentIndex(self.mode.findData("target"))
            self.target_name.setText("SN 1006")
            self.radius.setValue(60.0)
        elif preset == "gc":
            self.mode.setCurrentIndex(self.mode.findData("degrees"))
            self.ra_deg.setText("266.404996")
            self.dec_deg.setText("-28.936172")
            self.target_name.setText("Galactic Center")
            self.radius.setValue(60.0)
    def _toggle_advanced(self, visible: bool):
        self.advanced.setVisible(visible)
        self.advanced_toggle.setText(f"Advanced search  {'▾' if visible else '▸'}")
    def _emit_search(self):
        mode = self.mode.currentData()
        if mode == "target":
            ra_value = dec_value = ""
        elif mode == "degrees":
            ra_value, dec_value = self.ra_deg.text(), self.dec_deg.text()
        else:
            ra_value, dec_value = self.ra_sex.text(), self.dec_sex.text()
        self.search_requested.emit({
            "mode": mode,
            "target_name": self.target_name.text().strip(),
            "ra_value": ra_value,
            "dec_value": dec_value,
            "radius_arcmin": self.radius.value(),
            "selected_pairs": [p for p, c in self.instrument_checks.items() if c.isChecked()],
            "object_text": self.object_text.text().strip(),
            "observation_text": self.obsid_text.text().strip(),
            "date_start": self.date_start.text().strip(),
            "date_end": self.date_end.text().strip(),
            "limit": self.limit.value(),
        })
        self.accept()
class ObservationBrowserDialog(QDialog):
    add_requested = Signal(list)
    COLUMNS = (
        ("mission", "Mission"), ("instrument", "Instrument"),
        ("observation_id", "Obs ID"), ("object", "Object"),
        ("date_obs", "Date"), ("separation_deg", "Separation (deg)"),
    )
    def __init__(self, frame: pd.DataFrame, loaded_keys: set[str], parent=None, queued_keys=None):
        super().__init__(parent)
        self.setWindowTitle("Observation browser — add to workspace")
        self.resize(980, 620)
        self._frame = frame.reset_index(drop=True).copy()
        self._loaded_keys = set(loaded_keys)
        self._queued_keys = set(queued_keys or ())
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        self.summary = QLabel()
        self.summary.setObjectName("muted")
        header.addWidget(self.summary)
        header.addStretch(1)
        select_all = QPushButton("Select all not loaded")
        select_all.clicked.connect(self._select_all_not_loaded)
        header.addWidget(select_all)
        self.add_all_button = QPushButton("Load all matching observations")
        self.add_all_button.setObjectName("primary")
        self.add_all_button.setToolTip(
            "Queue every not-yet-loaded parquet observation returned by this sky search. "
            "Interactive views stay bounded; exact products can use every event."
        )
        self.add_all_button.clicked.connect(self._emit_add_all)
        header.addWidget(self.add_all_button)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self.table.clearSelection())
        header.addWidget(clear)
        self.add_button = QPushButton("Add 0 observations")
        self.add_button.setObjectName("primary")
        self.add_button.clicked.connect(self._emit_add)
        header.addWidget(self.add_button)
        root.addLayout(header)
        self.table = QTableWidget(0, len(self.COLUMNS) + 1)
        self.table.setHorizontalHeaderLabels(["State"] + [label for _, label in self.COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self._keys = []
        self.table.setRowCount(len(self._frame))
        for row_index, row in self._frame.iterrows():
            key = f"{row.get('mission','')}/{row.get('instrument','')}/{row.get('observation_id','')}"
            self._keys.append(key)
            loaded = key in self._loaded_keys
            state_item = QTableWidgetItem("LOADED" if loaded else ("LOADING" if key in self._queued_keys else ""))
            state_item.setData(Qt.UserRole, row_index)
            self.table.setItem(row_index, 0, state_item)
            for column_index, (column, _label) in enumerate(self.COLUMNS, start=1):
                value = row.get(column, "")
                text = "" if pd.isna(value) else (f"{value:.4f}" if isinstance(value, float) else str(value))
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, row_index)
                self.table.setItem(row_index, column_index, item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        root.addWidget(self.table, 1)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        root.addWidget(close)
        self._selection_changed()
    def _select_all_not_loaded(self):
        self.table.clearSelection()
        for visual_row in range(self.table.rowCount()):
            item = self.table.item(visual_row, 0)
            if item is None:
                continue
            original = int(item.data(Qt.UserRole))
            if self._keys[original] not in self._loaded_keys | self._queued_keys:
                self.table.selectionModel().select(
                    self.table.model().index(visual_row, 0),
                    QItemSelectionModel.Select | QItemSelectionModel.Rows,
                )
    def _selection_changed(self):
        selected = len(self.selected_keys())
        loaded = sum(key in self._loaded_keys for key in self._keys)
        self.summary.setText(
            f"{len(self._keys):,} search results  │  {loaded:,} already loaded  │  "
            f"{selected:,} new selected"
        )
        self.add_button.setText(f"Add {selected:,} observation{'s' if selected != 1 else ''}")
        available = sum(
            key not in self._loaded_keys and key not in self._queued_keys
            for key in self._keys
        )
        self.add_all_button.setText(f"Load all {available:,} matching")
        self.add_all_button.setEnabled(available > 0)
    def selected_keys(self) -> list[str]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        result = []
        for visual_row in rows:
            item = self.table.item(visual_row, 0)
            if item is None:
                continue
            original = int(item.data(Qt.UserRole))
            if 0 <= original < len(self._keys):
                key = self._keys[original]
                if key not in self._loaded_keys and key not in self._queued_keys:
                    result.append(key)
        return result
    def _emit_add(self):
        keys = self.selected_keys()
        if keys:
            self.add_requested.emit(keys)
            self.accept()
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Add observations", "Select at least one observation that is not already loaded.")
    def _emit_add_all(self):
        keys = [
            key for key in self._keys
            if key not in self._loaded_keys and key not in self._queued_keys
        ]
        if keys:
            self.add_requested.emit(keys)
            self.accept()
class DataLayersPanel(QWidget):
    search_clicked = Signal()
    visibility_changed = Signal(str, bool)
    visibility_many_changed = Signal(object, bool)
    observation_selected = Signal(str)
    render_mode_changed = Signal(str)
    slice_add_requested = Signal()
    slice_preset_requested = Signal(str)
    slice_selected = Signal(str)
    slice_visibility_changed = Signal(str, bool)
    slice_remove_requested = Signal(str)
    GROUP_KEYS_ROLE = int(Qt.UserRole) + 1
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(section_label("TARGET REGION"))
        target_card = QFrame()
        target_card.setObjectName("card")
        tbox = QVBoxLayout(target_card)
        self.target_title = QLabel("No target loaded")
        self.target_title.setObjectName("panelTitle")
        self.target_coords = QLabel("Search a target or coordinates")
        self.target_coords.setObjectName("muted")
        self.target_coords.setWordWrap(True)
        tbox.addWidget(self.target_title)
        tbox.addWidget(self.target_coords)
        search = QPushButton("Search / add observations")
        search.setObjectName("primary")
        search.clicked.connect(self.search_clicked)
        tbox.addWidget(search)
        root.addWidget(target_card)
        root.addWidget(section_label("LOADED OBSERVATIONS"))
        self.loaded_tree = QTreeWidget()
        self.loaded_tree.setHeaderHidden(True)
        self.loaded_tree.setUniformRowHeights(True)
        self.loaded_tree.itemChanged.connect(self._tree_changed)
        self.loaded_tree.currentItemChanged.connect(self._observation_selected)
        root.addWidget(self.loaded_tree, 4)
        root.addWidget(section_label("3D REPRESENTATION"))
        mode_row = QHBoxLayout()
        self.events_button = QPushButton("Events")
        self.density_button = QPushButton("Density")
        self.voxels_button = QPushButton("Voxels")
        for button in (self.events_button, self.density_button, self.voxels_button):
            button.setCheckable(True)
            button.setObjectName("modeButton")
            mode_row.addWidget(button)
        self.events_button.setChecked(True)
        self.events_button.clicked.connect(lambda: self._set_render_mode("events"))
        self.density_button.clicked.connect(lambda: self._set_render_mode("density"))
        self.voxels_button.clicked.connect(lambda: self._set_render_mode("voxels"))
        root.addLayout(mode_row)
        root.addWidget(section_label("ENERGY SLICES"))
        self.layers = QListWidget()
        self.layers.setUniformItemSizes(True)
        self.layers.itemChanged.connect(self._slice_item_changed)
        self.layers.currentItemChanged.connect(self._slice_selected)
        root.addWidget(self.layers, 2)
        preset_row = QHBoxLayout()
        self.slice_presets = QComboBox()
        self.slice_presets.addItem("Add reference bands…", "")
        self.slice_presets.addItem("Cas A — ASCA literature bands", "casa")
        self.slice_presets.currentIndexChanged.connect(self._slice_preset_selected)
        preset_row.addWidget(self.slice_presets, 1)
        add = QPushButton("+ Slice")
        add.clicked.connect(self.slice_add_requested)
        preset_row.addWidget(add)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_selected_slice)
        preset_row.addWidget(remove)
        root.addLayout(preset_row)
    def set_target(self, name: str, region, cache_hit: bool = False):
        self.target_title.setText(name or region.label or "Sky region")
        suffix = "  • cached" if cache_hit else ""
        self.target_coords.setText(
            f"RA {region.center_ra_deg:.5f}°\nDEC {region.center_dec_deg:+.5f}°\n"
            f"Radius {region.radius_deg * 60:.2f} arcmin{suffix}"
        )
    @staticmethod
    def _group_state(keys: list[str], visible_keys: set[str]):
        count = sum(key in visible_keys for key in keys)
        if count == 0:
            return Qt.Unchecked
        if count == len(keys):
            return Qt.Checked
        return Qt.PartiallyChecked
    def set_observations(self, observations, visible_keys: set[str] | None = None):
        """Mission → instrument → observation hierarchy with fast group toggles."""
        visible_keys = set(visible_keys or [])
        self.loaded_tree.blockSignals(True)
        self.loaded_tree.clear()
        grouped = defaultdict(lambda: defaultdict(list))
        for obs in observations:
            grouped[str(obs.record.mission).lower()][str(obs.record.instrument).lower()].append(obs)
        for mission in ("xrism", "hitomi", "suzaku", "asca"):
            instruments = grouped.get(mission, {})
            if not instruments:
                continue
            mission_keys = [
                f"{obs.record.mission}/{obs.record.instrument}/{obs.record.observation_id}"
                for rows in instruments.values() for obs in rows
            ]
            parent = QTreeWidgetItem([mission.upper()])
            parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable)
            parent.setData(0, self.GROUP_KEYS_ROLE, mission_keys)
            parent.setCheckState(0, self._group_state(mission_keys, visible_keys))
            parent.setExpanded(True)
            self.loaded_tree.addTopLevelItem(parent)
            for instrument, rows in sorted(instruments.items()):
                instrument_keys = [
                    f"{obs.record.mission}/{obs.record.instrument}/{obs.record.observation_id}"
                    for obs in rows
                ]
                instrument_item = QTreeWidgetItem([instrument.upper()])
                instrument_item.setFlags(instrument_item.flags() | Qt.ItemIsUserCheckable)
                instrument_item.setData(0, self.GROUP_KEYS_ROLE, instrument_keys)
                instrument_item.setCheckState(0, self._group_state(instrument_keys, visible_keys))
                instrument_item.setExpanded(True)
                parent.addChild(instrument_item)
                for obs in rows:
                    key = f"{obs.record.mission}/{obs.record.instrument}/{obs.record.observation_id}"
                    label = f"{obs.record.observation_id}   {obs.events_in_region:,} events"
                    child = QTreeWidgetItem([label])
                    child.setData(0, Qt.UserRole, key)
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.Checked if key in visible_keys else Qt.Unchecked)
                    instrument_item.addChild(child)
        self.loaded_tree.blockSignals(False)
    def sync_visibility_states(self, visible_keys: set[str]):
        """Update check marks without rebuilding the tree or emitting visibility signals."""
        visible_keys = set(visible_keys)
        self.loaded_tree.blockSignals(True)
        for top_index in range(self.loaded_tree.topLevelItemCount()):
            mission = self.loaded_tree.topLevelItem(top_index)
            mission_keys = list(mission.data(0, self.GROUP_KEYS_ROLE) or [])
            mission.setCheckState(0, self._group_state(mission_keys, visible_keys))
            for instrument_index in range(mission.childCount()):
                instrument = mission.child(instrument_index)
                instrument_keys = list(instrument.data(0, self.GROUP_KEYS_ROLE) or [])
                instrument.setCheckState(0, self._group_state(instrument_keys, visible_keys))
                for obs_index in range(instrument.childCount()):
                    child = instrument.child(obs_index)
                    key = child.data(0, Qt.UserRole)
                    if key:
                        child.setCheckState(0, Qt.Checked if str(key) in visible_keys else Qt.Unchecked)
        self.loaded_tree.blockSignals(False)
    def set_slices(self, slices, selected_uid: str | None):
        self.layers.blockSignals(True)
        self.layers.clear()
        for item in slices:
            row = QListWidgetItem(item.title)
            row.setData(Qt.UserRole, item.uid)
            row.setFlags(row.flags() | Qt.ItemIsUserCheckable)
            row.setCheckState(Qt.Checked if item.visible else Qt.Unchecked)
            row.setForeground(QBrush(QColor(item.color)))
            self.layers.addItem(row)
            if item.uid == selected_uid:
                self.layers.setCurrentItem(row)
        self.layers.blockSignals(False)
    def update_slice(self, slice_item, select: bool = False):
        """Update one slice row in place; used during live dragging to avoid rebuilding the list."""
        self.layers.blockSignals(True)
        for index in range(self.layers.count()):
            row = self.layers.item(index)
            if str(row.data(Qt.UserRole)) != str(slice_item.uid):
                continue
            row.setText(slice_item.title)
            row.setCheckState(Qt.Checked if slice_item.visible else Qt.Unchecked)
            row.setForeground(QBrush(QColor(slice_item.color)))
            if select:
                self.layers.setCurrentItem(row)
            break
        self.layers.blockSignals(False)
    def _set_group_children(self, item: QTreeWidgetItem, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for index in range(item.childCount()):
            child = item.child(index)
            child.setCheckState(0, state)
            self._set_group_children(child, checked)
    def _tree_changed(self, item: QTreeWidgetItem, _column: int):
        key = item.data(0, Qt.UserRole)
        if key:
            self.visibility_changed.emit(str(key), item.checkState(0) == Qt.Checked)
            return
        keys = item.data(0, self.GROUP_KEYS_ROLE)
        if not keys or item.checkState(0) == Qt.PartiallyChecked:
            return
        visible = item.checkState(0) == Qt.Checked
        self.loaded_tree.blockSignals(True)
        self._set_group_children(item, visible)
        self.loaded_tree.blockSignals(False)
        self.visibility_many_changed.emit(list(map(str, keys)), visible)
    def _observation_selected(self, current, _previous):
        if current is None:
            return
        key = current.data(0, Qt.UserRole)
        if key:
            self.observation_selected.emit(str(key))
    def _set_render_mode(self, mode: str):
        self.set_render_mode(mode)
        self.render_mode_changed.emit(mode)
    def set_render_mode(self, mode: str):
        self.events_button.blockSignals(True)
        self.density_button.blockSignals(True)
        self.voxels_button.blockSignals(True)
        self.events_button.setChecked(mode == "events")
        self.density_button.setChecked(mode == "density")
        self.voxels_button.setChecked(mode == "voxels")
        self.events_button.blockSignals(False)
        self.density_button.blockSignals(False)
        self.voxels_button.blockSignals(False)
    def _slice_preset_selected(self, index: int):
        preset = str(self.slice_presets.itemData(index) or "")
        if not preset:
            return
        self.slice_preset_requested.emit(preset)
        self.slice_presets.blockSignals(True)
        self.slice_presets.setCurrentIndex(0)
        self.slice_presets.blockSignals(False)
    def _slice_item_changed(self, item: QListWidgetItem):
        uid = item.data(Qt.UserRole)
        if uid:
            self.slice_visibility_changed.emit(str(uid), item.checkState() == Qt.Checked)
    def _slice_selected(self, current, _previous):
        if current is not None and current.data(Qt.UserRole):
            self.slice_selected.emit(str(current.data(Qt.UserRole)))
    def _remove_selected_slice(self):
        current = self.layers.currentItem()
        if current is not None and current.data(Qt.UserRole):
            self.slice_remove_requested.emit(str(current.data(Qt.UserRole)))
