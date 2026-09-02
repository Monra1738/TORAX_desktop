"""Small native widget helpers for display-only scalar image controls."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPixmap

from jaxa_udon3.desktop.science_views import SCALAR_PALETTE_LABELS, SCALAR_PALETTES


def _palette_icon(colors) -> QIcon:
    pixmap = QPixmap(64, 12)
    pixmap.fill(Qt.black)
    painter = QPainter(pixmap)
    gradient = QLinearGradient(0, 0, pixmap.width(), 0)
    for index, color in enumerate(colors):
        gradient.setColorAt(index / max(1, len(colors) - 1), QColor(color))
    painter.fillRect(pixmap.rect(), gradient)
    painter.end()
    return QIcon(pixmap)


def add_scalar_palette_items(combo):
    for key, colors in SCALAR_PALETTES.items():
        combo.addItem(_palette_icon(colors), SCALAR_PALETTE_LABELS[key], key)


def sync_scalar_display_controls(palette, stretch, brightness, contrast, state):
    for widget, value in (
        (brightness, round(float(state.image_brightness) * 100)),
        (contrast, round(float(state.image_contrast) * 100)),
    ):
        widget.blockSignals(True)
        widget.setValue(int(value))
        widget.blockSignals(False)
    for widget, value in ((palette, state.image_palette), (stretch, state.image_stretch)):
        index = widget.findData(str(value))
        widget.blockSignals(True)
        widget.setCurrentIndex(max(0, index))
        widget.blockSignals(False)
