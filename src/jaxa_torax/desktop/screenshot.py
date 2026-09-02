"""Reliable visible-workspace capture and a small in-app confirmation preview."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEventLoop, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)


def save_widget_screenshot(widget, destination: str | Path) -> Path:
    """Capture a visible Qt widget after pending paints have been processed."""
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not widget.isVisible():
        widget.show()
    widget.update()
    widget.repaint()
    application = QCoreApplication.instance()
    if application is not None:
        application.processEvents(QEventLoop.AllEvents, 250)
    pixmap = widget.grab()
    if pixmap.isNull() or pixmap.width() < 2 or pixmap.height() < 2:
        raise OSError("Qt returned an empty workspace image")
    if not pixmap.save(str(path), "PNG"):
        raise OSError(f"Could not write PNG screenshot: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Screenshot was not written: {path}")
    return path


def show_screenshot_preview(parent, path: str | Path) -> None:
    """Show the saved image and its resolved path without replacing the file."""
    path = Path(path).expanduser().resolve()
    dialog = QDialog(parent)
    dialog.setWindowTitle("Screenshot saved")
    dialog.resize(900, 650)
    layout = QVBoxLayout(dialog)
    image = QLabel()
    image.setAlignment(Qt.AlignCenter)
    pixmap = QPixmap(str(path))
    image.setPixmap(
        pixmap.scaled(860, 555, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    )
    layout.addWidget(image, 1)
    details = QLabel(f"Saved to:\n{path}\n{path.stat().st_size:,} bytes")
    details.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(details)
    buttons = QDialogButtonBox(QDialogButtonBox.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()
