"""Small Qt thread-pool helpers used to keep network/science work off the GUI thread."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    finished = Signal()
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str)


class Worker(QRunnable):
    def __init__(self, function: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception:
            try:
                self.signals.error.emit(traceback.format_exc())
            except RuntimeError:
                pass  # The window may have closed while native work completed.
        else:
            try:
                self.signals.result.emit(result)
            except RuntimeError:
                pass
        finally:
            try:
                self.signals.finished.emit()
            except RuntimeError:
                pass
