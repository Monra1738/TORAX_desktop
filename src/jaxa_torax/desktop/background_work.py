"""Generic main-window background operation plumbing."""

from PySide6.QtWidgets import QMessageBox

from jaxa_torax.desktop.workers import Worker


class BackgroundWorkMixin:
    def _run(
        self, function, *args, status="Working…", on_result=None,
        report_progress=False, **kwargs,
    ):
        self._busy_count += 1
        self.progress.setVisible(True)
        if self._load_session is None:
            self.progress.setRange(0, 0)
            self._update_status(status)
        worker = Worker(function, *args, **kwargs)
        if report_progress:
            worker.kwargs["progress_callback"] = worker.signals.progress.emit
            worker.signals.progress.connect(self._operation_progress)
        if on_result:
            worker.signals.result.connect(on_result)
        worker.signals.error.connect(self._worker_error)
        worker.signals.finished.connect(self._worker_finished)
        self.thread_pool.start(worker)

    def _operation_progress(self, completed: int, total: int, current: str):
        if self._load_session is not None:
            return
        self.progress.setRange(0, max(1, int(total)))
        self.progress.setValue(int(completed))
        self._update_status(f"Exact image {completed:,} / {total:,}  │  {current}")

    def _worker_error(self, trace: str):
        if self._load_session is None:
            self._update_status("Operation failed")
        QMessageBox.critical(self, "TORAX operation failed", trace[-5000:])

    def _worker_finished(self):
        self._busy_count = max(0, self._busy_count - 1)
        loading = self._load_session is not None and not self._load_session.cancelled
        self.progress.setVisible(self._busy_count > 0 or loading)

