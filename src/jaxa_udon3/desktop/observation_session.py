"""Progressive, failure-isolated observation preview loading."""

from __future__ import annotations

import errno
import logging
import time
import urllib.error
from logging.handlers import RotatingFileHandler
from threading import Lock
from traceback import format_exc
from uuid import uuid4

from PySide6.QtCore import QObject, Signal

from jaxa_udon3.desktop.data_controller import (
    COMBINED_PREVIEW_MAXIMUM,
    balanced_row_limits,
    load_observation_preview,
)
from jaxa_udon3.desktop.workers import Worker
from jaxa_udon3.infrastructure.science_core import APP_DATA_ROOT

MAX_LOAD_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = (0.2, 0.6, 1.5)
LOAD_DIAGNOSTIC_PATH = APP_DATA_ROOT / "logs" / "observation_loading.log"
_LOGGER = None
_LOGGER_LOCK = Lock()


class PreviewLoadError(RuntimeError):
    """Final preview error carrying automatic-retry context for the UI."""


def _diagnostic_logger():
    global _LOGGER
    with _LOGGER_LOCK:
        if _LOGGER is not None:
            return _LOGGER
        logger = logging.getLogger("jaxa_udon3.observation_loading")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            LOAD_DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                LOAD_DIAGNOSTIC_PATH,
                maxBytes=2 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        _LOGGER = logger
        return logger


def _write_diagnostic(level: int, message: str, *, include_traceback: bool = False):
    try:
        _diagnostic_logger().log(
            level,
            message + (f"\n{format_exc()}" if include_traceback else ""),
        )
    except Exception:
        # Diagnostics must never turn a recoverable preview problem into a load failure.
        pass


def is_transient_preview_error(error: Exception) -> bool:
    """Return whether retrying can plausibly succeed without changing the data."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or 500 <= error.code < 600
    if isinstance(error, (TimeoutError, ConnectionError, urllib.error.URLError)):
        return True
    if isinstance(error, OSError) and getattr(error, "errno", None) in {
        errno.EAGAIN,
        errno.EBUSY,
        errno.ECONNRESET,
        errno.ETIMEDOUT,
        errno.ECONNREFUSED,
    }:
        return True
    message = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in message
        for marker in (
            "different configuration than existing connections",
            "unique file handle conflict",
            "transaction conflict",
            "database is locked",
            "could not set lock",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection refused",
            "network is unreachable",
            "broken pipe",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
    )


def load_observation_preview_resilient(
    record,
    region,
    row_limit: int,
    *,
    retry_callback=None,
    sleep=time.sleep,
):
    """Load one preview with bounded retries for cache/network contention only."""
    key = record_key(record)
    for attempt in range(1, MAX_LOAD_ATTEMPTS + 1):
        try:
            observation = load_observation_preview(record, region, row_limit)
        except Exception as error:
            transient = is_transient_preview_error(error)
            final = not transient or attempt >= MAX_LOAD_ATTEMPTS
            _write_diagnostic(
                logging.ERROR if final else logging.WARNING,
                f"record={key} attempt={attempt}/{MAX_LOAD_ATTEMPTS} "
                f"transient={transient} error={type(error).__name__}: {error}",
                include_traceback=True,
            )
            if final:
                outcome = (
                    f"automatic retries exhausted after {attempt} attempts"
                    if transient
                    else "permanent/non-transient error"
                )
                raise PreviewLoadError(
                    f"{type(error).__name__}: {error} ({outcome}; "
                    f"diagnostic: {LOAD_DIAGNOSTIC_PATH})"
                ) from error
            next_attempt = attempt + 1
            if retry_callback is not None:
                retry_callback(key, next_attempt, MAX_LOAD_ATTEMPTS, str(error))
            sleep(RETRY_DELAYS_SECONDS[attempt - 1])
        else:
            if attempt > 1:
                _write_diagnostic(
                    logging.INFO,
                    f"record={key} recovered_on_attempt={attempt}/{MAX_LOAD_ATTEMPTS}",
                )
            return observation
    raise AssertionError("unreachable retry state")


def record_key(record) -> str:
    return f"{record.mission}/{record.instrument}/{record.observation_id}"


class ObservationLoadSession(QObject):
    observation_loaded = Signal(str, object)
    observation_failed = Signal(str, str, str)
    observation_retrying = Signal(str, str, int, int, str)
    progress_changed = Signal(str, int, int, str)
    session_finished = Signal(str, object, object)
    session_cancelled = Signal(str)

    def __init__(
        self,
        thread_pool,
        records,
        region,
        *,
        workspace_id: str = "",
        region_signature=None,
        existing_keys=(),
        combined_maximum: int = COMBINED_PREVIEW_MAXIMUM,
        parent=None,
    ):
        super().__init__(parent)
        self.session_id = uuid4().hex
        self.workspace_id = str(workspace_id or "")
        self.region = region
        self.region_signature = region_signature
        excluded = {str(key) for key in existing_keys}
        seen = set(excluded)
        self.records = []
        for record in records:
            key = record_key(record)
            if key not in seen:
                seen.add(key)
                self.records.append(record)
        self.ordered_record_queue = list(self.records)
        self.total_count = len(self.records)
        self.completed_count = 0
        self.successful_records: dict[str, object] = {}
        self.failed_records: dict[str, dict] = {}
        self.cancelled = False
        self._thread_pool = thread_pool
        self._next_index = 0
        self._active = 0
        self._workers: dict[str, Worker] = {}
        self._limits = balanced_row_limits(self.records, combined_maximum)

    @property
    def successful_keys(self) -> list[str]:
        return [record_key(record) for record in self.records if record_key(record) in self.successful_records]

    def start(self):
        if not self.records:
            self.session_finished.emit(self.session_id, [], {})
            return
        self._launch_available()

    def cancel(self):
        if self.cancelled:
            return
        self.cancelled = True
        self.session_cancelled.emit(self.session_id)

    def _launch_available(self):
        while not self.cancelled and self._active < 3 and self._next_index < self.total_count:
            index = self._next_index
            self._next_index += 1
            record = self.records[index]
            key = record_key(record)
            worker = Worker(
                load_observation_preview_resilient,
                record,
                self.region,
                self._limits[index],
                retry_callback=self._retrying,
            )
            worker.signals.result.connect(
                lambda observation, key=key: self._loaded(key, observation)
            )
            worker.signals.error.connect(
                lambda trace, key=key, record=record: self._failed(key, record, trace)
            )
            self._workers[key] = worker
            self._active += 1
            self._thread_pool.start(worker)

    def _retrying(self, key: str, attempt: int, maximum: int, message: str):
        if not self.cancelled:
            self.observation_retrying.emit(
                self.session_id, key, int(attempt), int(maximum), str(message)
            )

    def _loaded(self, key: str, observation):
        if key not in self._workers:
            return
        self._workers.pop(key, None)
        self._active = max(0, self._active - 1)
        if not self.cancelled:
            self.successful_records[key] = observation
            self.observation_loaded.emit(self.session_id, observation)
            self._completed(key)

    def _failed(self, key: str, record, trace: str):
        if key not in self._workers:
            return
        self._workers.pop(key, None)
        self._active = max(0, self._active - 1)
        if not self.cancelled:
            lines = [line.strip() for line in str(trace).splitlines() if line.strip()]
            message = lines[-1] if lines else "Unknown preview loading error"
            self.failed_records[key] = {
                "message": message,
                "source": str(getattr(record, "source", "unknown")),
                "parquet_path": str(getattr(record, "parquet_path", "")),
            }
            self.observation_failed.emit(self.session_id, key, message)
            self._completed(key)

    def _completed(self, key: str):
        self.completed_count += 1
        self.progress_changed.emit(
            self.session_id, self.completed_count, self.total_count, key
        )
        if self.completed_count >= self.total_count:
            self.session_finished.emit(
                self.session_id, self.successful_keys, dict(self.failed_records)
            )
            return
        self._launch_available()
