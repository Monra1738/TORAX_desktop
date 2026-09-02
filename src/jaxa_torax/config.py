"""Environment-driven configuration shared by every executable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_root: Path
    cache_limit_bytes: int

    @classmethod
    def load(cls) -> Settings:
        default_root = Path.cwd() / "var"
        return cls(
            data_root=Path(os.getenv("TORAX_DATA_ROOT", str(default_root))).expanduser().resolve(),
            cache_limit_bytes=max(
                0, int(os.getenv("TORAX_CACHE_LIMIT_BYTES", str(5 * 1024**3)))
            ),
        )

    @property
    def cache_dir(self) -> Path:
        return self.data_root / "data_cache"

    @property
    def export_dir(self) -> Path:
        return self.data_root / "exports"

    @property
    def duckdb_path(self) -> Path:
        return self.data_root / "darts_events.duckdb"
