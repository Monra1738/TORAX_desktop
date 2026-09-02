"""Seed a new persistent runtime volume from the packaged compatibility catalog."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from jaxa_torax.config import Settings


def ensure_runtime_seed(settings: Settings | None = None) -> bool:
    settings = settings or Settings.load()
    destination = settings.duckdb_path
    candidates = [
        Path(os.environ["TORAX_SEED_CATALOG"])
        if os.getenv("TORAX_SEED_CATALOG")
        else None,
    ]
    packaged = next((path for path in candidates if path is not None and path.exists()), None)
    if packaged is None or destination.exists() or destination.resolve() == packaged.resolve():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(packaged, destination)
    return True


def main() -> int:
    ensure_runtime_seed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
