"""Validated atomic filesystem publication."""

from __future__ import annotations

import os
import tempfile
from hashlib import sha256
from pathlib import Path


class ProductStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path_for(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("Product path escapes the configured root")
        return path

    def publish_bytes(self, relative_path: str, payload: bytes):
        destination = self.path_for(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".partial-", dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination, sha256(payload).hexdigest()
