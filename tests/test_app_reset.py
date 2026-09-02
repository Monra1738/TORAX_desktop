from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxa_torax.desktop.app_reset import clear_application_storage


class ApplicationResetTests(unittest.TestCase):
    def test_total_reset_removes_generated_state_but_preserves_user_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data_cache" / "mission").mkdir(parents=True)
            (root / "data_cache" / "mission" / "events.parquet").write_bytes(b"cache")
            (root / "logs").mkdir()
            (root / "logs" / "app.log").write_text("log", encoding="utf-8")
            (root / "darts_events.duckdb").write_bytes(b"database")
            (root / "darts_events.duckdb.wal").write_bytes(b"wal")
            (root / "exports").mkdir()
            export = root / "exports" / "result.png"
            export.write_bytes(b"result")
            (root / "data").mkdir()
            local = root / "data" / "local.parquet"
            local.write_bytes(b"local")

            removed = clear_application_storage(root)

            self.assertFalse((root / "data_cache").exists())
            self.assertFalse((root / "logs").exists())
            self.assertFalse((root / "darts_events.duckdb").exists())
            self.assertFalse((root / "darts_events.duckdb.wal").exists())
            self.assertTrue(export.exists())
            self.assertTrue(local.exists())
            self.assertGreaterEqual(len(removed), 4)

    def test_reset_rejects_a_filesystem_root(self):
        with self.assertRaises(ValueError):
            clear_application_storage(Path(Path.cwd().anchor))


if __name__ == "__main__":
    unittest.main()
