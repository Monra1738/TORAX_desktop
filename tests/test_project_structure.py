import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "jaxa_udon3"
sys.path.insert(0, str(ROOT / "src"))


class ProjectStructureTests(unittest.TestCase):
    def test_source_modules_are_small_and_parseable(self):
        oversized = {}
        for path in SOURCE.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            lines = len(source.splitlines())
            if lines > 650:
                oversized[str(path.relative_to(ROOT))] = lines
        self.assertEqual(oversized, {})

    def test_desktop_distribution_has_no_web_ui_or_demo_module(self):
        self.assertFalse((SOURCE / "ui").exists())
        self.assertFalse((SOURCE / "desktop" / "demo.py").exists())

    def test_legacy_monolith_modules_do_not_return(self):
        forbidden = {"test_2.py", "darts_backend.py", "trame_app.py", "science_backend.py"}
        active_names = {path.name for path in SOURCE.rglob("*.py")}
        self.assertTrue(forbidden.isdisjoint(active_names))

    def test_active_source_does_not_depend_on_removed_legacy_folders(self):
        offenders = []
        for path in SOURCE.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "try_1" in source or "from old" in source or "import old" in source:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
