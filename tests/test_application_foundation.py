import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxa_torax.domain import CircleSelection, RectangleSelection
from jaxa_torax.infrastructure.filesystem import ProductStore


class ApplicationFoundationTests(unittest.TestCase):
    def test_domain_models_are_deterministic_and_validate(self):
        self.assertEqual(
            CircleSelection(370.0, 20.0, 0.5).key(),
            CircleSelection(10.0, 20.0, 0.5).key(),
        )
        self.assertTrue(RectangleSelection(350.0, 10.0, -5.0, 5.0).crosses_ra_zero)

    def test_product_publication_is_confined_and_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProductStore(Path(temporary))
            path, checksum = store.publish_bytes("exports/example.csv", b"x\n1\n")
            self.assertEqual(path.read_bytes(), b"x\n1\n")
            self.assertEqual(len(checksum), 64)
            with self.assertRaises(ValueError):
                store.path_for("../../escape")


if __name__ == "__main__":
    unittest.main()
