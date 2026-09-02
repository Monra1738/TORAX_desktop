from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_3d_coordinates import responsive_axis_bounds


class Standalone3DCoordinateTests(unittest.TestCase):
    def test_energy_axis_uses_exact_global_limits_not_event_extrema(self):
        frame = pd.DataFrame(
            {
                "RA": [350.1, 350.6],
                "DEC": [58.65, 58.95],
                "KEV": [1.12, 1.87],
            }
        )
        self.assertEqual(
            responsive_axis_bounds(frame, 1.0, 2.0),
            (350.1, 350.6, 58.65, 58.95, 1.0, 2.0),
        )

    def test_reversed_global_limits_are_normalized(self):
        frame = pd.DataFrame(
            {"RA": [10.0], "DEC": [20.0], "KEV": [3.5]}
        )
        self.assertEqual(responsive_axis_bounds(frame, 4.0, 3.0)[4:], (3.0, 4.0))


if __name__ == "__main__":
    unittest.main()
