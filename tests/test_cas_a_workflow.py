import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxa_udon3.infrastructure.cas_a import (
    CAS_A_BANDS,
    CAS_A_REGION,
    cas_a_manifest,
    cas_a_record_specs,
)


class CasAWorkflowTests(unittest.TestCase):
    def test_reference_bands_and_region_match_theme_2_pdf(self):
        self.assertEqual(
            [(low, high) for low, high, _label in CAS_A_BANDS],
            [(1.55, 1.75), (1.75, 1.95), (2.35, 2.52), (3.93, 6.23)],
        )
        self.assertAlmostEqual(CAS_A_REGION.center_ra_deg, 350.8584)
        self.assertAlmostEqual(CAS_A_REGION.center_dec_deg, 58.8113)

    def test_real_data_manifest_is_deterministic_and_includes_both_resolve_ids(self):
        specs = cas_a_record_specs()
        keys = {spec.key for spec in specs}
        self.assertIn("xrism/resolve/000129000", keys)
        self.assertIn("xrism/resolve/000130000", keys)
        self.assertIn("asca/gis/50018000", keys)
        self.assertIn("asca/sis/50018000", keys)
        self.assertIn("asca/gis/50018010", keys)
        self.assertIn("asca/sis/50018010", keys)
        self.assertIn("suzaku/xis/507038010", keys)
        manifest = cas_a_manifest(specs, image_bins=240, output_dir=ROOT / "var" / "exports")
        self.assertEqual(manifest["image_bins"], 240)
        self.assertEqual(len(manifest["bands"]), 4)
        self.assertEqual(len(manifest["records"]), len(specs))

    def test_xtend_is_an_explicit_opt_in_for_large_files(self):
        base = {spec.key for spec in cas_a_record_specs()}
        wide = {spec.key for spec in cas_a_record_specs(include_xtend=True)}
        self.assertEqual(len(wide - base), 2)
        self.assertTrue({"xrism/xtend/000129000", "xrism/xtend/000130000"} <= wide)


if __name__ == "__main__":
    unittest.main()
