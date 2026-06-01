"""Tests cho scripts/drivers_250_fn_expansions.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_SCRIPT = _SCRIPTS / "drivers_250_fn_expansions.py"
_spec = importlib.util.spec_from_file_location("drivers_250_fn_expansions", _SCRIPT)
assert _spec and _spec.loader
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)


class TestDrivers250FnExpansions(unittest.TestCase):
    def test_expansion_constants(self):
        self.assertEqual(exp.SFAS_BBOX_EXPANSIONS, (1.0, 1.2, 1.4, 1.6))

    def test_crop_output_dir_baseline(self):
        p = exp.crop_output_dir(1.0, repo_root=ROOT)
        self.assertEqual(p, ROOT / "data" / "drivers_250_fn_cropped")

    def test_crop_output_dir_with_scale(self):
        p = exp.crop_output_dir(1.4, repo_root=ROOT)
        self.assertEqual(p, ROOT / "data/drivers_250_fn_cropped_sfas_exp1.4")

    def test_source_dataset_slug(self):
        self.assertEqual(exp.source_dataset_slug(1.0), "drivers_250_fn")
        self.assertEqual(exp.source_dataset_slug(1.2), "drivers_250_fn_exp1.2")

    def test_sample_csv_name(self):
        self.assertEqual(exp.sample_csv_name(1.0), "drivers_250_fn_sample.csv")
        self.assertEqual(exp.sample_csv_name(1.6), "drivers_250_fn_exp1.6_sample.csv")

    def test_dataset_config_path(self):
        p = exp.dataset_config_path(1.2, repo_root=ROOT)
        self.assertEqual(p, ROOT / "configs" / "dataset_drivers_250_fn_exp1.2.yaml")


if __name__ == "__main__":
    unittest.main()
