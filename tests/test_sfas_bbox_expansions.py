"""Tests cho scripts/sfas_bbox_expansions.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_SCRIPT = _SCRIPTS / "sfas_bbox_expansions.py"
_spec = importlib.util.spec_from_file_location("sfas_bbox_expansions", _SCRIPT)
assert _spec and _spec.loader
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)


class TestSfasBboxExpansions(unittest.TestCase):
    def test_expansion_constants(self):
        self.assertEqual(exp.SFAS_BBOX_EXPANSIONS, (1.6, 2.7))

    def test_drivers_crop_output_dir_baseline(self):
        p = exp.drivers_crop_output_dir(1.0, repo_root=ROOT)
        self.assertEqual(p, ROOT / "data" / "drivers_250_fn_cropped")

    def test_drivers_crop_output_dir_with_scale(self):
        p = exp.drivers_crop_output_dir(1.4, repo_root=ROOT)
        self.assertEqual(p, ROOT / "data/drivers_250_fn_cropped_sfas_exp1.4")

    def test_drivers_source_dataset_slug(self):
        self.assertEqual(exp.drivers_source_dataset_slug(1.0), "drivers_250_fn")
        self.assertEqual(exp.drivers_source_dataset_slug(1.2), "drivers_250_fn_exp1.2")

    def test_face_vn_crop_output_dir(self):
        p = exp.face_vn_crop_output_dir(1.6, repo_root=ROOT)
        self.assertEqual(p, ROOT / "data/face_antispoofing_vn_cropped_sfas_exp1.6")

    def test_face_vn_source_dataset_slug(self):
        self.assertEqual(exp.face_vn_source_dataset_slug(1.0), "face_antispoofing_vn")
        self.assertEqual(exp.face_vn_source_dataset_slug(2.7), "face_antispoofing_vn_exp2.7")

    def test_face_vn_input_root(self):
        p = exp.face_vn_input_root(repo_root=ROOT)
        self.assertEqual(p, ROOT / "data" / "face_antispoofing_vn")

    def test_drivers_dataset_config_path(self):
        p = exp.drivers_dataset_config_path(1.2, repo_root=ROOT)
        self.assertEqual(p, ROOT / "configs" / "dataset_drivers_250_fn_exp1.2.yaml")


if __name__ == "__main__":
    unittest.main()
