"""Tests drivers_250_fn expansion samples trong create_test_sample_annotation.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_SCRIPT = _SCRIPTS / "create_test_sample_annotation.py"
_spec = importlib.util.spec_from_file_location("create_test_sample_annotation", _SCRIPT)
assert _spec and _spec.loader
ann = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ann)


def _touch_crop_tree(crop_root: Path, name: str = "face.jpg") -> None:
    p = crop_root / "driver-uuid" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


class TestDrivers250FnSampleBuilders(unittest.TestCase):
    def test_build_all_expansions_returns_one_dataset_per_expansion(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for expansion in ann.SFAS_BBOX_EXPANSIONS:
                _touch_crop_tree(ann.drivers_crop_output_dir(expansion, repo_root=repo))

            with patch.object(ann, "REPO_ROOT", repo):
                built = ann._build_all_drivers_250_fn_samples()

            self.assertEqual(len(built), len(ann.SFAS_BBOX_EXPANSIONS))
            slugs = {slug for slug, _ in built}
            expected = {ann.drivers_source_dataset_slug(exp) for exp in ann.SFAS_BBOX_EXPANSIONS}
            self.assertEqual(slugs, expected)
            for slug, rows in built:
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["label"], "live")
                self.assertEqual(rows[0]["source_dataset"], slug)
                self.assertIn("sfas_exp=", rows[0]["note"])


if __name__ == "__main__":
    unittest.main()
