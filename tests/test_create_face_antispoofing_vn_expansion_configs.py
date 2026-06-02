"""Tests cho scripts/create_face_antispoofing_vn_expansion_configs.py."""

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

_SCRIPT = _SCRIPTS / "create_face_antispoofing_vn_expansion_configs.py"
_spec = importlib.util.spec_from_file_location("create_face_vn_expansion_configs", _SCRIPT)
assert _spec and _spec.loader
cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg_mod)


class TestFaceVnYamlForExpansion(unittest.TestCase):
    def test_contains_paths_and_slug(self):
        text = cfg_mod._yaml_for_expansion(1.6)
        self.assertIn("face_antispoofing_vn_exp1.6_sample", text)
        self.assertIn("data/sampled/face_antispoofing_vn_exp1.6_sample.csv", text)
        self.assertIn("image_root: data/face_antispoofing_vn_cropped_sfas_exp1.6", text)
        self.assertIn("source_dataset: face_antispoofing_vn_exp1.6", text)


class TestCreateFaceVnConfigsMain(unittest.TestCase):
    def test_writes_yaml_per_expansion(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            configs = repo / "configs"
            configs.mkdir()
            with patch.object(cfg_mod, "REPO_ROOT", repo):
                cfg_mod.main()

            names = sorted(p.name for p in configs.glob("dataset_face_antispoofing_vn*.yaml"))
            expected = [
                f"dataset_{cfg_mod.face_vn_source_dataset_slug(exp)}.yaml"
                for exp in cfg_mod.SFAS_BBOX_EXPANSIONS
            ]
            self.assertEqual(names, expected)


if __name__ == "__main__":
    unittest.main()
