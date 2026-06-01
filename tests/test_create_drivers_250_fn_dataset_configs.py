"""Tests cho scripts/create_drivers_250_fn_dataset_configs.py."""

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

_SCRIPT = _SCRIPTS / "create_drivers_250_fn_dataset_configs.py"
_spec = importlib.util.spec_from_file_location("create_drivers_250_fn_dataset_configs", _SCRIPT)
assert _spec and _spec.loader
cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg_mod)


class TestYamlForExpansion(unittest.TestCase):
    def test_contains_paths_and_slug(self):
        text = cfg_mod._yaml_for_expansion(1.2)
        self.assertIn("drivers_250_fn_exp1.2_sample", text)
        self.assertIn("data/sampled/drivers_250_fn_exp1.2_sample.csv", text)
        self.assertIn("data/drivers_250_fn_cropped_sfas_exp1.2", text)
        self.assertIn("source_dataset: drivers_250_fn_exp1.2", text)


class TestCreateConfigsMain(unittest.TestCase):
    def test_writes_four_yaml_files(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            configs = repo / "configs"
            configs.mkdir()
            with patch.object(cfg_mod, "REPO_ROOT", repo):
                cfg_mod.main()

            names = sorted(p.name for p in configs.glob("dataset_drivers_250_fn*.yaml"))
            self.assertEqual(
                names,
                [
                    "dataset_drivers_250_fn.yaml",
                    "dataset_drivers_250_fn_exp1.2.yaml",
                    "dataset_drivers_250_fn_exp1.4.yaml",
                    "dataset_drivers_250_fn_exp1.6.yaml",
                ],
            )


if __name__ == "__main__":
    unittest.main()
