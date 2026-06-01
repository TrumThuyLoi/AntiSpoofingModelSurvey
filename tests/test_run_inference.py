"""Tests cho scripts/run_inference.py."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"


def _load_run_inference_module():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))

    fake_run_batch = types.ModuleType("src.inference.run_batch")
    fake_run_batch.run_batch_inference = MagicMock(return_value=Path("/tmp/pred.csv"))

    fake_src = types.ModuleType("src")
    fake_inference = types.ModuleType("src.inference")
    sys.modules["src"] = fake_src
    sys.modules["src.inference"] = fake_inference
    sys.modules["src.inference.run_batch"] = fake_run_batch

    spec = importlib.util.spec_from_file_location(
        "run_inference",
        _SCRIPTS / "run_inference.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDiscoverConfigPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_run_inference_module()

    def test_discovers_repo_model_and_dataset_configs(self) -> None:
        models = self.mod.discover_config_paths(ROOT, "model")
        datasets = self.mod.discover_config_paths(ROOT, "dataset")
        self.assertGreaterEqual(len(models), 3)
        self.assertGreaterEqual(len(datasets), 7)
        self.assertTrue(all(p.name.startswith("model_") for p in models))
        self.assertTrue(all(p.name.startswith("dataset_") for p in datasets))


class TestRunInferenceMain(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_run_inference_module()

    def test_single_pair_calls_inference_once(self) -> None:
        with patch.object(self.mod, "parse_args") as mock_parse:
            mock_parse.return_value = MagicMock(
                all=False,
                dataset_config=ROOT / "configs/dataset_casia_fasd.yaml",
                model_config=ROOT / "configs/model_minifasnet.yaml",
                inference_config=ROOT / "configs/inference.yaml",
                annotation_csv=None,
            )
            rc = self.mod.main()

        self.assertEqual(rc, 0)
        self.mod.run_batch_inference.assert_called_once()

    def test_all_runs_model_times_dataset(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            configs = repo / "configs"
            configs.mkdir()
            (configs / "model_a.yaml").write_text("model_id: a\n", encoding="utf-8")
            (configs / "model_b.yaml").write_text("model_id: b\n", encoding="utf-8")
            (configs / "dataset_x.yaml").write_text("source_dataset: x\n", encoding="utf-8")
            (configs / "dataset_y.yaml").write_text("source_dataset: y\n", encoding="utf-8")

            with patch.object(self.mod, "REPO_ROOT", repo):
                with patch.object(self.mod, "parse_args") as mock_parse:
                    mock_parse.return_value = MagicMock(
                        all=True,
                        inference_config=repo / "configs/inference.yaml",
                        annotation_csv=None,
                    )
                    rc = self.mod.main()

            self.assertEqual(rc, 0)
            self.assertEqual(self.mod.run_batch_inference.call_count, 4)

    def test_all_raises_when_no_configs(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "configs").mkdir()

            with patch.object(self.mod, "REPO_ROOT", repo):
                with patch.object(self.mod, "parse_args") as mock_parse:
                    mock_parse.return_value = MagicMock(
                        all=True,
                        inference_config=repo / "configs/inference.yaml",
                        annotation_csv=None,
                    )
                    with self.assertRaises(ValueError):
                        self.mod.main()


if __name__ == "__main__":
    unittest.main()
