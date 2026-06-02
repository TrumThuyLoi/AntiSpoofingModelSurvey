"""Tests cho scripts/run_face_vn_expansion_benchmark.py."""

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


def _load_benchmark_module():
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
        "run_face_vn_expansion_benchmark",
        _SCRIPTS / "run_face_vn_expansion_benchmark.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFaceVnExpansionBenchmarkMain(unittest.TestCase):
    def setUp(self):
        self.bench = _load_benchmark_module()

    def test_runs_inference_and_eval_per_expansion(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            configs = repo / "configs"
            configs.mkdir()
            for expansion in self.bench.SFAS_BBOX_EXPANSIONS:
                self.bench.face_vn_dataset_config_path(expansion, repo_root=repo).write_text(
                    "name: x\n", encoding="utf-8"
                )

            model_cfg = repo / "configs/model.yaml"
            model_cfg.write_text("model_id: test\n", encoding="utf-8")

            with patch.object(self.bench, "REPO_ROOT", repo):
                with patch.object(
                    self.bench,
                    "parse_args",
                    return_value=MagicMock(
                        model_config=model_cfg,
                        inference_config=repo / "configs/inference.yaml",
                        eval_config=repo / "configs/evaluation.yaml",
                        skip_evaluation=False,
                    ),
                ):
                    with patch.object(self.bench, "subprocess") as mock_subprocess:
                        mock_subprocess.run.return_value = MagicMock(returncode=0)
                        rc = self.bench.main()

        self.assertEqual(rc, 0)
        n = len(self.bench.SFAS_BBOX_EXPANSIONS)
        self.assertEqual(self.bench.run_batch_inference.call_count, n)
        self.assertEqual(mock_subprocess.run.call_count, n)


if __name__ == "__main__":
    unittest.main()
