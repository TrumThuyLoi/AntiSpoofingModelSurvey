"""Tests cho src/models/minifasnet.py (MiniFASNetWrapper)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.minifasnet import MiniFASNetWrapper  # noqa: E402

MODEL_CONFIG = ROOT / "configs" / "model_minifasnet.yaml"
WEIGHTS_PATH = ROOT / "models" / "2.7_80x80_MiniFASNetV2.pth"
SUBMODULE_UTILITY = (
    ROOT / "third_party" / "Silent-Face-Anti-Spoofing" / "src" / "utility.py"
)
SAMPLE_IMAGE = ROOT / "data" / "raw" / "celeba_spoof" / "images" / "test" / "000000.jpg"


def _integration_ready() -> str | None:
    if not MODEL_CONFIG.is_file():
        return "Thiếu configs/model_minifasnet.yaml"
    if not WEIGHTS_PATH.is_file():
        return f"Thiếu weight: {WEIGHTS_PATH}"
    if not SUBMODULE_UTILITY.is_file():
        return f"Thiếu submodule: {SUBMODULE_UTILITY}"
    return None


def _sample_image_path() -> Path | None:
    if SAMPLE_IMAGE.is_file():
        return SAMPLE_IMAGE
    images_dir = ROOT / "data" / "raw" / "celeba_spoof" / "images" / "test"
    if not images_dir.is_dir():
        return None
    for path in sorted(images_dir.glob("*.jpg")):
        return path
    return None


class TestParseInputSize(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(MiniFASNetWrapper._parse_input_size({"input_size": [80, 80]}), (80, 80))

    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            MiniFASNetWrapper._parse_input_size({"input_size": [80]})

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            MiniFASNetWrapper._parse_input_size({"input_size": "80x80"})


class TestResolveDevice(unittest.TestCase):
    def test_prefer_cpu_over_gpu_config(self):
        wrapper = MiniFASNetWrapper.__new__(MiniFASNetWrapper)
        wrapper.prefer_cpu = True
        self.assertEqual(wrapper._resolve_device("gpu"), torch.device("cpu"))

    def test_cpu_config(self):
        wrapper = MiniFASNetWrapper.__new__(MiniFASNetWrapper)
        wrapper.prefer_cpu = False
        self.assertEqual(wrapper._resolve_device("cpu"), torch.device("cpu"))


class TestResolveWeightsPath(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapper = MiniFASNetWrapper.__new__(MiniFASNetWrapper)
        self.wrapper.repo_root = ROOT

    def test_relative_path_to_file(self):
        if not WEIGHTS_PATH.is_file():
            self.skipTest("Thiếu weight file")
        rel = str(WEIGHTS_PATH.relative_to(ROOT))
        resolved = self.wrapper._resolve_weights_path(rel)
        self.assertEqual(resolved, WEIGHTS_PATH.resolve())

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.wrapper._resolve_weights_path("models/minifasnet/missing.pth")


class TestInitFromModelYaml(unittest.TestCase):
    def test_reads_input_size_from_config(self):
        if not MODEL_CONFIG.is_file():
            self.skipTest("Thiếu configs/model_minifasnet.yaml")
        with MODEL_CONFIG.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "input_size" not in cfg:
            self.skipTest("configs/model_minifasnet.yaml chưa có input_size")

        reason = _integration_ready()
        if reason:
            self.skipTest(reason)

        wrapper = MiniFASNetWrapper(model_config_path=MODEL_CONFIG, prefer_cpu=True)
        expected = MiniFASNetWrapper._parse_input_size(cfg)
        self.assertEqual(wrapper.input_size, expected)
        self.assertEqual(wrapper.device, torch.device("cpu"))
        self.assertEqual(wrapper.weights_path.resolve(), WEIGHTS_PATH.resolve())


class TestMiniFASNetWrapperIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reason = _integration_ready()
        if reason:
            raise unittest.SkipTest(reason)
        cls.wrapper = MiniFASNetWrapper(model_config_path=MODEL_CONFIG, prefer_cpu=True)

    def test_load_only_once(self):
        self.wrapper.load()
        first = self.wrapper.model
        self.wrapper.load()
        self.assertIs(self.wrapper.model, first)
        self.assertTrue(self.wrapper._loaded)

    def test_preprocess_shape(self):
        image = _sample_image_path()
        if image is None:
            self.skipTest("Không có ảnh mẫu")
        tensor = self.wrapper.preprocess(image)
        self.assertEqual(tuple(tensor.shape), (1, 3, 80, 80))
        self.assertEqual(tensor.device.type, "cpu")

    def test_predict_output_keys_and_probs(self):
        image = _sample_image_path()
        if image is None:
            self.skipTest("Không có ảnh mẫu")

        out = self.wrapper.predict(image)
        for key in ("label_pred", "live_score", "spoof_score", "raw_output"):
            self.assertIn(key, out)

        self.assertIn(out["label_pred"], ("live", "spoof"))
        self.assertEqual(len(out["raw_output"]), 3)
        self.assertAlmostEqual(sum(out["raw_output"]), 1.0, places=5)
        self.assertGreaterEqual(out["live_score"], 0.0)
        self.assertGreaterEqual(out["spoof_score"], 0.0)
        self.assertAlmostEqual(out["live_score"] + out["spoof_score"], 1.0, places=5)

    def test_predict_batch_length(self):
        image = _sample_image_path()
        if image is None:
            self.skipTest("Không có ảnh mẫu")
        paths = [image, image]
        outs = self.wrapper.predict_batch(paths)
        self.assertEqual(len(outs), 2)
        self.assertIn("label_pred", outs[0])


class TestMiniFASNetWrapperBadConfig(unittest.TestCase):
    def test_missing_weights_in_temp_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "model_minifasnet.yaml"
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "minifasnet",
                        "weights_dir": "models/minifasnet/not_exist.pth",
                        "device": "cpu",
                        "input_size": [80, 80],
                        "threshold": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                MiniFASNetWrapper(model_config_path=cfg_path, prefer_cpu=True)


if __name__ == "__main__":
    unittest.main()
