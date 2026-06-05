"""Tests cho src/models/hairymax_onnx.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.hairymax_onnx import (  # noqa: E402
    HairymaxONNXWrapper,
    is_live_accepted,
    probs_to_prediction,
)

WEIGHTS = ROOT / "models" / "AntiSpoofing_bin_1.5_128.onnx"


def _sample_bgr() -> np.ndarray:
    img = np.zeros((64, 48, 3), dtype=np.uint8)
    img[..., 0] = 100
    img[..., 1] = 80
    img[..., 2] = 60
    return img


class TestHairymaxProbLogic(unittest.TestCase):
    def test_is_live_accepted_pass(self) -> None:
        self.assertTrue(is_live_accepted([0.9, 0.1], threshold=0.5))

    def test_is_live_accepted_fail_low_score(self) -> None:
        self.assertFalse(is_live_accepted([0.4, 0.6], threshold=0.5))

    def test_is_live_accepted_spoof_class(self) -> None:
        self.assertFalse(is_live_accepted([0.2, 0.8], threshold=0.5))

    def test_probs_to_prediction_live(self) -> None:
        out = probs_to_prediction([0.9, 0.1], threshold=0.5)
        self.assertEqual(out["label_pred"], "live")
        self.assertAlmostEqual(out["live_score"], 0.9)
        self.assertAlmostEqual(out["spoof_score"], 0.1)
        self.assertEqual(out["raw_output"], [0.9, 0.1])


class TestHairymaxONNXWrapperValidation(unittest.TestCase):
    def test_missing_weights_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            HairymaxONNXWrapper(
                weights_path=ROOT / "models" / "not_exist_hairymax.onnx",
                prefer_cpu=True,
            )

    @unittest.skipUnless(WEIGHTS.is_file(), f"Thiếu {WEIGHTS}")
    def test_missing_hairymax_dir_raises(self) -> None:
        wrapper = HairymaxONNXWrapper(
            weights_path=WEIGHTS,
            hairymax_dir=ROOT / "third_party" / "missing-hairymax",
            prefer_cpu=True,
        )
        with self.assertRaises(FileNotFoundError):
            wrapper.load()

    def test_invalid_ndarray_shape_raises(self) -> None:
        wrapper = HairymaxONNXWrapper.__new__(HairymaxONNXWrapper)
        with self.assertRaises(ValueError):
            wrapper._read_bgr(np.zeros((64, 64), dtype=np.uint8))


@unittest.skipUnless(WEIGHTS.is_file(), f"Thiếu {WEIGHTS}")
class TestHairymaxONNXWrapperPredict(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.img = Path(self.tmp_dir.name) / "sample.jpg"
        Image.fromarray(_sample_bgr()[..., ::-1], mode="RGB").save(self.img)
        self.wrapper = HairymaxONNXWrapper(
            weights_path=WEIGHTS,
            prefer_cpu=True,
        )
        mock_detector = MagicMock()
        mock_detector.return_value = [np.array([0.9, 0.1], dtype=np.float32)]
        self.wrapper._detector = mock_detector
        self.wrapper._loaded = True

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_predict_output_keys(self) -> None:
        out = self.wrapper.predict(self.img)
        for key in ("label_pred", "live_score", "spoof_score", "raw_output"):
            self.assertIn(key, out)
        self.assertEqual(out["label_pred"], "live")
        self.assertEqual(len(out["raw_output"]), 2)

    def test_predict_batch_length(self) -> None:
        outs = self.wrapper.predict_batch([self.img, _sample_bgr()])
        self.assertEqual(len(outs), 2)
        self.assertEqual(outs[0]["label_pred"], "live")


if __name__ == "__main__":
    unittest.main()
