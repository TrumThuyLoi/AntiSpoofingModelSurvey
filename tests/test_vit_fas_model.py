"""Tests cho src/models/vit_fas.py (ViTFASWrapper)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.vit_fas import ViTFASWrapper  # noqa: E402


class _DummyBinaryModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trả logits [spoof, live] theo quy ước wrapper.
        batch = x.shape[0]
        return torch.tensor([[-1.0, 1.0]], dtype=x.dtype, device=x.device).repeat(batch, 1)


def _create_torchscript_checkpoint(path: Path) -> None:
    model = _DummyBinaryModel().eval()
    example = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(path))


def _create_sample_image(path: Path) -> None:
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[..., 0] = 128
    arr[..., 1] = 64
    arr[..., 2] = 32
    Image.fromarray(arr, mode="RGB").save(path)


class TestViTFASWrapperBasic(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_dir.name)
        self.ckpt = self.tmp / "vitfas_vitb16_224x224.pth"
        self.img = self.tmp / "sample.jpg"
        _create_torchscript_checkpoint(self.ckpt)
        _create_sample_image(self.img)
        self.wrapper = ViTFASWrapper(
            weights_path=self.ckpt,
            input_size=(224, 224),
            threshold=0.5,
            prefer_cpu=True,
        )

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_load_only_once(self) -> None:
        self.wrapper.load()
        first = self.wrapper.model
        self.wrapper.load()
        self.assertIs(self.wrapper.model, first)
        self.assertTrue(self.wrapper._loaded)

    def test_preprocess_shape(self) -> None:
        tensor = self.wrapper.preprocess(self.img)
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        self.assertEqual(tensor.device.type, "cpu")

    def test_predict_output_keys(self) -> None:
        out = self.wrapper.predict(self.img)
        for key in ("label_pred", "live_score", "spoof_score", "raw_output"):
            self.assertIn(key, out)
        self.assertIn(out["label_pred"], ("live", "spoof"))
        self.assertEqual(len(out["raw_output"]), 2)
        self.assertGreater(out["live_score"], out["spoof_score"])
        self.assertEqual(out["label_pred"], "live")

    def test_predict_batch_length(self) -> None:
        outs = self.wrapper.predict_batch([self.img, self.img])
        self.assertEqual(len(outs), 2)
        self.assertIn("label_pred", outs[0])


class TestViTFASWrapperValidation(unittest.TestCase):
    def test_missing_weights_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ViTFASWrapper(weights_path=ROOT / "models" / "not_exist_vitfas.pth", prefer_cpu=True)

    def test_invalid_ndarray_shape_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = tmp_path / "dummy.pth"
            _create_torchscript_checkpoint(ckpt)
            wrapper = ViTFASWrapper(weights_path=ckpt, prefer_cpu=True)
            bad_input = np.zeros((224, 224), dtype=np.uint8)  # thiếu channel dim
            with self.assertRaises(ValueError):
                wrapper.preprocess(bad_input)


if __name__ == "__main__":
    unittest.main()

