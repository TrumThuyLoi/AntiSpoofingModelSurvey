"""Tests cho src/models/face_antispoof_onnx.py."""

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

from src.models.face_antispoof_onnx import FaceAntispoofONNXWrapper  # noqa: E402

UPSTREAM_MODEL_PY = (
    ROOT / "third_party" / "face-antispoof-onnx" / "src" / "minifasv2" / "model.py"
)


class _DummyBinaryModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.tensor([[1.0, -1.0]], dtype=x.dtype, device=x.device).repeat(batch, 1)


def _create_checkpoint_from_upstream(path: Path) -> None:
    """Tao checkpoint toi thieu dung format repo goc."""
    if not UPSTREAM_MODEL_PY.is_file():
        raise FileNotFoundError(f"Thieu upstream model: {UPSTREAM_MODEL_PY}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("fas_upstream_model_for_test", UPSTREAM_MODEL_PY)
    if spec is None or spec.loader is None:
        raise ImportError(f"Khong the load module tu {UPSTREAM_MODEL_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MultiFTNet(num_classes=2).eval()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "validation_accuracy": 0.0,
        "config": {"input_size": 128, "num_classes": 2, "ft_weight": 0.0},
    }
    torch.save(checkpoint, path)


def _create_sample_image(path: Path) -> None:
    img = np.zeros((180, 120, 3), dtype=np.uint8)
    img[..., 0] = 120
    img[..., 1] = 90
    img[..., 2] = 40
    Image.fromarray(img, mode="RGB").save(path)


@unittest.skipUnless(UPSTREAM_MODEL_PY.is_file(), "Thiếu third_party/face-antispoof-onnx")
class TestFaceAntispoofONNXWrapperBasic(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_dir.name)
        self.ckpt = self.tmp / "face_antispoof_onnx_best_9820.pth"
        self.img = self.tmp / "sample.jpg"
        _create_checkpoint_from_upstream(self.ckpt)
        _create_sample_image(self.img)
        self.wrapper = FaceAntispoofONNXWrapper(
            weights_path=self.ckpt,
            input_size=(128, 128),
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
        self.assertEqual(tuple(tensor.shape), (1, 3, 128, 128))
        self.assertEqual(tensor.device.type, "cpu")

    def test_predict_output_keys(self) -> None:
        self.wrapper.model = _DummyBinaryModel().to(self.wrapper.device).eval()
        self.wrapper._loaded = True
        out = self.wrapper.predict(self.img)
        for key in ("label_pred", "live_score", "spoof_score", "raw_output"):
            self.assertIn(key, out)
        self.assertIn(out["label_pred"], ("live", "spoof"))
        self.assertEqual(len(out["raw_output"]), 2)
        self.assertIsInstance(out["live_score"], float)
        self.assertIsInstance(out["spoof_score"], float)

    def test_predict_batch_length(self) -> None:
        self.wrapper.model = _DummyBinaryModel().to(self.wrapper.device).eval()
        self.wrapper._loaded = True
        outs = self.wrapper.predict_batch([self.img, self.img])
        self.assertEqual(len(outs), 2)
        self.assertIn("label_pred", outs[0])


class TestFaceAntispoofONNXWrapperValidation(unittest.TestCase):
    def test_missing_weights_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            FaceAntispoofONNXWrapper(
                weights_path=ROOT / "models" / "not_exist_face_antispoof_onnx.pth",
                prefer_cpu=True,
            )

    @unittest.skipUnless(UPSTREAM_MODEL_PY.is_file(), "Thiếu third_party/face-antispoof-onnx")
    def test_invalid_ndarray_shape_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = tmp_path / "dummy.pth"
            _create_checkpoint_from_upstream(ckpt)
            wrapper = FaceAntispoofONNXWrapper(weights_path=ckpt, prefer_cpu=True)
            bad_input = np.zeros((128, 128), dtype=np.uint8)  # thieu channel dim
            with self.assertRaises(ValueError):
                wrapper.preprocess(bad_input)


if __name__ == "__main__":
    unittest.main()

