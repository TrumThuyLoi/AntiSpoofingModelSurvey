"""Tests preprocessing MiniFASNet (resize + bgr_to_chw_float) so với submodule."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.minifasnet import (  # noqa: E402
    preprocess_bgr,
    resize_bgr,
    bgr_to_chw_float,
)


def _load_crop_image_class():
    module_path = (
        ROOT
        / "third_party"
        / "Silent-Face-Anti-Spoofing"
        / "src"
        / "generate_patches.py"
    )
    if not module_path.is_file():
        raise unittest.SkipTest(f"Thiếu submodule: {module_path}")
    spec = importlib.util.spec_from_file_location("sfas_generate_patches", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CropImage


def _reference_pre_cropped(bgr: np.ndarray, out_w: int = 80, out_h: int = 80) -> torch.Tensor:
    """Pipeline gốc: crop=False + functional.to_tensor numpy (không /255)."""
    CropImage = _load_crop_image_class()
    cropper = CropImage()
    resized = cropper.crop(
        org_img=bgr,
        bbox=[0, 0, 1, 1],
        scale=2.7,
        out_w=out_w,
        out_h=out_h,
        crop=False,
    )
    pic = resized
    if pic.ndim == 2:
        pic = pic.reshape((pic.shape[0], pic.shape[1], 1))
    return torch.from_numpy(pic.transpose((2, 0, 1))).float()


def _sample_bgr() -> np.ndarray | None:
    images_dir = ROOT / "data" / "raw" / "celeba_spoof" / "images" / "test"
    if not images_dir.is_dir():
        return None
    for path in sorted(images_dir.glob("*.jpg"))[:1]:
        img = cv2.imread(str(path))
        if img is not None:
            return img
    return None


class TestResizeBgr(unittest.TestCase):
    def test_output_shape(self):
        bgr = np.zeros((120, 80, 3), dtype=np.uint8)
        out = resize_bgr(bgr, (80, 80))
        self.assertEqual(out.shape, (80, 80, 3))

    def test_opencv_width_height_order(self):
        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        out = resize_bgr(bgr, (60, 40))
        self.assertEqual(out.shape, (60, 40, 3))


class TestBgrToChwFloat(unittest.TestCase):
    def test_shape_and_dtype(self):
        bgr = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
        tensor = bgr_to_chw_float(bgr)
        self.assertEqual(tuple(tensor.shape), (3, 80, 80))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreater(tensor.max().item(), 1.0)


class TestMatchesSubmodule(unittest.TestCase):
    def test_resize_and_tensor_match_reference(self):
        bgr = _sample_bgr()
        if bgr is None:
            self.skipTest("Không có ảnh mẫu trong data/raw/celeba_spoof/images/test")
        ref = _reference_pre_cropped(bgr, 80, 80)
        ours = preprocess_bgr(bgr, (80, 80))
        self.assertTrue(torch.equal(ref, ours))


if __name__ == "__main__":
    unittest.main()
