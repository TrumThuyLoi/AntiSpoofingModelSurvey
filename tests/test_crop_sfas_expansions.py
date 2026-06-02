"""Tests cho scripts/crop_sfas_expansions.py (mock detector)."""

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

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_PIL = importlib.util.find_spec("PIL") is not None


def _inject_heavy_import_stubs() -> None:
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda iterable, **kwargs: iterable
    sys.modules.setdefault("tqdm", tqdm_mod)

    if _HAS_NUMPY:
        import numpy as np

        cv2_mod = MagicMock()
        cv2_mod.imread.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2_mod.cvtColor.side_effect = lambda img, code: img
    else:
        cv2_mod = MagicMock()
    sys.modules.setdefault("cv2", cv2_mod)
    sys.modules.setdefault("datasets", MagicMock())
    sys.modules.setdefault("dotenv", MagicMock(load_dotenv=MagicMock()))


def _load_crop_module():
    _inject_heavy_import_stubs()
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))

    spec = importlib.util.spec_from_file_location(
        "crop_sfas_expansions",
        _SCRIPTS / "crop_sfas_expansions.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if _HAS_PIL:
        from PIL import Image as PILImage

        mod.vn.detect_and_crop_rgb = MagicMock(
            return_value=PILImage.new("RGB", (8, 8), (1, 2, 3))
        )
    else:
        mod.vn.detect_and_crop_rgb = MagicMock(return_value=MagicMock())
    mod.vn._load_sfas_detection = MagicMock(return_value=MagicMock())
    return mod


@unittest.skipUnless(_HAS_NUMPY and _HAS_PIL, "Cần numpy và Pillow")
class TestCropDrivers(unittest.TestCase):
    def setUp(self):
        self.crop_mod = _load_crop_module()

    def test_writes_each_expansion_directory(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            inp = repo / "data" / "drivers_250_FN" / "uid" / "a.jpg"
            inp.parent.mkdir(parents=True, exist_ok=True)
            inp.write_bytes(b"x")

            with patch.object(self.crop_mod, "REPO_ROOT", repo):
                with patch.object(self.crop_mod, "DRIVERS_INPUT_ROOT", repo / "data" / "drivers_250_FN"):
                    stats = self.crop_mod.crop_drivers(skip_existing=False)

            self.assertEqual(len(stats), len(self.crop_mod.SFAS_BBOX_EXPANSIONS))
            for expansion in self.crop_mod.SFAS_BBOX_EXPANSIONS:
                out = self.crop_mod.drivers_crop_output_dir(expansion, repo_root=repo)
                self.assertTrue((out / "uid" / "a.jpg").is_file())

    def test_missing_drivers_input_raises(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with patch.object(self.crop_mod, "REPO_ROOT", repo):
                with patch.object(self.crop_mod, "DRIVERS_INPUT_ROOT", repo / "missing"):
                    with self.assertRaises(FileNotFoundError):
                        self.crop_mod.crop_drivers()


@unittest.skipUnless(_HAS_NUMPY and _HAS_PIL, "Cần numpy và Pillow")
class TestCropFaceVn(unittest.TestCase):
    def setUp(self):
        self.crop_mod = _load_crop_module()

    def test_crop_face_vn_preserves_relative_path(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src = repo / "data/face_antispoofing_vn/test_photo/live/000.jpg"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(b"x")

            with patch.object(self.crop_mod, "REPO_ROOT", repo):
                with patch.object(
                    self.crop_mod,
                    "FACE_VN_INPUT_ROOT",
                    repo / "data/face_antispoofing_vn",
                ):
                    stats = self.crop_mod.crop_face_antispoofing_vn(skip_existing=False)

            out = self.crop_mod.face_vn_crop_output_dir(1.6, repo_root=repo)
            self.assertTrue((out / "test_photo/live/000.jpg").is_file())
            self.assertEqual(stats[1.6]["ok"], 1)


if __name__ == "__main__":
    unittest.main()
