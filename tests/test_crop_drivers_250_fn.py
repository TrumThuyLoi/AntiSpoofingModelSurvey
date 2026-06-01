"""Tests cho scripts/crop_drivers_250_fn.py (mock detector, không quét full dataset)."""

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
        "crop_drivers_250_fn",
        _SCRIPTS / "crop_drivers_250_fn.py",
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
class TestCropAllExpansions(unittest.TestCase):
    def setUp(self):
        self.crop_mod = _load_crop_module()

    def test_writes_each_expansion_directory(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            inp = repo / "data" / "drivers_250_FN" / "uid" / "a.jpg"
            inp.parent.mkdir(parents=True, exist_ok=True)
            inp.write_bytes(b"x")

            with patch.object(self.crop_mod, "REPO_ROOT", repo):
                with patch.object(self.crop_mod, "INPUT_ROOT", repo / "data" / "drivers_250_FN"):
                    stats = self.crop_mod._crop_all_expansions(skip_existing=False)

            self.assertEqual(len(stats), 4)
            for expansion in self.crop_mod.SFAS_BBOX_EXPANSIONS:
                out = self.crop_mod.crop_output_dir(expansion, repo_root=repo)
                dest = out / "uid" / "a.jpg"
                self.assertTrue(dest.is_file(), f"missing {dest}")

            expansions_called = {
                c.kwargs.get("bbox_expansion")
                for c in self.crop_mod.vn.detect_and_crop_rgb.call_args_list
            }
            self.assertEqual(expansions_called, {1.0, 1.2, 1.4, 1.6})

    def test_missing_input_raises(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with patch.object(self.crop_mod, "REPO_ROOT", repo):
                with patch.object(self.crop_mod, "INPUT_ROOT", repo / "missing"):
                    with self.assertRaises(FileNotFoundError):
                        self.crop_mod._crop_all_expansions()


if __name__ == "__main__":
    unittest.main()
