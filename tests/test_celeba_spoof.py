"""Tests cho celeba_spoof (unittest, không tải Hugging Face)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.datasets.celeba_spoof import (
    HF_REPO_ID,
    _is_download_complete,
    _normalize_label,
    _write_manifest,
    download_raw_dataset,
)


class TestCelebaSpoof(unittest.TestCase):
    def test_normalize_label(self):
        self.assertEqual(_normalize_label(None, "live"), "live")
        self.assertEqual(_normalize_label("1spoof", None), "spoof")
        with self.assertRaises(ValueError):
            _normalize_label("bad", None)

    def test_is_download_complete(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            for name in ("meta", "annotations", "images"):
                (root / name).mkdir()
            (root / "annotations" / "raw.csv").write_text("x")
            _write_manifest(root / "meta", {"repo_id": HF_REPO_ID, "num_images": 2})
            self.assertTrue(_is_download_complete(root, 2))
            self.assertFalse(_is_download_complete(root, 3))

    @patch("src.datasets.celeba_spoof._iter_hf_rows", return_value=([], 2))
    @patch("src.datasets.celeba_spoof.find_project_root")
    def test_download_skipped_when_complete(self, mock_root, _mock_hf):
        with TemporaryDirectory() as d:
            root = Path(d)
            mock_root.return_value = root
            out = root / "raw"
            for name in ("meta", "annotations", "images"):
                (out / name).mkdir(parents=True)
            (out / "annotations" / "raw.csv").write_text("")
            _write_manifest(
                out / "meta",
                {"repo_id": HF_REPO_ID, "num_images": 2, "live_count": 1, "spoof_count": 1},
            )
            result = download_raw_dataset(out)
            self.assertTrue(result.skipped)
            self.assertEqual(result.num_images, 2)


if __name__ == "__main__":
    unittest.main()
