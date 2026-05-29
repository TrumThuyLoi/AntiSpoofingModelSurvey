"""Tests cho hf_raw — CelebA-Spoof và CASIA-FASD (unittest, không tải Hugging Face)."""

from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

_LABEL_ALIASES = {
    "live": "live",
    "0live": "live",
    "spoof": "spoof",
    "1spoof": "spoof",
}


def _install_specs_stub() -> None:
    @dataclass(frozen=True)
    class HFRawDatasetSpec:
        source_dataset: str
        repo_id: str
        dataset_page: str
        default_raw_rel_dir: str
        hf_split: str = "test"
        image_column: str = "cropped_image"
        label_field: str = "labels"
        label_name_field: str = "labelNames"
        label_aliases: dict[str, str] | None = None
        token_env_var: str | None = None

    @dataclass(frozen=True)
    class DownloadResult:
        output_dir: Path
        annotation_path: Path
        num_images: int
        label_counts: dict[str, int]
        skipped: bool

    celeba_spec = HFRawDatasetSpec(
        source_dataset="celeba_spoof",
        repo_id="nguyenkhoa/celeba-spoof-for-face-antispoofing-test",
        dataset_page=(
            "https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test"
        ),
        default_raw_rel_dir="data/raw/celeba_spoof",
        label_aliases=_LABEL_ALIASES,
    )
    casia_spec = HFRawDatasetSpec(
        source_dataset="casia_fasd",
        repo_id="vu-hong-quang/casia_fasd",
        dataset_page="https://huggingface.co/datasets/vu-hong-quang/casia_fasd",
        default_raw_rel_dir="data/raw/casia_fasd",
        label_aliases=_LABEL_ALIASES,
        token_env_var="HUGGINGFACE_TOKEN",
    )

    specs_mod = types.ModuleType("src.datasets.specs")
    specs_mod.HFRawDatasetSpec = HFRawDatasetSpec
    specs_mod.DownloadResult = DownloadResult
    specs_mod.CELEBA_SPOOF_SPEC = celeba_spec
    specs_mod.CASIA_FASD_SPEC = casia_spec

    for name in ("src", "src.datasets"):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg

    sys.modules.pop("src.datasets.specs", None)
    sys.modules.pop("src.datasets.hf_raw", None)
    sys.modules["src.datasets.specs"] = specs_mod


def _import_hf_raw():
    try:
        from src.datasets import hf_raw as mod

        return mod
    except TypeError:
        _install_specs_stub()
        from src.datasets import hf_raw as mod

        return mod


hf_raw = _import_hf_raw()
from src.datasets.specs import CASIA_FASD_SPEC, CELEBA_SPOOF_SPEC

_DATASET_SPECS = (
    ("celeba_spoof", CELEBA_SPOOF_SPEC),
    ("casia_fasd", CASIA_FASD_SPEC),
)


def _mock_hf_dataset(num_rows: int) -> MagicMock:
    hf_ds = MagicMock()
    hf_ds.__len__ = MagicMock(return_value=num_rows)
    return hf_ds


def _prepare_complete_raw_dir(out: Path, spec) -> None:
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    (out / "images" / spec.hf_split).mkdir(parents=True, exist_ok=True)
    (out / "annotations" / "raw.csv").write_text("x")
    hf_raw._write_manifest(
        out / "meta",
        {
            "repo_id": spec.repo_id,
            "split": spec.hf_split,
            "num_images": 2,
            "live_count": 1,
            "spoof_count": 1,
        },
    )


class TestHFRawNormalize(unittest.TestCase):
    def test_normalize_label(self):
        for name, spec in _DATASET_SPECS:
            with self.subTest(dataset=name):
                self.assertEqual(
                    hf_raw._normalize_label(spec, None, "live"),
                    "live",
                )
                self.assertEqual(
                    hf_raw._normalize_label(spec, "1spoof", None),
                    "spoof",
                )
                with self.assertRaises(ValueError):
                    hf_raw._normalize_label(spec, "bad", None)


class TestHFRawDownloadComplete(unittest.TestCase):
    def test_is_download_complete(self):
        for name, spec in _DATASET_SPECS:
            with self.subTest(dataset=name):
                hf_ds = _mock_hf_dataset(2)
                with TemporaryDirectory() as d:
                    out = Path(d)
                    _prepare_complete_raw_dir(out, spec)
                    self.assertTrue(
                        hf_raw._is_download_complete(out, spec, hf_ds)
                    )
                    hf_ds.__len__.return_value = 3
                    self.assertFalse(
                        hf_raw._is_download_complete(out, spec, hf_ds)
                    )

    def test_wrong_repo_id_in_manifest(self):
        spec = CELEBA_SPOOF_SPEC
        hf_ds = _mock_hf_dataset(2)
        with TemporaryDirectory() as d:
            out = Path(d)
            _prepare_complete_raw_dir(out, spec)
            hf_raw._write_manifest(
                out / "meta",
                {
                    "repo_id": "other/repo",
                    "split": spec.hf_split,
                    "num_images": 2,
                },
            )
            self.assertFalse(hf_raw._is_download_complete(out, spec, hf_ds))


class TestHFRawDownloadSkipped(unittest.TestCase):
    @patch("src.datasets.hf_raw._iter_hf_rows")
    @patch("src.datasets.hf_raw.find_project_root")
    def test_download_skipped_when_complete(self, mock_root, mock_iter_hf):
        for name, spec in _DATASET_SPECS:
            with self.subTest(dataset=name):
                hf_ds = _mock_hf_dataset(2)
                mock_iter_hf.reset_mock()
                mock_iter_hf.return_value = (hf_ds, 2)

                with TemporaryDirectory() as d:
                    root = Path(d)
                    mock_root.return_value = root
                    out = root / spec.default_raw_rel_dir
                    _prepare_complete_raw_dir(out, spec)

                    result = hf_raw.download_raw_dataset(spec)

                    self.assertTrue(result.skipped)
                    self.assertEqual(result.num_images, 2)
                    self.assertEqual(result.output_dir, out)
                    mock_iter_hf.assert_called_once_with(spec=spec)

    @patch.dict("os.environ", {}, clear=True)
    def test_casia_iter_hf_rows_requires_token_env(self):
        spec = CASIA_FASD_SPEC
        self.assertEqual(spec.token_env_var, "HUGGINGFACE_PRIVATE_DATASET_TOKEN")
        with self.assertRaises(ValueError):
            hf_raw._iter_hf_rows(spec)


if __name__ == "__main__":
    unittest.main()
