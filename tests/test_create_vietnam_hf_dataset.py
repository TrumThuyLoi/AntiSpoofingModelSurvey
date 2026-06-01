"""Tests cho scripts/create_vietnam_hf_dataset.py (không quét full dataset / không push HF thật)."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPT = ROOT / "scripts" / "create_vietnam_hf_dataset.py"
_spec = importlib.util.spec_from_file_location("create_vietnam_hf_dataset", _SCRIPT)
assert _spec and _spec.loader
vn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vn)

SFAS_UTILITY = (
    ROOT / "third_party" / "Silent-Face-Anti-Spoofing" / "src" / "anti_spoof_predict.py"
)


def _write_bgr(path: Path, shape: tuple[int, int, int] = (200, 160, 3)) -> None:
    img = np.random.randint(0, 255, shape, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _mock_detector(bbox: list[int] | None = None):
    det = MagicMock()

    def get_bbox(img):
        if bbox is None:
            h, w = img.shape[:2]
            return [w // 4, h // 4, w // 2, h // 2]
        return bbox

    det.get_bbox.side_effect = get_bbox
    return det


def _write_cropped_sample_tree(root: Path) -> None:
    for split, label_dir in (("train_photo", "live"), ("test_photo", "not_live")):
        folder = root / split / label_dir
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), (10, 20, 30)).save(folder / "sample.jpg")


class TestIterVietnamImageRecords(unittest.TestCase):
    def test_yields_labels_and_splits(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "test_photo" / "live" / "a.jpg"
            spoof = root / "train_photo" / "not_live" / "b.png"
            live.parent.mkdir(parents=True)
            spoof.parent.mkdir(parents=True)
            live.touch()
            spoof.touch()

            rows = list(vn.iter_vietnam_image_records(root))
            self.assertEqual(len(rows), 2)
            by_name = {r["path"].name: r for r in rows}
            self.assertEqual(by_name["a.jpg"]["labels"], "0live")
            self.assertEqual(by_name["a.jpg"]["labelNames"], "live")
            self.assertEqual(by_name["a.jpg"]["split"], "test_photo")
            self.assertEqual(by_name["b.png"]["labels"], "1spoof")
            self.assertEqual(by_name["b.png"]["labelNames"], "spoof")
            self.assertEqual(by_name["b.png"]["label_dir"], "not_live")

    def test_skips_non_images(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "test_photo" / "live"
            d.mkdir(parents=True)
            (d / "x.txt").touch()
            self.assertEqual(list(vn.iter_vietnam_image_records(root)), [])


class TestCropBgrFaceSfasExpanded(unittest.TestCase):
    def test_expansion_1_matches_tight_crop(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = 255
        bbox = [25, 25, 50, 50]
        tight = vn._crop_bgr_face(img, bbox)
        expanded = vn._crop_bgr_face_sfas_expanded(img, bbox, 1.0)
        self.assertIsNotNone(tight)
        self.assertIsNotNone(expanded)
        np.testing.assert_array_equal(tight, expanded)

    def test_expansion_larger_than_1_increases_patch(self):
        sfas_patches = ROOT / "third_party/Silent-Face-Anti-Spoofing/src/generate_patches.py"
        if not sfas_patches.is_file():
            self.skipTest("Thiếu SFAS submodule")

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = 255
        bbox = [50, 50, 100, 100]
        tight = vn._crop_bgr_face(img, bbox)
        assert tight is not None
        expanded = vn._crop_bgr_face_sfas_expanded(img, bbox, 1.4)
        self.assertIsNotNone(expanded)
        assert expanded is not None
        self.assertGreater(expanded.shape[0], tight.shape[0])
        self.assertGreater(expanded.shape[1], tight.shape[1])


class TestCropBgrFace(unittest.TestCase):
    def test_crops_center_region(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = 255
        patch = vn._crop_bgr_face(img, [25, 25, 50, 50])
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch.shape, (50, 50, 3))
        self.assertEqual(int(patch[0, 0, 0]), 255)

    def test_zero_bbox_clamped_to_one_pixel(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        patch = vn._crop_bgr_face(img, [0, 0, 0, 0])
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch.shape, (1, 1, 3))

    def test_clamps_to_image_bounds(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        patch = vn._crop_bgr_face(img, [-10, -10, 100, 100])
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch.shape[0], 50)
        self.assertEqual(patch.shape[1], 50)


class TestOutputPath(unittest.TestCase):
    def test_relative_layout(self):
        record = {
            "split": "test_photo",
            "label_dir": "live",
            "path": Path("/tmp/foo/bar.jpg"),
        }
        out = vn._output_path(record, Path("/out"))
        self.assertEqual(out, Path("/out/test_photo/live/bar.jpg"))


class TestDetectAndCropRgb(unittest.TestCase):
    def test_returns_rgb_image(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "face.jpg"
            _write_bgr(path)
            img = vn.detect_and_crop_rgb(path, _mock_detector())
            self.assertIsInstance(img, Image.Image)
            assert img is not None
            self.assertEqual(img.mode, "RGB")
            self.assertGreater(img.size[0], 0)
            self.assertGreater(img.size[1], 0)

    def test_missing_file_returns_none(self):
        self.assertIsNone(
            vn.detect_and_crop_rgb(Path("/nonexistent/x.jpg"), _mock_detector())
        )

    def test_bad_bbox_returns_none(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "face.jpg"
            _write_bgr(path)
            self.assertIsNone(vn.detect_and_crop_rgb(path, _mock_detector([0, 0, 0, 0])))

    @patch.object(vn, "_crop_bgr_face_sfas_expanded")
    def test_passes_bbox_expansion_to_crop(self, mock_crop):
        mock_crop.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "face.jpg"
            _write_bgr(path)
            out = vn.detect_and_crop_rgb(path, _mock_detector(), bbox_expansion=1.4)
            self.assertIsInstance(out, Image.Image)
            mock_crop.assert_called_once()
            self.assertEqual(mock_crop.call_args[0][2], 1.4)


class TestProcessAndSaveCrops(unittest.TestCase):
    @patch.object(vn, "_load_sfas_detection")
    def test_writes_jpeg_and_skip_existing(self, mock_load):
        mock_load.return_value = _mock_detector()
        with TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in"
            out = Path(tmp) / "out"
            src = inp / "test_photo" / "live" / "one.jpg"
            src.parent.mkdir(parents=True)
            _write_bgr(src)

            stats1 = vn.process_and_save_crops(
                input_root=inp, output_root=out, skip_existing=True
            )
            self.assertEqual(stats1, {"total": 1, "ok": 1, "skip": 0, "fail": 0})
            dest = out / "test_photo" / "live" / "one.jpg"
            self.assertTrue(dest.is_file())

            stats2 = vn.process_and_save_crops(
                input_root=inp, output_root=out, skip_existing=True
            )
            self.assertEqual(stats2, {"total": 1, "ok": 0, "skip": 1, "fail": 0})

    @patch.object(vn, "_load_sfas_detection")
    def test_fail_detect_counted_in_stats(self, mock_load):
        mock_load.return_value = _mock_detector([0, 0, 0, 0])
        with TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in"
            out = Path(tmp) / "out"
            src = inp / "test_photo" / "live" / "bad.jpg"
            src.parent.mkdir(parents=True)
            _write_bgr(src)

            stats = vn.process_and_save_crops(input_root=inp, output_root=out)
            self.assertEqual(stats, {"total": 1, "ok": 0, "skip": 0, "fail": 1})
            self.assertFalse((out / "test_photo" / "live" / "bad.jpg").exists())

    @patch.object(vn, "_load_sfas_detection")
    def test_total_includes_ok_and_fail(self, mock_load):
        with TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in"
            out = Path(tmp) / "out"
            ok_src = inp / "train_photo" / "live" / "ok.jpg"
            fail_src = inp / "train_photo" / "not_live" / "fail.jpg"
            ok_src.parent.mkdir(parents=True)
            fail_src.parent.mkdir(parents=True)
            _write_bgr(ok_src)
            _write_bgr(fail_src)

            calls = {"n": 0}

            def get_bbox(img):
                calls["n"] += 1
                if calls["n"] == 1:
                    h, w = img.shape[:2]
                    return [w // 4, h // 4, w // 2, h // 2]
                return [0, 0, 0, 0]

            det = MagicMock()
            det.get_bbox.side_effect = get_bbox
            mock_load.return_value = det

            stats = vn.process_and_save_crops(input_root=inp, output_root=out)
            self.assertEqual(stats, {"total": 2, "ok": 1, "skip": 0, "fail": 1})


class TestRecordsForPhotoSplit(unittest.TestCase):
    def test_filters_split_and_existing_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_cropped_sample_tree(root)

            train_rows = vn._records_for_photo_split(root, "train_photo")
            test_rows = vn._records_for_photo_split(root, "test_photo")

            self.assertEqual(len(train_rows), 1)
            self.assertEqual(train_rows[0]["labels"], "0live")
            self.assertEqual(len(test_rows), 1)
            self.assertEqual(test_rows[0]["labels"], "1spoof")


class TestBuildVietnamHfDataset(unittest.TestCase):
    def test_schema_and_row(self):
        with TemporaryDirectory() as tmp:
            os.environ["HF_DATASETS_CACHE"] = tmp
            root = Path(tmp) / "cropped"
            _write_cropped_sample_tree(root)
            records = vn._records_for_photo_split(root, "train_photo")

            ds = vn.build_vietnam_hf_dataset_from_records(records)
            self.assertEqual(len(ds), 1)
            self.assertIn("cropped_image", ds.features)
            self.assertIn("labels", ds.features)
            self.assertIn("labelNames", ds.features)

            row = ds[0]
            self.assertEqual(row["labelNames"], "live")
            self.assertEqual(row["labels"], 0)


class TestHfToken(unittest.TestCase):
    def test_missing_token_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                vn._hf_token()
            self.assertIn("HUGGINGFACE_PRIVATE_DATASET_TOKEN", str(ctx.exception))

    def test_reads_private_dataset_token(self):
        with patch.dict(
            os.environ,
            {"HUGGINGFACE_PRIVATE_DATASET_TOKEN": "hf_private"},
            clear=True,
        ):
            self.assertEqual(vn._hf_token(), "hf_private")


class TestPushVietnamHfSplits(unittest.TestCase):
    def test_push_train_and_test_splits(self):
        with TemporaryDirectory() as tmp:
            cropped = Path(tmp) / "cropped"
            _write_cropped_sample_tree(cropped)

            mock_ds = MagicMock()
            with patch.object(
                vn, "build_vietnam_hf_dataset_from_records", return_value=mock_ds
            ) as mock_build:
                counts = vn.push_vietnam_hf_splits(
                    "org/face_antispoofing_vn",
                    cropped_root=cropped,
                    token="hf_test_token",
                )

            self.assertEqual(counts, {"train": 1, "test": 1})
            self.assertEqual(mock_build.call_count, 2)
            self.assertEqual(mock_ds.push_to_hub.call_count, 2)

            pushed_splits = [
                c.kwargs["split"] for c in mock_ds.push_to_hub.call_args_list
            ]
            self.assertEqual(set(pushed_splits), {"train", "test"})
            for call in mock_ds.push_to_hub.call_args_list:
                self.assertEqual(call.kwargs["private"], True)
                self.assertEqual(call.kwargs["max_shard_size"], "500MB")
                self.assertEqual(call.kwargs["token"], "hf_test_token")

    def test_missing_cropped_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            vn.push_vietnam_hf_splits(
                "org/repo",
                cropped_root=Path("/nonexistent/cropped"),
                token="tok",
            )

    def test_empty_split_raises(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train_photo" / "live").mkdir(parents=True)
            with self.assertRaises(ValueError):
                vn.push_vietnam_hf_splits(
                    "org/repo",
                    cropped_root=root,
                    token="tok",
                    photo_splits=("train_photo",),
                )


class TestMain(unittest.TestCase):
    @patch.object(vn, "push_vietnam_hf_splits", return_value={"train": 1, "test": 1})
    @patch.object(
        vn,
        "process_and_save_crops",
        return_value={"total": 2, "ok": 2, "skip": 0, "fail": 0},
    )
    def test_runs_crop_then_push(self, mock_crop, mock_push):
        with TemporaryDirectory() as tmp:
            input_root = Path(tmp) / "in"
            input_root.mkdir()
            with patch.object(vn, "INPUT_ROOT", input_root):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("VIETNAM_HF_REPO_ID", None)
                    vn.main()

        mock_crop.assert_called_once_with()
        mock_push.assert_called_once()
        self.assertEqual(mock_push.call_args[0][0], vn._DEFAULT_HF_REPO_ID)

    @patch.object(vn, "push_vietnam_hf_splits")
    @patch.object(vn, "process_and_save_crops")
    def test_uses_repo_id_from_env(self, mock_crop, mock_push):
        mock_crop.return_value = {"total": 0, "ok": 0, "skip": 0, "fail": 0}
        mock_push.return_value = {}
        with TemporaryDirectory() as tmp:
            input_root = Path(tmp) / "in"
            input_root.mkdir()
            with patch.object(vn, "INPUT_ROOT", input_root):
                with patch.dict(
                    os.environ,
                    {"VIETNAM_HF_REPO_ID": "myorg/custom_vn"},
                    clear=False,
                ):
                    vn.main()
        mock_push.assert_called_once_with("myorg/custom_vn")

    def test_missing_input_root_raises(self):
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_input"
            with patch.object(vn, "INPUT_ROOT", missing):
                with self.assertRaises(FileNotFoundError):
                    vn.main()


class TestVietnamHfConstants(unittest.TestCase):
    def test_default_repo_and_features(self):
        self.assertEqual(vn._DEFAULT_HF_REPO_ID, "vu-hong-quang/face_antispoofing_vn")
        self.assertIn("cropped_image", vn.VIETNAM_HF_FEATURES)
        self.assertIn("labels", vn.VIETNAM_HF_FEATURES)
        self.assertIn("labelNames", vn.VIETNAM_HF_FEATURES)


class TestPhotoToHfSplitMapping(unittest.TestCase):
    def test_mapping_matches_plan(self):
        self.assertEqual(vn._PHOTO_TO_HF_SPLIT["train_photo"], "train")
        self.assertEqual(vn._PHOTO_TO_HF_SPLIT["test_photo"], "test")


@unittest.skipUnless(SFAS_UTILITY.is_file(), "Thiếu third_party/Silent-Face-Anti-Spoofing")
class TestSfasDetectionIntegration(unittest.TestCase):
    def test_load_detection(self):
        det = vn._load_sfas_detection()
        self.assertEqual(det.detector_confidence, 0.6)

    def test_detect_on_synthetic_image(self):
        img = np.zeros((320, 240, 3), dtype=np.uint8)
        cv2.rectangle(img, (60, 40), (180, 220), (128, 128, 128), -1)
        det = vn._load_sfas_detection()
        bbox = det.get_bbox(img)
        self.assertEqual(len(bbox), 4)
        self.assertGreater(bbox[2], 0)
        self.assertGreater(bbox[3], 0)


if __name__ == "__main__":
    unittest.main()
