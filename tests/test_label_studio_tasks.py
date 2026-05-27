"""Tests cho create_label_studio_task.py — dùng unittest, không đọc dataset thật."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from create_label_studio_task import (
    _image_url_from_image_path,
    build_tasks_from_rows,
    load_raw_annotations,
    write_tasks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_ROWS: List[Dict[str, str]] = [
    {
        "image_path": "data/raw/casia_fasd/images/test/000000.jpg",
        "label": "live",
        "source_dataset": "casia_fasd",
        "split": "test",
        "is_valid": "True",
        "note": "",
    },
    {
        "image_path": "data/raw/casia_fasd/images/test/000001.jpg",
        "label": "spoof",
        "source_dataset": "casia_fasd",
        "split": "test",
        "is_valid": "True",
        "note": "",
    },
    {
        "image_path": "data/raw/celeba_spoof/images/test/000001.jpg",
        "label": "spoof",
        "source_dataset": "celeba_spoof",
        "split": "test",
        "is_valid": "True",
        "note": "",
    },
]


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    cols = ["image_path", "label", "source_dataset", "split", "is_valid", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# _image_url_from_image_path
# ---------------------------------------------------------------------------


class TestImageUrl(unittest.TestCase):
    def test_standard_path(self) -> None:
        """Path bắt đầu bằng 'data/' → bỏ prefix và dùng ?d=."""
        url = _image_url_from_image_path("data/raw/casia_fasd/images/test/000000.jpg")
        self.assertEqual(url, "/data/local-files/?d=raw/casia_fasd/images/test/000000.jpg")

    def test_leading_dot_slash_stripped(self) -> None:
        url = _image_url_from_image_path("./data/raw/celeba_spoof/images/test/abc.jpg")
        self.assertEqual(url, "/data/local-files/?d=raw/celeba_spoof/images/test/abc.jpg")

    def test_path_without_data_prefix(self) -> None:
        """Path không có prefix 'data/' → dùng toàn bộ path làm relative."""
        url = _image_url_from_image_path("raw/casia_fasd/images/test/000000.jpg")
        self.assertEqual(url, "/data/local-files/?d=raw/casia_fasd/images/test/000000.jpg")

    def test_url_format(self) -> None:
        url = _image_url_from_image_path("data/raw/casia_fasd/images/test/000000.jpg")
        self.assertTrue(url.startswith("/data/local-files/?d="), msg=f"Sai format URL: {url}")


# ---------------------------------------------------------------------------
# build_tasks_from_rows
# ---------------------------------------------------------------------------


class TestBuildTasks(unittest.TestCase):
    def test_empty_rows(self) -> None:
        self.assertEqual(build_tasks_from_rows([]), [])

    def test_task_count_matches_rows(self) -> None:
        tasks = build_tasks_from_rows(_VALID_ROWS)
        self.assertEqual(len(tasks), len(_VALID_ROWS))

    def test_task_schema(self) -> None:
        """Mỗi task phải có data.image, meta với đủ key."""
        tasks = build_tasks_from_rows(_VALID_ROWS)
        required_meta = {"image_path", "source_dataset", "split", "label_original"}
        for task in tasks:
            with self.subTest(task=task):
                self.assertIn("data", task)
                self.assertIn("meta", task)
                self.assertIn("image", task["data"])
                for key in required_meta:
                    self.assertIn(key, task["meta"], msg=f"meta.{key} thiếu")

    def test_label_original_matches_label_in_row(self) -> None:
        tasks = build_tasks_from_rows(_VALID_ROWS)
        for row, task in zip(_VALID_ROWS, tasks):
            self.assertEqual(task["meta"]["label_original"], row["label"])

    def test_source_dataset_and_split_preserved(self) -> None:
        tasks = build_tasks_from_rows(_VALID_ROWS)
        for row, task in zip(_VALID_ROWS, tasks):
            self.assertEqual(task["meta"]["source_dataset"], row["source_dataset"])
            self.assertEqual(task["meta"]["split"], row["split"])

    def test_image_path_preserved_in_meta(self) -> None:
        tasks = build_tasks_from_rows(_VALID_ROWS)
        for row, task in zip(_VALID_ROWS, tasks):
            self.assertEqual(task["meta"]["image_path"], row["image_path"])

    def test_image_url_is_consistent_with_image_path(self) -> None:
        tasks = build_tasks_from_rows(_VALID_ROWS)
        for task in tasks:
            expected = _image_url_from_image_path(task["meta"]["image_path"])
            self.assertEqual(task["data"]["image"], expected)

    def test_missing_required_columns_raises(self) -> None:
        bad_rows = [{"image_path": "x.jpg", "label": "live"}]  # thiếu source_dataset, split
        with self.assertRaises(KeyError):
            build_tasks_from_rows(bad_rows)

    def test_valid_labels_only(self) -> None:
        """label_original phải là live, spoof, hoặc unknown."""
        tasks = build_tasks_from_rows(_VALID_ROWS)
        allowed = {"live", "spoof", "unknown"}
        for task in tasks:
            self.assertIn(task["meta"]["label_original"], allowed)


# ---------------------------------------------------------------------------
# load_raw_annotations
# ---------------------------------------------------------------------------


class TestLoadRawAnnotations(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _csv(self, rows: List[Dict[str, str]]) -> Path:
        p = self.tmp / "raw.csv"
        _write_csv(p, rows)
        return p

    def test_loads_all_rows(self) -> None:
        p = self._csv(_VALID_ROWS)
        rows = load_raw_annotations(p)
        self.assertEqual(len(rows), len(_VALID_ROWS))

    def test_limit_respected(self) -> None:
        p = self._csv(_VALID_ROWS)
        rows = load_raw_annotations(p, limit=2)
        self.assertEqual(len(rows), 2)

    def test_limit_larger_than_csv(self) -> None:
        p = self._csv(_VALID_ROWS)
        rows = load_raw_annotations(p, limit=1000)
        self.assertEqual(len(rows), len(_VALID_ROWS))

    def test_file_not_found_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_raw_annotations(self.tmp / "nonexistent.csv")

    def test_columns_present(self) -> None:
        p = self._csv(_VALID_ROWS)
        rows = load_raw_annotations(p)
        for col in ("image_path", "label", "source_dataset", "split"):
            self.assertIn(col, rows[0], msg=f"Cột '{col}' thiếu")


# ---------------------------------------------------------------------------
# write_tasks + round-trip
# ---------------------------------------------------------------------------


class TestWriteTasks(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_file_created(self) -> None:
        tasks = build_tasks_from_rows(_VALID_ROWS)
        out = self.tmp / "tasks.json"
        write_tasks(tasks, out)
        self.assertTrue(out.is_file())

    def test_output_is_valid_json_array(self) -> None:
        tasks = build_tasks_from_rows(_VALID_ROWS)
        out = self.tmp / "tasks.json"
        write_tasks(tasks, out)
        with out.open(encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertIsInstance(loaded, list)

    def test_round_trip_preserves_all_tasks(self) -> None:
        tasks = build_tasks_from_rows(_VALID_ROWS)
        out = self.tmp / "tasks.json"
        write_tasks(tasks, out)
        with out.open(encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(len(loaded), len(tasks))

    def test_round_trip_label_original(self) -> None:
        """label_original setelah round-trip harus sama persis."""
        tasks = build_tasks_from_rows(_VALID_ROWS)
        out = self.tmp / "tasks.json"
        write_tasks(tasks, out)
        with out.open(encoding="utf-8") as f:
            loaded = json.load(f)
        for orig, rt in zip(_VALID_ROWS, loaded):
            self.assertEqual(rt["meta"]["label_original"], orig["label"])

    def test_parent_dir_created_if_missing(self) -> None:
        tasks = build_tasks_from_rows(_VALID_ROWS)
        out = self.tmp / "subdir" / "tasks.json"
        self.assertFalse(out.parent.exists())
        write_tasks(tasks, out)
        self.assertTrue(out.is_file())


# ---------------------------------------------------------------------------
# Consistency: tasks.json so với raw.csv (kiểm tra nhãn, URL, duplicate)
# ---------------------------------------------------------------------------


class TestTasksConsistency(unittest.TestCase):
    """
    Mô phỏng check_tasks_against_raw bằng cách build task từ mock rows
    rồi kiểm tra từng điều kiện mà pipeline DVX yêu cầu.
    """

    def _tasks_from_rows(self, rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        return build_tasks_from_rows(rows)

    def test_no_duplicate_image_paths(self) -> None:
        tasks = self._tasks_from_rows(_VALID_ROWS)
        paths = [t["meta"]["image_path"] for t in tasks]
        self.assertEqual(len(paths), len(set(paths)), "Có image_path bị lặp")

    def test_label_matches_raw_csv(self) -> None:
        tasks = self._tasks_from_rows(_VALID_ROWS)
        by_path = {row["image_path"]: row for row in _VALID_ROWS}
        for task in tasks:
            ip = task["meta"]["image_path"]
            self.assertIn(ip, by_path, f"{ip} không tìm thấy trong raw.csv mock")
            self.assertEqual(
                task["meta"]["label_original"],
                by_path[ip]["label"],
                f"Nhãn không khớp cho {ip}",
            )

    def test_source_dataset_matches_raw_csv(self) -> None:
        tasks = self._tasks_from_rows(_VALID_ROWS)
        by_path = {row["image_path"]: row for row in _VALID_ROWS}
        for task in tasks:
            ip = task["meta"]["image_path"]
            self.assertEqual(
                task["meta"]["source_dataset"],
                by_path[ip]["source_dataset"],
            )

    def test_split_matches_raw_csv(self) -> None:
        tasks = self._tasks_from_rows(_VALID_ROWS)
        by_path = {row["image_path"]: row for row in _VALID_ROWS}
        for task in tasks:
            ip = task["meta"]["image_path"]
            self.assertEqual(task["meta"]["split"], by_path[ip]["split"])

    def test_image_url_consistent_with_image_path(self) -> None:
        tasks = self._tasks_from_rows(_VALID_ROWS)
        for task in tasks:
            expected_url = _image_url_from_image_path(task["meta"]["image_path"])
            self.assertEqual(
                task["data"]["image"],
                expected_url,
                f"URL không khớp cho {task['meta']['image_path']}",
            )

    def test_label_mismatch_detected(self) -> None:
        """Giả lập task có nhãn sai, kiểm tra code phát hiện được."""
        tasks = self._tasks_from_rows(_VALID_ROWS)
        # Sửa nhãn của task đầu tiên thành nhãn sai
        tasks[0]["meta"]["label_original"] = "live" if _VALID_ROWS[0]["label"] == "spoof" else "spoof"
        by_path = {row["image_path"]: row for row in _VALID_ROWS}
        mismatches = [
            t for t in tasks
            if by_path[t["meta"]["image_path"]]["label"] != t["meta"]["label_original"]
        ]
        self.assertGreater(len(mismatches), 0, "Không phát hiện được nhãn sai")

    def test_duplicate_image_path_detected(self) -> None:
        """Giả lập duplicate task, kiểm tra code phát hiện được."""
        rows_with_dup = _VALID_ROWS + [_VALID_ROWS[0]]
        tasks = self._tasks_from_rows(rows_with_dup)
        paths = [t["meta"]["image_path"] for t in tasks]
        duplicates = [p for p in paths if paths.count(p) > 1]
        self.assertGreater(len(duplicates), 0, "Không phát hiện được duplicate")


if __name__ == "__main__":
    unittest.main()
