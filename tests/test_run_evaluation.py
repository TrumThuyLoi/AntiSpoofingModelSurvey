"""Tests cho resolve_dataset_slug và output metrics theo dataset."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_evaluation import (  # noqa: E402
    _evaluate_threshold,
    _write_summary_csv,
    _write_threshold_json,
    parse_predictions_metadata,
    resolve_dataset_slug,
)
from run_evaluation import EvalStats  # noqa: E402


def _write_predictions_csv(path: Path, *, source_dataset: str | None = "celeba_spoof") -> None:
    lines = []
    if source_dataset is not None:
        lines.append(f"# source_dataset={source_dataset}\n")
    lines.append("image_path,label_true,label_pred,live_score,spoof_score,error\n")
    lines.append("data/raw/x.jpg,live,live,0.9,0.1,\n")
    lines.append("data/raw/y.jpg,spoof,spoof,0.1,0.9,\n")
    path.write_text("".join(lines), encoding="utf-8")


class TestResolveDatasetSlug(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_from_latest_filename(self) -> None:
        p = self.tmp / "casia_fasd_latest.csv"
        _write_predictions_csv(p, source_dataset=None)
        self.assertEqual(resolve_dataset_slug(p), "casia_fasd")

    def test_from_run_timestamp_filename(self) -> None:
        p = self.tmp / "run_celeba_spoof_20260115_120000.csv"
        _write_predictions_csv(p, source_dataset=None)
        self.assertEqual(resolve_dataset_slug(p), "celeba_spoof")

    def test_from_metadata_comment(self) -> None:
        p = self.tmp / "predictions.csv"
        _write_predictions_csv(p, source_dataset="casia_fasd")
        self.assertEqual(resolve_dataset_slug(p), "casia_fasd")

    def test_cli_override(self) -> None:
        p = self.tmp / "anything.csv"
        _write_predictions_csv(p)
        self.assertEqual(resolve_dataset_slug(p, "manual_name"), "manual_name")

    def test_unresolvable_raises(self) -> None:
        p = self.tmp / "predictions.csv"
        _write_predictions_csv(p, source_dataset=None)
        with self.assertRaises(ValueError):
            resolve_dataset_slug(p)


class TestParsePredictionsMetadata(unittest.TestCase):
    def test_reads_comment_header(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            path.write_text(
                "# source_dataset=celeba_spoof\n"
                "# model_name=MiniFASNet\n"
                "image_path,label_true,live_score,error\n",
                encoding="utf-8",
            )
            meta = parse_predictions_metadata(path)
            self.assertEqual(meta["source_dataset"], "celeba_spoof")
            self.assertEqual(meta["model_name"], "MiniFASNet")


class TestMetricsOutputLayout(unittest.TestCase):
    def test_json_includes_provenance(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "metrics_threshold_0.5.json"
            stats = EvalStats(total_rows=2, kept_rows=2, kept_live_rows=1, kept_spoof_rows=1)
            result = _evaluate_threshold(
                [
                    {"label_true": "live", "live_score": "0.9"},
                    {"label_true": "spoof", "live_score": "0.2"},
                ],
                0.5,
            )
            provenance = {
                "source_dataset": "celeba_spoof",
                "predictions_path": "reports/predictions/celeba_spoof_latest.csv",
                "evaluated_at": "2026-01-01T00:00:00",
            }
            _write_threshold_json(out, result, stats, provenance)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["provenance"]["source_dataset"], "celeba_spoof")
            self.assertIn("metrics", payload)

    def test_summary_csv_written(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "metrics_summary.csv"
            stats = EvalStats(kept_rows=2, kept_live_rows=1, kept_spoof_rows=1)
            results = [_evaluate_threshold([{"label_true": "live", "live_score": "0.9"}], 0.5)]
            _write_summary_csv(out, stats, results)
            self.assertTrue(out.is_file())
            self.assertIn("threshold", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
