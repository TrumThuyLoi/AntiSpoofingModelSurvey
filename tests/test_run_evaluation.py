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
    _plot_apcer_bpcer_curve,
    _write_summary_csv,
    _write_threshold_json,
    discover_config_paths,
    parse_predictions_metadata,
    resolve_dataset_slug,
    source_dataset_from_config,
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

    def test_from_models_namespace_latest(self) -> None:
        p = self.tmp / "predictions" / "celeba_spoof" / "latest.csv"
        p.parent.mkdir(parents=True)
        _write_predictions_csv(p, source_dataset=None)
        self.assertEqual(resolve_dataset_slug(p), "celeba_spoof")

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


class TestDiscoverConfigPaths(unittest.TestCase):
    def test_discovers_repo_configs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertGreaterEqual(len(discover_config_paths(root, "model")), 3)
        self.assertGreaterEqual(len(discover_config_paths(root, "dataset")), 7)


class TestSourceDatasetFromConfig(unittest.TestCase):
    def test_reads_source_dataset(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_x.yaml"
            path.write_text("source_dataset: casia_fasd\n", encoding="utf-8")
            self.assertEqual(source_dataset_from_config(path), "casia_fasd")


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

    def test_apcer_bpcer_curve_png_written(self) -> None:
        rows = [
            {"label_true": "live", "live_score": "0.9"},
            {"label_true": "spoof", "live_score": "0.2"},
        ]
        results = [_evaluate_threshold(rows, t) for t in (0.3, 0.5, 0.7)]
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "apcer_bpcer_vs_threshold.png"
            _plot_apcer_bpcer_curve(out, results, dataset_slug="celeba_spoof", model_id="test_model")
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 500)


class TestRunEvaluationAll(unittest.TestCase):
    def test_all_evaluates_existing_pairs_only(self) -> None:
        from unittest.mock import MagicMock, patch

        import run_evaluation as mod

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            configs = repo / "configs"
            configs.mkdir()
            (configs / "evaluation.yaml").write_text("thresholds: [0.5]\n", encoding="utf-8")
            (configs / "model_a.yaml").write_text("model_id: model_a\n", encoding="utf-8")
            (configs / "model_b.yaml").write_text("model_id: model_b\n", encoding="utf-8")
            (configs / "dataset_x.yaml").write_text("source_dataset: dataset_x\n", encoding="utf-8")
            (configs / "dataset_y.yaml").write_text("source_dataset: dataset_y\n", encoding="utf-8")

            pred_root = repo / "reports" / "models"
            for mid, ds in (("model_a", "dataset_x"), ("model_b", "dataset_y")):
                pdir = pred_root / mid / "predictions" / ds
                pdir.mkdir(parents=True)
                _write_predictions_csv(pdir / "latest.csv", source_dataset=ds)

            with patch.object(mod, "REPO_ROOT", repo):
                with patch.object(mod, "parse_args") as mock_parse:
                    mock_parse.return_value = MagicMock(
                        all=True,
                        predictions=None,
                        config=repo / "configs/evaluation.yaml",
                        dataset=None,
                        flat_output=False,
                        model_id=None,
                    )
                    with patch.object(mod, "evaluate_predictions") as mock_eval:
                        rc = mod.main()

            self.assertEqual(rc, 0)
            self.assertEqual(mock_eval.call_count, 2)

    def test_all_raises_without_configs(self) -> None:
        from unittest.mock import MagicMock, patch

        import run_evaluation as mod

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "configs").mkdir()
            (repo / "configs/evaluation.yaml").write_text("thresholds: [0.5]\n", encoding="utf-8")

            with patch.object(mod, "REPO_ROOT", repo):
                with patch.object(mod, "parse_args") as mock_parse:
                    mock_parse.return_value = MagicMock(
                        all=True,
                        predictions=None,
                        config=repo / "configs/evaluation.yaml",
                        dataset=None,
                        flat_output=False,
                        model_id=None,
                    )
                    with self.assertRaises(ValueError):
                        mod.main()


if __name__ == "__main__":
    unittest.main()
