"""Tests cho reports layout (model_id, paths)."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.reports.layout import (  # noqa: E402
    ModelReportPaths,
    derive_model_id_from_weights,
    model_report_paths,
    resolve_model_id,
    sanitize_model_id,
)


class TestModelId(unittest.TestCase):
    def test_explicit_model_id(self) -> None:
        cfg = {
            "model_id": "minifasnet_v2_2p7",
            "name": "minifasnet",
            "weights_dir": "models/minifasnet/2.7_80x80_MiniFASNetV2.pth",
        }
        self.assertEqual(resolve_model_id(cfg, ROOT), "minifasnet_v2_2p7")

    def test_derive_from_weights(self) -> None:
        weights = ROOT / "models/minifasnet/2.7_80x80_MiniFASNetV2.pth"
        if not weights.is_file():
            self.skipTest(f"Thiếu {weights}")
        cfg = {"name": "minifasnet", "weights_dir": str(weights.relative_to(ROOT))}
        self.assertEqual(resolve_model_id(cfg, ROOT), derive_model_id_from_weights("minifasnet", weights))

    def test_sanitize_dots(self) -> None:
        self.assertEqual(sanitize_model_id("Foo.Bar-1"), "foopbar-1")


class TestModelReportPaths(unittest.TestCase):
    def test_paths_under_models_namespace(self) -> None:
        paths = ModelReportPaths("minifasnet_v2_2p7", ROOT)
        self.assertEqual(
            paths.predictions_latest("celeba_spoof"),
            ROOT / "reports/models/minifasnet_v2_2p7/predictions/celeba_spoof/latest.csv",
        )
        self.assertEqual(
            paths.metrics_dir("casia_fasd"),
            ROOT / "reports/models/minifasnet_v2_2p7/metrics/casia_fasd",
        )

    def test_from_model_yaml(self) -> None:
        model_yaml = ROOT / "configs/model.yaml"
        if not model_yaml.is_file():
            self.skipTest("Thiếu configs/model.yaml")
        import yaml

        with model_yaml.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        paths = model_report_paths(cfg, ROOT)
        self.assertEqual(paths.model_id, "minifasnet_v2_2p7")


if __name__ == "__main__":
    unittest.main()
