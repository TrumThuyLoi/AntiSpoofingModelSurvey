"""Đường dẫn artifact báo cáo theo model_id — mỗi model một cây riêng dưới reports/models/."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs").is_dir():
        raise FileNotFoundError(f"Không tìm thấy configs/ tại {root}")
    return root


def sanitize_model_id(value: str) -> str:
    """Chuẩn hóa model_id an toàn cho tên thư mục."""
    slug = value.strip().lower().replace(".", "p")
    slug = re.sub(r"[^\w-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug or not _MODEL_ID_RE.match(slug):
        raise ValueError(f"model_id không hợp lệ sau chuẩn hóa: {value!r} → {slug!r}")
    return slug


def derive_model_id_from_weights(model_name: str, weights_path: Path) -> str:
    """
    Fallback (B): ``name`` + stem weight.

    VD. minifasnet + ``2.7_80x80_MiniFASNetV2.pth`` → ``minifasnet_2p7_80x80_minifasnetv2``
    """
    name = (model_name or "model").strip().lower()
    stem = weights_path.stem.replace(".", "p")
    stem = re.sub(r"[^\w]+", "_", stem).strip("_").lower()
    return sanitize_model_id(f"{name}_{stem}")


def resolve_model_id(model_cfg: dict[str, Any], repo_root: Path | None = None) -> str:
    """
    Xác định model_id: (A) ``model_cfg['model_id']`` hoặc (B) derive từ name + weights_dir.
    """
    root = repo_root or _repo_root()
    explicit = model_cfg.get("model_id")
    if explicit and str(explicit).strip():
        return sanitize_model_id(str(explicit))

    weights_value = model_cfg.get("weights_dir")
    if not weights_value:
        raise ValueError("model config thiếu weights_dir (cần cho derive model_id).")

    weights_path = Path(str(weights_value))
    if not weights_path.is_absolute():
        weights_path = root / weights_path
    if not weights_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy weight: {weights_path}")

    return derive_model_id_from_weights(str(model_cfg.get("name", "model")), weights_path)


@dataclass(frozen=True)
class ModelReportPaths:
    """``reports/models/<model_id>/predictions|metrics/<source_dataset>/``."""

    model_id: str
    repo_root: Path

    @property
    def model_root(self) -> Path:
        return self.repo_root / "reports" / "models" / self.model_id

    def predictions_dir(self, source_dataset: str) -> Path:
        return self.model_root / "predictions" / source_dataset

    def predictions_latest(self, source_dataset: str) -> Path:
        return self.predictions_dir(source_dataset) / "latest.csv"

    def metrics_dir(self, source_dataset: str) -> Path:
        return self.model_root / "metrics" / source_dataset

    def manifest_path(self) -> Path:
        return self.model_root / "model_manifest.json"

    def write_manifest(self, payload: dict[str, Any]) -> None:
        path = self.manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def model_report_paths(model_cfg: dict[str, Any], repo_root: Path | None = None) -> ModelReportPaths:
    root = repo_root or _repo_root()
    model_id = resolve_model_id(model_cfg, root)
    return ModelReportPaths(model_id=model_id, repo_root=root)
