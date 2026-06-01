#!/usr/bin/env python3
"""CLI chạy batch inference MiniFASNet — đọc annotation CSV, ghi predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.run_batch import run_batch_inference


def discover_config_paths(repo_root: Path, prefix: str) -> list[Path]:
    """Liệt kê configs/<prefix>_*.yaml (sorted)."""
    configs_dir = repo_root / "configs"
    return sorted(configs_dir.glob(f"{prefix}_*.yaml"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy batch inference trên annotation CSV và ghi run_YYYYMMDD_HHMMSS.csv.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=REPO_ROOT / "configs/dataset.yaml",
        help="Đường dẫn dataset config (mặc định: configs/dataset.yaml).",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs/model_minifasnet.yaml",
        help="Đường dẫn model config (mặc định: configs/model_minifasnet.yaml).",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=REPO_ROOT / "configs/inference.yaml",
        help="Đường dẫn inference config (mặc định: configs/inference.yaml).",
    )
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=None,
        help="Override annotation_path trong dataset config.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Chạy inference cho mọi configs/model_*.yaml × configs/dataset_*.yaml.",
    )
    return parser.parse_args()


def _run_one(
    *,
    dataset_config: Path,
    model_config: Path,
    inference_config: Path,
    annotation_csv: Path | None,
    repo_root: Path,
) -> Path:
    return run_batch_inference(
        dataset_config_path=dataset_config,
        model_config_path=model_config,
        inference_config_path=inference_config,
        annotation_csv=annotation_csv,
        repo_root=repo_root,
    )


def main() -> int:
    args = parse_args()
    inference_config = (
        args.inference_config
        if args.inference_config.is_absolute()
        else REPO_ROOT / args.inference_config
    )

    if args.all:
        model_configs = discover_config_paths(REPO_ROOT, "model")
        dataset_configs = discover_config_paths(REPO_ROOT, "dataset")
        if not model_configs:
            raise ValueError(f"Không tìm thấy configs/model_*.yaml trong {REPO_ROOT / 'configs'}")
        if not dataset_configs:
            raise ValueError(f"Không tìm thấy configs/dataset_*.yaml trong {REPO_ROOT / 'configs'}")

        for model_config in model_configs:
            for dataset_config in dataset_configs:
                print(f"\n=== {model_config.name} × {dataset_config.name} ===")
                out_path = _run_one(
                    dataset_config=dataset_config,
                    model_config=model_config,
                    inference_config=inference_config,
                    annotation_csv=args.annotation_csv,
                    repo_root=REPO_ROOT,
                )
                print(out_path)
        return 0

    out_path = _run_one(
        dataset_config=args.dataset_config,
        model_config=args.model_config,
        inference_config=inference_config,
        annotation_csv=args.annotation_csv,
        repo_root=REPO_ROOT,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
