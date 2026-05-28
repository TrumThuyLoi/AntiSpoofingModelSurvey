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
        default=REPO_ROOT / "configs/model.yaml",
        help="Đường dẫn model config (mặc định: configs/model.yaml).",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = run_batch_inference(
        dataset_config_path=args.dataset_config,
        model_config_path=args.model_config,
        inference_config_path=args.inference_config,
        annotation_csv=args.annotation_csv,
        repo_root=REPO_ROOT,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
