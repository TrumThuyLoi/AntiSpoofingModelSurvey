#!/usr/bin/env python3
"""Inference + evaluation một model trên toàn bộ dataset drivers_250_fn (mọi expansion)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from drivers_250_fn_expansions import SFAS_BBOX_EXPANSIONS, dataset_config_path, source_dataset_slug
from src.inference.run_batch import run_batch_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy inference + evaluation cho mọi expansion drivers_250_fn.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs/model_minifasnet.yaml",
        help="Model config (mặc định MiniFASNet).",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=REPO_ROOT / "configs/inference.yaml",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=REPO_ROOT / "configs/evaluation.yaml",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Chỉ chạy inference, bỏ qua evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_config = args.model_config if args.model_config.is_absolute() else REPO_ROOT / args.model_config
    inference_config = (
        args.inference_config
        if args.inference_config.is_absolute()
        else REPO_ROOT / args.inference_config
    )
    eval_config = args.eval_config if args.eval_config.is_absolute() else REPO_ROOT / args.eval_config

    for expansion in SFAS_BBOX_EXPANSIONS:
        dataset_config = dataset_config_path(expansion, repo_root=REPO_ROOT)
        if not dataset_config.is_file():
            raise FileNotFoundError(
                f"Thiếu {dataset_config}. Chạy: "
                "python3 scripts/create_drivers_250_fn_dataset_configs.py"
            )

        slug = source_dataset_slug(expansion)
        print(f"\n=== expansion={expansion:g} source_dataset={slug} ===")

        pred_path = run_batch_inference(
            dataset_config_path=dataset_config,
            model_config_path=model_config,
            inference_config_path=inference_config,
            repo_root=REPO_ROOT,
        )
        print(f"predictions: {pred_path}")

        if args.skip_evaluation:
            continue

        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/run_evaluation.py"),
                "--config",
                str(eval_config),
                "--predictions",
                str(pred_path),
                "--dataset",
                slug,
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
