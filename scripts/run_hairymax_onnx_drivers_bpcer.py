#!/usr/bin/env python3
"""BPCER trên drivers_250_fn (ảnh crop SFAS) với ONNX hairymax — tich_hop_onnx_hairymax.md."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from sfas_bbox_expansions import drivers_sample_csv_name  # noqa: E402

HAIRYMAX_DEFAULT = REPO_ROOT / "third_party/Face-AntiSpoofing-hairymax"
DEFAULT_EXPANSION = 1.5

DEFAULT_MODELS: tuple[tuple[str, Path], ...] = (
    ("pretrained", REPO_ROOT / "models/AntiSpoofing_bin_1.5_128.onnx"),
    ("retrained", REPO_ROOT / "models/retrained_AntiSpoofing_bin_1.5_128.onnx"),
)


def _load_antispoof(hairymax_dir: Path):
    src = hairymax_dir / "src"
    if not (src / "FaceAntiSpoofing.py").is_file():
        raise FileNotFoundError(
            f"Thiếu {src / 'FaceAntiSpoofing.py'}. Clone:\n"
            "  git clone --depth 1 https://github.com/hairymax/Face-AntiSpoofing.git "
            f"{hairymax_dir}",
        )
    sys.path.insert(0, str(hairymax_dir))
    from src.FaceAntiSpoofing import AntiSpoof  # noqa: E402

    return AntiSpoof


def is_live_accepted(probs, *, threshold: float) -> bool:
    """Quy ước hairymax binary: class 0 = live, chấp nhận nếu argmax==0 và p[0] > threshold."""
    p = np.asarray(probs).reshape(-1)
    if p.size == 0:
        return False
    label = int(np.argmax(p))
    score = float(p[0])
    return label == 0 and score > threshold


def evaluate_model(
    *,
    name: str,
    weights: Path,
    sample_csv: Path,
    threshold: float,
    hairymax_dir: Path,
) -> tuple[int, int] | None:
    if not weights.is_file():
        print(f"{name}: skip — thiếu {weights}")
        return None

    AntiSpoof = _load_antispoof(hairymax_dir)
    det = AntiSpoof(str(weights), model_img_size=128)

    fn = n = 0
    with sample_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = (row.get("image_path") or "").strip()
            if not rel:
                continue
            img_path = Path(rel) if Path(rel).is_absolute() else REPO_ROOT / rel
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            pred = det([img])[0]
            n += 1
            if not is_live_accepted(pred, threshold=threshold):
                fn += 1

    bpcer = fn / n if n else 0.0
    print(
        f"{name} ({weights.name}): n={n} FN={fn} "
        f"BPCER={bpcer:.4f} ({fn}/{n}) @ threshold={threshold:g}",
    )
    return fn, n


def _sample_csv_for_expansion(expansion: float) -> Path:
    return REPO_ROOT / "data/sampled" / drivers_sample_csv_name(expansion)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tính BPCER drivers trên ONNX hairymax (AntiSpoofing_bin_1.5_128).",
    )
    p.add_argument(
        "--expansion",
        type=float,
        default=DEFAULT_EXPANSION,
        help="SFAS bbox expansion cho sample CSV (mặc định: 1.5, khớp tên model).",
    )
    p.add_argument(
        "--sample-csv",
        type=Path,
        default=None,
        help="Override CSV sample (mặc định: data/sampled/drivers_250_fn_exp<expansion>_sample.csv).",
    )
    p.add_argument(
        "--model",
        type=Path,
        action="append",
        help="Đường dẫn .onnx (lặp lại). Mặc định: pretrained + retrain trong models/.",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--hairymax-dir",
        type=Path,
        default=HAIRYMAX_DEFAULT,
        help="Clone hairymax/Face-AntiSpoofing.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_csv is not None:
        sample_csv = args.sample_csv if args.sample_csv.is_absolute() else REPO_ROOT / args.sample_csv
    else:
        sample_csv = _sample_csv_for_expansion(args.expansion)
    hairymax_dir = args.hairymax_dir if args.hairymax_dir.is_absolute() else REPO_ROOT / args.hairymax_dir

    if not sample_csv.is_file():
        raise FileNotFoundError(
            f"Thiếu {sample_csv}. Chạy: "
            "python3 scripts/create_test_sample_annotation.py --dataset drivers_250_fn "
            f"(expansion {args.expansion:g})",
        )

    if args.model:
        models = [(p.stem, p if p.is_absolute() else REPO_ROOT / p) for p in args.model]
    else:
        models = list(DEFAULT_MODELS)

    print(f"expansion: {args.expansion:g}")
    print(f"sample: {sample_csv}")
    print(f"hairymax: {hairymax_dir}")
    ran = 0
    for name, weights in models:
        if evaluate_model(
            name=name,
            weights=weights,
            sample_csv=sample_csv,
            threshold=args.threshold,
            hairymax_dir=hairymax_dir,
        ) is not None:
            ran += 1

    if ran == 0:
        print("Không có model nào được chạy (thiếu file .onnx).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
