#!/usr/bin/env python3
"""Crop mặt cho data/drivers_250_FN → data/drivers_250_fn_cropped."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO_ROOT / "data" / "drivers_250_FN"
OUTPUT_ROOT = REPO_ROOT / "data" / "drivers_250_fn_cropped"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_spec = importlib.util.spec_from_file_location(
    "create_vietnam_hf_dataset",
    REPO_ROOT / "scripts" / "create_vietnam_hf_dataset.py",
)
vn = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(vn)


def iter_driver_images(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        yield path


def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Không tìm thấy: {INPUT_ROOT}")

    detector = vn._load_sfas_detection()
    stats = {"total": 0, "ok": 0, "skip": 0, "fail": 0}

    for src in tqdm(list(iter_driver_images(INPUT_ROOT)), desc="Crop", unit="img"):
        stats["total"] += 1
        rel = src.relative_to(INPUT_ROOT)
        dest = OUTPUT_ROOT / rel

        if dest.is_file():
            stats["skip"] += 1
            continue

        cropped = vn.detect_and_crop_rgb(src, detector)
        if cropped is None:
            stats["fail"] += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(dest, format="JPEG", quality=95)
        stats["ok"] += 1

    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(
        f"Crop: images={stats['total']} saved={stats['ok']} "
        f"skipped={stats['skip']} fail_detect={stats['fail']}"
    )


if __name__ == "__main__":
    main()
