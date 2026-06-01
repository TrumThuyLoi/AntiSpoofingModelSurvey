#!/usr/bin/env python3
"""Crop mặt SFAS cho data/drivers_250_FN — sinh đủ thư mục expansion 1.0, 1.2, 1.4, 1.6."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from drivers_250_fn_expansions import SFAS_BBOX_EXPANSIONS, crop_output_dir
INPUT_ROOT = REPO_ROOT / "data" / "drivers_250_FN"
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


def _crop_all_expansions(*, skip_existing: bool = True) -> dict[float, dict[str, int]]:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Không tìm thấy: {INPUT_ROOT}")

    detector = vn._load_sfas_detection()
    sources = list(iter_driver_images(INPUT_ROOT))
    stats_by_exp: dict[float, dict[str, int]] = {}

    for expansion in SFAS_BBOX_EXPANSIONS:
        output_root = crop_output_dir(expansion, repo_root=REPO_ROOT)
        stats = {"total": 0, "ok": 0, "skip": 0, "fail": 0}
        desc = f"Crop exp={expansion:g}"
        for src in tqdm(sources, desc=desc, unit="img"):
            stats["total"] += 1
            rel = src.relative_to(INPUT_ROOT)
            dest = output_root / rel

            if skip_existing and dest.is_file():
                stats["skip"] += 1
                continue

            cropped = vn.detect_and_crop_rgb(src, detector, bbox_expansion=expansion)
            if cropped is None:
                stats["fail"] += 1
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(dest, format="JPEG", quality=95)
            stats["ok"] += 1

        stats_by_exp[expansion] = stats
        print(f"exp={expansion:g} -> {output_root}")
        print(
            f"  images={stats['total']} saved={stats['ok']} "
            f"skipped={stats['skip']} fail_detect={stats['fail']}"
        )

    return stats_by_exp


def main() -> None:
    print(f"Input: {INPUT_ROOT}")
    _crop_all_expansions()


if __name__ == "__main__":
    main()
