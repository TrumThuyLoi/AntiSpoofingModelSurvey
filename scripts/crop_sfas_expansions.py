#!/usr/bin/env python3
"""SFAS detect+crop cho drivers_250_FN và face_antispoofing_vn (mọi expansion)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from sfas_bbox_expansions import (
    SFAS_BBOX_EXPANSIONS,
    drivers_crop_output_dir,
    face_vn_crop_output_dir,
    face_vn_input_root,
)

DRIVERS_INPUT_ROOT = REPO_ROOT / "data" / "drivers_250_FN"
FACE_VN_INPUT_ROOT = face_vn_input_root(repo_root=REPO_ROOT)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_spec = importlib.util.spec_from_file_location(
    "create_vietnam_hf_dataset",
    REPO_ROOT / "scripts" / "create_vietnam_hf_dataset.py",
)
vn = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(vn)


def _iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            yield path


def crop_drivers(*, skip_existing: bool = True) -> dict[float, dict[str, int]]:
    if not DRIVERS_INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Không tìm thấy: {DRIVERS_INPUT_ROOT}")

    detector = vn._load_sfas_detection()
    sources = list(_iter_images(DRIVERS_INPUT_ROOT))
    stats_by_exp: dict[float, dict[str, int]] = {}

    print(f"\n=== drivers_250_FN ({len(sources)} ảnh) ===")
    for expansion in SFAS_BBOX_EXPANSIONS:
        output_root = drivers_crop_output_dir(expansion, repo_root=REPO_ROOT)
        stats = {"total": 0, "ok": 0, "skip": 0, "fail": 0}
        for src in tqdm(sources, desc=f"drivers exp={expansion:g}", unit="img"):
            stats["total"] += 1
            rel = src.relative_to(DRIVERS_INPUT_ROOT)
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


def crop_face_antispoofing_vn(*, skip_existing: bool = True) -> dict[float, dict[str, int]]:
    if not FACE_VN_INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Không tìm thấy: {FACE_VN_INPUT_ROOT}")

    detector = vn._load_sfas_detection()
    sources = list(_iter_images(FACE_VN_INPUT_ROOT))
    stats_by_exp: dict[float, dict[str, int]] = {}

    print(f"\n=== face_antispoofing_vn ({len(sources)} ảnh) ===")
    for expansion in SFAS_BBOX_EXPANSIONS:
        output_root = face_vn_crop_output_dir(expansion, repo_root=REPO_ROOT)
        stats = {"total": 0, "ok": 0, "skip": 0, "fail": 0}
        for src in tqdm(sources, desc=f"face_vn exp={expansion:g}", unit="img"):
            stats["total"] += 1
            rel = src.relative_to(FACE_VN_INPUT_ROOT)
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
    print(f"Expansions: {SFAS_BBOX_EXPANSIONS}")
    print(f"Drivers input: {DRIVERS_INPUT_ROOT}")
    print(f"Face VN input: {FACE_VN_INPUT_ROOT}")
    crop_drivers()
    crop_face_antispoofing_vn()


if __name__ == "__main__":
    main()
