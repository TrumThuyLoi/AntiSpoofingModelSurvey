#!/usr/bin/env python3
"""Tạo file <dataset>_sample.csv cho celeba_spoof, casia_fasd, face_antispoofing_vn và drivers_250_fn."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from drivers_250_fn_expansions import SFAS_BBOX_EXPANSIONS, crop_output_dir, source_dataset_slug

RAW_ROOT = REPO_ROOT / "data" / "raw"
OUT_ROOT = REPO_ROOT / "data" / "sampled"
FIELDNAMES = ["image_path", "label", "source_dataset", "split", "is_valid", "note"]
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SUPPORTED_DATASETS = ("celeba_spoof", "casia_fasd", "face_antispoofing_vn", "drivers_250_fn")


def _read_rows(dataset: str) -> list[dict[str, str]]:
    path = RAW_ROOT / dataset / "annotations" / "raw.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_valid_true(row: dict[str, str]) -> bool:
    return str(row.get("is_valid", "")).strip().lower() == "true"


def _sample_celeba(rows: list[dict[str, str]], k: int = 2000) -> list[dict[str, str]]:
    valid_rows = [r for r in rows if _is_valid_true(r)]
    live = [r for r in valid_rows if r.get("label") == "live"]
    spoof = [r for r in valid_rows if r.get("label") == "spoof"]
    if len(live) < k or len(spoof) < k:
        raise ValueError(
            f"Không đủ dữ liệu celeba_spoof (live={len(live)}, spoof={len(spoof)}, cần mỗi loại={k})."
        )
    return random.sample(live, k) + random.sample(spoof, k)


def _sample_casia(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if _is_valid_true(r)]


def _sample_face_antispoofing_vn(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Toàn bộ dòng is_valid=true từ raw test (hf_raw → data/raw/face_antispoofing_vn/)."""
    return _sample_casia(rows)


def _sample_drivers_250_fn_from_crop(
    crop_root: Path,
    *,
    expansion: float,
) -> list[dict[str, str]]:
    """Toàn bộ ảnh trong một thư mục crop — ground truth live (bộ FN)."""
    if not crop_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục crop: {crop_root}")

    slug = source_dataset_slug(expansion)
    rows: list[dict[str, str]] = []
    for path in sorted(crop_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        rows.append(
            {
                "image_path": str(path.relative_to(REPO_ROOT)),
                "label": "live",
                "source_dataset": slug,
                "split": "all",
                "is_valid": "True",
                "note": f"sfas_exp={expansion:g}",
            }
        )
    if not rows:
        raise ValueError(f"Không có ảnh hợp lệ trong {crop_root}")
    return rows


def _build_all_drivers_250_fn_samples() -> list[tuple[str, list[dict[str, str]]]]:
    """Một lần chạy tạo CSV cho mọi expansion (1.0, 1.2, 1.4, 1.6)."""
    out: list[tuple[str, list[dict[str, str]]]] = []
    for expansion in SFAS_BBOX_EXPANSIONS:
        crop_root = crop_output_dir(expansion, repo_root=REPO_ROOT)
        rows = _sample_drivers_250_fn_from_crop(crop_root, expansion=expansion)
        out.append((source_dataset_slug(expansion), rows))
    return out


def _write_rows(dataset: str, rows: list[dict[str, str]]) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{dataset}_sample.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    return out_path


def _build_drivers_250_fn_sample() -> list[tuple[str, list[dict[str, str]]]]:
    return _build_all_drivers_250_fn_samples()


def _build_celeba_sample() -> tuple[str, list[dict[str, str]]]:
    return "celeba_spoof", _sample_celeba(_read_rows("celeba_spoof"), k=2000)


def _build_casia_sample() -> tuple[str, list[dict[str, str]]]:
    return "casia_fasd", _sample_casia(_read_rows("casia_fasd"))


def _build_face_antispoofing_vn_sample() -> tuple[str, list[dict[str, str]]]:
    return "face_antispoofing_vn", _sample_face_antispoofing_vn(_read_rows("face_antispoofing_vn"))


_BUILDERS = {
    "celeba_spoof": _build_celeba_sample,
    "casia_fasd": _build_casia_sample,
    "face_antispoofing_vn": _build_face_antispoofing_vn_sample,
    "drivers_250_fn": _build_drivers_250_fn_sample,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo data/sampled/<dataset>_sample.csv.")
    parser.add_argument(
        "--dataset",
        choices=_SUPPORTED_DATASETS,
        default=None,
        help="Chỉ tạo sample cho một dataset (mặc định: tất cả dataset có dữ liệu).",
    )
    return parser.parse_args()


def main() -> None:
    random.seed(42)
    args = parse_args()

    datasets = [args.dataset] if args.dataset else list(_SUPPORTED_DATASETS)
    for name in datasets:
        built = _BUILDERS[name]()
        if name == "drivers_250_fn":
            for dataset, rows in built:
                out_path = _write_rows(dataset, rows)
                print(f"[OK] {dataset}: {len(rows)} -> {out_path}")
        else:
            dataset, rows = built
            out_path = _write_rows(dataset, rows)
            print(f"[OK] {dataset}: {len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
