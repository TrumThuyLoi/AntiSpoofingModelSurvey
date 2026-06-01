#!/usr/bin/env python3
"""Ghi configs/dataset_drivers_250_fn*.yaml cho mọi mức SFAS bbox expansion."""

from __future__ import annotations

from pathlib import Path

from drivers_250_fn_expansions import (
    SFAS_BBOX_EXPANSIONS,
    crop_output_dir,
    dataset_config_path,
    sample_csv_name,
    source_dataset_slug,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _yaml_for_expansion(expansion: float) -> str:
    slug = source_dataset_slug(expansion)
    crop_rel = crop_output_dir(expansion, repo_root=REPO_ROOT).relative_to(REPO_ROOT)
    ann_rel = Path("data/sampled") / sample_csv_name(expansion)
    return f"""name: {slug}_sample
annotation_path: {ann_rel.as_posix()}
image_root: {crop_rel.as_posix()}
source_dataset: {slug}
default_split: all
labels:
  live: live
  spoof: spoof
  unknown: unknown
"""


def main() -> None:
    for expansion in SFAS_BBOX_EXPANSIONS:
        path = dataset_config_path(expansion, repo_root=REPO_ROOT)
        path.write_text(_yaml_for_expansion(expansion), encoding="utf-8")
        print(f"[OK] {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
