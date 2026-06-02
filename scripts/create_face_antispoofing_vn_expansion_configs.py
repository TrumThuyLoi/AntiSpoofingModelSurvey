#!/usr/bin/env python3
"""Ghi configs/dataset_face_antispoofing_vn_exp*.yaml cho mọi mức SFAS bbox expansion."""

from __future__ import annotations

from pathlib import Path

from sfas_bbox_expansions import (
    REPO_ROOT,
    SFAS_BBOX_EXPANSIONS,
    face_vn_crop_output_dir,
    face_vn_dataset_config_path,
    face_vn_sample_csv_name,
    face_vn_source_dataset_slug,
)


def _yaml_for_expansion(expansion: float) -> str:
    slug = face_vn_source_dataset_slug(expansion)
    crop_rel = face_vn_crop_output_dir(expansion, repo_root=REPO_ROOT).relative_to(REPO_ROOT)
    ann_rel = Path("data/sampled") / face_vn_sample_csv_name(expansion)
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
        path = face_vn_dataset_config_path(expansion, repo_root=REPO_ROOT)
        path.write_text(_yaml_for_expansion(expansion), encoding="utf-8")
        print(f"[OK] {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
