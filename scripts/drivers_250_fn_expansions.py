"""Hằng số expansion SFAS và đường dẫn dataset drivers_250_fn."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Một lần chạy crop / benchmark dùng đủ bốn mức này.
SFAS_BBOX_EXPANSIONS: tuple[float, ...] = (1.0, 1.2, 1.4, 1.5, 1.6, 2.7, 4.0)


def expansion_tag(expansion: float) -> str:
    return f"{expansion:g}".replace(".", ".")


def crop_output_dir(expansion: float, *, repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    if expansion == 1.0:
        return root / "data" / "drivers_250_fn_cropped"
    return root / f"data/drivers_250_fn_cropped_sfas_exp{expansion_tag(expansion)}"


def source_dataset_slug(expansion: float) -> str:
    if expansion == 1.0:
        return "drivers_250_fn"
    return f"drivers_250_fn_exp{expansion_tag(expansion)}"


def sample_csv_name(expansion: float) -> str:
    slug = source_dataset_slug(expansion)
    return f"{slug}_sample.csv"


def dataset_config_path(expansion: float, *, repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    slug = source_dataset_slug(expansion)
    return root / "configs" / f"dataset_{slug}.yaml"
