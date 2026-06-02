"""Hằng số SFAS bbox expansion và đường dẫn dataset (drivers_250_fn, face_antispoofing_vn)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ablation crop/benchmark tự động; baseline exp 1.0 dùng config riêng (drivers_250_fn / face_antispoofing_vn).
SFAS_BBOX_EXPANSIONS: tuple[float, ...] = (1.6, 2.7)

FACE_VN_INPUT_REL = Path("data/face_antispoofing_vn")


def expansion_tag(expansion: float) -> str:
    return f"{expansion:g}".replace(".", ".")


# --- drivers_250_fn ---


def drivers_crop_output_dir(expansion: float, *, repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    if expansion == 1.0:
        return root / "data" / "drivers_250_fn_cropped"
    return root / f"data/drivers_250_fn_cropped_sfas_exp{expansion_tag(expansion)}"


def drivers_source_dataset_slug(expansion: float) -> str:
    if expansion == 1.0:
        return "drivers_250_fn"
    return f"drivers_250_fn_exp{expansion_tag(expansion)}"


def drivers_sample_csv_name(expansion: float) -> str:
    return f"{drivers_source_dataset_slug(expansion)}_sample.csv"


def drivers_dataset_config_path(expansion: float, *, repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    slug = drivers_source_dataset_slug(expansion)
    return root / "configs" / f"dataset_{slug}.yaml"


# Aliases (drivers scripts / tests cũ)
crop_output_dir = drivers_crop_output_dir
source_dataset_slug = drivers_source_dataset_slug
sample_csv_name = drivers_sample_csv_name
dataset_config_path = drivers_dataset_config_path


# --- face_antispoofing_vn (input: data/face_antispoofing_vn, giống drivers_250_FN) ---


def face_vn_input_root(*, repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    return root / FACE_VN_INPUT_REL


def _face_vn_exp_tag(expansion: float) -> str:
    """Tag thư mục/slug (1.0 → '1.0', không rút thành '1')."""
    return "1.0" if expansion == 1.0 else expansion_tag(expansion)


def face_vn_crop_output_dir(expansion: float, *, repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    return root / f"data/face_antispoofing_vn_cropped_sfas_exp{_face_vn_exp_tag(expansion)}"


def face_vn_source_dataset_slug(expansion: float) -> str:
    if expansion == 1.0:
        return "face_antispoofing_vn"
    return f"face_antispoofing_vn_exp{_face_vn_exp_tag(expansion)}"


def face_vn_sample_csv_name(expansion: float) -> str:
    return f"{face_vn_source_dataset_slug(expansion)}_sample.csv"


def face_vn_dataset_config_path(expansion: float, *, repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    slug = face_vn_source_dataset_slug(expansion)
    return root / "configs" / f"dataset_{slug}.yaml"
