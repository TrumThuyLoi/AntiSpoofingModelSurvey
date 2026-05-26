"""CelebA-Spoof sub-dataset (Hugging Face) — download and raw layout."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HF_REPO_ID = "nguyenkhoa/celeba-spoof-for-face-antispoofing-test"
HF_DATASET_PAGE = (
    "https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test"
)
DEFAULT_RAW_REL_DIR = "data/raw/celeba-spoof"
SOURCE_DATASET = "celeba-spoof"
MANIFEST_FILENAME = "download_manifest.json"

LABEL_ALIASES: dict[str, str] = {
    "live": "live",
    "0live": "live",
    "spoof": "spoof",
    "1spoof": "spoof",
}


@dataclass(frozen=True)
class DownloadResult:
    """Kết quả sau khi tải raw dataset."""

    output_dir: Path
    annotation_path: Path
    num_images: int
    label_counts: dict[str, int]
    skipped: bool


def find_project_root(start: Path | None = None) -> Path:
    """Tìm root repo (thư mục chứa configs/ và src/)."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.append(Path(__file__).resolve().parents[2])
    candidates.append(Path.cwd().resolve())

    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if (root / "configs").is_dir() and (root / "src").is_dir():
            return root

    raise FileNotFoundError(
        "Không tìm thấy project root. Chạy từ thư mục gốc DVX hoặc truyền output_dir tuyệt đối."
    )


def _normalize_label(raw_label: Any, raw_label_name: Any) -> str:
    """Chuẩn hóa nhãn HF về live / spoof."""
    for value in (raw_label_name, raw_label):
        if value is None:
            continue
        key = str(value).strip().lower()
        if key in LABEL_ALIASES:
            return LABEL_ALIASES[key]
    raise ValueError(f"Không nhận dạng được nhãn: labels={raw_label!r}, labelNames={raw_label_name!r}")


def _resolve_output_dir(output_dir: Path | str | None) -> Path:
    if output_dir is None:
        root = find_project_root()
        return root / DEFAULT_RAW_REL_DIR
    path = Path(output_dir)
    if not path.is_absolute():
        path = find_project_root() / path
    return path.resolve()


def _load_manifest(meta_dir: Path) -> dict[str, Any] | None:
    manifest_path = meta_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(meta_dir: Path, payload: dict[str, Any]) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return manifest_path


def _is_download_complete(output_dir: Path, expected_rows: int | None) -> bool:
    manifest = _load_manifest(output_dir / "meta")
    annotation = output_dir / "annotations" / "raw.csv"
    images_dir = output_dir / "images"

    if manifest is None or not annotation.is_file() or not images_dir.is_dir():
        return False

    if expected_rows is not None and manifest.get("num_images") != expected_rows:
        return False

    if manifest.get("repo_id") != HF_REPO_ID:
        return False

    return True


def _iter_hf_rows(repo_id: str, split: str) -> tuple[Any, int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Cần cài package `datasets`: pip install datasets"
        ) from exc

    logger.info("Đang tải metadata từ Hugging Face: %s (split=%s)", repo_id, split)
    dataset = load_dataset(repo_id, split=split)
    return dataset, len(dataset)


def _save_image(image: Any, dest: Path) -> tuple[bool, str]:
    failed_reason = ""
    if image is None:
        failed_reason = "Image is None"
        return False, failed_reason

    if not hasattr(image, "save"):
        failed_reason = f"Image does not have save method: {type(image)}"
        return False, failed_reason

    try:
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")

        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, format="JPEG", quality=95)
        return True, failed_reason
    except Exception as e:
        failed_reason = f"Failed to save image {dest} because {e}"
        return False, failed_reason


def _export_rows(
    rows: Any,
    *,
    images_dir: Path,
    split: str,
    project_root: Path,
    log_every: int = 5000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {"live": 0, "spoof": 0}

    for idx, row in enumerate(rows):
        label = _normalize_label(row.get("labels"), row.get("labelNames"))
        rel_image = Path(DEFAULT_RAW_REL_DIR) / "images" / split / f"{idx:06d}.jpg"
        abs_image = project_root / rel_image

        is_saved, failed_reason = _save_image(row["cropped_image"], abs_image)
        
        if not is_saved:
            logger.error(failed_reason)
            label = "unknown"
        else:
            logger.info(f"Saved image {idx:06d}.jpg to {abs_image} successfully")

        records.append(
            {
                "image_path": rel_image.as_posix(),
                "label": label,
                "source_dataset": SOURCE_DATASET,
                "split": split,
                "is_valid": is_saved,
                "note": failed_reason,
            }
        )
        label_counts[label] = label_counts.get(label, 0) + 1

        if log_every > 0 and (idx + 1) % log_every == 0:
            logger.info("Đã xử lý %d ảnh...", idx + 1)

    return records, label_counts


def _write_annotation_csv(annotation_path: Path, records: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Cần cài pandas để ghi annotation CSV") from exc

    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(annotation_path, index=False)


def download_raw_dataset(
    output_dir: Path | str | None = None,
    *,
    repo_id: str = HF_REPO_ID,
    split: str = "test",
    force: bool = False,
    log_every: int = 5000,
) -> DownloadResult:
    """
    Tải sub-dataset CelebA-Spoof từ Hugging Face và lưu vào data/raw/celeba-spoof.

    Cấu trúc sau khi tải:
        data/raw/celeba-spoof/
          images/test/000000.jpg ...
          annotations/raw.csv
          meta/download_manifest.json

    Args:
        output_dir: Thư mục đích (mặc định data/raw/celeba-spoof từ project root).
        repo_id: Hugging Face dataset id.
        split: Split HF (mặc định test — ~67k ảnh).
        force: Tải lại dù đã có manifest hợp lệ.
        log_every: Tần suất log tiến độ (số ảnh).

    Returns:
        DownloadResult với đường dẫn annotation và thống kê nhãn.
    """
    project_root = find_project_root()
    out = _resolve_output_dir(output_dir)
    images_dir = out / "images" / split
    annotation_path = out / "annotations" / "raw.csv"
    meta_dir = out / "meta"

    hf_dataset, num_rows = _iter_hf_rows(repo_id, split)

    if not force and _is_download_complete(out, num_rows):
        manifest = _load_manifest(meta_dir) or {}
        logger.info("Raw dataset đã tồn tại tại %s — bỏ qua (dùng force=True để tải lại)", out)
        return DownloadResult(
            output_dir=out,
            annotation_path=annotation_path,
            num_images=int(manifest.get("num_images", num_rows)),
            label_counts={
                "live": int(manifest.get("live_count", 0)),
                "spoof": int(manifest.get("spoof_count", 0)),
            },
            skipped=True,
        )

    if force and out.exists():
        logger.warning("force=True: ghi đè ảnh/annotation hiện có trong %s", out)

    logger.info(
        "Bắt đầu export %d ảnh (~%.1f GB) — có thể mất vài phút đến hàng giờ tùy mạng",
        num_rows,
        4.95,
    )

    records, label_counts = _export_rows(
        hf_dataset,
        images_dir=images_dir,
        split=split,
        project_root=project_root,
        log_every=log_every,
    )

    _write_annotation_csv(annotation_path, records)
    _write_manifest(
        meta_dir,
        {
            "repo_id": repo_id,
            "dataset_page": HF_DATASET_PAGE,
            "split": split,
            "num_images": len(records),
            "live_count": label_counts.get("live", 0),
            "spoof_count": label_counts.get("spoof", 0),
            "annotation_path": annotation_path.relative_to(project_root).as_posix(),
            "images_dir": images_dir.relative_to(project_root).as_posix(),
        },
    )

    logger.info(
        "Hoàn tất: %d ảnh (live=%d, spoof=%d) → %s",
        len(records),
        label_counts.get("live", 0),
        label_counts.get("spoof", 0),
        out,
    )

    return DownloadResult(
        output_dir=out,
        annotation_path=annotation_path,
        num_images=len(records),
        label_counts=label_counts,
        skipped=False,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Tải CelebA-Spoof sub-dataset từ Hugging Face vào data/raw/celeba-spoof"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Thư mục đích (mặc định: {DEFAULT_RAW_REL_DIR})",
    )
    parser.add_argument("--split", default="test", help="HF split (mặc định: test)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải lại dù đã có dữ liệu",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    result = download_raw_dataset(
        args.output_dir,
        split=args.split,
        force=args.force,
    )
    print(
        f"output_dir={result.output_dir}\n"
        f"annotation={result.annotation_path}\n"
        f"num_images={result.num_images}\n"
        f"labels={result.label_counts}\n"
        f"skipped={result.skipped}"
    )


if __name__ == "__main__":
    main()
