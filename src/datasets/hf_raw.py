from __future__ import annotations

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Any
from datasets import Dataset
from dotenv import load_dotenv
from .specs import HFRawDatasetSpec, DownloadResult, DATASET_SPECS

load_dotenv()

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "download_manifest.json"

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

def _iter_hf_rows(spec: HFRawDatasetSpec) -> tuple[Any, int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Cần cài package `datasets`: pip install datasets"
        ) from exc

    logger.info("Đang tải metadata từ Hugging Face: %s (split=%s)", spec.repo_id, spec.hf_split)
    
    dataset = None
    if spec.token_env_var is None:
        dataset = load_dataset(spec.repo_id, split=spec.hf_split)
    elif os.getenv(spec.token_env_var) is None:
        raise ValueError(f"Không tải được dataset từ Hugging Face vì không có token trong env var {spec.token_env_var}")
    else:
        dataset = load_dataset(spec.repo_id, split=spec.hf_split, token=os.getenv(spec.token_env_var))

    return dataset, len(dataset)

def _is_download_complete(output_dir: Path, spec: HFRawDatasetSpec, hf_dataset: Dataset) -> bool:
    manifest = _load_manifest(output_dir / "meta")
    annotation = output_dir / "annotations" / "raw.csv"
    images_dir = output_dir / "images" / spec.hf_split

    if manifest is None or not annotation.is_file() or not images_dir.is_dir():
        return False

    if not manifest.get("repo_id") == spec.repo_id:
        return False

    if not manifest.get("split") == spec.hf_split:
        return False

    if not manifest.get("num_images") == len(hf_dataset):
        return False

    return True

def _normalize_label(spec: HFRawDatasetSpec, raw_label: Any, raw_label_name: Any) -> str:
    """Chuẩn hóa nhãn HF về live / spoof."""
    for value in (raw_label_name, raw_label):
        if value is None:
            continue
        key = str(value).strip().lower()
        if key in spec.label_aliases:
            return spec.label_aliases[key]
    raise ValueError(f"Không nhận dạng được nhãn: labels={raw_label!r}, labelNames={raw_label_name!r}")


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
    spec: HFRawDatasetSpec,
    project_root: Path,
    log_every: int = 5000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {"live": 0, "spoof": 0}

    for idx, row in enumerate(rows):
        label = _normalize_label(spec=spec, raw_label=row.get(spec.label_field), raw_label_name=row.get(spec.label_name_field))
        rel_image = Path(spec.default_raw_rel_dir) / "images" / spec.hf_split / f"{idx:06d}.jpg"
        abs_image = project_root / rel_image

        is_saved, failed_reason = _save_image(row[spec.image_column], abs_image)
        
        if not is_saved:
            logger.error(failed_reason)
            label = "unknown"
        else:
            logger.info(f"Saved image {idx:06d}.jpg to {abs_image} successfully")

        records.append(
            {
                "image_path": rel_image.as_posix(),
                "label": label,
                "source_dataset": spec.source_dataset,
                "split": spec.hf_split,
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

def download_raw_dataset(
    spec: HFRawDatasetSpec,
    force: bool = False,
    log_every: int = 5000
) -> DownloadResult:
    """
    Tải sub-dataset <source_dataset> từ Hugging Face và lưu vào data/raw/<source_dataset>.

    Cấu trúc sau khi tải:
        data/raw/<source_dataset>/
          images/<hf_split>/000000.jpg ...
          annotations/raw.csv
          meta/download_manifest.json

    Args:
        spec: Cấu hình dataset từ Hugging Face.
        force: Tải lại dù đã có manifest hợp lệ.
        log_every: Tần suất log tiến độ (số ảnh).

    Returns:
        DownloadResult với đường dẫn annotation và thống kê nhãn.
    """
    project_root = find_project_root()
    out = project_root / spec.default_raw_rel_dir
    images_dir = out / "images" / spec.hf_split
    annotation_path = out / "annotations" / "raw.csv"
    meta_dir = out / "meta"
    
    hf_dataset, num_rows = _iter_hf_rows(spec=spec)

    if not force and _is_download_complete(output_dir=out, spec=spec, hf_dataset=hf_dataset):
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
        spec=spec,
        project_root=project_root,
        log_every=log_every,
    )

    _write_annotation_csv(annotation_path, records)
    _write_manifest(
        meta_dir,
        {
            "repo_id": spec.repo_id,
            "dataset_page": spec.dataset_page,
            "split": spec.hf_split,
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
    parser = argparse.ArgumentParser(description="Tải raw dataset từ Hugging Face")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_SPECS),
        help="Tên dataset (celeba_spoof | casia_fasd)",
    )
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    spec = DATASET_SPECS[args.dataset]
    result = download_raw_dataset(spec, force=args.force, log_every=args.log_every)
    print(result)

if __name__ == "__main__":
    main()