#!/usr/bin/env python3
"""Crop mặt (SFAS) rồi push Hugging Face (train + test). Chạy: python3 scripts/create_vietnam_hf_dataset.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from datasets import ClassLabel, Dataset, Features, Image as HFImage, Value
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
SFAS_ROOT = REPO_ROOT / "third_party" / "Silent-Face-Anti-Spoofing"
INPUT_ROOT = REPO_ROOT / "data" / "face_antispoofing_vn"
OUTPUT_ROOT = REPO_ROOT / "data" / "face_antispoofing_vn_cropped"

_SPLITS = ("train_photo", "test_photo")
_LABEL_MAP = {
    "live": ("0live", "live"),
    "not_live": ("1spoof", "spoof"),
}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_DEFAULT_HF_REPO_ID = "vu-hong-quang/face_antispoofing_vn"
_PHOTO_TO_HF_SPLIT = {"train_photo": "train", "test_photo": "test"}

VIETNAM_HF_FEATURES = Features(
    {
        "cropped_image": HFImage(),
        "labels": ClassLabel(names=["0live", "1spoof"]),
        "labelNames": Value("string"),
    }
)


def _load_sfas_detection():
    """Import class Detection từ submodule gốc (cần cwd SFAS lúc load Caffe model)."""
    if not SFAS_ROOT.is_dir():
        raise FileNotFoundError(f"Thiếu submodule: {SFAS_ROOT}")

    sfas_src = str(SFAS_ROOT)
    if sfas_src not in sys.path:
        sys.path.insert(0, sfas_src)

    prev_cwd = os.getcwd()
    os.chdir(SFAS_ROOT)
    try:
        from src.anti_spoof_predict import Detection
    finally:
        os.chdir(prev_cwd)

    prev_cwd = os.getcwd()
    os.chdir(SFAS_ROOT)
    try:
        return Detection()
    finally:
        os.chdir(prev_cwd)


def iter_vietnam_image_records(
    input_root: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Bước 1: duyệt train_photo / test_photo, live / not_live."""
    root = input_root or INPUT_ROOT
    for split in _SPLITS:
        for label_dir, (labels, label_names) in _LABEL_MAP.items():
            folder = root / split / label_dir
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                yield {
                    "path": path,
                    "split": split,
                    "label_dir": label_dir,
                    "labels": labels,
                    "labelNames": label_names,
                }


def _crop_bgr_face(img: np.ndarray, bbox: list[int]) -> np.ndarray | None:
    x, y, w, h = bbox
    height, width = img.shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    patch = img[y : y + h, x : x + w]
    return patch if patch.size else None


def _crop_bgr_face_sfas_expanded(
    img: np.ndarray,
    bbox: list[int],
    bbox_expansion: float,
) -> np.ndarray | None:
    """Crop theo SFAS CropImage._get_new_box (scale = bbox expansion). Không resize."""
    if bbox_expansion == 1.0:
        return _crop_bgr_face(img, bbox)

    src_h, src_w = img.shape[:2]
    sfas_src = str(SFAS_ROOT)
    if sfas_src not in sys.path:
        sys.path.insert(0, sfas_src)

    prev_cwd = os.getcwd()
    os.chdir(SFAS_ROOT)
    try:
        from src.generate_patches import CropImage
    finally:
        os.chdir(prev_cwd)

    left_top_x, left_top_y, right_bottom_x, right_bottom_y = CropImage._get_new_box(
        src_w,
        src_h,
        bbox,
        bbox_expansion,
    )
    patch = img[left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1]
    return patch if patch.size else None


def detect_and_crop_rgb(
    image_path: Path,
    detector: Any,
    *,
    bbox_expansion: float = 1.0,
) -> Image.Image | None:
    """Bước 2: detect + crop (không resize về input MiniFASNet)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    bbox = detector.get_bbox(img)
    if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
        return None

    patch = _crop_bgr_face_sfas_expanded(img, bbox, bbox_expansion)
    if patch is None:
        return None

    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _output_path(record: dict[str, Any], output_root: Path) -> Path:
    rel = Path(record["split"]) / record["label_dir"] / record["path"].name
    return output_root / rel


def process_and_save_crops(
    *,
    input_root: Path | None = None,
    output_root: Path | None = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Chạy detect+crop và ghi JPEG RGB ra output_root (cùng cấu trúc thư mục)."""
    out = output_root or OUTPUT_ROOT
    detector = _load_sfas_detection()
    stats = {"total": 0, "ok": 0, "skip": 0, "fail": 0}
    records = list(iter_vietnam_image_records(input_root))

    for record in tqdm(records, desc="Crop", unit="img"):
        stats["total"] += 1
        dest = _output_path(record, out)
        if skip_existing and dest.is_file():
            stats["skip"] += 1
            continue

        cropped = detect_and_crop_rgb(record["path"], detector)
        if cropped is None:
            stats["fail"] += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(dest, format="JPEG", quality=95)
        stats["ok"] += 1

    return stats


def _records_for_photo_split(
    cropped_root: Path,
    photo_split: str,
) -> list[dict[str, Any]]:
    return [
        r
        for r in iter_vietnam_image_records(cropped_root)
        if r["split"] == photo_split and r["path"].is_file()
    ]


def _vietnam_hf_records_generator(records: list[dict[str, Any]]):
    for record in records:
        with Image.open(record["path"]) as img:
            cropped = img.convert("RGB") if img.mode != "RGB" else img.copy()
        yield {
            "cropped_image": cropped,
            "labels": record["labels"],
            "labelNames": record["labelNames"],
        }


def build_vietnam_hf_dataset_from_records(
    records: list[dict[str, Any]],
    *,
    features: Features = VIETNAM_HF_FEATURES,
) -> Dataset:
    return Dataset.from_generator(
        _vietnam_hf_records_generator,
        gen_kwargs={"records": records},
        features=features,
    )


def _hf_token() -> str:
    token = os.getenv("HUGGINGFACE_PRIVATE_DATASET_TOKEN")
    if not token:
        raise ValueError("Thiếu HUGGINGFACE_PRIVATE_DATASET_TOKEN trong .env")
    return token


def push_vietnam_hf_splits(
    repo_id: str,
    *,
    cropped_root: Path | None = None,
    token: str | None = None,
    photo_splits: tuple[str, ...] = _SPLITS,
    private: bool = True,
) -> dict[str, int]:
    """Bước 3: Dataset.from_generator + push_to_hub cho train / test."""
    root = cropped_root or OUTPUT_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy ảnh đã crop: {root}")

    hub_token = token or _hf_token()
    counts: dict[str, int] = {}

    for photo_split in tqdm(photo_splits, desc="Push HF", unit="split"):
        hf_split = _PHOTO_TO_HF_SPLIT[photo_split]
        records = _records_for_photo_split(root, photo_split)
        if not records:
            raise ValueError(f"Không có ảnh cho {photo_split} trong {root}")

        ds = build_vietnam_hf_dataset_from_records(records)
        ds.push_to_hub(
            repo_id,
            token=hub_token,
            split=hf_split,
            private=private,
            max_shard_size="500MB",
        )
        counts[hf_split] = len(records)
        print(f"Pushed split={hf_split} n={len(records)} -> {repo_id}")

    return counts


def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {INPUT_ROOT}")

    stats = process_and_save_crops()
    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(
        f"Crop: images={stats['total']} saved={stats['ok']} "
        f"skipped={stats['skip']} fail_detect={stats['fail']}"
    )

    repo_id = os.getenv("VIETNAM_HF_REPO_ID", _DEFAULT_HF_REPO_ID)
    counts = push_vietnam_hf_splits(repo_id)
    print(f"HF repo={repo_id} counts={counts}")


if __name__ == "__main__":
    main()
