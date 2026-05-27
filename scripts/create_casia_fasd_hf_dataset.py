"""Chuẩn bị dataset CASIA-FASD trên Hugging Face (WIP)."""
from __future__ import annotations

import os
import requests
import PIL.Image
from datasets import ClassLabel, Dataset, Features, Image, Value, load_dataset
from dotenv import load_dotenv
from pathlib import Path
from typing import Any

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

_COLOR_IMAGE_DIRS = (
    "data/raw/kaggle/casia_fasd/test_img/color",
    "data/raw/kaggle/casia_fasd/train_img/color",
)
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})

CASIA_FASD_FEATURES = Features(
    {
        "cropped_image": Image(),
        "labels": ClassLabel(names=["0live", "1spoof"]),
        "labelNames": Value("string"),
    }
)


def get_casia_fasd_color_image_records() -> list[dict[str, Any]]:
    """Thu thập metadata ảnh từ test_img/color và train_img/color."""
    list_image_records = []
    project_root = Path(__file__).resolve().parents[1]
    for rel_dir in _COLOR_IMAGE_DIRS:
        directory = project_root / rel_dir
        if not directory.is_dir():
            continue

        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                labels = "unknown"
                lableNames = "unknown"
                if "_real." in path.name:
                    labels = "0live"
                    lableNames = "live"
                elif "_fake." in path.name:
                    labels = "1spoof"
                    lableNames = "spoof"

                list_image_records.append(
                    {
                        "path": path,
                        "labels": labels,
                        "lableNames": lableNames,
                    }
                )

    return list_image_records


def _casia_fasd_records_generator(records: list[dict[str, Any]]):
    for record in records:
        if record["labels"] not in ("0live", "1spoof"):
            continue
        with PIL.Image.open(record["path"]) as img:
            cropped = img.convert("RGB") if img.mode != "RGB" else img.copy()
        yield {
            "cropped_image": cropped,
            "labels": record["labels"],
            "labelNames": record["lableNames"],
        }


def build_casia_fasd_dataset_from_records(
    records: list[dict[str, Any]],
    *,
    features: Features = CASIA_FASD_FEATURES,
) -> Dataset:
    """Tạo Hugging Face Dataset từ list records (path, labels, label_names)."""
    return Dataset.from_generator(
        _casia_fasd_records_generator,
        gen_kwargs={"records": records},
        features=features,
    )

def verify_casia_fasd_test_split() -> Dataset:
    if not HUGGINGFACE_TOKEN:
        raise ValueError("HUGGINGFACE_TOKEN is not set")
        
    ds = load_dataset(
        "vu-hong-quang/casia_fasd",
        split="test",
        token=HUGGINGFACE_TOKEN,
    )
    return ds

if __name__ == "__main__":
    test_split = verify_casia_fasd_test_split()
    print(test_split, test_split.features, len(test_split))
    if len(test_split) >= 4063:
        print("Test split already exists!")
        exit(0)

    records = get_casia_fasd_color_image_records()
    ds = build_casia_fasd_dataset_from_records(records)
    
    if not HUGGINGFACE_TOKEN:
        raise ValueError("HUGGINGFACE_TOKEN is not set")
    
    ds.push_to_hub(
        "vu-hong-quang/casia_fasd",
        token=HUGGINGFACE_TOKEN,
        split="test",
        private=True,
        max_shard_size="500MB",
    )
