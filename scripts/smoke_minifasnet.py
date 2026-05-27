#!/usr/bin/env python3
"""Smoke test preprocess MiniFASNet trên một ảnh (theo device trong configs/model.yaml)."""

import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.minifasnet import preprocess_path
from src.utils.device import get_torch_device_from_model_config

CELEBA_RAW_CSV = REPO_ROOT / "data/raw/celeba_spoof/annotations/raw.csv"


def sample_smoke_image_paths(
    raw_csv: Path | None = None,
    n_per_label: int = 5,
    seed: int | None = None,
) -> list[str]:
    """5 live + 5 spoof từ raw.csv (is_valid=True), trả về list image_path."""
    path = raw_csv or CELEBA_RAW_CSV
    live: list[str] = []
    spoof: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("is_valid") != "True":
                continue
            p = row["image_path"]
            if row.get("label") == "live":
                live.append(p)
            elif row.get("label") == "spoof":
                spoof.append(p)
    rng = random.Random(seed)
    k = n_per_label
    return rng.sample(live, min(k, len(live))) + rng.sample(spoof, min(k, len(spoof)))


def main() -> int:
    device = get_torch_device_from_model_config()
    print(f"device: {device}")

    image_paths = sample_smoke_image_paths(seed=534)
    for img_path in image_paths:
        image_path = Path(img_path)
        if not image_path.is_file():
            print(f"FAIL  không tìm thấy ảnh: {image_path}")
            return 1

        t = preprocess_path(image_path, add_batch_dim=True, device=device)
        print(f"tensor: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
        print(f"min={t.min().item():.1f} max={t.max().item():.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
