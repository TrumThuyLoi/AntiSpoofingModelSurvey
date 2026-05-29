#!/usr/bin/env python3
"""Tạo file <dataset>_sample.csv cho celeba_spoof, casia_fasd và face_antispoofing_vn."""

from __future__ import annotations

import csv
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
OUT_ROOT = REPO_ROOT / "data" / "sampled"
FIELDNAMES = ["image_path", "label", "source_dataset", "split", "is_valid", "note"]


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


def _write_rows(dataset: str, rows: list[dict[str, str]]) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{dataset}_sample.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    return out_path


def main() -> None:
    random.seed(42)

    celeba_rows = _read_rows("celeba_spoof")
    casia_rows = _read_rows("casia_fasd")
    vn_rows = _read_rows("face_antispoofing_vn")

    celeba_sample = _sample_celeba(celeba_rows, k=2000)
    casia_sample = _sample_casia(casia_rows)
    vn_sample = _sample_face_antispoofing_vn(vn_rows)

    celeba_out = _write_rows("celeba_spoof", celeba_sample)
    casia_out = _write_rows("casia_fasd", casia_sample)
    vn_out = _write_rows("face_antispoofing_vn", vn_sample)

    print(f"[OK] celeba_spoof: {len(celeba_sample)} -> {celeba_out}")
    print(f"[OK] casia_fasd: {len(casia_sample)} -> {casia_out}")
    print(f"[OK] face_antispoofing_vn: {len(vn_sample)} -> {vn_out}")


if __name__ == "__main__":
    main()
