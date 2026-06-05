#!/usr/bin/env python3
"""Tải pretrained weights vào models/."""

from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

WEIGHTS = [
    (
        "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/"
        "master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
        MODELS / "2.7_80x80_MiniFASNetV2.pth",
    ),
    (
        "https://huggingface.co/ArchitRastogi/vit-spoof-detection-pda/resolve/main/"
        "best_model_run_eif1jakb.pth",
        MODELS / "vitfas_vitb16_224x224.pth",
    ),
    (
        "https://raw.githubusercontent.com/facenox/face-antispoof-onnx/main/"
        "models/best/98.20/best_model.pth",
        MODELS / "face_antispoof_onnx_best_9820.pth",
    ),
    (
        "https://github.com/hairymax/Face-AntiSpoofing/raw/main/"
        "saved_models/AntiSpoofing_bin_1.5_128.onnx",
        MODELS / "AntiSpoofing_bin_1.5_128.onnx",
    ),
]


def download(url: str, dest: Path) -> None:
    if dest.is_file():
        print(f"skip {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with dest.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def main() -> None:
    for url, dest in WEIGHTS:
        download(url, dest)


if __name__ == "__main__":
    main()
