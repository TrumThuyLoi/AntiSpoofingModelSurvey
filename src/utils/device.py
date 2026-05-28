"""Resolve torch.device từ configs/model_minifasnet.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import torch
import yaml

PathLike = Union[str, Path]


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs").is_dir():
        raise FileNotFoundError(f"Không tìm thấy configs/ tại {root}")
    return root


def load_model_config(config_path: PathLike | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else _project_root() / "configs" / "model_minifasnet.yaml"
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root YAML phải là dict")
    return data


def resolve_torch_device(
    device_spec: str,
    *,
    cuda_index: int = 0,
    fail_if_cuda_unavailable: bool = True,
) -> torch.device:
    """
    Map config ``device`` → ``torch.device``.

    - ``gpu`` / ``cuda`` → ``cuda:{cuda_index}`` nếu CUDA khả dụng
    - ``cpu`` → ``cpu``
    """
    spec = device_spec.strip().lower()
    if spec in ("gpu", "cuda"):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{cuda_index}")
        msg = (
            "configs/model_minifasnet.yaml yêu cầu GPU nhưng torch.cuda.is_available()=False. "
            "Kiểm tra driver NVIDIA trên Windows, WSL2 GPU, và torch bản CUDA."
        )
        if fail_if_cuda_unavailable:
            raise RuntimeError(msg)
        return torch.device("cpu")
    if spec == "cpu":
        return torch.device("cpu")
    raise ValueError(f"device không hỗ trợ: {device_spec!r} (dùng cpu, gpu, hoặc cuda)")


def get_torch_device_from_model_config(
    config_path: PathLike | None = None,
    *,
    cuda_index: int = 0,
    fail_if_cuda_unavailable: bool = True,
) -> torch.device:
    cfg = load_model_config(config_path)
    spec = str(cfg.get("device", "cpu"))
    return resolve_torch_device(
        spec,
        cuda_index=cuda_index,
        fail_if_cuda_unavailable=fail_if_cuda_unavailable,
    )
