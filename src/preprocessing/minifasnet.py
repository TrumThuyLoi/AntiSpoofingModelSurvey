"""
Preprocessing MiniFASNet cho ảnh đã crop mặt (pre_cropped).

Bước 5.1: resize về input_size (mặc định 80×80), giống CropImage.crop(..., crop=False).
Bước 5.2: bgr_to_chw_float — khớp nhánh numpy của
           third_party/.../src/data_io/functional.to_tensor (fork Minivision):
           HWC uint8 → CHW float, giữ pixel ~0–255 (không chia 255).
           Docstring class ToTensor trong repo gốc ghi [0,1] nhưng code thực tế không /255.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import yaml

SizeHW = Tuple[int, int]
PathLike = Union[str, Path]


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs").is_dir():
        raise FileNotFoundError(f"Không tìm thấy configs/ tại {root}")
    return root


def load_input_size_from_model_config(
    config_path: PathLike | None = None,
) -> SizeHW:
    """Đọc input_size [height, width] từ configs/model.yaml."""
    path = Path(config_path) if config_path else _project_root() / "configs" / "model.yaml"
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get("input_size", [80, 80])
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"input_size không hợp lệ trong {path}: {raw!r}")
    height, width = int(raw[0]), int(raw[1])
    return height, width


def read_bgr_image(path: PathLike) -> np.ndarray:
    """Đọc ảnh BGR uint8; raise nếu không đọc được."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cv2.imread thất bại: {path}")
    return image


def resize_bgr(image: np.ndarray, input_size: SizeHW) -> np.ndarray:
    """
    Bước 5.1 — Resize ảnh BGR (pre-cropped).

    Khớp ``CropImage.crop(..., crop=False)``: ``cv2.resize(org_img, (out_w, out_h))``.
    OpenCV dùng (width, height); input_size là (height, width).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Ảnh BGR kỳ vọng H×W×3, nhận shape {image.shape}")
    height, width = input_size
    if height <= 0 or width <= 0:
        raise ValueError(f"input_size phải dương: {input_size}")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def bgr_to_chw_float(image: np.ndarray) -> torch.Tensor:
    """
    Bước 5.2 — Đổi layout BGR numpy → tensor float CHW cho MiniFASNet.

    Không chuẩn hóa về [0, 1]. Khớp ``functional.to_tensor`` (numpy) của Minivision:
    ``transpose(2,0,1)`` rồi ``.float()``; giá trị vẫn trong khoảng pixel gốc (~0–255).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Kỳ vọng numpy.ndarray, nhận {type(image)}")
    pic = image
    if pic.ndim == 2:
        pic = pic.reshape((pic.shape[0], pic.shape[1], 1))
    if pic.ndim != 3:
        raise ValueError(f"Ảnh kỳ vọng 2D hoặc 3D, nhận shape {pic.shape}")
    return torch.from_numpy(pic.transpose((2, 0, 1))).float()


def preprocess_bgr(
    image: np.ndarray,
    input_size: SizeHW = (80, 80),
) -> torch.Tensor:
    """Resize rồi ``bgr_to_chw_float``; trả tensor (C, H, W) float."""
    resized = resize_bgr(image, input_size)
    return bgr_to_chw_float(resized)


def preprocess_path(
    path: PathLike,
    input_size: SizeHW | None = None,
    *,
    config_path: PathLike | None = None,
    add_batch_dim: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Đọc ảnh từ disk → preprocess.

    Nếu ``input_size`` None, đọc từ ``configs/model.yaml`` (hoặc ``config_path``).
    ``add_batch_dim=True`` → shape (1, C, H, W).
    ``device`` — nếu set, tensor được ``.to(device)`` (vd. cuda:0).
    """
    size = input_size if input_size is not None else load_input_size_from_model_config(config_path)
    tensor = preprocess_bgr(read_bgr_image(path), size)
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def preprocess_paths(
    paths: Sequence[PathLike],
    input_size: SizeHW | None = None,
    *,
    config_path: PathLike | None = None,
    add_batch_dim: bool = False,
    device: torch.device | str | None = None,
) -> list[torch.Tensor | None]:
    """
    Preprocess danh sách ảnh; phần tử None nếu lỗi (log qua exception message nếu cần).
    """
    size = input_size if input_size is not None else load_input_size_from_model_config(config_path)
    results: list[torch.Tensor | None] = []
    for path in paths:
        try:
            tensor = preprocess_bgr(read_bgr_image(path), size)
            if add_batch_dim:
                tensor = tensor.unsqueeze(0)
            if device is not None:
                tensor = tensor.to(device)
            results.append(tensor)
        except (FileNotFoundError, ValueError):
            results.append(None)
    return results
