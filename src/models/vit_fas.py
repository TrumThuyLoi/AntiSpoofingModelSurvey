"""Minimal ViT-FAS wrapper with unified inference interface.

This wrapper is intentionally small-scope:
- no extra dependencies
- no environment changes
- interface-compatible: load / predict / predict_batch
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class ViTFASWrapper:
    """Wrapper thống nhất cho ViT-FAS inference.

    Notes:
    - Ưu tiên load checkpoint dạng TorchScript (`torch.jit.load`).
    - Nếu không phải TorchScript, thử `torch.load`:
      - nếu trả về `nn.Module` -> dùng trực tiếp
      - nếu trả về dict có key `model` là `nn.Module` -> dùng model đó
    """

    def __init__(
        self,
        weights_path: str | Path,
        *,
        input_size: tuple[int, int] = (224, 224),
        threshold: float = 0.5,
        prefer_cpu: bool = True,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.weights_path = self._resolve_weights_path(weights_path)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.threshold = float(threshold)
        self.device = torch.device(
            "cpu" if prefer_cpu or not torch.cuda.is_available() else "cuda:0"
        )

        self.model: nn.Module | None = None
        self._loaded = False

    def _resolve_weights_path(self, value: str | Path) -> Path:
        p = Path(value)
        if not p.is_absolute():
            p = self.repo_root / p
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy weight file: {p}")
        return p

    def _load_model_from_checkpoint(self) -> nn.Module:
        # Flow giống repo gốc: ViTFaceAntiSpoofing(TestConfig) + checkpoint['model_state_dict'].
        repo_test_py = self.repo_root / "third_party" / "vit-spoof-detection-pda" / "test.py"
        if not repo_test_py.is_file():
            raise FileNotFoundError(f"Không tìm thấy file model repo gốc: {repo_test_py}")

        spec = importlib.util.spec_from_file_location("vit_spoof_detection_test", repo_test_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Không thể load module từ {repo_test_py}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model_cls = getattr(mod, "ViTFaceAntiSpoofing", None)
        config_cls = getattr(mod, "TestConfig", None)
        if model_cls is None or config_cls is None:
            raise ImportError("Không tìm thấy ViTFaceAntiSpoofing/TestConfig trong repo gốc.")

        repo_cfg = config_cls()
        repo_cfg.device = str(self.device)
        repo_cfg.num_classes = 2
        repo_cfg.img_size = int(self.input_size[0])
        model = model_cls(repo_cfg)

        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint phải là dict theo format repo gốc.")

        if isinstance(checkpoint.get("model_state_dict"), dict):
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint.get("state_dict"), dict):
            state_dict = checkpoint["state_dict"]
        elif checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise ValueError("Checkpoint không có model_state_dict/state_dict hợp lệ.")

        cleaned_state_dict: dict[str, Any] = {}
        for k, v in state_dict.items():
            new_key = k[7:] if isinstance(k, str) and k.startswith("module.") else k
            cleaned_state_dict[new_key] = v

        load_result = model.load_state_dict(cleaned_state_dict, strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
        logger.warning(
            "ViT checkpoint load summary: missing_keys=%d, unexpected_keys=%d",
            len(missing_keys),
            len(unexpected_keys),
        )
        if missing_keys:
            logger.warning("Missing keys (sample): %s", missing_keys[:10])
        if unexpected_keys:
            logger.warning("Unexpected keys (sample): %s", unexpected_keys[:10])
        model.eval()
        return model

        raise ValueError(
            "Checkpoint không ở dạng hỗ trợ cho luồng ViT-FAS gốc."
        )

    def load(self) -> None:
        if self._loaded:
            return
        self.model = self._load_model_from_checkpoint().to(self.device)
        self._loaded = True

    def preprocess(self, image_or_path: str | Path | np.ndarray) -> torch.Tensor:
        if isinstance(image_or_path, (str, Path)):
            image = Image.open(image_or_path).convert("RGB")
        elif isinstance(image_or_path, np.ndarray):
            if image_or_path.ndim != 3 or image_or_path.shape[2] != 3:
                raise ValueError("np.ndarray input phải có shape [H, W, 3].")
            image = Image.fromarray(image_or_path.astype(np.uint8), mode="RGB")
        else:
            raise TypeError(f"Kiểu input không hỗ trợ: {type(image_or_path)}")

        image = image.resize(self.input_size, Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor

    def _decode_logits(self, logits: torch.Tensor) -> tuple[float, float]:
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        if probs.shape[0] < 2:
            raise ValueError("Output model phải có ít nhất 2 class (spoof/live).")

        # Quy ước: class 1 = live, class 0 = spoof (2 class đầu tiên).
        live_score = float(probs[1])
        spoof_score = float(probs[0])
        return live_score, spoof_score

    def predict(self, image_or_path: str | Path | np.ndarray) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        assert self.model is not None

        batch = self.preprocess(image_or_path)
        with torch.no_grad():
            output = self.model(batch)

        if isinstance(output, dict) and "logits" in output:
            logits = output["logits"]
        elif hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Output model không phải torch.Tensor.")

        live_score, spoof_score = self._decode_logits(logits)
        label_pred = "live" if live_score >= self.threshold else "spoof"
        return {
            "label_pred": label_pred,
            "live_score": live_score,
            "spoof_score": spoof_score,
            "raw_output": logits.detach().cpu().numpy()[0].tolist(),
        }

    def predict_batch(self, images_or_paths: list[str | Path | np.ndarray]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for item in images_or_paths:
            outputs.append(self.predict(item))
        return outputs

