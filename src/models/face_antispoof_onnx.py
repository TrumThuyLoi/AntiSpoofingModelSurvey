"""Thin wrapper around face-antispoof-onnx PyTorch checkpoint.

This wrapper intentionally reuses the upstream model definition from:
`third_party/face-antispoof-onnx/src/minifasv2/model.py`
to keep behavior as close to the original repository as possible.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


class FaceAntispoofONNXWrapper:
    """Minimal interface-compatible wrapper: load / predict / predict_batch."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        input_size: tuple[int, int] = (128, 128),
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

        self.model: torch.nn.Module | None = None
        self._loaded = False

    def _resolve_weights_path(self, value: str | Path) -> Path:
        p = Path(value)
        if not p.is_absolute():
            p = self.repo_root / p
        if not p.is_file():
            raise FileNotFoundError(f"Khong tim thay weight file: {p}")
        return p

    def _load_upstream_model_class(self):
        model_py = (
            self.repo_root
            / "third_party"
            / "face-antispoof-onnx"
            / "src"
            / "minifasv2"
            / "model.py"
        )
        if not model_py.is_file():
            raise FileNotFoundError(f"Khong tim thay model.py cua repo goc: {model_py}")

        spec = importlib.util.spec_from_file_location("fas_onnx_minifasv2_model", model_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Khong the load module tu {model_py}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_cls = getattr(module, "MultiFTNet", None)
        if model_cls is None:
            raise ImportError("Khong tim thay class MultiFTNet trong repo goc.")
        return model_cls

    @staticmethod
    def _kernel_from_input_size(input_size: int) -> tuple[int, int]:
        # Same formula as upstream: src/minifasv2/config.py:get_kernel
        k = (int(input_size) + 15) // 16
        return (k, k)

    def load(self) -> None:
        if self._loaded:
            return

        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint khong dung format dict.")

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint thieu model_state_dict.")

        # Match architecture to training config in checkpoint.
        ckpt_cfg = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
        conv6_weight = state_dict.get("model.conv_6_dw.conv.weight")
        if isinstance(conv6_weight, torch.Tensor) and conv6_weight.ndim == 4:
            conv6_kernel = (int(conv6_weight.shape[-2]), int(conv6_weight.shape[-1]))
        else:
            ckpt_input_size = int(ckpt_cfg.get("input_size", self.input_size[0]))
            conv6_kernel = self._kernel_from_input_size(ckpt_input_size)

        model_cls = self._load_upstream_model_class()
        model = model_cls(num_classes=2, conv6_kernel=conv6_kernel)

        load_result = model.load_state_dict(state_dict, strict=True)
        if load_result is not None:
            # load_state_dict may return IncompatibleKeys in some torch versions.
            missing = list(getattr(load_result, "missing_keys", []))
            unexpected = list(getattr(load_result, "unexpected_keys", []))
            if missing or unexpected:
                raise ValueError(
                    f"Khong khop checkpoint: missing={len(missing)}, unexpected={len(unexpected)}"
                )

        self.model = model.to(self.device).eval()
        self._loaded = True

    @staticmethod
    def _letterbox_reflect_rgb(img: np.ndarray, target_size: int) -> np.ndarray:
        """Mirror the upstream preprocess (resize + reflect pad + /255 + CHW)."""
        old_h, old_w = img.shape[:2]
        ratio = float(target_size) / max(old_h, old_w)
        scaled_h, scaled_w = int(old_h * ratio), int(old_w * ratio)
        interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, (scaled_w, scaled_h), interpolation=interpolation)

        delta_w = target_size - scaled_w
        delta_h = target_size - scaled_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0

    def preprocess(self, image_or_path: str | Path | np.ndarray) -> torch.Tensor:
        if isinstance(image_or_path, (str, Path)):
            bgr = cv2.imread(str(image_or_path))
            if bgr is None:
                raise ValueError(f"cv2.imread that bai: {image_or_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_or_path, np.ndarray):
            if image_or_path.ndim != 3 or image_or_path.shape[2] != 3:
                raise ValueError("np.ndarray input phai co shape [H, W, 3].")
            rgb = image_or_path.astype(np.uint8)
        else:
            raise TypeError(f"Kieu input khong ho tro: {type(image_or_path)}")

        size = int(self.input_size[0])
        arr = self._letterbox_reflect_rgb(rgb, size)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def predict(self, image_or_path: str | Path | np.ndarray) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        assert self.model is not None

        batch = self.preprocess(image_or_path)
        with torch.no_grad():
            logits = self.model(batch)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if not isinstance(logits, torch.Tensor):
                raise TypeError("Output model khong phai tensor.")
            logits_row = logits[0]
            probs_row = torch.softmax(logits_row, dim=0)
            logits_np = logits_row.detach().cpu().numpy()
            probs_np = probs_row.detach().cpu().numpy()

        # Follow upstream logic in src/inference/inference.py:
        # class 0 = real/live, class 1 = spoof.
        logit_live = float(logits_np[0])
        logit_spoof = float(logits_np[1])
        live_score = float(probs_np[0])
        spoof_score = float(probs_np[1])
        is_live = (logit_live - logit_spoof) >= self.threshold
        label_pred = "live" if is_live else "spoof"

        return {
            "label_pred": label_pred,
            "live_score": live_score,
            "spoof_score": spoof_score,
            "raw_output": logits_np.tolist(),
        }

    def predict_batch(self, images_or_paths: list[str | Path | np.ndarray]) -> list[dict[str, Any]]:
        return [self.predict(item) for item in images_or_paths]

